"""
Evaluate all models and print a comparison table.

Usage
-----
uv run python scripts/evaluate.py
uv run python scripts/evaluate.py --config configs/default.yaml --sample 20000
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml
from loguru import logger

from rec_sys.baselines import (
    AgeSegmentedPopularityRecommender,
    BaselineConfig,
    GlobalPopularityRecommender,
    RecentPopularityRecommender,
    RepurchaseRecommender,
)
from rec_sys.cf_model import CFConfig, ItemCFRecommender
from rec_sys.data_utils import canonical_split
from rec_sys.model import ModelConfig, TwoStageLGBMRanker, load_data
from rec_sys.baselines import top_popular_articles
import polars as pl
# ─────────────────────────────────────────────────────────────
# Args
# ─────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate all recommendation models")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--sample", type=int, default=50_000)
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────


def main() -> None:
    args = parse_args()

    with open(args.config) as f:
        raw = yaml.safe_load(f)

    data_dir = Path(raw.get("data", {}).get("data_dir", "data"))
    sample = args.sample

    logger.info("Loading data …")

    tx, customers, articles = load_data(ModelConfig(data_dir=data_dir))

    # ─────────────────────────────────────────────
    # Unified split (single source of truth)
    # ─────────────────────────────────────────────

    train_full, val_tx, test_tx, val_gt, test_gt = canonical_split(tx)

    logger.info(f"Split ready — train={len(train_full):,}, test_users={len(test_gt):,}")

    results: dict[str, float] = {}

    # ─────────────────────────────────────────────
    # Baselines
    # ─────────────────────────────────────────────

    bl_cfg = BaselineConfig(**raw.get("baselines", {}))
    global_top = top_popular_articles(
        pl.from_pandas(train_full),
        n=bl_cfg.popularity_top_n,
    )

    baselines = [
        GlobalPopularityRecommender(global_top),
        RecentPopularityRecommender(recent_weeks=bl_cfg.recent_weeks),
        RepurchaseRecommender(),
        AgeSegmentedPopularityRecommender(
            age_bins=bl_cfg.age_bins,
            age_labels=bl_cfg.age_labels,
        ),
    ]

    for model in baselines:
        model.fit(train_full)

        results[model.name] = model.evaluate(
            test_gt,
            k=bl_cfg.k,
            sample=sample,
        )

    # ─────────────────────────────────────────────
    # Collaborative Filtering
    # ─────────────────────────────────────────────

    cf_cfg = CFConfig(**raw.get("cf", {}))

    cf = ItemCFRecommender(cf_cfg)
    cf.fit(train_full)

    results["item_cf"] = cf.evaluate(
        test_gt,
        k=cf_cfg.k,
        sample=sample,
    )

    # ─────────────────────────────────────────────
    # Two-stage LGBM
    # ─────────────────────────────────────────────

    m_cfg = ModelConfig(**raw.get("model", {}))
    m_cfg.data_dir = data_dir

    lgbm = TwoStageLGBMRanker(m_cfg)
    lgbm.fit(train_full)

    results["two_stage_lgbm"] = lgbm.evaluate(
        test_gt,
        prediction_date=test_tx["t_dat"].min(),
        k=m_cfg.k,
        sample=sample,
    )

    # ─────────────────────────────────────────────
    # Report
    # ─────────────────────────────────────────────

    logger.info("\n" + "=" * 52)
    logger.info(f"{'Model':<40} {'MAP@K':>8}")
    logger.info("=" * 52)

    for name, score in sorted(results.items(), key=lambda x: -x[1]):
        marker = " ◄" if name == "two_stage_lgbm" else ""
        logger.info(f"{name:<40} {score:>8.6f}{marker}")

    logger.info("=" * 52)

    # Save results to file
    import json
    output_path = Path("outputs/evaluation_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved evaluation results → {output_path}")


if __name__ == "__main__":
    main()
