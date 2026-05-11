"""Train the two-stage LightGBM ranker and save the model.

Usage
-----
    uv run python scripts/train.py
    uv run python scripts/train.py --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import yaml
from loguru import logger

from rec_sys.model import ModelConfig, TwoStageLGBMRanker, load_data
from rec_sys.data_utils import make_splits_lgbm as make_splits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Two-Stage LGBM Ranker")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--output", type=str, default="outputs/model.pkl")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    with open(args.config) as f:
        raw = yaml.safe_load(f)

    cfg = ModelConfig(**raw.get("model", {}))
    cfg.data_dir = Path(raw.get("data", {}).get("data_dir", "data"))

    tx, customers, articles = load_data(cfg)

    train_full, val_tx, test_tx, val_gt, test_gt = make_splits(tx)

    model = TwoStageLGBMRanker(cfg)
    model.fit(train_full)

    prediction_date = test_tx["t_dat"].min()

    score = model.evaluate(
        test_gt=test_gt,
        prediction_date=prediction_date,
        k=cfg.k,
    )
    logger.info(f"Test MAP@{cfg.k} = {score:.6f}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "lgbm_model": model.model,
        "cfg": model.cfg,
    }
    with open(output_path, "wb") as f:
        pickle.dump(payload, f)
    logger.info(f"Saved model payload → {output_path} ({output_path.stat().st_size / 1e6:.1f} MB)")

    try:
        logger.info("Extracting feature importance...")
        lgbm_obj = model.model

        if hasattr(lgbm_obj, "feature_name_"):
            feats = lgbm_obj.feature_name_
            imps = lgbm_obj.feature_importances_
        else:
            feats = lgbm_obj.feature_name()
            imps = lgbm_obj.feature_importance(importance_type="gain")

        importance_df = pd.DataFrame(
            {"feature": feats, "importance": imps}
        ).sort_values(by="importance", ascending=False)

        # Tạo thư mục figures
        fig_dir = Path("figures")
        fig_dir.mkdir(exist_ok=True)

        # Vẽ biểu đồ
        plt.figure(figsize=(10, 8))
        top_df = importance_df.head(20)

        plt.barh(top_df["feature"], top_df["importance"])
        plt.gca().invert_yaxis()
        plt.title(f"Top 20 Feature Importance - MAP: {score:.4f}")
        plt.xlabel("Importance Score")
        plt.ylabel("Features")
        plt.tight_layout()

        # Lưu ảnh
        fig_path = fig_dir / "feature_importance.png"
        plt.savefig(fig_path)
        logger.info(f"Feature importance plot saved → {fig_path}")

    except Exception as e:
        # Nếu có lỗi khi vẽ ảnh, logger sẽ báo nhưng model đã được lưu an toàn ở Bước 1
        logger.error(f"Failed to generate feature importance plot: {e}")


if __name__ == "__main__":
    main()
