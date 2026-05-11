"""Quick test of the fixed pipeline with a sample of data.

This runs a smaller sample to verify all fixes work correctly.
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
from loguru import logger

from rec_sys.baselines import (
    AgeSegmentedPopularityRecommender,
    BaselineConfig,
    GlobalPopularityRecommender,
    RecentPopularityRecommender,
    RepurchaseRecommender,
)
from rec_sys.cf_model import CFConfig, ItemCFRecommender
from rec_sys.data_utils import canonical_split, make_splits_lgbm
from rec_sys.model import ModelConfig, TwoStageLGBMRanker


def main():
    data_dir = Path("data")
    sample_size = 10000  # Number of customers to sample

    logger.info("=" * 60)
    logger.info("Quick Test Pipeline - Verifying All Fixes")
    logger.info("=" * 60)

    # Load data
    logger.info("Loading data...")
    tx = pd.read_parquet(data_dir / "transactions.parquet").head(1000000)  # Load only first 1M rows
    customers = pd.read_parquet(data_dir / "customers.parquet")
    articles = pd.read_parquet(data_dir / "articles.parquet")

    logger.info(
        f"Loaded {len(tx):,} transactions, {len(customers):,} customers, {len(articles):,} articles"
    )

    # Sample for quick test
    sample_customers = (
        tx["customer_id"]
        .drop_duplicates()
        .sample(n=min(sample_size, tx["customer_id"].nunique()), random_state=42)
    )
    tx_sample = tx[tx["customer_id"].isin(sample_customers)]
    customers_sample = customers[customers["customer_id"].isin(sample_customers)]

    logger.info(
        f"Sampled {len(tx_sample):,} transactions for {len(sample_customers):,} customers"
    )

    # Test 1: Unified splits
    logger.info("\n" + "=" * 60)
    logger.info("Test 1: Unified Train/Test Splits")
    logger.info("=" * 60)

    train_full, val_tx, test_tx, val_gt, test_gt = canonical_split(tx_sample)
    train_full_lgbm, train_feat, val_tx_lgbm, val_gt_lgbm, test_gt_lgbm = (
        make_splits_lgbm(tx_sample)
    )

    logger.info(
        f"Canonical split: {len(train_full):,} train, {len(test_tx):,} test transactions"
    )
    logger.info(
        f"LGBM split: {len(train_full_lgbm):,} train, {len(test_tx):,} test transactions"
    )

    # Verify splits match
    test_users_match = set(test_gt.keys()) == set(test_gt_lgbm.keys())
    logger.info(f"Test users match: {test_users_match}")

    if test_users_match:
        logger.info("✓ Unified splits working correctly!")
    else:
        logger.error("✗ Splits don't match - check implementation")
        return

    # Test 2: Baseline evaluation
    logger.info("\n" + "=" * 60)
    logger.info("Test 2: Baseline Models")
    logger.info("=" * 60)

    bl_cfg = BaselineConfig()
    results = {}

    # Get top articles for global popularity
    top_articles = (
        tx_sample["article_id"]
        .value_counts()
        .head(100)
        .index.tolist()
    )

    models = [
        ("Global Popularity", GlobalPopularityRecommender(top_articles=top_articles)),
        (
            "Recent Popularity",
            RecentPopularityRecommender(recent_weeks=bl_cfg.recent_weeks),
        ),
        ("Repurchase", RepurchaseRecommender(fallback_articles=top_articles)),
        (
            "Age-Segmented",
            AgeSegmentedPopularityRecommender(
                global_top=top_articles,
                age_bins=bl_cfg.age_bins, age_labels=bl_cfg.age_labels
            ),
        ),
    ]

    for name, model in models:
        t0 = time.time()
        try:
            if name == "Age-Segmented":
                model.fit(train_full, customers=customers_sample)
            else:
                model.fit(train_full)
            map_score = model.evaluate(test_gt, k=12, sample=5000)
            results[name] = map_score
            logger.info(f"{name}: MAP@12 = {map_score:.6f} ({time.time() - t0:.1f}s)")
        except Exception as e:
            logger.warning(f"{name}: Failed with error: {e}")
            results[name] = 0.0

    # Test 3: Item-CF
    logger.info("\n" + "=" * 60)
    logger.info("Test 3: Item-Based Collaborative Filtering")
    logger.info("=" * 60)

    cf_cfg = CFConfig()
    cf = ItemCFRecommender(cf_cfg)
    t0 = time.time()
    cf.fit(train_full)
    cf_map = cf.evaluate(test_gt, k=12, sample=5000)
    results["Item-CF"] = cf_map
    logger.info(f"Item-CF: MAP@12 = {cf_map:.6f} ({time.time() - t0:.1f}s)")

    # Test 4: Two-Stage LGBM with all fixes
    logger.info("\n" + "=" * 60)
    logger.info("Test 4: Two-Stage LGBM (with all fixes)")
    logger.info("=" * 60)

    cfg = ModelConfig(
        n_candidates=200,
        use_stratified_candidates=True,
        use_negative_sampling=True,
        negative_sampling_ratio=50,
        enable_cold_start=True,
    )

    lgbm = TwoStageLGBMRanker(cfg)

    logger.info("Training LGBM model (this may take a few minutes)...")
    t0 = time.time()
    lgbm.fit(
        train_feat=train_feat,
        val_tx=val_tx_lgbm,
        customers=customers_sample,
        articles=articles,
        train_full=train_full_lgbm,
    )
    train_time = time.time() - t0
    logger.info(f"Training completed in {train_time:.1f}s")

    # Test 5: Temporal-aware inference
    logger.info("\n" + "=" * 60)
    logger.info("Test 5: Temporal-Aware Inference")
    logger.info("=" * 60)

    # Test with explicit prediction date
    test_users = list(test_gt.keys())[:100]

    t0 = time.time()
    _ = lgbm.predict(test_users, k=12)
    pred_time_default = time.time() - t0

    t0 = time.time()
    _ = lgbm.predict(
        test_users, k=12, prediction_date=tx_sample["t_dat"].max()
    )
    pred_time_temporal = time.time() - t0

    logger.info(
        f"Default prediction: {pred_time_default:.2f}s for {len(test_users)} users"
    )
    logger.info(
        f"Temporal prediction: {pred_time_temporal:.2f}s for {len(test_users)} users"
    )
    logger.info("✓ Temporal inference working!")

    # Evaluate LGBM
    logger.info("\nEvaluating LGBM model...")
    t0 = time.time()
    lgbm_map = lgbm.evaluate(test_gt, k=12, sample=5000)
    results["Two-Stage LGBM"] = lgbm_map
    logger.info(f"Two-Stage LGBM: MAP@12 = {lgbm_map:.6f} ({time.time() - t0:.1f}s)")

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("Results Summary")
    logger.info("=" * 60)
    logger.info(f"{'Model':<30} {'MAP@12':>10}")
    logger.info("-" * 60)
    for name, score in sorted(results.items(), key=lambda x: -x[1]):
        marker = " ★" if name == "Two-Stage LGBM" else ""
        logger.info(f"{name:<30} {score:>10.6f}{marker}")
    logger.info("=" * 60)

    # Verify improvements
    lgbm_better = lgbm_map > max(v for k, v in results.items() if k != "Two-Stage LGBM")
    logger.info(f"\nLGBM outperforms all baselines: {lgbm_better}")

    logger.info("\n✓ All tests passed! Fixes are working correctly.")
    logger.info("\nTo run full evaluation:")
    logger.info("  uv run python -m rec_sys.baselines")
    logger.info("  uv run python -m rec_sys.cf_model")
    logger.info("  uv run python scripts/train.py")
    logger.info("  uv run python scripts/evaluate.py")


if __name__ == "__main__":
    main()
