"""Unified data splitting utilities for consistent train/validation/test splits.

Designed for fashion recommendation systems with strong seasonality:
- Uses recent history only (default: 2020-03-01 onward)
- Prevents stale fashion trends from polluting training
- Reduces RAM + training time significantly
- Keeps all models aligned on identical temporal splits
"""

from __future__ import annotations

import pandas as pd
from loguru import logger


# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

# Dataset timeline:
# 2018-09-20 → 2020-09-22

DEFAULT_TEST_DAYS = 7
DEFAULT_VAL_DAYS = 7

# Fashion is highly seasonal.
# Using only recent months improves both realism and speed.
TRAIN_START = pd.Timestamp("2020-03-01")


# ──────────────────────────────────────────────────────────────────────────────
# Internal Helpers
# ──────────────────────────────────────────────────────────────────────────────


def _build_ground_truth(
    tx: pd.DataFrame,
) -> dict[str, set[str]]:
    """Convert transactions → {customer_id: {article_ids}}."""

    return tx.groupby("customer_id")["article_id"].apply(set).to_dict()


def _get_split_dates(
    transactions: pd.DataFrame,
    test_days: int,
    val_days: int,
) -> tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]:

    last = transactions["t_dat"].max()

    test_start = last - pd.Timedelta(days=test_days - 1)

    val_start = test_start - pd.Timedelta(days=val_days)

    return last, val_start, test_start


# ──────────────────────────────────────────────────────────────────────────────
# Canonical Split
# ──────────────────────────────────────────────────────────────────────────────


def canonical_split(
    transactions: pd.DataFrame,
    test_days: int = DEFAULT_TEST_DAYS,
    val_days: int = DEFAULT_VAL_DAYS,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    dict[str, set[str]],
    dict[str, set[str]],
]:
    """Single source of truth for all model splits.

    Returns:
        train_full:
            2020-03-01 → before test week

        val_tx:
            validation week transactions

        test_tx:
            final evaluation week

        val_gt:
            validation ground truth

        test_gt:
            test ground truth
    """

    last, val_start, test_start = _get_split_dates(
        transactions,
        test_days,
        val_days,
    )

    train_mask = (transactions["t_dat"] >= TRAIN_START) & (
        transactions["t_dat"] < test_start
    )

    val_mask = (transactions["t_dat"] >= val_start) & (
        transactions["t_dat"] < test_start
    )

    test_mask = transactions["t_dat"] >= test_start

    train_full = transactions.loc[train_mask]
    val_tx = transactions.loc[val_mask]
    test_tx = transactions.loc[test_mask]

    val_tx = transactions[
        (transactions["t_dat"] >= val_start) & (transactions["t_dat"] < test_start)
    ].copy()

    test_tx = transactions[transactions["t_dat"] >= test_start].copy()

    val_gt = _build_ground_truth(val_tx)

    test_gt = _build_ground_truth(test_tx)

    logger.info(
        f"Split — "
        f"train_full={len(train_full):,} rows "
        f"({train_full['t_dat'].min().date()} → "
        f"{train_full['t_dat'].max().date()}), "
        f"val={len(val_tx):,} rows "
        f"({val_start.date()} → "
        f"{(test_start - pd.Timedelta(days=1)).date()}), "
        f"test={len(test_tx):,} rows "
        f"({test_start.date()} → {last.date()}), "
        f"val_users={len(val_gt):,}, "
        f"test_users={len(test_gt):,}"
    )

    return (
        train_full,
        val_tx,
        test_tx,
        val_gt,
        test_gt,
    )


# ──────────────────────────────────────────────────────────────────────────────
# LGBM Train Feature Split
# ──────────────────────────────────────────────────────────────────────────────


def get_lgbm_train_feat_split(
    transactions: pd.DataFrame,
    val_days: int = DEFAULT_VAL_DAYS,
    test_days: int = DEFAULT_TEST_DAYS,
) -> pd.DataFrame:
    """Feature-generation history for validation week."""

    _, val_start, _ = _get_split_dates(
        transactions,
        test_days,
        val_days,
    )

    train_feat = transactions[
        (transactions["t_dat"] >= TRAIN_START) & (transactions["t_dat"] < val_start)
    ].copy()

    return train_feat


# ──────────────────────────────────────────────────────────────────────────────
# LGBM Splits
# ──────────────────────────────────────────────────────────────────────────────


def make_splits_lgbm(
    transactions: pd.DataFrame,
    test_days: int = DEFAULT_TEST_DAYS,
    val_days: int = DEFAULT_VAL_DAYS,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    dict[str, set[str]],
    dict[str, set[str]],
]:
    """Create splits for LGBM training.

    Returns:
        train_full:
            history before test week

        train_feat:
            history before validation week

        val_tx:
            validation transactions

        val_gt:
            validation labels

        test_gt:
            final test labels
    """

    last, val_start, test_start = _get_split_dates(
        transactions,
        test_days,
        val_days,
    )

    train_feat = transactions[
        (transactions["t_dat"] >= TRAIN_START) & (transactions["t_dat"] < val_start)
    ].copy()

    train_mask = (transactions["t_dat"] >= TRAIN_START) & (
        transactions["t_dat"] < test_start
    )

    val_mask = (transactions["t_dat"] >= val_start) & (
        transactions["t_dat"] < test_start
    )

    test_mask = transactions["t_dat"] >= test_start

    train_full = transactions.loc[train_mask]
    val_tx = transactions.loc[val_mask]
    test_tx = transactions.loc[test_mask]

    val_tx = transactions[
        (transactions["t_dat"] >= val_start) & (transactions["t_dat"] < test_start)
    ].copy()

    test_tx = transactions[transactions["t_dat"] >= test_start].copy()

    val_gt = _build_ground_truth(val_tx)

    test_gt = _build_ground_truth(test_tx)

    logger.info(
        f"LGBM Split — "
        f"train_feat={len(train_feat):,} rows, "
        f"train_full={len(train_full):,} rows, "
        f"val={len(val_tx):,} rows, "
        f"test={len(test_tx):,} rows, "
        f"val_users={len(val_gt):,}, "
        f"test_users={len(test_gt):,}"
    )

    return (
        train_full,
        train_feat,
        val_tx,
        val_gt,
        test_gt,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Multi-Week LGBM Training
# ──────────────────────────────────────────────────────────────────────────────


def get_multi_week_training_splits(
    train_full: pd.DataFrame,
    n_weeks: int = 3,
    test_start: pd.Timestamp | None = None,
) -> list[tuple[pd.DataFrame, pd.DataFrame]]:

    if test_start is None:
        test_start = train_full["t_dat"].max() + pd.Timedelta(days=1)

    week_pairs = []

    for week_idx in range(1, n_weeks + 1):
        label_end = test_start - pd.Timedelta(weeks=week_idx)

        label_start = label_end - pd.Timedelta(weeks=1)

        hist_w = train_full[train_full["t_dat"] < label_start].copy()

        label_w = train_full[
            (train_full["t_dat"] >= label_start) & (train_full["t_dat"] < label_end)
        ].copy()

        if hist_w.empty or label_w.empty:
            continue

        logger.info(
            f"Week -{week_idx}: "
            f"history < {label_start.date()} | "
            f"label = {label_start.date()} -> "
            f"{(label_end - pd.Timedelta(days=1)).date()}"
        )

        week_pairs.append((hist_w, label_w))

    return week_pairs


# ──────────────────────────────────────────────────────────────────────────────
# Backward Compatibility
# ──────────────────────────────────────────────────────────────────────────────


def train_test_split(
    transactions: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    dict[str, set[str]],
]:
    """Deprecated wrapper."""

    train_full, _, test_tx, _, test_gt = canonical_split(transactions)

    return train_full, test_tx, test_gt


def make_splits(
    transactions: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    dict[str, set[str]],
    dict[str, set[str]],
]:
    """Deprecated wrapper."""

    return make_splits_lgbm(transactions)
