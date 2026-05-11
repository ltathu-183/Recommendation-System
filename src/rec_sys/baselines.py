"""Fast recommendation baselines for H&M Fashion dataset.

Optimizations:
- Polars for fast preprocessing/groupby/sorting
- Minimal parquet column loading
- Cached popularity rankings
- Faster repurchase history construction
- Lower memory footprint
- Cleaner architecture

Metric:
    MAP@12
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
from loguru import logger
from pydantic import BaseModel, field_validator

from rec_sys.data_utils import canonical_split


# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────


class BaselineConfig(BaseModel):
    data_dir: Path = Path("data")

    k: int = 12
    recent_weeks: int = 2

    age_bins: list[int] = [15, 25, 35, 45, 55, 65, 100]
    age_labels: list[str] = [
        "16-24",
        "25-34",
        "35-44",
        "45-54",
        "55-64",
        "65+",
    ]

    sample_eval: int = 50_000
    popularity_top_n: int = 100

    @field_validator("k")
    @classmethod
    def validate_k(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("k must be positive")
        return v


# ──────────────────────────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────────────────────────


def _ap_at_k(
    predicted: list[str],
    actual: set[str],
    k: int,
) -> float:
    """Average precision at K for a single user."""

    if not actual:
        return 0.0

    hits = 0
    score = 0.0

    for i, item in enumerate(predicted[:k], start=1):
        if item in actual:
            hits += 1
            score += hits / i

    return score / min(len(actual), k)


def map_at_k(
    predictions: dict[str, list[str]],
    ground_truth: dict[str, set[str]],
    k: int = 12,
) -> float:
    """Mean Average Precision at K."""

    scores = [
        _ap_at_k(predictions.get(uid, []), gt, k) for uid, gt in ground_truth.items()
    ]

    return float(np.mean(scores))


# ──────────────────────────────────────────────────────────────────────────────
# Data Loading
# ──────────────────────────────────────────────────────────────────────────────


def load_data(
    cfg: BaselineConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load only required columns."""

    logger.info("Loading parquet files...")

    transactions = pd.read_parquet(
        cfg.data_dir / "transactions.parquet",
        columns=["customer_id", "article_id", "t_dat"],
    )

    customers = pd.read_parquet(
        cfg.data_dir / "customers.parquet",
        columns=["customer_id", "age"],
    )

    # categorical compression
    for col in ["customer_id", "article_id"]:
        transactions[col] = transactions[col].astype("category")

    customers["customer_id"] = customers["customer_id"].astype("category")

    logger.info(
        f"Loaded transactions={len(transactions):,}, customers={len(customers):,}"
    )

    return transactions, customers


def train_test_split(
    transactions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, set[str]]]:
    """Canonical split wrapper."""

    train, _val_tx, test_tx, _val_gt, test_gt = canonical_split(transactions)

    return train, test_tx, test_gt


# ──────────────────────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────────────────────


def top_popular_articles(
    df: pl.DataFrame,
    n: int = 100,
) -> list[str]:
    """Fast top-N article extraction."""

    return (
        df.group_by("article_id")
        .len()
        .sort("len", descending=True)
        .head(n)["article_id"]
        .to_list()
    )


# ──────────────────────────────────────────────────────────────────────────────
# Base Recommender
# ──────────────────────────────────────────────────────────────────────────────


class BaseRecommender:
    name: str = "base"

    def fit(self, train: pl.DataFrame, **kwargs: Any) -> None:
        raise NotImplementedError

    def predict(
        self,
        customer_ids: list[str],
        k: int = 12,
    ) -> dict[str, list[str]]:
        raise NotImplementedError

    def evaluate(
        self,
        ground_truth: dict[str, set[str]],
        k: int = 12,
        sample: int | None = None,
    ) -> float:

        if sample and sample < len(ground_truth):
            rng = np.random.default_rng(42)

            sampled_users = rng.choice(
                list(ground_truth.keys()),
                size=sample,
                replace=False,
            )

            gt = {u: ground_truth[u] for u in sampled_users}

        else:
            gt = ground_truth

        preds = self.predict(list(gt.keys()), k=k)

        score = map_at_k(preds, gt, k=k)

        logger.info(f"[{self.name}] MAP@{k} = {score:.6f}")

        return score


# ──────────────────────────────────────────────────────────────────────────────
# Baseline 1: Global Popularity
# ──────────────────────────────────────────────────────────────────────────────


class GlobalPopularityRecommender(BaseRecommender):
    name = "global_popularity"

    def __init__(self, top_articles: list[str]):
        self._top_articles = top_articles

    def fit(self, train: pl.DataFrame, **kwargs: Any) -> None:
        logger.info(f"[{self.name}] fitted")

    def predict(
        self,
        customer_ids: list[str],
        k: int = 12,
    ) -> dict[str, list[str]]:

        recs = self._top_articles[:k]

        return {uid: recs for uid in customer_ids}


# ──────────────────────────────────────────────────────────────────────────────
# Baseline 2: Recent Popularity
# ──────────────────────────────────────────────────────────────────────────────


class RecentPopularityRecommender(BaseRecommender):
    name = "recent_popularity"

    def __init__(
        self,
        recent_weeks: int = 2,
        top_n: int = 100,
    ):
        self.recent_weeks = recent_weeks
        self.top_n = top_n

        self._top_articles: list[str] = []

    def fit(self, train: pl.DataFrame, **kwargs: Any) -> None:

        max_date = train["t_dat"].max()

        cutoff = max_date - pd.Timedelta(weeks=self.recent_weeks)

        recent = train.filter(pl.col("t_dat") >= cutoff)

        self._top_articles = top_popular_articles(
            recent,
            n=self.top_n,
        )

        logger.info(f"[{self.name}] fitted ({recent.height:,} rows)")

    def predict(
        self,
        customer_ids: list[str],
        k: int = 12,
    ) -> dict[str, list[str]]:

        recs = self._top_articles[:k]

        return {uid: recs for uid in customer_ids}


# ──────────────────────────────────────────────────────────────────────────────
# Baseline 3: Repurchase
# ──────────────────────────────────────────────────────────────────────────────


class RepurchaseRecommender(BaseRecommender):
    name = "repurchase"

    def __init__(
        self,
        fallback_articles: list[str],
        top_n: int = 100,
    ):
        self.top_n = top_n

        self._fallback = fallback_articles
        self._user_history: dict[str, list[str]] = {}

    def fit(self, train: pl.DataFrame, **kwargs: Any) -> None:

        histories = train.group_by("customer_id").agg(
            pl.col("article_id")
            .sort_by("t_dat", descending=True)
            .unique(maintain_order=True)
            .head(self.top_n)
        )

        self._user_history = dict(
            zip(
                histories["customer_id"].to_list(),
                histories["article_id"].to_list(),
            )
        )

        logger.info(f"[{self.name}] fitted ({len(self._user_history):,} users)")

    def predict(
        self,
        customer_ids: list[str],
        k: int = 12,
    ) -> dict[str, list[str]]:

        preds: dict[str, list[str]] = {}

        for uid in customer_ids:
            history = self._user_history.get(uid, [])[:k]

            if len(history) < k:
                history_set = set(history)

                extras = [a for a in self._fallback if a not in history_set]

                history = history + extras[: k - len(history)]

            preds[uid] = history

        return preds


# ──────────────────────────────────────────────────────────────────────────────
# Baseline 4: Age Segmented Popularity
# ──────────────────────────────────────────────────────────────────────────────


class AgeSegmentedPopularityRecommender(BaseRecommender):
    name = "age_segmented_popularity"

    def __init__(
        self,
        global_top: list[str],
        age_bins: list[int],
        age_labels: list[str],
        top_n: int = 100,
    ):
        self.global_top = global_top
        self.age_bins = age_bins
        self.age_labels = age_labels
        self.top_n = top_n

        self._segment_tops: dict[str, list[str]] = {}
        self._customer_segment: dict[str, str] = {}

    def fit(
        self,
        train: pl.DataFrame,
        customers: pd.DataFrame | None = None,
        **kwargs: Any,
    ) -> None:

        if customers is None:
            raise ValueError("customers required")

        customers = customers.copy()

        customers["age_group"] = pd.cut(
            customers["age"],
            bins=self.age_bins,
            labels=self.age_labels,
            right=True,
        )

        customers = customers.dropna(subset=["age_group"])

        self._customer_segment = (
            customers.set_index("customer_id")["age_group"].astype(str).to_dict()
        )

        # SMALL dataframe → convert once
        cust_pl = pl.from_pandas(customers[["customer_id", "age_group"]]).with_columns(
            pl.col("age_group").cast(pl.String)
        )

        # JOIN instead of map_elements
        tx_age = train.join(
            cust_pl,
            on="customer_id",
            how="left",
        )

        # Count popularity directly
        popular = (
            tx_age.drop_nulls("age_group")
            .group_by(["age_group", "article_id"])
            .len()
            .sort(
                ["age_group", "len"],
                descending=[False, True],
            )
        )

        # top-N per segment
        segment_top = (
            popular.group_by("age_group")
            .head(self.top_n)
            .group_by("age_group")
            .agg(pl.col("article_id"))
        )

        self._segment_tops = {
            row["age_group"]: row["article_id"]
            for row in segment_top.iter_rows(named=True)
        }

        logger.info(f"[{self.name}] fitted ({len(self._segment_tops)} segments)")

    def predict(
        self,
        customer_ids: list[str],
        k: int = 12,
    ) -> dict[str, list[str]]:

        preds: dict[str, list[str]] = {}

        for uid in customer_ids:
            seg = self._customer_segment.get(uid)

            top = (
                self._segment_tops.get(seg, self.global_top) if seg else self.global_top
            )

            preds[uid] = top[:k]

        return preds


# ──────────────────────────────────────────────────────────────────────────────
# Runner
# ──────────────────────────────────────────────────────────────────────────────


def run_all_baselines(
    cfg: BaselineConfig | None = None,
) -> dict[str, float]:

    if cfg is None:
        cfg = BaselineConfig()

    transactions_pd, customers = load_data(cfg)

    train_pd, _test, ground_truth = train_test_split(transactions_pd)

    # convert once
    train = pl.from_pandas(train_pd)

    logger.info("Computing shared popularity rankings...")

    global_top = top_popular_articles(
        train,
        n=cfg.popularity_top_n,
    )

    models: list[BaseRecommender] = [
        GlobalPopularityRecommender(global_top),
        RecentPopularityRecommender(
            recent_weeks=cfg.recent_weeks,
            top_n=cfg.popularity_top_n,
        ),
        RepurchaseRecommender(
            fallback_articles=global_top,
            top_n=cfg.popularity_top_n,
        ),
        AgeSegmentedPopularityRecommender(
            global_top=global_top,
            age_bins=cfg.age_bins,
            age_labels=cfg.age_labels,
            top_n=cfg.popularity_top_n,
        ),
    ]

    results: dict[str, float] = {}

    for model in models:
        logger.info(f"─── Fitting {model.name} ───")

        if model.name == "age_segmented_popularity":
            model.fit(train, customers=customers)

        else:
            model.fit(train)

        score = model.evaluate(
            ground_truth,
            k=cfg.k,
            sample=cfg.sample_eval,
        )

        results[model.name] = score

    logger.info("─── Results ───")

    for name, score in sorted(
        results.items(),
        key=lambda x: -x[1],
    ):
        logger.info(f"{name:<35} MAP@{cfg.k} = {score:.6f}")

    return results


if __name__ == "__main__":
    run_all_baselines()
