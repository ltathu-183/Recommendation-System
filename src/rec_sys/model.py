"""Two-stage recommendation model.

Leakage-safe timeline
---------------------
train_feat : before validation week
val_tx     : validation week labels
test_tx    : final evaluation week

NO future interactions are used when:
- generating candidates
- building features
- training LightGBM
- inference

Optimized for:
- low RAM usage
- vectorized operations
- no apply(axis=1)
- consistent temporal splits
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from loguru import logger
from pydantic import BaseModel

from rec_sys.baselines import map_at_k
from rec_sys.candidate_generation import sample_hard_negatives
from rec_sys.data_utils import (
    canonical_split,
    get_multi_week_training_splits,
)

# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────


class ModelConfig(BaseModel):
    data_dir: Path = Path("data")
    model_dir: Path = Path("artifacts")

    k: int = 12

    n_candidates: int = 100
    n_train_weeks: int = 2

    negative_sampling_ratio: int = 4

    sample_eval: int = 20_000

    save_model: bool = True

    lgbm_params: dict[str, Any] = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_child_samples": 50,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "n_estimators": 400,
        "n_jobs": -1,
        "verbose": -1,
        "random_state": 42,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────────────────


def load_data(
    cfg: ModelConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    logger.info("Loading parquet files...")

    tx = pd.read_parquet(cfg.data_dir / "transactions.parquet")
    customers = pd.read_parquet(cfg.data_dir / "customers.parquet")
    articles = pd.read_parquet(cfg.data_dir / "articles.parquet")

    tx = tx[
        [
            "customer_id",
            "article_id",
            "t_dat",
            "price",
            "sales_channel_id",
        ]
    ]

    logger.info("Optimizing memory...")

    if not pd.api.types.is_datetime64_any_dtype(tx["t_dat"]):
        tx["t_dat"] = pd.to_datetime(tx["t_dat"])

    tx["customer_id"] = tx["customer_id"].astype("category")
    tx["article_id"] = tx["article_id"].astype("int32")
    tx["price"] = tx["price"].astype("float32")
    tx["sales_channel_id"] = tx["sales_channel_id"].astype("int8")

    logger.info(
        f"transactions={len(tx):,} "
        f"customers={len(customers):,} "
        f"articles={len(articles):,}"
    )

    return tx, customers, articles


# ──────────────────────────────────────────────────────────────────────────────
# Candidate generation
# ──────────────────────────────────────────────────────────────────────────────


def generate_candidates(
    history: pd.DataFrame,
    target_users: list[str],
    n_candidates: int,
    prediction_date: pd.Timestamp,
) -> pd.DataFrame:
    """
    Leakage-safe candidate generation.

    Candidate sources:
    - recent global popularity
    - user repurchases
    """

    # ──────────────────────────────────────
    # history before prediction only
    # ──────────────────────────────────────

    hist = history.loc[history["t_dat"] < prediction_date]

    # ──────────────────────────────────────
    # recent window
    # ──────────────────────────────────────

    recent_cutoff = prediction_date - pd.Timedelta(days=14)

    recent = hist.loc[hist["t_dat"] >= recent_cutoff]

    # ──────────────────────────────────────
    # global popularity candidates
    # ──────────────────────────────────────

    popular_items = (
        recent["article_id"].value_counts().head(n_candidates).index.tolist()
    )

    popular_df = pd.DataFrame(
        {
            "customer_id": np.repeat(
                target_users,
                len(popular_items),
            ),
            "article_id": (popular_items * len(target_users)),
            "source": "popular",
        }
    )

    # ──────────────────────────────────────
    # user repurchase candidates
    # ──────────────────────────────────────

    repurchase_df = (
        hist.loc[hist["customer_id"].isin(target_users)]
        .sort_values("t_dat")
        .groupby("customer_id")
        .tail(20)[["customer_id", "article_id"]]
        .drop_duplicates()
    )

    repurchase_df["source"] = "repurchase"

    # ──────────────────────────────────────
    # combine candidate sources
    # ──────────────────────────────────────

    cands = pd.concat(
        [
            popular_df,
            repurchase_df,
        ],
        ignore_index=True,
    )

    cands = cands.drop_duplicates(subset=["customer_id", "article_id"])

    # ──────────────────────────────────────
    # optimize dtypes
    # ──────────────────────────────────────

    cands["article_id"] = cands["article_id"].astype("int32")

    return cands.reset_index(drop=True)


# ──────────────────────────────────────────────────────────────────────────────
# Features
# ──────────────────────────────────────────────────────────────────────────────


FEATURE_COLS = [
    "user_total_tx",
    "user_avg_price",
    "user_days_since_last_tx",
    "user_tx_7d",
    "art_total_tx",
    "art_pop_1w",
    "art_avg_price",
    "ua_purchase_count",
    "ua_days_since_last",
    "price_diff",
]


def build_features(
    history: pd.DataFrame,
    candidates: pd.DataFrame,
    prediction_date: pd.Timestamp,
) -> pd.DataFrame:
    """
    Leakage-safe features.
    """

    hist = history.loc[history["t_dat"] < prediction_date]

    # ──────────────────────────────────────
    # User features
    # ──────────────────────────────────────

    user_feats = (
        hist.groupby("customer_id")
        .agg(
            user_total_tx=("article_id", "count"),
            user_avg_price=("price", "mean"),
            user_last_tx=("t_dat", "max"),
        )
        .reset_index()
    )

    user_feats["user_days_since_last_tx"] = (
        prediction_date - user_feats["user_last_tx"]
    ).dt.days.astype("float32")

    user_feats = user_feats.drop(columns=["user_last_tx"])

    # user tx 7d

    tx7 = (
        hist.loc[hist["t_dat"] >= prediction_date - pd.Timedelta(days=7)]
        .groupby("customer_id")
        .size()
        .rename("user_tx_7d")
        .reset_index()
    )

    user_feats = user_feats.merge(
        tx7,
        on="customer_id",
        how="left",
    )

    # ──────────────────────────────────────
    # Article features
    # ──────────────────────────────────────

    art_feats = (
        hist.groupby("article_id")
        .agg(
            art_total_tx=("customer_id", "count"),
            art_avg_price=("price", "mean"),
        )
        .reset_index()
    )

    art_1w = (
        hist.loc[hist["t_dat"] >= prediction_date - pd.Timedelta(days=7)]
        .groupby("article_id")
        .size()
        .rename("art_pop_1w")
        .reset_index()
    )

    art_feats = art_feats.merge(
        art_1w,
        on="article_id",
        how="left",
    )

    # ──────────────────────────────────────
    # User-item features
    # ──────────────────────────────────────

    ua = (
        hist.groupby(["customer_id", "article_id"])
        .agg(
            ua_purchase_count=("t_dat", "count"),
            ua_last_tx=("t_dat", "max"),
        )
        .reset_index()
    )

    ua["ua_days_since_last"] = (prediction_date - ua["ua_last_tx"]).dt.days.astype(
        "float32"
    )

    ua = ua.drop(columns=["ua_last_tx"])

    # ──────────────────────────────────────
    # Merge
    # ──────────────────────────────────────

    df = candidates.merge(
        user_feats,
        on="customer_id",
        how="left",
    )

    df = df.merge(
        art_feats,
        on="article_id",
        how="left",
    )

    df = df.merge(
        ua,
        on=["customer_id", "article_id"],
        how="left",
    )

    # ──────────────────────────────────────
    # Derived
    # ──────────────────────────────────────

    df["price_diff"] = np.abs(df["user_avg_price"] - df["art_avg_price"])

    # ──────────────────────────────────────
    # Fill NA
    # ──────────────────────────────────────

    for c in FEATURE_COLS:
        if c in df.columns:
            df[c] = df[c].fillna(0)

    # optimize dtypes

    for c in FEATURE_COLS:
        df[c] = df[c].astype("float32")

    return df


# ──────────────────────────────────────────────────────────────────────────────
# Model
# ──────────────────────────────────────────────────────────────────────────────


class TwoStageLGBMRanker:
    name = "two_stage_lgbm"

    def __init__(
        self,
        cfg: ModelConfig | None = None,
    ) -> None:
        self.cfg = cfg or ModelConfig()

        self.model: lgb.LGBMRanker | None = None

        self.train_history: pd.DataFrame | None = None

    def _make_training_frame(
        self,
        history: pd.DataFrame,
        label_tx: pd.DataFrame,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

        prediction_date = label_tx["t_dat"].min()

        target_users = label_tx["customer_id"].unique().tolist()

        # ──────────────────────────────────
        # Generate candidates
        # ──────────────────────────────────

        cands = generate_candidates(
            history=history,
            target_users=target_users,
            n_candidates=self.cfg.n_candidates,
            prediction_date=prediction_date,
        )

        # ──────────────────────────────────
        # Build labels
        # ──────────────────────────────────

        positive_df = (
            label_tx[["customer_id", "article_id"]].drop_duplicates().assign(label=1)
        )

        cands = cands.merge(
            positive_df,
            on=["customer_id", "article_id"],
            how="left",
        )

        cands["label"] = cands["label"].fillna(0).astype("int8")

        # ──────────────────────────────────
        # Negative sampling
        # ──────────────────────────────────

        cands = sample_hard_negatives(
            cands,
            pos_multiplier=self.cfg.negative_sampling_ratio,
            random_state=42,
        )

        # IMPORTANT:
        # sample_hard_negatives may drop rows,
        # so label must already exist BEFORE sampling.
        # Do NOT rebuild labels again afterward.

        # ──────────────────────────────────
        # Build features
        # ──────────────────────────────────

        df = build_features(
            history=history,
            candidates=cands,
            prediction_date=prediction_date,
        )

        df["label"] = cands["label"].values.astype("int8")
        # ──────────────────────────────────
        # Sort for LightGBM groups
        # ──────────────────────────────────

        df = df.sort_values(
            "customer_id",
            kind="stable",
        )

        groups = (
            df.groupby(
                "customer_id",
                sort=False,
            )
            .size()
            .values.astype(np.int32)
        )

        # ──────────────────────────────────
        # Final matrices
        # ──────────────────────────────────

        X = df[FEATURE_COLS].to_numpy(dtype=np.float32)

        y = df["label"].to_numpy(dtype=np.int8)

        return X, y, groups

    def fit(
        self,
        train_full: pd.DataFrame,
    ) -> None:

        self.train_history = train_full

        logger.info("Building training data...")

        all_X = []
        all_y = []
        all_groups = []

        week_pairs = get_multi_week_training_splits(
            train_full=train_full,
            n_weeks=self.cfg.n_train_weeks,
        )

        for i, (hist_w, label_w) in enumerate(
            week_pairs,
            1,
        ):
            logger.info(f"Week {i}")

            X_w, y_w, g_w = self._make_training_frame(
                hist_w,
                label_w,
            )

            all_X.append(X_w)
            all_y.append(y_w)
            all_groups.append(g_w)

        X_train = np.vstack(all_X)
        y_train = np.concatenate(all_y)
        groups_train = np.concatenate(all_groups)

        logger.info(f"Training rows={len(X_train):,} positives={y_train.sum():,}")

        self.model = lgb.LGBMRanker(**self.cfg.lgbm_params)

        self.model.fit(
            X_train,
            y_train,
            group=groups_train,
            feature_name=FEATURE_COLS,
            callbacks=[lgb.log_evaluation(period=50)],
            eval_at=[12],
        )

        logger.info("Training complete.")

        if self.cfg.save_model:
            self.save()

    def predict(
        self,
        customer_ids: list[str],
        prediction_date: pd.Timestamp,
        k: int = 12,
    ) -> dict[str, list[int]]:

        assert self.model is not None
        assert self.train_history is not None

        cands = generate_candidates(
            self.train_history,
            customer_ids,
            self.cfg.n_candidates,
            prediction_date,
        )

        df = build_features(
            self.train_history,
            cands,
            prediction_date,
        )

        X = df[FEATURE_COLS].to_numpy(dtype=np.float32)

        df["score"] = self.model.predict(X)

        preds = {}

        for uid, grp in df.groupby("customer_id"):
            topk = grp.nlargest(k, "score")["article_id"].astype(int).tolist()

            preds[str(uid)] = topk

        return preds

    def evaluate(
        self,
        test_gt: dict[str, set[int]],
        prediction_date: pd.Timestamp,
        k: int = 12,
        sample: int | None = None,
    ) -> float:

        # optional subsampling
        if sample is not None and sample < len(test_gt):
            rng = np.random.default_rng(42)

            sampled_users = rng.choice(
                list(test_gt.keys()),
                size=sample,
                replace=False,
            )

            eval_gt = {u: test_gt[u] for u in sampled_users}

        else:
            eval_gt = test_gt

        preds = self.predict(
            customer_ids=list(eval_gt.keys()),
            prediction_date=prediction_date,
            k=k,
        )

        score = map_at_k(
            preds,
            eval_gt,
            k=k,
        )

        logger.info(f"MAP@{k} = {score:.6f}")

        return score

    def save(self) -> None:

        assert self.model is not None

        self.cfg.model_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        path = self.cfg.model_dir / "two_stage_lgbm.pkl"

        with open(path, "wb") as f:
            pickle.dump(
                {
                    "model": self.model,
                    "features": FEATURE_COLS,
                    "config": self.cfg.model_dump(mode="json"),
                },
                f,
            )

        logger.info(f"Saved model → {path}")

    def load(self, path: Path) -> None:

        with open(path, "rb") as f:
            obj = pickle.load(f)

        self.model = obj["model"]
        self.cfg = ModelConfig(**obj["config"])

        logger.info(f"Loaded model ← {path}")


# ──────────────────────────────────────────────────────────────────────────────
# Runner
# ──────────────────────────────────────────────────────────────────────────────


def run() -> dict[str, float]:

    cfg = ModelConfig()

    tx, customers, articles = load_data(cfg)

    (
        train_full,
        val_tx,
        test_tx,
        val_gt,
        test_gt,
    ) = canonical_split(tx)

    logger.info("Training model...")

    model = TwoStageLGBMRanker(cfg)

    model.fit(
        train_full=train_full,
    )

    logger.info("Evaluating...")

    prediction_date = test_tx["t_dat"].min()

    score = model.evaluate(
        test_gt=test_gt,
        prediction_date=prediction_date,
        k=cfg.k,
    )

    return {
        "two_stage_lgbm": score,
    }


if __name__ == "__main__":
    run()
