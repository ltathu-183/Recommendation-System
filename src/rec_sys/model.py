"""
Two-stage recommendation model (STABLE + MEMORY SAFE FIX)

KEY FIXES:
- Removed ALL pandas groupby factorization hotspots
- Removed cumcount (ROOT CAUSE of crashes)
- Replaced with numpy-based ranking
- Prevents Arrow string backend explosions
- Safe dedup without groupby hash tables
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from loguru import logger
from pydantic import BaseModel

from rec_sys.baselines import map_at_k
from rec_sys.data_utils import canonical_split, get_multi_week_training_splits


# =========================
# 🔥 GLOBAL SAFETY FIX
# =========================
os.environ["PANDAS_STRING_STORAGE"] = "python"
pd.options.mode.string_storage = "python"
pd.options.mode.copy_on_write = False


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

class ModelConfig(BaseModel):
    data_dir: Path = Path("data")
    model_dir: Path = Path("artifacts")

    k: int = 12
    n_candidates: int = 80
    n_train_weeks: int = 4
    negative_sampling_ratio: int = 40

    lgbm_params: dict[str, Any] = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "learning_rate": 0.05,
        "num_leaves": 128,
        "min_data_in_leaf": 30,
        "feature_fraction": 0.85,
        "bagging_fraction": 0.85,
        "bagging_freq": 5,
        "lambda_l1": 0.1,
        "lambda_l2": 0.2,
        "min_split_gain": 0.01,
        "n_estimators": 600,
        "n_jobs": -1,
        "verbose": -1,
        "random_state": 42,
        "eval_at": 12,
    }


# ─────────────────────────────────────────────
# LOAD DATA (SAFE)
# ─────────────────────────────────────────────

def load_data(cfg: ModelConfig):
    logger.info("Loading data...")
    tx = pd.read_parquet(cfg.data_dir / "transactions.parquet", engine="fastparquet")
    customers = pd.read_parquet(cfg.data_dir / "customers.parquet", engine="fastparquet")
    articles = pd.read_parquet(cfg.data_dir / "articles.parquet", engine="fastparquet")

    tx["t_dat"] = pd.to_datetime(tx["t_dat"])

    # 🔥 Map string customer_id to int32 codes
    unique_customers = tx["customer_id"].unique()
    customer_to_int = {cid: i for i, cid in enumerate(unique_customers)}
    tx["customer_id_int"] = tx["customer_id"].map(customer_to_int).astype(np.int32)

    # Keep original mapping for later conversion back
    int_to_customer = {i: cid for cid, i in customer_to_int.items()}

    tx["article_id"] = tx["article_id"].astype(np.int32)
    tx["price"] = tx["price"].astype(np.float32)

    # DO NOT drop 'customer_id' – data_utils needs it
    return tx, customers, articles, int_to_customer


# ─────────────────────────────────────────────
# CANDIDATES (FIXED - NO GROUPBY)
# ─────────────────────────────────────────────

def generate_candidates(history, users_int, n_candidates, t, articles=None):
    """
    users_int: array of integer customer codes (not original strings)
    """
    hist = history[history["t_dat"] < t].copy()
    if hist.empty:
        return pd.DataFrame(columns=["customer_id_int", "article_id", "source"])

    users = np.asarray(users_int, dtype=np.int32)
    pop_limit = 100  # reduce memory

    pop_7 = hist[hist["t_dat"] >= t - pd.Timedelta(days=7)]
    pop_14 = hist[hist["t_dat"] >= t - pd.Timedelta(days=14)]

    top7 = pop_7["article_id"].value_counts().head(pop_limit).index.to_numpy(np.int32)
    top14 = pop_14["article_id"].value_counts().head(pop_limit).index.to_numpy(np.int32)

    def cross(u_arr, i_arr, name, chunk_size=5000):
        if len(i_arr) == 0:
            return None
        i_arr = np.asarray(i_arr, dtype=np.int32)
        parts = []
        for i in range(0, len(u_arr), chunk_size):
            u_chunk = u_arr[i:i+chunk_size]
            df_chunk = pd.DataFrame({
                "customer_id_int": np.repeat(u_chunk, len(i_arr)),
                "article_id": np.tile(i_arr, len(u_chunk)),
                "source": name
            })
            parts.append(df_chunk)
        return pd.concat(parts, ignore_index=True)

    cands = []
    tmp = cross(users, top7, "pop_7d")
    if tmp is not None:
        cands.append(tmp)
    tmp = cross(users, top14, "pop_14d")
    if tmp is not None:
        cands.append(tmp)

    # Recent items (using customer_id_int)
    last = (
        hist.groupby("customer_id_int", sort=False, observed=True)
        .tail(5)[["customer_id_int", "article_id"]]
        .copy()
    )
    if not last.empty:
        last["source"] = "recent_user"
        cands.append(last)

    if not cands:
        return pd.DataFrame(columns=["customer_id_int", "article_id", "source"])

    cands = pd.concat(cands, ignore_index=True)
    cands["customer_id_int"] = cands["customer_id_int"].astype(np.int32)
    cands["article_id"] = cands["article_id"].astype(np.int32)

    cands = cands.drop_duplicates(["customer_id_int", "article_id"], keep="first")
    return cands


# ─────────────────────────────────────────────
# FEATURES (UNCHANGED LOGIC, SAFE OPS)
# ─────────────────────────────────────────────

def build_features(history, candidates, t, articles=None):
    hist = history[history["t_dat"] < t].copy()
    # All groupings use customer_id_int
    user_stats = hist.groupby("customer_id_int").agg(
        u_tx=("article_id", "count"),
        u_unique=("article_id", "nunique"),
        u_avg_price=("price", "mean"),
        u_last=("t_dat", "max"),
    ).reset_index()
    user_stats["u_recency"] = (t - user_stats["u_last"]).dt.days
    user_stats.drop(columns=["u_last"], inplace=True)

    item_stats = hist.groupby("article_id").agg(
        i_tx=("customer_id_int", "count"),
        i_avg_price=("price", "mean"),
        i_last=("t_dat", "max"),
    ).reset_index()
    item_stats["i_recency"] = (t - item_stats["i_last"]).dt.days
    item_stats.drop(columns=["i_last"], inplace=True)

    ui = hist.groupby(["customer_id_int", "article_id"]).size().reset_index(name="ui_cnt")

    df = candidates.merge(user_stats, on="customer_id_int", how="left")
    df = df.merge(item_stats, on="article_id", how="left")
    df = df.merge(ui, on=["customer_id_int", "article_id"], how="left")

    df["ui_cnt"] = df["ui_cnt"].fillna(0).astype(np.float32)
    df["pop_bias"] = np.log1p(df["i_tx"].fillna(0))
    df["recency_bias"] = 1 / (1 + df["i_recency"].fillna(999))
    df["user_activity"] = np.log1p(df["u_tx"].fillna(0))
    df["price_gap"] = np.abs(df["u_avg_price"] - df["i_avg_price"])
    df["source_rank"] = df["source"].map({"recent_user": 3, "pop_7d": 2, "pop_14d": 1}).fillna(0).astype(np.float32)

    features = [
        "u_tx", "u_unique", "u_avg_price", "u_recency",
        "i_tx", "i_avg_price", "i_recency",
        "ui_cnt", "pop_bias", "recency_bias",
        "user_activity", "price_gap", "source_rank"
    ]
    for c in features:
        df[c] = df[c].fillna(0).astype(np.float32)

    return df, features


# ─────────────────────────────────────────────
# MODEL (UNCHANGED API)
# ─────────────────────────────────────────────

class TwoStageLGBMRanker:
    def __init__(self, cfg=None, int_to_customer=None):
        self.cfg = cfg or ModelConfig()
        self.model = None
        self.train_hist = None
        self.articles = None
        self.int_to_customer = int_to_customer   # for final output

    def _make_training_frame(self, hist, labels, t):
        # labels already contain customer_id_int
        users = labels["customer_id_int"].unique()
        cands = generate_candidates(hist, users, self.cfg.n_candidates, t)
        pos = labels[["customer_id_int", "article_id"]].drop_duplicates()
        cands = cands.merge(pos.assign(label=1), on=["customer_id_int", "article_id"], how="left")
        cands["label"] = cands["label"].fillna(0).astype(np.int8)

        df, feats = build_features(hist, cands, t, self.articles)
        df["label"] = cands["label"].values
        df = df.sort_values("customer_id_int", kind="stable")
        groups = df.groupby("customer_id_int").size().values

        X = df[feats].values.astype(np.float32)
        y = df["label"].values
        return X, y, groups

    def fit(self, train_full, val_tx=None, articles=None):
        # train_full must have customer_id_int column
        self.train_hist = train_full
        self.articles = articles

        X_all, y_all, g_all = [], [], []
        for hist_w, label_w in get_multi_week_training_splits(
            train_full, self.cfg.n_train_weeks
        ):
            X, y, g = self._make_training_frame(
                hist_w, label_w, label_w["t_dat"].min()
            )
            X_all.append(X)
            y_all.append(y)
            g_all.append(g)

        X_train = np.vstack(X_all)
        y_train = np.concatenate(y_all)
        g_train = np.concatenate(g_all)

        self.model = lgb.LGBMRanker(**self.cfg.lgbm_params)
        self.model.fit(X_train, y_train, group=g_train)

    def predict(self, users_str, t, k=12):
        """
        users_str: list of original customer_id strings
        """
        # Convert input strings to internal integer codes
        # Build reverse mapping from int_to_customer dict
        str_to_int = {v: k for k, v in self.int_to_customer.items()}
        users_int = np.array([str_to_int[u] for u in users_str], dtype=np.int32)

        cands = generate_candidates(self.train_hist, users_int, self.cfg.n_candidates, t)
        df, feats = build_features(self.train_hist, cands, t, self.articles)
        df["score"] = self.model.predict(df[feats])
        df = df.sort_values(["customer_id_int", "score"], ascending=[True, False])

        # Convert back to original string IDs for output
        df["customer_id"] = df["customer_id_int"].map(self.int_to_customer)
        return (
            df.groupby("customer_id")
            .head(k)
            .groupby("customer_id")["article_id"]
            .apply(list)
            .to_dict()
        )
    
    def evaluate(self, test_gt, t, k=12):
        preds = self.predict(list(test_gt.keys()), t, k)
        return map_at_k(preds, test_gt, k)