"""Item-based Collaborative Filtering.

Algorithm
---------
1. Filter to recent interactions only (temporal honesty).
2. Build sparse user-item matrix X.
3. Compute sparse item-item cosine similarity:
       S = normalize(X.T @ X)
4. Keep only top-K neighbors per item.
5. Predict by summing neighbor similarities of purchased items.
6. Fall back to recent popularity for cold-start users.

Leakage Prevention
------------------
- Training ONLY uses transactions before test_start.
- Similarity matrix is built ONLY from train_full.
- Recent filtering is applied INSIDE train_full only.
- No future interactions are used anywhere.

"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from loguru import logger
from pydantic import BaseModel, field_validator
from sklearn.preprocessing import normalize

from rec_sys.baselines import map_at_k
from rec_sys.data_utils import canonical_split


# ─────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────


class CFConfig(BaseModel):
    data_dir: Path = Path("data")

    k: int = 12

    # use only recent history
    recent_weeks: int = 16

    # remove ultra-rare items
    min_item_support: int = 20

    # sparse similarity neighbors
    top_similar: int = 100

    # popularity fallback
    fallback_n: int = 100

    # evaluation
    sample_eval: int = 50_000

    @field_validator("k")
    @classmethod
    def validate_k(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("k must be positive")
        return v


# ─────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────


class ItemCFRecommender:
    """Sparse item-item collaborative filtering."""

    name = "item_cf"

    def __init__(self, cfg: CFConfig | None = None) -> None:
        self.cfg = cfg or CFConfig()

        self._item_sim: sp.csr_matrix | None = None

        self._item_to_idx: dict[str, int] = {}
        self._idx_to_item: dict[int, str] = {}

        self._user_items: dict[str, np.ndarray] = {}

        self._fallback: list[str] = []

    # ─────────────────────────────────────────────────────────

    def fit(self, train_full: pd.DataFrame) -> None:
        """Fit sparse item-item similarity."""

        t0 = time.time()

        logger.info("Preparing CF training data...")

        # =====================================================
        # TEMPORAL FILTERING (NO LEAKAGE)
        # =====================================================

        t_max = train_full["t_dat"].max()

        cutoff = t_max - pd.Timedelta(weeks=self.cfg.recent_weeks)

        recent = train_full[train_full["t_dat"] >= cutoff].copy()

        logger.info(f"Recent interactions: {len(recent):,}")

        # =====================================================
        # ITEM SUPPORT FILTERING
        # =====================================================

        item_counts = recent["article_id"].value_counts()

        valid_items = item_counts[item_counts >= self.cfg.min_item_support].index

        recent = recent[recent["article_id"].isin(valid_items)]

        logger.info(f"Filtered interactions: {len(recent):,}")

        # =====================================================
        # INDEXING
        # =====================================================

        items = recent["article_id"].unique().tolist()
        users = recent["customer_id"].unique().tolist()

        self._item_to_idx = {a: i for i, a in enumerate(items)}

        self._idx_to_item = {i: a for a, i in self._item_to_idx.items()}

        user_to_idx = {u: i for i, u in enumerate(users)}

        n_users = len(users)
        n_items = len(items)

        logger.info(f"Sparse matrix: {n_users:,} users × {n_items:,} items")

        # =====================================================
        # BUILD SPARSE USER-ITEM MATRIX
        # =====================================================

        rows = recent["customer_id"].map(user_to_idx).values
        cols = recent["article_id"].map(self._item_to_idx).values

        data = np.ones(len(recent), dtype=np.float32)

        X = sp.csr_matrix(
            (data, (rows, cols)),
            shape=(n_users, n_items),
            dtype=np.float32,
        )

        # binarize
        X.data[:] = 1.0

        logger.info(f"Matrix nnz = {X.nnz:,}")

        # =====================================================
        # COSINE NORMALIZATION
        # =====================================================

        logger.info("Normalizing matrix...")

        X_norm = normalize(X, norm="l2", axis=0)

        # =====================================================
        # SPARSE ITEM-ITEM SIMILARITY
        # =====================================================

        logger.info("Computing sparse item similarity...")

        # IMPORTANT:
        # stays sparse, NO .toarray()
        S = (X_norm.T @ X_norm).tocsr()

        logger.info(f"Similarity nnz before pruning: {S.nnz:,}")

        # remove diagonal
        S.setdiag(0)
        S.eliminate_zeros()

        # =====================================================
        # KEEP ONLY TOP-K NEIGHBORS
        # =====================================================

        logger.info(f"Keeping top-{self.cfg.top_similar} neighbors...")

        S = self._keep_top_k(S, self.cfg.top_similar)

        self._item_sim = S.tocsr()

        logger.info(f"Similarity nnz after pruning: {self._item_sim.nnz:,}")

        # =====================================================
        # STORE USER HISTORY
        # =====================================================

        self._user_items = (
            recent.groupby("customer_id")["article_id"]
            .apply(
                lambda s: np.array(
                    [
                        self._item_to_idx[a]
                        for a in s.unique()
                        if a in self._item_to_idx
                    ],
                    dtype=np.int32,
                )
            )
            .to_dict()
        )

        # =====================================================
        # FALLBACK POPULARITY
        # =====================================================

        self._fallback = (
            recent["article_id"].value_counts().head(self.cfg.fallback_n).index.tolist()
        )

        logger.info(f"[{self.name}] fitted in {time.time() - t0:.1f}s")

    # ─────────────────────────────────────────────────────────

    @staticmethod
    def _keep_top_k(
        matrix: sp.csr_matrix,
        k: int,
    ) -> sp.csr_matrix:
        """Keep top-k values per row."""

        matrix = matrix.tolil()

        for i in range(matrix.shape[0]):
            row_data = matrix.data[i]

            if len(row_data) <= k:
                continue

            row_indices = matrix.rows[i]

            top_idx = np.argsort(row_data)[-k:]

            matrix.data[i] = [row_data[j] for j in top_idx]

            matrix.rows[i] = [row_indices[j] for j in top_idx]

        return matrix.tocsr()

    # ─────────────────────────────────────────────────────────

    def predict(
        self,
        customer_ids: list[str],
        k: int = 12,
    ) -> dict[str, list[str]]:

        assert self._item_sim is not None

        preds: dict[str, list[str]] = {}

        for uid in customer_ids:
            history = self._user_items.get(uid)

            # cold start
            if history is None or len(history) == 0:
                preds[uid] = self._fallback[:k]
                continue

            # aggregate similarity rows
            scores = np.asarray(self._item_sim[history].sum(axis=0)).ravel()

            # remove purchased items
            scores[history] = 0.0

            if scores.sum() == 0:
                preds[uid] = self._fallback[:k]
                continue

            top_idx = np.argpartition(scores, -k)[-k:]

            top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]

            recs = [self._idx_to_item[i] for i in top_idx if scores[i] > 0]

            # fill fallback
            if len(recs) < k:
                seen = {self._idx_to_item[i] for i in history}

                extra = [a for a in self._fallback if a not in seen and a not in recs]

                recs.extend(extra[: k - len(recs)])

            preds[uid] = recs[:k]

        return preds

    # ─────────────────────────────────────────────────────────

    def evaluate(
        self,
        ground_truth: dict[str, set[str]],
        k: int = 12,
        sample: int | None = None,
    ) -> float:

        if sample and sample < len(ground_truth):
            rng = np.random.default_rng(42)

            users = rng.choice(
                list(ground_truth.keys()),
                size=sample,
                replace=False,
            )

            gt = {u: ground_truth[u] for u in users}

        else:
            gt = ground_truth

        logger.info(f"Evaluating on {len(gt):,} users...")

        preds = self.predict(
            list(gt.keys()),
            k=k,
        )

        score = map_at_k(
            preds,
            gt,
            k=k,
        )

        logger.info(f"[{self.name}] MAP@{k} = {score:.6f}")

        return score


# ─────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────


def run(
    cfg: CFConfig | None = None,
) -> float:

    cfg = cfg or CFConfig()

    logger.info("Loading transactions...")

    tx = pd.read_parquet(cfg.data_dir / "transactions.parquet")
    tx = tx[
        [
            "customer_id",
            "article_id",
            "t_dat",
        ]
    ]

    # =========================================================
    # CANONICAL SPLIT (NO LEAKAGE)
    # =========================================================

    train_full, _, _, _, test_gt = canonical_split(tx)

    logger.info(f"Train rows: {len(train_full):,}")

    model = ItemCFRecommender(cfg)

    model.fit(train_full)

    score = model.evaluate(
        test_gt,
        k=cfg.k,
        sample=cfg.sample_eval,
    )

    return score


if __name__ == "__main__":
    run()
