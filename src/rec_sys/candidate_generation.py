"""Candidate generation with stratified sampling and negative sampling.

This module implements:
1. Stratified candidate selection to ensure diversity across sources
2. Negative sampling for training efficiency
3. Cold-start candidate generation
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd


# ── Stratified Candidate Selection ───────────────────────────────────────────


def stratified_candidate_selection(
    candidates: pd.DataFrame,
    n_candidates: int,
    source_quotas: dict[str, int] | None = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """Select candidates ensuring minimum representation from each source.

    This addresses the issue where deduplication prioritizes repurchase candidates,
    limiting diversity and preventing the ranker from learning across signal types.

    Args:
        candidates: DataFrame with columns [customer_id, article_id, source]
        n_candidates: Total candidates per user
        source_quotas: Dict mapping source names to minimum quotas.
                      If None, uses default quotas.
        random_state: Random seed for reproducibility

    Returns:
        DataFrame with stratified candidates
    """
    # Default quotas: ensure diverse candidate sources
    if source_quotas is None:
        source_quotas = {
            "repurchase_short": min(50, n_candidates // 4),
            "repurchase_long": min(30, n_candidates // 6),
            "product_code_repurchase": min(20, n_candidates // 8),
            "popular_global": min(40, n_candidates // 4),
            "popular_segment": min(20, n_candidates // 8),
            "category_popular": min(20, n_candidates // 8),
        }

    selected_rows = []

    for uid, grp in candidates.groupby("customer_id"):
        user_candidates = []

        # First, ensure minimum from each source
        for source, quota in source_quotas.items():
            source_cands = grp[grp["source"] == source]
            if len(source_cands) > 0:
                # Take up to quota from this source
                n_take = min(quota, len(source_cands))
                sampled = source_cands.head(n_take)  # Already ordered by relevance
                user_candidates.append(sampled)

        # Fill remaining with best available (by source priority order)
        remaining = n_candidates - sum(len(c) for c in user_candidates)
        if remaining > 0:
            # Get already selected article IDs
            selected_ids = set()
            for c in user_candidates:
                selected_ids.update(c["article_id"].tolist())

            # Get remaining candidates not yet selected
            remaining_cands = grp[~grp["article_id"].isin(selected_ids)]

            # Sort by source priority and recency/popularity
            source_priority = {
                "repurchase_short": 6,
                "repurchase_long": 5,
                "product_code_repurchase": 4,
                "category_popular": 3,
                "popular_segment": 2,
                "popular_global": 1,
            }
            remaining_cands = remaining_cands.copy()
            remaining_cands["source_priority"] = (
                remaining_cands["source"].map(source_priority).fillna(0)
            )
            remaining_cands = remaining_cands.sort_values(
                "source_priority", ascending=False
            ).head(remaining)

            user_candidates.append(remaining_cands.drop(columns=["source_priority"]))

        # Combine and limit to n_candidates
        if user_candidates:
            combined = pd.concat(user_candidates, ignore_index=True)
            combined = combined.head(n_candidates)
            selected_rows.append(combined)

    if not selected_rows:
        return candidates.head(0)  # Empty DataFrame with same columns

    return pd.concat(selected_rows, ignore_index=True)


# ── Negative Sampling ──────────────────────────────────────────────────────────


def sample_hard_negatives(
    df: pd.DataFrame,
    pos_multiplier: int = 50,
    random_state: int = 42,
) -> pd.DataFrame:
    """Sample hard negatives to balance training data.

    Current training uses all non-purchased candidates as negatives,
    resulting in 98-99% negative labels. This function samples a
    manageable number of hard negatives per positive example.

    Hard negatives are selected based on:
    1. Popularity (popular items user didn't buy)
    2. Same category as purchased items
    3. Recent activity

    Args:
        df: DataFrame with columns including [customer_id, article_id, label]
        pos_multiplier: Number of negatives to sample per positive
        random_state: Random seed

    Returns:
        Balanced DataFrame with sampled negatives
    """
    dfs = []

    for uid, grp in df.groupby("customer_id"):
        pos = grp[grp["label"] == 1]
        neg = grp[grp["label"] == 0]

        if len(pos) == 0:
            # No positives - keep a small sample of negatives for training
            n_keep = min(10, len(neg))
            if n_keep > 0:
                dfs.append(neg.sample(n=n_keep, random_state=random_state))
            continue

        if len(neg) == 0:
            # No negatives - just keep positives
            dfs.append(pos)
            continue

        # Calculate how many negatives to sample
        n_neg_samples = min(len(neg), len(pos) * pos_multiplier)

        if n_neg_samples >= len(neg):
            # Keep all negatives if we don't have enough
            dfs.append(pd.concat([pos, neg]))
            continue

        # Sample hard negatives using stratified approach
        # 1. Popular negatives (50%)
        # 2. Category-matched negatives (30%)
        # 3. Random recent negatives (20%)

        neg_samples = []

        # Sort by candidate source priority for "hardness"
        source_priority = {
            "repurchase_short": 6,  # User bought similar recently
            "repurchase_long": 5,  # User bought before
            "product_code_repurchase": 4,  # Same product, different variant
            "category_popular": 3,  # Popular in user's category
            "popular_segment": 2,  # Popular in user's age group
            "popular_global": 1,  # Globally popular
        }

        neg_sorted = neg.copy()
        neg_sorted["source_score"] = neg_sorted["source"].map(source_priority).fillna(0)

        # Take from each stratum
        n_popular = int(n_neg_samples * 0.5)
        n_category = int(n_neg_samples * 0.3)
        n_random = n_neg_samples - n_popular - n_category

        # Popular negatives (high source score)
        popular_neg = neg_sorted.sort_values("source_score", ascending=False).head(
            n_popular
        )
        neg_samples.append(popular_neg)

        # Category matched (medium source score)
        remaining = neg_sorted[~neg_sorted.index.isin(popular_neg.index)]
        category_neg = remaining.sort_values("source_score", ascending=False).head(
            n_category
        )
        neg_samples.append(category_neg)

        # Random from remaining
        remaining2 = remaining[~remaining.index.isin(category_neg.index)]
        if len(remaining2) > 0:
            n_random_actual = min(n_random, len(remaining2))
            random_neg = remaining2.sample(n=n_random_actual, random_state=random_state)
            neg_samples.append(random_neg)

        # Combine
        sampled_neg = pd.concat(neg_samples).drop(
            columns=["source_score"], errors="ignore"
        )
        dfs.append(pd.concat([pos, sampled_neg]))

    return pd.concat(dfs).reset_index(drop=True)


# ── Cold-Start Candidate Generation ───────────────────────────────────────────


def generate_cold_start_candidates(
    target_users: list[str],
    customers: pd.DataFrame,
    articles: pd.DataFrame,
    recent_transactions: pd.DataFrame,
    n_candidates: int = 200,
    age_bins: list[int] | None = None,
    age_labels: list[str] | None = None,
) -> pd.DataFrame:
    """Generate diverse candidates for cold-start users (no purchase history).

    Instead of giving all cold-start users the same popularity-based candidates,
    this generates personalized candidates based on demographics and trends.

    Args:
        target_users: List of customer IDs with no history
        customers: Customer demographics DataFrame
        articles: Article metadata DataFrame
        recent_transactions: Recent transaction history
        n_candidates: Number of candidates per user
        age_bins: Age bin boundaries for segmentation
        age_labels: Labels for age groups

    Returns:
        DataFrame with cold-start candidates
    """
    if age_bins is None:
        age_bins = [15, 25, 35, 45, 55, 65, 100]
    if age_labels is None:
        age_labels = ["16-24", "25-34", "35-44", "45-54", "55-64", "65+"]

    # Get user demographics
    users_df = customers[customers["customer_id"].isin(target_users)].copy()

    # Add age groups
    users_df["age_group"] = pd.cut(
        users_df["age"], bins=age_bins, labels=age_labels, right=True
    ).astype(str)
    users_df.loc[users_df["age"].isna(), "age_group"] = "unknown"

    # Pre-compute popularity by segment
    t_max = recent_transactions["t_dat"].max()
    recent_2w = recent_transactions[
        recent_transactions["t_dat"] >= t_max - pd.Timedelta(weeks=2)
    ]

    # Age-group popular items
    tx_age = recent_2w.merge(
        users_df[["customer_id", "age_group"]], on="customer_id", how="left"
    )
    age_popular = {}
    for grp, grp_df in tx_age.groupby("age_group", observed=True):
        age_popular[str(grp)] = (
            grp_df["article_id"].value_counts().head(100).index.tolist()
        )

    # Category trending (diverse categories)
    art_pg = articles.set_index("article_id")["product_group_name"].to_dict()
    recent_2w_pg = recent_2w.copy()
    recent_2w_pg["product_group"] = recent_2w_pg["article_id"].map(art_pg)

    # Get top items from diverse categories
    category_trending = (
        recent_2w_pg.groupby("product_group")["article_id"]
        .apply(lambda x: x.value_counts().head(5).index.tolist())
        .explode()
        .dropna()
        .unique()
        .tolist()[:50]
    )

    # Global trending
    global_trending = recent_2w["article_id"].value_counts().head(100).index.tolist()

    # Generate candidates per user
    candidates = []

    for _, user in users_df.iterrows():
        uid = user["customer_id"]
        age_group = user["age_group"]

        # Personalized components
        age_items = age_popular.get(age_group, global_trending)[:50]
        cat_items = category_trending[:30]
        global_items = global_trending[:30]

        # Combine with diversification
        user_cands = []
        seen = set()

        # Interleave from different sources
        for i in range(max(len(age_items), len(cat_items), len(global_items))):
            if i < len(age_items) and age_items[i] not in seen:
                user_cands.append(
                    {
                        "customer_id": uid,
                        "article_id": age_items[i],
                        "source": "cold_start_age",
                    }
                )
                seen.add(age_items[i])

            if i < len(cat_items) and cat_items[i] not in seen:
                user_cands.append(
                    {
                        "customer_id": uid,
                        "article_id": cat_items[i],
                        "source": "cold_start_category",
                    }
                )
                seen.add(cat_items[i])

            if i < len(global_items) and global_items[i] not in seen:
                user_cands.append(
                    {
                        "customer_id": uid,
                        "article_id": global_items[i],
                        "source": "cold_start_global",
                    }
                )
                seen.add(global_items[i])

            if len(user_cands) >= n_candidates:
                break

        candidates.extend(user_cands[:n_candidates])

    return pd.DataFrame(candidates)


# ── Candidate Recall Analysis ─────────────────────────────────────────────────


def analyze_candidate_recall(
    candidates: pd.DataFrame,
    ground_truth: dict[str, set[str]],
    k: int = 12,
) -> dict:
    """Analyze candidate recall and diversity metrics.

    Args:
        candidates: DataFrame with [customer_id, article_id, source]
        ground_truth: Dict of user -> set of purchased items
        k: Top-k for recall calculation

    Returns:
        Dictionary of recall metrics
    """
    per_user_recall = []
    source_hits = defaultdict(int)
    source_totals = defaultdict(int)

    for uid in ground_truth:
        gt_items = ground_truth[uid]
        user_cands = candidates[candidates["customer_id"] == uid]

        # Overall recall
        cand_items = set(user_cands["article_id"])
        hits = len(cand_items & gt_items)
        per_user_recall.append(hits / len(gt_items) if gt_items else 0)

        # Per-source recall
        for source, grp in user_cands.groupby("source"):
            source_items = set(grp["article_id"])
            source_hits[source] += len(source_items & gt_items)
            source_totals[source] += len(gt_items)

    metrics = {
        "mean_recall": np.mean(per_user_recall),
        "median_recall": np.median(per_user_recall),
        "users_with_candidates": len(set(candidates["customer_id"])),
        "total_candidates": len(candidates),
        "avg_candidates_per_user": len(candidates)
        / candidates["customer_id"].nunique(),
        "source_distribution": candidates["source"].value_counts().to_dict(),
        "source_recall": {
            source: source_hits[source] / source_totals[source]
            if source_totals[source] > 0
            else 0
            for source in source_totals
        },
    }

    return metrics
