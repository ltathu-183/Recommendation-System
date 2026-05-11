"""Feature engineering utilities with consistent missing value handling.

This module ensures train/inference consistency by using the same imputation
strategy in both phases.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from loguru import logger


# ── Feature Imputation Rules ──────────────────────────────────────────────────

# Define imputation rules: column -> value or callable
FEATURE_IMPUTES: dict[str, float | int | Callable[[pd.DataFrame, str], pd.Series]] = {
    # User features
    "user_total_tx": 0,
    "user_unique_articles": 0,
    "user_avg_price": 0.0,
    "user_price_std": 0.0,
    "user_days_since_last_tx": 9999,  # Large value for "never purchased"
    "user_tx_2w": 0,
    "user_online_ratio": 0.5,  # Neutral value
    "age": lambda df, col: df[col].fillna(df[col].median()),
    "age_group_enc": -1,  # Unknown
    "FN": 0,
    "Active": 0,
    "club_active": 0,
    "news_regular": 0,
    # Article features
    "art_total_tx": 0,
    "art_unique_customers": 0,
    "art_avg_price": 0.0,
    "art_days_since_last_sale": 9999,  # Large value for "never sold"
    "art_trend_score": 0.0,
    "art_category_pop_2w": 0,
    "art_pop_1w": 0,
    "art_pop_2w": 0,
    "art_pop_4w": 0,
    "product_type_no": -1,
    "product_group_enc": -1,
    "graphical_appearance_no": -1,
    "colour_group_code": -1,
    "index_group_no": -1,
    "section_no": -1,
    "garment_group_no": -1,
    # User-article features
    "ua_has_purchased": 0,
    "ua_purchase_count": 0,
    "ua_days_since_purchase": 9999,  # Large value for "never purchased"
    "ua_same_product_code": 0,
    "ua_category_purchases": 0,
    "ua_price_affinity": 0.0,
    "candidate_source_enc": 0,
}


def impute_features(df: pd.DataFrame, impute_rules: dict | None = None) -> pd.DataFrame:
    """Apply consistent imputation to features.

    Args:
        df: DataFrame with features
        impute_rules: Dictionary mapping column names to imputation values or callables.
                     If None, uses default FEATURE_IMPUTES.

    Returns:
        DataFrame with imputed values
    """
    df = df.copy()
    rules = impute_rules or FEATURE_IMPUTES

    for col, impute_val in rules.items():
        if col not in df.columns:
            continue

        if callable(impute_val):
            df[col] = impute_val(df, col)
        else:
            df[col] = df[col].fillna(impute_val)

    return df


def get_feature_columns(
    df: pd.DataFrame,
    exclude_cols: set[str] | None = None,
    include_dtypes: tuple | None = None,
) -> list[str]:
    """Dynamically determine feature columns for model training.

    Args:
        df: DataFrame containing features
        exclude_cols: Set of columns to exclude (e.g., IDs, labels)
        include_dtypes: Tuple of numpy/pandas dtypes to include

    Returns:
        List of feature column names
    """
    if exclude_cols is None:
        exclude_cols = {
            "customer_id",
            "article_id",
            "source",
            "label",
            "score",
            "t_dat",
            "week",
            "price",
            "sales_channel_id",
        }

    if include_dtypes is None:
        include_dtypes = (np.number,)

    features = []
    for col in df.columns:
        if col in exclude_cols:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            features.append(col)

    return features


def validate_features(
    df: pd.DataFrame,
    feature_cols: list[str],
    raise_on_missing: bool = True,
) -> dict[str, int]:
    """Validate feature columns and report missing values.

    Args:
        df: DataFrame to validate
        feature_cols: Expected feature columns
        raise_on_missing: If True, raises error for missing columns

    Returns:
        Dictionary of column -> missing count
    """
    missing_cols = set(feature_cols) - set(df.columns)
    if missing_cols and raise_on_missing:
        raise ValueError(f"Missing feature columns: {missing_cols}")

    missing_counts = {}
    for col in feature_cols:
        if col in df.columns:
            missing_counts[col] = df[col].isna().sum()

    total_missing = sum(missing_counts.values())
    if total_missing > 0:
        logger.warning(
            f"Found {total_missing:,} missing values across {len([c for c, v in missing_counts.items() if v > 0])} feature columns"
        )
        # Log top columns with missing values
        sorted_missing = sorted(missing_counts.items(), key=lambda x: -x[1])[:5]
        for col, count in sorted_missing:
            if count > 0:
                logger.warning(f"  {col}: {count:,} missing ({count / len(df):.1%})")

    return missing_counts


def build_features_with_imputation(
    df: pd.DataFrame,
    feature_cols: list[str] | None = None,
    impute_rules: dict | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Build features with consistent imputation.

    This is the main entry point for feature preparation that ensures
    train/inference consistency.

    Args:
        df: DataFrame with raw features
        feature_cols: List of feature columns (auto-detected if None)
        impute_rules: Imputation rules (defaults if None)

    Returns:
        Tuple of (imputed_df, feature_columns)
    """
    # Auto-detect features if not specified
    if feature_cols is None:
        feature_cols = get_feature_columns(df)
        logger.info(f"Auto-detected {len(feature_cols)} feature columns")

    # Validate features
    validate_features(df, feature_cols)

    # Apply imputation
    df_imputed = impute_features(df, impute_rules)

    return df_imputed, feature_cols


# ── Legacy compatibility ─────────────────────────────────────────────────────


def fill_missing_features(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Legacy function for backward compatibility.

    Uses the new imputation system internally.
    """
    df = df.copy()

    # Ensure all feature columns exist
    for col in feature_cols:
        if col not in df.columns:
            # Get default value from imputation rules
            default_val = FEATURE_IMPUTES.get(col, 0)
            if callable(default_val):
                default_val = 0
            df[col] = default_val

    # Apply imputation
    df = impute_features(df)

    return df
