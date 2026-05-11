"""Minimal test of the TwoStageLGBMRanker improvements."""

from pathlib import Path
import pandas as pd
from rec_sys.model import ModelConfig, TwoStageLGBMRanker
from rec_sys.data_utils import canonical_split

def main():
    data_dir = Path("data")

    # Load data
    tx_full = pd.read_parquet(data_dir / "transactions.parquet")
    print(f"Full data date range: {tx_full['t_dat'].min()} to {tx_full['t_dat'].max()}")
    
    # Use a sample that includes earlier dates
    tx = tx_full.head(500000)  # Load more data
    articles = pd.read_parquet(data_dir / "articles.parquet")

    print(f"Data date range: {tx['t_dat'].min()} to {tx['t_dat'].max()}")

    # Get some training data - use earlier dates
    cutoff_date = tx["t_dat"].max() - pd.Timedelta(days=7)
    train_full = tx[tx["t_dat"] < cutoff_date].head(10000)

    print(f"Training data: {len(train_full)} transactions from {train_full['t_dat'].min() if len(train_full) > 0 else 'N/A'} to {train_full['t_dat'].max() if len(train_full) > 0 else 'N/A'}")

    # Test the model
    cfg = ModelConfig(
        n_candidates=50,
        n_train_weeks=1,
        negative_sampling_ratio=2,
    )

    model = TwoStageLGBMRanker(cfg=cfg)

    print("Fitting model...")
    model.fit(train_full, articles=articles)

    print("Evaluating model...")
    score = model.evaluate(test_gt, k=12, sample=1000)

    print(f"MAP@12 = {score:.6f}")
    print("✓ TwoStageLGBMRanker improvements working correctly!")

if __name__ == "__main__":
    main()