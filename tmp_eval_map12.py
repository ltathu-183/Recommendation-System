from pathlib import Path
import pandas as pd
from rec_sys.data_utils import make_splits_lgbm
from rec_sys.model import ModelConfig, TwoStageLGBMRanker

DATA_DIR = Path('data')
print('Loading transaction subset...')

# Load a manageable subset of the recent data for evaluation
cols = ['customer_id', 'article_id', 't_dat', 'price']
tx = pd.read_parquet(DATA_DIR / 'transactions.parquet', columns=cols)
tx = tx[tx['t_dat'] >= pd.Timestamp('2020-03-01')]
print(f'Filtered rows after 2020-03-01: {len(tx):,}')
if len(tx) > 500_000:
    tx = tx.head(500_000)
    print(f'Sampled first 500k rows for evaluation: {len(tx):,}')

train_full, train_feat, val_tx, val_gt, test_gt = make_splits_lgbm(tx)
print(f'train_full={len(train_full):,}, val={len(val_tx):,}, test_users={len(test_gt):,}')

cfg = ModelConfig(n_candidates=100, n_train_weeks=2, negative_sampling_ratio=4)
model = TwoStageLGBMRanker(cfg=cfg)
print('Fitting model...')
articles = pd.read_parquet(DATA_DIR / 'articles.parquet')
model.fit(train_full, articles=articles)
print('Evaluating model...')
score = model.evaluate(test_gt, k=12, sample=2000)
print(f'MAP@12 = {score:.6f}')
