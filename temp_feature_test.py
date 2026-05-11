import pandas as pd
from rec_sys.model import generate_candidates, build_features

hist = pd.DataFrame({
    'customer_id': ['c1', 'c1', 'c2', 'c2'],
    'article_id': [1, 2, 2, 3],
    't_dat': pd.to_datetime(['2020-09-01', '2020-09-10', '2020-09-08', '2020-09-15']),
    'price': [10.0, 20.0, 15.0, 30.0],
})
articles = pd.DataFrame({
    'article_id': [1, 2, 3],
    'product_code': ['p1', 'p2', 'p2'],
    'product_group_name': ['A', 'A', 'B'],
})
target_users = ['c1', 'c2']
pred = pd.Timestamp('2020-09-16')

cands = generate_candidates(hist, target_users, 10, pred, articles=articles)
print('cands', cands.to_dict(orient='records'))

df = build_features(hist, cands, pred, articles=articles)
print('features', df.head().to_dict(orient='records'))
