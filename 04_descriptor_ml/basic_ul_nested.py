"""Nested version of the forward selection in basic_ul.py: features are chosen inside each outer training set only."""
import numpy as np, pandas as pd, collections
from scipy.stats import spearmanr
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from common import *
import basic_ul_data as B
d, y, g, POOL = B.d, B.y, B.g, B.POOL
fast = lambda: make_pipeline(StandardScaler(), KernelRidge(kernel='rbf', alpha=0.1, gamma=0.1))
def inner_cv(cols, idx):
    X = d[cols].values.astype(float)[idx]; yy = y[idx]; gg = g[idx]; p = np.zeros(len(yy))
    for tr, te in LeaveOneGroupOut().split(X, yy, gg): p[te] = fast().fit(X[tr], yy[tr]).predict(X[te])
    e = np.abs(p - yy); return e.mean(), e.std() / np.sqrt(len(e))
pred = np.zeros(len(y)); picks = []
for tr, te in LeaveOneGroupOut().split(d, y, g):
    chosen, path = [], []
    for step in range(7):
        res = [(f, *inner_cv(chosen + [f], tr)) for f in POOL if f not in chosen]
        f, m, se = min(res, key=lambda r: r[1]); chosen.append(f); path.append((m, se))
    k = int(np.argmin([p[0] for p in path])); thr = path[k][0] + path[k][1]
    k1 = min(i for i, p in enumerate(path) if p[0] <= thr) + 1
    cols = chosen[:k1]; picks.append(cols)
    pred[te] = B.krr().fit(d[cols].values.astype(float)[tr], y[tr]).predict(d[cols].values.astype(float)[te])
m, lo, hi = boot_mae(pred - y)
print(f'nested forward selection: MAE {m:.3f} [{lo:.3f}-{hi:.3f}], R2 {r2(y, pred):.2f}, rho {spearmanr(y, pred)[0]:.2f}, baseline {np.abs(y - y.mean()).mean():.3f}')
print('features per fold (count of folds):', collections.Counter(f for p in picks for f in p).most_common())
print('set sizes:', [len(p) for p in picks])
pd.DataFrame({'fold': range(len(picks)), 'features': [', '.join(p) for p in picks]}).to_csv(f'{OUT}/basic_ul_nested_picks.csv', index=False)
