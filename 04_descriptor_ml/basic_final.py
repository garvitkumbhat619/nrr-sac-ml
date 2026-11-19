"""Numbers for the thesis section on U_L from basic features (unseen site, isomers grouped)."""
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from scipy.optimize import least_squares
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from common import *
import basic_ul_data as B
d, y, g = B.d, B.y, B.g
sp = list(LeaveOneGroupOut().split(d, y, g))
def cv(cols, yy):
    X = d[cols].values.astype(float); p = np.zeros(len(yy))
    for tr, te in sp: p[te] = B.krr().fit(X[tr], yy[tr]).predict(X[te])
    return p
def row(label, k, p):
    m, lo, hi = boot_mae(p - y); top = len(set(np.argsort(-y)[:20]) & set(np.argsort(-p)[:20]))
    return dict(model=label, n_features=k, MAE=m, lo=lo, hi=hi, R2=r2(y, p), rho=spearmanr(y, p)[0], top20=top)
F4 = ['group', 'period', 'CN', 'n_C']
P = {}
P['metal2'] = cv(['group', 'period'], y)
P['basic4'] = cv(F4, y)
s1 = d.dG_N2_to_NNH_end.values; s6 = d.dG_NH2_to_NH3.values
P['basic4_steps'] = -np.maximum(cv(F4, s1), cv(F4, s6))
P['all18'] = cv(METAL + COORD, y)
x = d.dE_NNH_end.values; pl = np.zeros(len(y))
for tr, te in sp:
    a, b = np.polyfit(x[tr], y[tr], 1); pl[te] = a * x[te] + b
P['dENNH'] = pl
rows = [row('Metal only (group, period)', 2, P['metal2']), row('Four basic features', 4, P['basic4']),
        row('Four basic features, two steps', 4, P['basic4_steps']), row('All 18 descriptors', 18, P['all18']),
        row('UMA ΔE(*NNH) only', 1, P['dENNH'])]
T = pd.DataFrame(rows); T['baseline'] = np.abs(y - y.mean()).mean()
cls = (y >= -0.9).astype(int); X = d[F4].values.astype(float); pr = np.zeros(len(y))
for tr, te in sp:
    m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(X[tr], cls[tr]); pr[te] = m.predict_proba(X[te])[:, 1]
auc = roc_auc_score(cls, pr)
# how many of the 96 promising SACs are found among the 96 highest predicted (precision at n)
k = cls.sum(); hit = cls[np.argsort(-pr)[:k]].mean()
T.to_csv(f'{OUT}/basic_final_models.csv', index=False)
pd.DataFrame({k: v for k, v in P.items()}).assign(SAC=d.SAC, family=d.family, UL=y).to_csv(f'{OUT}/basic_final_preds.csv', index=False)
print(T.round(3).to_string(index=False)); print(f'classification U_L >= -0.9 V: {k} of {len(y)} SACs; AUC {auc:.2f}; precision at {k}: {hit:.2f}')
