"""Can U_L be predicted from basic features only (no free energies, no adsorbate calculations), with few features?

Target: UMA U_L (best of distal/alternating, 331 SACs). Validation: unseen site, isomers held out together (13 folds).
Pools: 8 metal + 10 site descriptors + two literature descriptors:
  phi_NC  = theta_d (n_N chi_N + n_C chi_C + n_S chi_S) / chi_M          (Xu, Cheng, Cao, Zeng, Nat. Catal. 2018)
  phi_Xu  = N_d (chi_site + 1 - chi_M) / sqrt(chi_site chi_M)            (Xu et al. 2025, Eq. 8)
Writes out/basic_ul_single.csv, out/basic_ul_forward.csv, out/basic_ul_summary.csv
"""
import numpy as np, pandas as pd
from scipy.optimize import least_squares
from scipy.stats import spearmanr
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import LeaveOneGroupOut, GridSearchCV, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from common import *

df = load_graphene()
df['phi_NC'] = df.Nd * (df.n_N * CHI['N'] + df.n_C * CHI['C'] + df.n_S * CHI['S']) / df.chi_M
df['phi_Xu'] = df.Nd * (df.chi_site + 1 - df.chi_M) / np.sqrt(df.chi_site * df.chi_M)
d = df[df.UL.notna()].reset_index(drop=True); y = d.UL.values; g = d.site_group.values
POOL = METAL + COORD + ['phi_NC', 'phi_Xu']
base = np.abs(y - y.mean()).mean()
splits = list(LeaveOneGroupOut().split(d, y, g))


def krr():
    return GridSearchCV(make_pipeline(StandardScaler(), KernelRidge(kernel='rbf')),
                        {'kernelridge__alpha': [0.03, 0.1, 0.3, 1], 'kernelridge__gamma': [0.03, 0.1, 0.3]},
                        cv=KFold(4, shuffle=True, random_state=0), scoring='neg_mean_absolute_error')


def cv_krr(cols):
    X = d[cols].values.astype(float); p = np.zeros(len(y))
    for tr, te in splits:
        p[te] = krr().fit(X[tr], y[tr]).predict(X[te])
    return p


def stats(p, label, k):
    m, lo, hi = boot_mae(p - y)
    top = len(set(np.argsort(-y)[:20]) & set(np.argsort(-p)[:20]))
    return dict(model=label, n_features=k, MAE=m, lo=lo, hi=hi, R2=r2(y, p), rho=spearmanr(y, p)[0], top20=top, baseline=base)


# 1. one basic feature at a time, as a volcano (two lines) and with KRR
vol = lambda q, x: -np.maximum(q[0] * x + q[1], q[2] * x + q[3])
single = []
for f in POOL:
    x = d[f].values.astype(float); pv = np.zeros(len(y))
    for tr, te in splits:
        q = least_squares(lambda q: vol(q, x[tr]) - y[tr], [0.1, 1.0, -0.1, 1.0]).x; pv[te] = vol(q, x[te])
    pk = cv_krr([f])
    single.append(dict(feature=f, MAE_volcano=np.abs(pv - y).mean(), R2_volcano=r2(y, pv), MAE_krr=np.abs(pk - y).mean(), R2_krr=r2(y, pk)))
single = pd.DataFrame(single).sort_values('MAE_krr'); single.to_csv(f'{OUT}/basic_ul_single.csv', index=False)
pd.set_option('display.width', 200); print(single.round(3).head(10).to_string(index=False))

# 2. forward selection (KRR), one-standard-error rule
chosen, path = [], []
for step in range(8):
    best = None
    for f in POOL:
        if f in chosen: continue
        p = cv_krr(chosen + [f]); e = np.abs(p - y)
        if best is None or e.mean() < best[1]: best = (f, e.mean(), e.std() / np.sqrt(len(e)), p)
    chosen.append(best[0]); path.append(dict(step=step + 1, added=best[0], features=', '.join(chosen), MAE=best[1], se=best[2], R2=r2(y, best[3])))
    print(path[-1], flush=True)
path = pd.DataFrame(path); path.to_csv(f'{OUT}/basic_ul_forward.csv', index=False)
kmin = path.MAE.idxmin(); thr = path.MAE[kmin] + path.se[kmin]
k1se = int(path[path.MAE <= thr].step.min())
print(f'minimum MAE {path.MAE[kmin]:.3f} at {kmin + 1} features; 1-SE rule picks {k1se}: {path.features[k1se - 1]}')

summ = [stats(cv_krr(POOL[:18]), 'all 18 basic descriptors', 18)]
for k in sorted({1, 2, 3, k1se}):
    summ.append(stats(cv_krr(path.features[k - 1].split(', ')), f'forward selection, {k}: {path.features[k - 1]}', k))
summ = pd.DataFrame(summ)

# 3. classification of the promising region (U_L >= -0.9 V) with the selected basic features
cls = (y >= -0.9).astype(int); cols = path.features[k1se - 1].split(', ')
X = d[cols].values.astype(float); pr = np.zeros(len(y))
for tr, te in splits:
    if cls[tr].min() == cls[tr].max(): pr[te] = cls[tr].mean(); continue
    m = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000)).fit(X[tr], cls[tr]); pr[te] = m.predict_proba(X[te])[:, 1]
auc = roc_auc_score(cls, pr)
print(f'promising region U_L >= -0.9 V: {cls.sum()} of {len(cls)} SACs; AUC with {k1se} basic features (logistic) = {auc:.2f}')
summ.to_csv(f'{OUT}/basic_ul_summary.csv', index=False)
print(summ.round(3).to_string(index=False))
