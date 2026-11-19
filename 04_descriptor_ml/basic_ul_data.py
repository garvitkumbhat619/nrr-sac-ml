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


