"""Shared data and models for the revised Chapter 6 analysis (same dataset and settings as coordination_ML.ipynb)."""
import os, numpy as np, pandas as pd, warnings
from pathlib import Path
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, LeaveOneGroupOut, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
warnings.filterwarnings('ignore')

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
FIN = str(ROOT / '03_screening')                       # UMA_476_full.csv, metal_descriptors.csv, xu_reference.csv
CN = str(ROOT / '05_carbon_nitride_test' / 'results')  # cn_merged.csv, relaxed_bare/
OUT = str(HERE / 'out')

MD = pd.read_csv(f'{FIN}/metal_descriptors.csv')
CHI = {'N': 3.04, 'S': 2.58, 'C': 2.55}; RCOV = {'N': 0.71, 'S': 1.05, 'C': 0.76}
METAL = ['Z', 'period', 'group', 'Nd', 'chi_M', 'IE1', 'r_atom', 'Ecoh']
COORD = ['n_N', 'n_S', 'n_C', 'CN', 'chi_site', 'r_site', 'EN_diff', 'is_PEN', 'is_HEX', 'is_OPP']
FEATS = {'metal only': METAL, 'coordination only': COORD, 'metal + coordination': METAL + COORD}


def add_site_descriptors(d):
    d = d.copy()
    d['chi_site'] = (d.n_N * CHI['N'] + d.n_S * CHI['S'] + d.n_C * CHI['C']) / d.CN
    d['r_site'] = (d.n_N * RCOV['N'] + d.n_S * RCOV['S'] + d.n_C * RCOV['C']) / d.CN
    d['EN_diff'] = d.chi_M - d.chi_site
    for iso in ('PEN', 'HEX', 'OPP'):
        d[f'is_{iso}'] = d.Support.str.endswith(iso).astype(int)
    return d


def load_graphene():
    df = pd.read_csv(f'{FIN}/UMA_476_full.csv').drop(columns=['U0_diss_V', 'n_e'], errors='ignore').merge(MD, on='Metal')
    df = df[df.status_bare == 'ok'].copy()
    df = add_site_descriptors(df)
    # isomers of the same composition form one group, so that a held-out site never has its isomer in training
    df['site_group'] = df.Support.replace({'S2N2-PEN': 'S2N2', 'S2N2-HEX': 'S2N2', 'S2N2-OPP': 'S2N2',
                                           'N2C2-PEN': 'N2C2', 'N2C2-HEX': 'N2C2', 'N2C2-OPP': 'N2C2'})
    return df.reset_index(drop=True)


def target_masks(df):
    okN = (df.status_N2_end == 'ok') & (df.status_NNH_end == 'ok')
    return {
        'Ef': (df.Ef_cal.notna(), 'Ef_cal'),
        'step': (okN & df.dG_N2_to_NNH_end.notna(), 'dG_N2_to_NNH_end'),
        'UL': (df.UL.notna(), 'UL'),
        'dGH': ((df.status_H == 'ok') & df['dG_H*'].notna(), 'dG_H*'),
    }


def models():
    return {
        'KRR': GridSearchCV(make_pipeline(StandardScaler(), KernelRidge(kernel='rbf')),
                            {'kernelridge__alpha': [0.01, 0.1, 1], 'kernelridge__gamma': [0.01, 0.03, 0.1]},
                            cv=KFold(4, shuffle=True, random_state=0), scoring='neg_mean_absolute_error'),
        'RF': RandomForestRegressor(n_estimators=300, min_samples_leaf=2, random_state=0, n_jobs=-1),
        'XGB': XGBRegressor(n_estimators=400, learning_rate=0.05, max_depth=4, subsample=0.8, colsample_bytree=0.8,
                            random_state=0, n_jobs=4),
    }


def oof(X, y, groups, name, test_mask=None):
    """Out-of-fold predictions; groups None = random 5-fold. test_mask limits which groups are held out."""
    p = np.full(len(y), np.nan)
    if groups is None:
        splits = KFold(5, shuffle=True, random_state=0).split(X)
    else:
        splits = LeaveOneGroupOut().split(X, y, groups)
    for tr, te in splits:
        if test_mask is not None:
            te = te[test_mask[te]]
            if len(te) == 0:
                continue
        m = models()[name]; m.fit(X[tr], y[tr]); p[te] = m.predict(X[te])
    return p


def boot_mae(err, n=2000, seed=0):
    """95% percentile bootstrap CI of the MAE (resampling catalysts)."""
    rng = np.random.default_rng(seed); a = np.abs(np.asarray(err)); a = a[~np.isnan(a)]
    b = np.array([a[rng.integers(0, len(a), len(a))].mean() for _ in range(n)])
    return a.mean(), np.percentile(b, 2.5), np.percentile(b, 97.5)


def boot_diff(e1, e2, n=2000, seed=0):
    """CI of MAE(e1) - MAE(e2) on the same catalysts (paired)."""
    rng = np.random.default_rng(seed); a1, a2 = np.abs(e1), np.abs(e2)
    k = ~(np.isnan(a1) | np.isnan(a2)); a1, a2 = a1[k], a2[k]
    idx = [rng.integers(0, len(a1), len(a1)) for _ in range(n)]
    b = np.array([a1[i].mean() - a2[i].mean() for i in idx])
    return a1.mean() - a2.mean(), np.percentile(b, 2.5), np.percentile(b, 97.5)


def r2(y, p):
    return 1 - ((p - y) ** 2).sum() / ((y - y.mean()) ** 2).sum()
