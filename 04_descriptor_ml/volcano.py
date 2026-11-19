"""Item 1: scaling relations, a two-descriptor volcano for U_L, and descriptor ML for the two adsorption energies.
Writes out/scaling.csv, out/volcano_cv.csv, out/adsorb_ml.csv, out/volcano_points.csv, out/volcano_fit.json
"""
import json, numpy as np, pandas as pd
from scipy.optimize import least_squares
from scipy.stats import spearmanr, linregress
from sklearn.model_selection import LeaveOneGroupOut
from common import *

df = load_graphene()
G_NH3 = -0.17
PATHS = {'distal': (['N2_end', 'NNH_end', 'NNH2', 'N', 'NH', 'NH2', 'NH3'], [0, 0, 0, 1, 1, 1, 1]),
         'alternating': (['N2_end', 'NNH_end', 'NHNH', 'NHNH2', 'NH2NH2', 'NH2', 'NH3'], [0, 0, 0, 0, 0, 1, 1])}


def pds(r):
    best = None
    for p, (seq, out) in PATHS.items():
        if not all(r.get(f'status_{k}') == 'ok' and pd.notna(r.get(f'dG_{k}')) for k in seq):
            continue
        st = [r[f'dG_{k}'] + n * G_NH3 for k, n in zip(seq, out)]
        steps = np.diff(st); k = int(np.argmax(steps)); ul = -steps[k]
        if best is None or ul > best[0]:
            best = (ul, f'{seq[k]}->{seq[k + 1]}')
    return pd.Series(best if best else (np.nan, None), index=['UL_chk', 'PDS'])


df = pd.concat([df.drop(columns=['PDS', 'UL_chk'], errors='ignore'), df.apply(pds, axis=1)], axis=1)
V = df[df.UL.notna() & (df.status_NNH_end == 'ok') & (df.status_NH2 == 'ok')].reset_index(drop=True)
print('U_L reproduced:', np.abs(V.UL - V.UL_chk).max().round(4), '| n =', len(V))
cls = V.PDS.map(lambda s: 'first step' if s == 'N2_end->NNH_end' else ('*NH2→*NH3' if s == 'NH2->NH3' else 'other'))
print('PDS classes:', cls.value_counts().to_dict())

# ---------------------------------------------------------------- scaling relations (UMA and DFT)
xu = pd.read_csv(f'{FIN}/xu_reference.csv')
cn = pd.read_csv(f'{CN}/cn_merged.csv'); cn = cn[cn.status_bare == 'ok']
okNNH = (df.status_N2_end == 'ok') & (df.status_NNH_end == 'ok')
okNH = (df.status_NH2 == 'ok') & (df.status_NH3 == 'ok')
sc = []
def add(name, data, x, y, src):
    d = data[[x, y]].dropna(); lr = linregress(d[x], d[y])
    sc.append(dict(relation=name, source=src, n=len(d), slope=lr.slope, intercept=lr.intercept, r=lr.rvalue,
                   rmse=np.sqrt(((d[y] - lr.intercept - lr.slope * d[x]) ** 2).mean())))
add('dE(*NNH) vs dE(*N2)', df[okNNH], 'dE_N2_end', 'dE_NNH_end', 'UMA, 17 graphene sites')
add('dG(*N2->*NNH) vs dE(*NNH)', df[okNNH], 'dE_NNH_end', 'dG_N2_to_NNH_end', 'UMA, 17 graphene sites')
add('dG(*NH2->*NH3) vs dE(*NH2)', df[okNH], 'dE_NH2', 'dG_NH2_to_NH3', 'UMA, 17 graphene sites')
add('dE(*NNH) vs dE(*N2)', xu, 'xu_dE_N2_end', 'xu_dE_NNH_end', 'DFT, Xu 2025 (S/N)')
add('dG(*N2->*NNH) vs dE(*NNH)', xu, 'xu_dE_NNH_end', 'xu_dG_N2_to_NNH_end', 'DFT, Xu 2025 (S/N)')
add('dG(*NH2->*NH3) vs dE(*NH2)', xu, 'xu_dE_NH2', 'xu_dG_NH2_to_NH3', 'DFT, Xu 2025 (S/N)')
add('dG(*N2->*NNH) vs dE(*NNH)', cn, 'dE_N2H', 'dG_N2_to_N2H', 'DFT, Xu 2024 (C/N)')
add('dG(*N2->*NNH) vs dE(*NNH)', cn, 'dE_NNH_end', 'dG_N2_to_NNH_end', 'UMA, carbon nitrides')
sc = pd.DataFrame(sc); sc.to_csv(f'{OUT}/scaling.csv', index=False)
pd.set_option('display.width', 220); print(sc.round(3).to_string(index=False))

# ---------------------------------------------------------------- volcano: U_L = -max(a1 x1 + b1, a2 x2 + b2)
f = lambda p, a, b: -np.maximum(p[0] * a + p[1], p[2] * b + p[3])
def fit(x1, x2, y):
    return least_squares(lambda p: f(p, x1, x2) - y, [0.5, 1.0, -0.2, 0.3]).x
x1, x2, y = V.dE_NNH_end.values, V.dE_NH2.values, V.UL.values
P_all = fit(x1, x2, y)
print('volcano (all data): U_L = -max(%.2f dE(*NNH) %+.2f, %.2f dE(*NH2) %+.2f)' % tuple(P_all))
grp = V.site_group.values

# ML for the two adsorption energies from metal + site descriptors (all SACs with that state)
ADS = {'dE(*NNH)': ('dE_NNH_end', df.status_NNH_end == 'ok'), 'dE(*NH2)': ('dE_NH2', df.status_NH2 == 'ok')}
am, ADS_OOF = [], {}
for t, (col, mask) in ADS.items():
    d = df[mask & df[col].notna()].reset_index(drop=True); yy = d[col].values
    for fs, cols in FEATS.items():
        X = d[cols].values.astype(float)
        for mn in models():
            p = oof(X, yy, d.site_group.values, mn); m_, lo, hi = boot_mae(p - yy)
            am.append(dict(target=t, features=fs, model=mn, n=len(yy), MAE=m_, lo=lo, hi=hi, R2=r2(yy, p),
                           baseline=np.abs(yy - yy.mean()).mean()))
            ADS_OOF[(t, fs, mn)] = pd.Series(p, index=d.SAC)
am = pd.DataFrame(am); am.to_csv(f'{OUT}/adsorb_ml.csv', index=False)
amb = am.loc[am.groupby(['target', 'features']).MAE.idxmin()]
print(amb.round(3).to_string(index=False))
b1 = amb[(amb.target == 'dE(*NNH)') & (amb.features == 'metal + coordination')].model.iloc[0]
b2 = amb[(amb.target == 'dE(*NH2)') & (amb.features == 'metal + coordination')].model.iloc[0]

# predicted adsorption energies for the hierarchical model must also be out-of-site: refit per held-out group
X_all = df[METAL + COORD].values.astype(float)
pred = {k: np.full(len(V), np.nan) for k in ('volcano_UMA', 'volcano_ML', 'direct_ML', 'one_desc')}
dmask = {t: (mask & df[col].notna()).values for t, (col, mask) in ADS.items()}
Xv = V[METAL + COORD].values.astype(float)
for gname in np.unique(grp):
    te = grp == gname; tr = ~te
    P = fit(x1[tr], x2[tr], y[tr])
    pred['volcano_UMA'][te] = f(P, x1[te], x2[te])
    lr = linregress(x1[tr], y[tr]); pred['one_desc'][te] = lr.intercept + lr.slope * x1[te]
    hat = []
    for t, (col, _), bm in (('dE(*NNH)', ADS['dE(*NNH)'], b1), ('dE(*NH2)', ADS['dE(*NH2)'], b2)):
        trm = dmask[t] & (df.site_group.values != gname)
        m = models()[bm]; m.fit(X_all[trm], df[col].values[trm]); hat.append(m.predict(Xv[te]))
    pred['volcano_ML'][te] = f(P, hat[0], hat[1])
    m = models()['KRR']; m.fit(Xv[tr], y[tr]); pred['direct_ML'][te] = m.predict(Xv[te])

def topk(p, k=20):
    t = set(np.argsort(-y)[:k]); q = set(np.argsort(-p)[:k]); return len(t & q)
vc = []
for k, lab in (('direct_ML', 'Descriptors → U_L (KRR)'), ('volcano_ML', 'Descriptors → ΔE(*NNH), ΔE(*NH2) → volcano'),
               ('volcano_UMA', 'UMA ΔE(*NNH), ΔE(*NH2) → volcano'), ('one_desc', 'UMA ΔE(*NNH) only (linear)')):
    p = pred[k]; m_, lo, hi = boot_mae(p - y)
    vc.append(dict(model=lab, key=k, n=len(y), MAE=m_, lo=lo, hi=hi, R2=r2(y, p), rho=spearmanr(y, p)[0], top20=topk(p),
                   baseline=np.abs(y - y.mean()).mean()))
vc = pd.DataFrame(vc); vc.to_csv(f'{OUT}/volcano_cv.csv', index=False)
print(vc.round(3).to_string(index=False))
for a, b in (('volcano_ML', 'direct_ML'), ('volcano_UMA', 'direct_ML')):
    print(a, '-', b, np.round(boot_diff(pred[a] - y, pred[b] - y), 3))

# relation between the two descriptors (for the 1D volcano drawing)
lr12 = linregress(x1, x2)
json.dump(dict(P_all=list(P_all), x2_on_x1=[lr12.slope, lr12.intercept, lr12.rvalue], b1=b1, b2=b2),
          open(f'{OUT}/volcano_fit.json', 'w'))
V.assign(PDS_class=cls.values, **{k: v for k, v in pred.items()})[
    ['SAC', 'family', 'Support', 'site_group', 'dE_NNH_end', 'dE_NH2', 'UL', 'PDS', 'PDS_class'] + list(pred)].to_csv(f'{OUT}/volcano_points.csv', index=False)
print('PDS of the finalists:', V[V.SAC.isin(['Re@N2C2-PEN', 'Re@N2C2-OPP', 'Mo@N2C2-PEN'])][['SAC', 'UL', 'PDS']].to_string(index=False))
