"""Item 3: Chapter 6 evaluation with isomer-grouped hold-out, a coordination-only ablation and bootstrap CIs.
Writes out/cv_all.csv, out/cv_best.csv, out/paired.csv, out/expansion.csv, out/oof.pkl
"""
import os, pickle, numpy as np, pandas as pd
from scipy.stats import spearmanr
from common import *

os.makedirs(OUT, exist_ok=True)
df = load_graphene()
TM = target_masks(df)
SCHEMES = {'random 5-fold': None, 'unseen site (isomers grouped)': 'site_group', 'unseen site (old, per isomer)': 'Support',
           'unseen family': 'family'}
rows, OOF = [], {}
for t, (mask, col) in TM.items():
    d = df[mask].reset_index(drop=True); y = d[col].values
    for fs, cols in FEATS.items():
        X = d[cols].values.astype(float)
        for sch, gcol in SCHEMES.items():
            if sch.startswith('unseen site (old') and fs != 'metal + coordination':
                continue
            g = None if gcol is None else d[gcol].values
            for mn in models():
                p = oof(X, y, g, mn); OOF[(t, fs, sch, mn)] = p
                mae, lo, hi = boot_mae(p - y)
                rows.append(dict(target=t, features=fs, scheme=sch, model=mn, n=len(y), MAE=mae, lo=lo, hi=hi,
                                 R2=r2(y, p), rho=spearmanr(y, p)[0], baseline=np.abs(y - y.mean()).mean()))
        print(t, 'done', flush=True)
res = pd.DataFrame(rows); res.to_csv(f'{OUT}/cv_all.csv', index=False)
best = res.loc[res.groupby(['target', 'features', 'scheme']).MAE.idxmin()]
best.to_csv(f'{OUT}/cv_best.csv', index=False)
pd.set_option('display.width', 220)
print(best.round(3).to_string(index=False))

# spread between the three models in each cell (is the "best model" choice meaningful?)
spread = res.groupby(['target', 'features', 'scheme']).MAE.agg(lambda s: s.max() - s.min()).rename('model_spread')
print(spread.round(3).to_string())

# paired comparisons on the same catalysts with the best model of each cell
pr = []
for t, (mask, col) in TM.items():
    y = df[mask][col].values
    for sch in ('random 5-fold', 'unseen site (isomers grouped)', 'unseen family'):
        b = best[(best.target == t) & (best.scheme == sch)].set_index('features').model
        e = {fs: OOF[(t, fs, sch, b[fs])] - y for fs in FEATS}
        for a_, b_ in (('metal only', 'metal + coordination'), ('coordination only', 'metal + coordination')):
            dlt, lo, hi = boot_diff(e[a_], e[b_])
            pr.append(dict(target=t, scheme=sch, comparison=f'{a_} - both', dMAE=dlt, lo=lo, hi=hi))
pr = pd.DataFrame(pr); pr.to_csv(f'{OUT}/paired.csv', index=False)
print(pr.round(3).to_string(index=False))

# ---------------------------------------------------------------- with vs without expansion (isomers grouped)
NEW = ~df.family.str.startswith('S/N')
ex = []
for t, (mask, col) in TM.items():
    d = df[mask].reset_index(drop=True); y = d[col].values; X = d[METAL + COORD].values.astype(float)
    new = (~d.family.str.startswith('S/N')).values
    for mn in ('RF', 'XGB'):
        m = models()[mn]; m.fit(X[~new], y[~new]); p0 = m.predict(X[new])
        for sch, gcol in (('isomers grouped', 'site_group'), ('old, per isomer', 'Support')):
            p1 = oof(X, y, d[gcol].values, mn, test_mask=new)[new]
            a0, l0, h0 = boot_mae(p0 - y[new]); a1, l1, h1 = boot_mae(p1 - y[new])
            dd = boot_diff(p0 - y[new], p1 - y[new])
            ex.append(dict(target=t, model=mn, scheme=sch, n=int(new.sum()), MAE_without=a0, lo0=l0, hi0=h0, R2_without=r2(y[new], p0),
                           rho_without=spearmanr(y[new], p0)[0], MAE_with=a1, lo1=l1, hi1=h1, R2_with=r2(y[new], p1),
                           rho_with=spearmanr(y[new], p1)[0], gain=dd[0], gain_lo=dd[1], gain_hi=dd[2],
                           baseline=np.abs(y[new] - y[new].mean()).mean()))
            if mn == 'RF' and sch == 'isomers grouped':
                OOF[(t, 'expansion_with', mn)] = (d.SAC[new].values, p1); OOF[(t, 'expansion_without', mn)] = (d.SAC[new].values, p0)
ex = pd.DataFrame(ex); ex.to_csv(f'{OUT}/expansion.csv', index=False)
print(ex.round(3).to_string(index=False))
pickle.dump(OOF, open(f'{OUT}/oof.pkl', 'wb'))
