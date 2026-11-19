"""Item 2: end-to-end test of the descriptor models on an unseen family against DFT (Xu et al. 2024 carbon nitrides).

Three ways of training the same descriptor models, all tested on the carbon-nitride SACs:
  (a) published DFT data: Xu 2025 S/N SACs (the data available before the expansion)
  (b) UMA labels on the 459 graphene SACs of this work (17 sites)
  (c) as (b) plus the UMA carbon-nitride data of the other four sheets (leave-one-sheet-out)
Writes out/cn_e2e.csv and out/cn_e2e_pred.csv
"""
import glob, numpy as np, pandas as pd
from ase.io import read
from ase.neighborlist import natural_cutoffs
from scipy.stats import spearmanr
from common import *

g = load_graphene()
best = pd.read_csv(f'{OUT}/cv_best.csv')
pick = lambda t, fs: best[(best.target == t) & (best.features == fs) & (best.scheme == 'unseen family')].model.iloc[0]

# ---------------------------------------------------------------- carbon-nitride descriptor frame
cn = pd.read_csv(f'{CN}/cn_merged.csv')
cn = cn[cn.status_bare == 'ok'].copy()
cnt = []
for sac in cn.SAC:
    a = read(f'{CN}/relaxed_bare/{sac}/bare.extxyz'); i = len(a) - 1
    d = a.get_distances(i, range(i), mic=True); sym = np.array(a.get_chemical_symbols()[:i])
    nb = np.where((d < d.min() + 0.35) & (d < 2.7))[0]          # first coordination shell of the metal
    cnt.append(dict(SAC=sac, n_N=int((sym[nb] == 'N').sum()), n_C=int((sym[nb] == 'C').sum())))
cn = cn.drop(columns=[c for c in ['n_N', 'n_C', 'n_S', 'CN'] + METAL if c in cn], errors='ignore').merge(pd.DataFrame(cnt), on='SAC').merge(MD, on='Metal')
cn['n_S'] = 0; cn['CN'] = cn.n_N + cn.n_C
cn = add_site_descriptors(cn)
cn['family'] = 'C/N'; cn['site_group'] = cn.Support
cn['Ef_cal'] = 0.832 * cn.Ef_eV - 0.208
print('carbon-nitride SACs:', len(cn), '| CN values:', cn.CN.value_counts().sort_index().to_dict())

# targets: (UMA column on graphene, UMA column on C/N, DFT column on C/N, Xu 2025 DFT column on S/N)
T = {'step': ('dG_N2_to_NNH_end', 'dG_N2_to_NNH_end', 'dG_N2_to_N2H', 'xu_dG_N2_to_NNH_end'),
     'dGH': ('dG_H*', 'dG_H', 'dG_H_xu', 'xu_dG_H'),
     'Ef': ('Ef_cal', 'Ef_cal', 'Ef_ref', 'xu_Ef_eV')}
TM = target_masks(g)
xu = pd.read_csv(f'{FIN}/xu_reference.csv')
sn = g[g.family.str.startswith('S/N')].merge(xu[['SAC'] + [v[3] for v in T.values()]], on='SAC')

rows, preds = [], []
for t, (cg, cc, cd, cx) in T.items():
    test = cn.dropna(subset=[cd]).reset_index(drop=True); yd = test[cd].values
    def rec(label, fs, p, ref='DFT'):
        yy = yd if ref == 'DFT' else test[cc].values
        k = ~np.isnan(yy) & ~np.isnan(p); m_, lo, hi = boot_mae(p[k] - yy[k])
        rows.append(dict(target=t, train=label, features=fs, ref=ref, n=int(k.sum()), MAE=m_, lo=lo, hi=hi, R2=r2(yy[k], p[k]),
                         rho=spearmanr(yy[k], p[k])[0], baseline=np.abs(yy[k] - yy[k].mean()).mean()))
    # raw UMA on the carbon nitrides (the floor for anything trained on UMA labels)
    rec('UMA calculation itself', '-', test[cc].values.astype(float))
    for fs in ('metal only', 'metal + coordination'):
        cols = FEATS[fs]; mn = pick(t, fs)
        # (a) published DFT data (Xu 2025 S/N)
        s = sn.dropna(subset=[cx]); m = models()[mn]; m.fit(s[cols].values.astype(float), s[cx].values)
        pa = m.predict(test[cols].values.astype(float)); rec('published DFT (Xu 2025, 7 S/N sites)', fs, pa)
        # (b) UMA labels, 17 graphene sites
        mask, col = TM[t]; s = g[mask]; m = models()[mn]; m.fit(s[cols].values.astype(float), s[col].values)
        pb = m.predict(test[cols].values.astype(float)); rec('UMA, 17 graphene sites', fs, pb); rec('UMA, 17 graphene sites', fs, pb, ref='UMA')
        # (c) + UMA carbon nitrides of the other sheets
        pc = np.full(len(test), np.nan)
        cu = cn.dropna(subset=[cc])
        for sheet in test.Support.unique():
            tr = pd.concat([s[cols + [col]].rename(columns={col: 'y'}), cu[cu.Support != sheet][cols + [cc]].rename(columns={cc: 'y'})])
            m = models()[mn]; m.fit(tr[cols].values.astype(float), tr.y.values)
            te = (test.Support == sheet).values; pc[te] = m.predict(test.loc[te, cols].values.astype(float))
        rec('UMA, 17 graphene sites + other C/N sheets', fs, pc); rec('UMA, 17 graphene sites + other C/N sheets', fs, pc, ref='UMA')
        if fs == 'metal + coordination':
            preds.append(test[['SAC', 'Metal', 'Support', cd]].rename(columns={cd: 'dft'}).assign(target=t, published=pa, uma17=pb, uma17_cn=pc, uma_calc=test[cc].values))
res = pd.DataFrame(rows); res.to_csv(f'{OUT}/cn_e2e.csv', index=False)
pd.concat(preds).to_csv(f'{OUT}/cn_e2e_pred.csv', index=False)
pd.set_option('display.width', 230)
print(res.round(3).to_string(index=False))
