# ML-assisted screening of single-atom catalysts for electrochemical nitrogen reduction

Code and data for the B.Tech. project report *Machine Learning-Assisted Screening of Single-Atom Catalysts for
Electrochemical Nitrogen Reduction* (Garvit Kumbhat, Department of Chemical Engineering, IIT Indore, 2026;
guide: Dr. Naveen Agrawal).

The workflow:
1. Train ML models on published DFT data and evaluate them without leakage.
2. Expand the dataset from 196 to 476 single-atom catalysts (SACs) with the universal ML potential UMA.
3. Validate and correct UMA against DFT.
4. Screen for stability, activity and selectivity.
5. Analyse which descriptors control the results.
6. Check the finalists with Quantum ESPRESSO.

## Contents

| Folder | Report section | What it contains |
|---|---|---|
| `01_published_data/` | Chapter 3 | Activity and stability ML on the DFT data of Xu et al. (2025, 2024): notebooks, merged dataset, results |
| `02_uma_expansion/` | Chapter 4 | UMA notebooks (Colab and Kaggle), the worker bundle (`nrr_kaggle_bundle.zip`), the 476-SAC results (`UMA_476_full.csv`), the relaxed structures (`relaxed_broadened.zip`: 280 new-site SACs, all states; `relaxed_SN_pathway.zip`: S/N pathway intermediates) and the first validation tables |
| `03_screening/` | Chapter 5, Section 4.3.4 | `final_screening.ipynb`: Δ-learning corrections, four-stage screen, finalists |
| `04_descriptor_ml/` | Chapter 6 | `coordination_ML.ipynb` and the revised analysis scripts (see below) with their outputs in `out/` |
| `05_carbon_nitride_test/` | Section 4.3.5 | UMA notebooks for the 140 carbon-nitride SACs, the input bundle, `cn_compare.py`, `retrain_with_cn.py`, results and relaxed bare structures |
| `06_dft_qe/` | Sections 4.3.6, 5.7 | Quantum ESPRESSO notebooks (Kaggle CPU), bundles and DFT results: the three finalists (single points), and in `run2_newsites_and_relax/` the 16 new-site SACs, partial relaxations of the finalists and `pf_dftval.py` |

The interactive viewer of the 17 Mo sites is at https://garvitkumbhat619.github.io/mo-site-explorer/.

## Re-running the Chapter 6 analysis

```bash
cd 04_descriptor_ml
pip install numpy pandas scikit-learn xgboost scipy ase
python eval_fix.py   # Sections 6.2, 6.3, 6.6: descriptor sets, isomer-grouped hold-out, bootstrap CIs, expansion test
python volcano.py    # Section 6.4: scaling relations, volcano, ML for the adsorption energies
python cn_e2e.py     # Section 6.7: descriptor models tested against carbon-nitride DFT (needs eval_fix.py first)
python basic_ul.py   # Section 6.8: U_L from basic features (single features, forward selection)
python basic_ul_nested.py   # nested forward selection (honest estimate)
python basic_final.py       # Table 6.4 numbers
```

All paths are relative to the repository. Each script writes its tables to `out/`. A full run takes a few minutes on a laptop.

## Re-running the GPU and DFT parts
- **UMA (Chapter 4, Section 4.3.5):**
  - Upload the bundle zip as a Kaggle dataset, or put it in Google Drive for Colab.
  - Open the matching notebook and add a Hugging Face token with access to `facebook/UMA` as the secret `HF_TOKEN`.
  - Run all cells.
  - The notebooks resume from their own output if a session stops.
- **Quantum ESPRESSO (Sections 4.3.6 and 5.7):**
  - Finalists: upload `nrr_qe_bundle.zip` as a Kaggle dataset and run `QE_finalists_Kaggle.ipynb` (CPU, about 4–5 h).
  - New sites and relaxations: upload `run2_newsites_and_relax/nrr_qe2_bundle.zip` and run `QE_validation_part1_Kaggle.ipynb`, `QE_validation_part2_Kaggle.ipynb` and `QE_relax_finalists_Kaggle.ipynb` at the same time (CPU, one session each). `QE_anchors_Xu_Kaggle.ipynb` checks the QE setup on three of Xu's SACs (about 2 h).
  - A run that stops at the time limit can be resumed by adding its previous output as an input.
  - `python pf_dftval.py` in `run2_newsites_and_relax/` redraws Figure 4.7 and writes the numbers of Table 4.9 from `results/`.

## Data sources
- L. Xu et al., S/N-coordinated SACs: DFT data from the supporting information (2025).
- L. Xu et al., *J. Mater. Chem. A* 12, 33053 (2024): carbon-nitride SACs. The binding energies were digitised from Fig. 2a.
- Dissolution potentials: CRC Handbook. Cohesive energies: Kittel.

See the report for full references and conventions.
