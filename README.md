# Predicting Fluorescent Protein Spectral Properties from AlphaFold2 Structures

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18048544.svg)](https://doi.org/10.5281/zenodo.18048544)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![ORCID](https://img.shields.io/badge/ORCID-0009--0000--3445--5239-green.svg)](https://orcid.org/0009-0000-3445-5239)

**A machine learning approach for predicting excitation and emission wavelengths of fluorescent proteins using structural features extracted from AlphaFold2-predicted structures.**

---

## Abstract

This work presents a computational pipeline for predicting spectral properties of fluorescent proteins (FPs) from three-dimensional structures predicted by AlphaFold2. Using a curated dataset of 517 FPs from FPbase with corresponding ColabFold-generated structures, we demonstrate that predicted structures contain sufficient information for spectral prediction.

The optimal model (Extra Trees with 113 features) achieves:
- **17.13 nm MAE** for emission wavelength (λ<sub>em</sub>)
- **21.44 nm MAE** for excitation wavelength (λ<sub>ex</sub>)
- **R² = 0.737** (73.7% variance explained)

SHAP analysis identifies the **ψ dihedral angle of the chromophore tyrosine** as the most predictive structural descriptor, consistent with established photophysical theory linking chromophore geometry to electronic transition energies.

---

## Quick Start

### 1. Clone and Install

```bash
git clone https://github.com/yannbanas/fp-spectra.git
cd fp-spectra
pip install -r requirements.txt
```

### 2. Extract Structure Files

The PDB structure files are compressed to reduce repository size. Extract them before running the pipeline:

```bash
# Linux/Mac
cd data/structures
unzip fp-spectra-structures-v1.0.0.zip
cd ../..

# Windows (PowerShell)
Expand-Archive -Path "data\structures\fp-spectra-structures-v1.0.0.zip" -DestinationPath "data\structures\"
```

### 3. Use Pre-trained Models

```python
import joblib
import pandas as pd

# Load best model
model = joblib.load('models/extra_trees_advanced_em_max.joblib')

# Load features (must match training schema - 113 features)
features = pd.read_csv('data/processed/dataset_test.csv')
X = features.drop(columns=['name', 'ex_max', 'em_max'], errors='ignore')

# Predict emission wavelength
predictions = model.predict(X)
print(f"Predicted λem: {predictions[:5]} nm")
```

---

## Key Findings

| Finding | Description |
|---------|-------------|
| **Local > Global** | ESM-2 embeddings restricted to ±15 residues around the chromophore outperform global sequence embeddings |
| **Classical ML > Deep Learning** | Extra Trees outperforms GNN and MLP architectures on this dataset (n=415 training samples) |
| **ψ angle is paramount** | Chromophore ψ dihedral angle exhibits highest SHAP importance (14.51) |
| **AlphaFold2 suffices** | Predicted structures enable spectral prediction without experimental crystallography |

---

## Methodology

### Data Sources

| Source | Description | Count |
|--------|-------------|-------|
| [FPbase](https://www.fpbase.org) | Spectral properties (λ<sub>ex</sub>, λ<sub>em</sub>, QY) | 1,040 FPs |
| [ColabFold](https://github.com/sokrypton/ColabFold) | AlphaFold2 structure predictions | 676 structures |
| **Final dataset** | Intersection with complete data | 517 FPs |

### Feature Engineering

**113 total features** organized in three categories:

#### 1. Structural Features (83)

| Category | Features | Description |
|----------|----------|-------------|
| Chromophore geometry | φ, ψ, ω, χ1, χ2 | Dihedral angles of chromophore tripeptide |
| Interatomic distances | Cα-Cα, OH-backbone | Geometric constraints and H-bond potential |
| Planarity | Ring RMSD, conjugated RMSD | Deviation from ideal planar geometry |
| Environment | Neighbor count, H-bonds, cavity volume | Local protein environment within 8Å |
| Global | Radius of gyration, pLDDT, SS content | Whole-protein descriptors |
| Sequence | AA frequencies, GRAVY, length | Compositional features |

#### 2. ESM-2 Local Embeddings (30)

- **Window**: ±15 residues around chromophore (31 residues)
- **Dimensionality**: PCA reduction from 1280 to 30 components
- **Rationale**: Concentrates evolutionary signal on spectroscopically relevant region

### Model Comparison

| Model | λ<sub>ex</sub> MAE (nm) | λ<sub>em</sub> MAE (nm) | R² |
|-------|-------------------------|-------------------------|-----|
| Random Forest | 21.93 | 19.28 | 0.713 |
| XGBoost | 23.28 | 19.29 | 0.694 |
| Extra Trees | 21.44 | 18.47 | 0.718 |
| Gradient Boosting | 22.15 | 18.73 | 0.712 |
| ESM-2 Global + ET | 22.35 | 18.57 | 0.731 |
| Multi-Task MLP | 25.46 | 19.53 | 0.717 |
| Graph Neural Network | 22.63 | 21.06 | 0.706 |
| Stacking Ensemble | 31.12 | 22.22 | 0.685 |
| **Advanced + ESM-2 Local** | **21.75** | **17.13** | **0.737** |

---

## Installation

### Requirements

- Python ≥ 3.8
- ~4 GB disk space (ESM-2 model weights)

### Dependencies

```
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
xgboost>=1.5.0
shap>=0.40.0
matplotlib>=3.4.0
biopython>=1.79
torch>=1.10.0
transformers>=4.15.0
scipy>=1.7.0
joblib>=1.1.0
```

---

## Full Pipeline Execution

```bash
# 1. Data collection from FPbase
python 01_fpbase_collection.py

# 2. Generate FASTA for ColabFold
python 02_generate_fasta.py

# 3. [External] Run ColabFold for structure prediction
#    Extract results to data/structures/

# 4. Extract structural features
python 03_extract_features.py

# 5. Train baseline models
python 04_train_baseline.py

# 6. SHAP interpretability analysis
python 05_shap_analysis.py

# 7. Hyperparameter tuning
python 06_hyperparameter_tuning.py

# 8. Ensemble methods
python 07_ensemble_stacking.py

# 9. ESM-2 embeddings
python 08_esm2_embeddings.py

# 10. Deep learning approaches
python 09_deep_learning_gnn.py

# 11. Advanced features + ESM-2 local (best performance)
python 10_chromophore_advanced.py

# 12. Generate final report
python 11_rapport_final_complet.py
```

---

## Repository Structure

```
fp-spectra/
├── data/
│   ├── fasta/                              # FASTA sequences (15 batch files)
│   ├── logs/                               # Processing logs
│   ├── processed/
│   │   ├── dataset_final.csv               # Complete dataset
│   │   ├── dataset_train.csv               # Training set (n=415)
│   │   ├── dataset_test.csv                # Test set (n=102)
│   │   ├── esm2_embeddings_train.npy       # ESM-2 embeddings
│   │   ├── esm2_embeddings_test.npy
│   │   ├── fpbase_curated.csv              # Curated FPbase data
│   │   └── fpbase_stats.json               # Dataset statistics
│   ├── raw/
│   │   └── fpbase.json                     # Raw FPbase export
│   └── structures/
│       └── fp-spectra-structures-v1.0.0.zip  # 676 PDB files (extract before use)
├── models/
│   ├── extra_trees_advanced_em_max.joblib  # Best model (emission)
│   ├── extra_trees_advanced_ex_max.joblib  # Best model (excitation)
│   ├── extra_trees_esm2_*.joblib           # ESM-2 models
│   ├── rf_*.joblib                         # Random Forest models
│   ├── stacking_*.joblib                   # Stacking models
│   └── xgb_*.joblib                        # XGBoost models
├── reports/
│   ├── RAPPORT_FINAL_COMPLET.html          # Complete HTML report
│   ├── RAPPORT_FINAL_MASTER.html           # Master thesis report
│   ├── shap_*.png                          # SHAP visualizations
│   ├── *_comparison.png                    # Model comparisons
│   └── *.html                              # Individual reports
├── 01_fpbase_collection.py
├── 02_generate_fasta.py
├── 03_extract_features.py
├── 04_train_baseline.py
├── 05_shap_analysis.py
├── 06_hyperparameter_tuning.py
├── 07_ensemble_stacking.py
├── 08_esm2_embeddings.py
├── 09_deep_learning_gnn.py
├── 10_chromophore_advanced.py
├── 11_rapport_final_complet.py
├── .gitignore
├── CITATION.cff
├── LICENSE
├── README.md
└── requirements.txt
```

---

## Interpretability Analysis

SHAP (SHapley Additive exPlanations) analysis quantifies feature contributions:

| Rank | Feature | SHAP Importance | Physical Interpretation |
|------|---------|-----------------|------------------------|
| 1 | `chrom_psi_tyr` | 14.51 | ψ angle modulates π-conjugation planarity and HOMO-LUMO gap |
| 2 | `seq_freq_D` | 7.62 | Aspartate residues affect excited state stabilization |
| 3 | `seq_gravy` | 7.20 | Hydrophobicity influences chromophore microenvironment |
| 4 | `seq_freq_N` | 6.89 | Asparagine forms H-bonds with chromophore |
| 5 | `chrom_chi1_tyr` | 5.43 | Tyrosine side-chain orientation affects conjugation extent |

The predominance of the ψ dihedral angle confirms theoretical predictions: chromophore planarity directly determines the extent of π-electron delocalization, modulating the energy gap between ground and excited electronic states.

---

## Limitations and Scope

### Known Limitations

| Limitation | Impact | Mitigation |
|------------|--------|------------|
| Dataset size (n=517) | Constrains deep learning approaches | Use classical ML with regularization |
| GFP-like bias | FPbase overrepresents *Aequorea*-derived FPs | Stratified sampling; caution with novel scaffolds |
| Static structures | AlphaFold2 provides single conformer | Consider MD-derived features in future work |
| ±17 nm accuracy | Insufficient for precise spectral engineering | Use for screening, not fine-tuning |

### Recommended Applications

| Application | Suitability |
|-------------|-------------|
| High-throughput variant screening | Excellent |
| Color family classification | Excellent |
| Pre-selection for experimental validation | Good |
| Structure-spectrum relationship studies | Good |
| Precise wavelength engineering (<10 nm) | Not recommended |
| Quantum yield prediction | Not applicable |

---

## Citation

If you use this work in your research, please cite:

```bibtex
@software{banas2025fpspectra,
  author       = {Banas, Yann},
  title        = {{FP-Spectra: Predicting Fluorescent Protein Spectral 
                   Properties from AlphaFold2 Structures}},
  year         = {2025},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.18048544},
  url          = {https://github.com/yannbanas/fp-spectra}
}
```

See [CITATION.cff](CITATION.cff) for machine-readable citation metadata.

---

## Related Work

This project is part of the **NPHP** (Neuro-Photonic Hybrid Processor) research program investigating biophotonic interfaces for neural systems.

### Key References

1. Jumper, J. et al. (2021). Highly accurate protein structure prediction with AlphaFold. *Nature* **596**, 583–589. [doi:10.1038/s41586-021-03819-2](https://doi.org/10.1038/s41586-021-03819-2)

2. Lambert, T.J. (2019). FPbase: a community-editable fluorescent protein database. *Nature Methods* **16**, 277–278. [doi:10.1038/s41592-019-0352-8](https://doi.org/10.1038/s41592-019-0352-8)

3. Lin, Z. et al. (2023). Evolutionary-scale prediction of atomic-level protein structure with a language model. *Science* **379**, 1123–1130. [doi:10.1126/science.ade2574](https://doi.org/10.1126/science.ade2574)

4. Mirdita, M. et al. (2022). ColabFold: making protein folding accessible to all. *Nature Methods* **19**, 679–682. [doi:10.1038/s41592-022-01488-1](https://doi.org/10.1038/s41592-022-01488-1)

5. Lundberg, S.M. & Lee, S.-I. (2017). A unified approach to interpreting model predictions. *Advances in Neural Information Processing Systems* **30**.

---

## License

This work is licensed under a [Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/).

You are free to share and adapt this material for any purpose, provided appropriate credit is given.

---

## Author

**Yann Banas**

| | |
|---|---|
| ORCID | [0009-0000-3445-5239](https://orcid.org/0009-0000-3445-5239) |
| GitHub | [@yannbanas](https://github.com/yannbanas) |
| Email | yann.banas@banastechnologie.cloud |
| LinkedIn | [yann-banas-440a63156](https://www.linkedin.com/in/yann-banas-440a63156) |
| Project | NPHP (Neuro-Photonic Hybrid Processor) |

---

## Acknowledgments

- [FPbase](https://www.fpbase.org) community for maintaining the fluorescent protein database
- [DeepMind](https://www.deepmind.com) and [ColabFold](https://github.com/sokrypton/ColabFold) teams for democratizing structure prediction
- [Meta AI](https://ai.facebook.com) for the ESM-2 protein language model
- [SHAP](https://github.com/slundberg/shap) developers for interpretability tools