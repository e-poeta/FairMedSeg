<div align="center">
  
## Divergence-Aware Training with Automatic Subgroup Mitigation for Breast Tumor Segmentation

[**Eleonora Poeta**](https://scholar.google.com/citations?user=NVUrRyoAAAAJ&hl=it)<sup>1</sup> · [**Luisa Vargas**](https://scholar.google.com/citations?user=vxrAZhsAAAAJ&hl=it)<sup>2</sup> · [**Daniele Falcetta**](https://2falx.github.io/danielefalcetta/)<sup>2</sup> · [**Vincenzo Marciano'**](https://scholar.google.com/citations?user=Ga_uQ98AAAAJ&hl=it)<sup>2</sup>

[**Eliana Pastor**](https://elianap.github.io/)<sup>1</sup>
[**Tania Cerquitelli**](https://smartdata.polito.it/members/tania-cerquitelli)<sup>1</sup> · [**Elena Baralis**](https://smartdata.polito.it/members/elena-baralis/)<sup>1</sup> · [**Maria A. Zuluaga**](https://zuluaga.eurecom.io/)<sup>2</sup>

<sup>1</sup>Politecnico di Torino, Italy&emsp;<sup>2</sup>EURECOM, France

**🏆 Winner of the *Best Paper Award* at** [MAMA-MIA Challenge at MICCAI 2025](https://www.ub.edu/mama-mia/challenge/) 

<a href="https://doi.org/10.1007/978-3-032-05559-0_6"><img src="https://img.shields.io/badge/Paper-Springer-%2300ADEF?style=flat&logo=springer&logoColor=%2300ADEF&labelColor=white" alt="Paper DOI"></a>
</div>

## 📌 Overview
Deep learning models for breast tumor segmentation in DCE-MRI often exhibit performance disparities across different demographic and clinical subgroups. This research addresses these fairness concerns by proposing a subgroup-aware in-processing mitigation strategy.

Key Features:
- Divergence-based Regularization: Integrates a fairness-aware loss directly into the training loop.

- Interpretable Metadata: Leverages clinical data (age, menopausal status, breast density) to identify underperforming subgroups.

- Dynamic Loss Weighting: Automatically assigns higher weights to samples from subgroups that diverge from average performance.

- No Post-processing Required: Improves fairness and segmentation quality during the training phase without needing external data.

## 📊 Results
Our method was evaluated on the **MAMA-MIA Challenge dataset**, demonstrating:

- Significant improvements in fairness scores across subgroups;

- Maintained or improved segmentation quality (Dice score) for the overall population;

- Enhanced clinical trustworthiness by reducing performance gaps in harder-to-segment subpopulations.

--- 
## ⚡️ Fast setup with uv (recommended)

We recommend using **[`uv`](https://github.com/astral-sh/uv)** for a fast, reproducible, and dependency-safe setup.

### 0️⃣ Prerequisites

- **Python 3.10** (recommended)
- **Git**

### 1️⃣ Clone the repository

```bash
git clone https://github.com/robustml-eurecom/FairMedSeg.git
cd FairMedSeg
```
### 2️⃣ Create and activate the virtual environment

```bash
uv init
source .venv/bin/activate
```
### 3️⃣ Install Torch and all the other dependencies

```bash
uv add torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
uv add -r requirements.txt
uv pip install pyradiomics==3.0.1 --no-build-isolation
uv sync
```

## 📁 Data layout expected by the loaders

The MONAI data loader (`dataloader/load_monai_metadata.py`) expects the dataset to be organized as follows:

- **Images**  
  One folder per patient ID, containing the DCE-MRI phase volumes (`.nii.gz`).

- **Segmentations**  
  One segmentation file per patient, named `<patient_id>.nii.gz`.

- **Metadata**  
  An Excel file containing clinical and imaging information, with at least a `patient_id` column.

- **Train / test split**  
  A CSV file specifying the dataset split, with `train_split` and `test_split` columns listing patient IDs.

Example directory structure

```text
<DATA_ROOT>/
├── images/
│   └── PAT_0001/
│       ├── phase_0.nii.gz
│       ├── phase_1.nii.gz
│       ├── phase_2.nii.gz
│       └── ...
├── segmentations/
│   └── expert/
│       ├── PAT_0001.nii.gz
│       └── ...
├── patient_info_files/
│   ├── PAT_0001.json
│   └── ...
├── clinical_and_imaging_info.xlsx
└── train_test_splits.csv
```

## 🗣️ Citation
If you use this repository, please cite the associated paper:

```bibtex
@inproceedings{poeta2025divergence,
  title     = {Divergence-Aware Training with Automatic Subgroup Mitigation for Breast Tumor Segmentation},
  author    = {Poeta, Eleonora and Vargas, Luisa and Falcetta, Daniele and Marciano', Vincenzo and Pastor, Eliana and Cerquitelli, Tania and Baralis, Elena and Zuluaga, Maria A.},
  booktitle = {Artificial Intelligence and Imaging for Diagnostic and Treatment Challenges in Breast Care (Deep-Breath 2025)},
  series    = {Lecture Notes in Computer Science},
  volume    = {16142},
  pages     = {52--62},
  year      = {2025},
  publisher = {Springer},
  doi       = {10.1007/978-3-032-05559-0_6}
```
