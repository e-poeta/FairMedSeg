# Divergence-aware training with automatic subgroup mitigation for breast tumor segmentation

**🏆 Winner of the *Best Paper Award* at MAMA-MIA Challenge held in conjunction with MICCAI 2025**

This repository contains the official implementation of the paper:

"Divergence-aware training with automatic subgroup mitigation for breast tumor segmentation"

## 📌 Overview
Deep learning models for breast tumor segmentation in DCE-MRI often exhibit performance disparities across different demographic and clinical subgroups. This research addresses these fairness concerns by proposing a subgroup-aware in-processing mitigation strategy.

Key Features:
- Divergence-based Regularization: Integrates a fairness-aware loss directly into the training loop.

- Interpretable Metadata: Leverages clinical data (age, menopausal status, breast density) to identify underperforming subgroups.

- Dynamic Loss Weighting: Automatically assigns higher weights to samples from subgroups that diverge from average performance.

- No Post-processing Required: Improves fairness and segmentation quality during the training phase without needing external data.


## 🗣️ Citation

If you use this repository, please cite the associated paper:

Divergence-Aware Training with Automatic Subgroup Mitigation for Breast Tumor Segmentation. Deep-Breath 2025, Lecture Notes in Computer Science, vol. 16142, pp. 52–62, Springer. First online: 21 September 2025. DOI: [10.1007/978-3-032-05559-0_6](https://doi.org/10.1007/978-3-032-05559-0_6).

```
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
}
