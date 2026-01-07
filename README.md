# 🧠 MAMA-MIA Challenge: Breast Tumor Segmentation on DCE-MRI

We are Team AIH-MAMA from *AI4Health@EURECOM* and *Politecnico di Torino, Italy*.

This repository contains the complete pipeline for participating in the MAMA-MIA Challenge, focused on automatic breast tumor segmentation from DCE-MRI scans.


## Environment Setup

```bash
uv venv ahimama --python 3.10
source ahimama/bin/activate
python -m ensurepip --upgrade
uv pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install numpy setuptools wheel
pip install pyradiomics==3.0.1 --no-build-isolation
uv pip install -r requirements.txt
