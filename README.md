# mmsegmentation-bayes
This repo is a fork of [open-mmlab mmsegmentation](https://github.com/open-mmlab/mmsegmentation). It adds support for training and evaluating **Laplace-approximated Bayesian models** for **pre-trained openmmlab segmentation models**.


Upstream repo [README](https://github.com/open-mmlab/mmsegmentation/blob/main/README.md). 

## 🚀 New Features 

- **mcglm API**: new class `mmseg.apis.MCGLM` to run pixel-level segmentation prediction uncertainty
- **Uncertainty evaluation metrics**: new functions `mmseg.apis.avu` for uncertainty quality evaluation using scores AvU and distribution separation
- **Fisher training**: training script `tools/fisher.py` for training any pre-trained segmentation model

## Installation

```bash
pip install -U openmim
mim install mmcv

git clone https://github.com/romiebanerjee/mmengine-bayes
pip install -e mmengine-bayes/.

git clone https://github.com/romiebanerjee/mmsegmentation-bayes
pip install -e mmsegmentation-bayes/.
```
```python 
import sys
sys.path.append('/path/to/mmsegmentation-bayes')
sys.path.append('/path/to/mmengine-bayes')
import mmcv, mmengine, mmseg
```

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/romiebanerjee/mmsegmentation-bayes/blob/main/mmlab_bayes.ipynb)  [Tutorial Notebook](https://github.com/romiebanerjee/mmsegmentation-bayes/blob/main/mmlab_bayes.ipynb)

## Usage

### Estimate Fisher of a pre-trained model

```bash
python tools/fisher.py --config /path/to/model/config.py --work_dir /path/to/work/dir --ckpt /path/to/model/ckpt
```
### Run test uncertainty over validation dataset
```bash
python tools/test_unc.py -config /path/to/model/config.py --work_dir /path/to/work/dir --ckpt /path/to/model/ckpt --curvature_ckpt /path/to/kfac/state/dict/ckpt
```
### Uncertainty demo
```bash
python seg_unc_demo.py --image demo.png --output results.png --iters 5 --show
```



