# SAFE-Mixer: A Spatial-Frequency Mutual Gating Module for Ultrasound Lesion Segmentation

This repository contains the implementation of SAFE-Mixer for ultrasound lesion segmentation. The proposed module integrates wavelet-derived frequency cues and spatial structural priors through mutual gating. The segmentation framework uses a U-Net backbone with Spatial Reduction Attention (SRA) and hierarchical SAFE-Mixer deployment.

## Environment

The experimental environment reported in the manuscript is Python 3.9.25, PyTorch 2.0.0, CUDA 11.8, and an NVIDIA RTX A6000 GPU.

Install the dependencies in a Python 3.9 environment:

```bash
pip install -r requirements.txt
```

The requirements file specifies a proposed compatible environment. Dependency versions other than those reported in the manuscript have not been verified against the original training environment. A CUDA-capable GPU and compatible NVIDIA driver are required.

## Datasets

- [TN3K](https://github.com/haifangong/TRFE-Net-for-thyroid-nodule-segmentation): 3,493 thyroid ultrasound images.
- [DDTI](https://fidopt.io/dataset/DDTI/): Digital Database of Thyroid Ultrasound Images (637 images used in the study).
- [BUSI](https://www.kaggle.com/aryashah2k/breast-ultrasound-images-dataset): 647 benign and malignant images used in the study.

Download the datasets from their respective sources and prepare paired images and binary lesion masks. Each image and its corresponding mask must have the same filename stem. Masks should use pixel values 0 and 255.

```text
inputs/
├── TN3K/
│   ├── images/
│   │   └── sample.jpg
│   └── masks/
│       └── 0/
│           └── sample.png
├── DDTI/
│   ├── images/
│   └── masks/0/
└── BUSI/
    ├── images/
    └── masks/0/
```

The independent clinical dataset is not distributed in this repository.

## Training and Evaluation

Run the following command from the repository root:

```bash
python main.py --dataset TN3K --name TN3K_SAFE_41 --img_ext .jpg --mask_ext .png --epochs 300 --batch_size 8 --split_seed 41 --num_workers 0
```

Adjust the dataset name and file extensions to match your prepared data. The command above uses an example split seed; use the original experimental seeds and sample lists to reproduce the reported runs exactly.

The code uses an 80/20 development/test split and reserves 10% of the development subset for validation. The best checkpoint is selected according to validation IoU and evaluated on the held-out test subset. Repeat with the desired split seeds and distinct run names, and average the test results.

Default training settings: input size 256 × 256, Adam optimizer, learning rate 1e-4, weight decay 1e-4, cosine annealing with minimum learning rate 1e-5, and 0.5 BCE + Dice loss.

The run configuration, split lists, training logs, best checkpoint, and test metrics are saved in `checkpoint/<name>/`.

The HD95 value in the training utility is an auxiliary metric; the manuscript uses a separate HD95 evaluation procedure. That separate evaluator is not included in this version, so the logged HD95 should not be treated as a reproduction of the manuscript's HD95 results.

## Model

The main architecture is implemented in `src/network/Best.py`. SRA is used in encoder stages 2–4 with reduction ratios 8, 4, and 2. SAFE-Mixer is deployed at the bottleneck and decoder stages D4, D2, and D1 with 7, 7, 2, and 1 blocks, respectively.

## Acknowledgements

The training framework is adapted from [CMU-Net](https://github.com/FengheTan9/CMU-Net), which acknowledges helper functions from [UNeXt](https://github.com/jeya-maria-jose/UNeXt-pytorch). The original license and copyright notice are retained.
