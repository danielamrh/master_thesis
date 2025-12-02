# Master Thesis: Sparse Sensor IMU-UWB Fusion for Precise Upper Limb Motion Tracking
### Author: Daniel Amrhein
Copyright (C) 2025 Daniel Amrhein

This repository contains the implementation of a deep learning framework, utilizing Transformer and LSTM architectures, for accurate arm pose estimation. The model fuses inertial (IMU) and ultra-wideband (UWB) sensor data, primarily trained and evaluated on synthetic AMASS motion data and real-world UIP data.

## ⚠️ Important Note on Data and Model Files

Due to GitHub's file size restrictions (even with Git LFS), large model weights, SMPL model files, and preprocessed datasets have been removed from the repository history.

## 🚀 Getting Started

### 1. Prerequisites

Python 3.8+

PyTorch (CUDA recommended)

Standard scientific libraries (NumPy, Matplotlib)

Articulate 

Fairmotion 

*** 2. Installation

Clone the repository:
```
git clone git@github.com:danielamrh/master_thesis.git
cd master_thesis
```

(Note: Use SSH (git@...) for stability.)

Install dependencies:
```
pip install torch numpy matplotlib tqdm
```

### 3. Data Setup

After downloading the external files (see "Important Note" above), ensure your local file structure matches the paths defined in config_amass.py.

The required data structure is:
```
master_thesis/
├── data/
│   ├── smplx_models/  <-- Contains SMPL/SMPLH/SMPLX files
│   ├── preprocess/    <-- Contains AMASS_processed data
│   └── ...
├── UIP_dataset/
│   └── UIP_DB_Dataset/
│       ├── train.pt   <-- Downloaded UIP Training Data
│       └── test.pt    <-- Downloaded UIP Test Data
└── ...
```

## 📚 Citations and Acknowledgements

R. Armani, C. Qian, J. Jiang, and C. Holz, "Ultra Inertial Poser: Scalable Motion Capture and Tracking from Sparse Inertial Sensors and Ultra-Wideband Ranging," Apr. 2024. [Online]. Available: http://​arxiv.org​/​pdf/​2404.19541.

N. Mahmood, N. Ghorbani, N. F. Troje, G. Pons-Moll, and M. J. Black, "AMASS: Archive of Motion Capture as Surface Shapes," Apr. 2019. [Online]. Available: https://​arxiv.org​/​pdf/​1904.03278. 

M. Loper, N. Mahmood, J. Romero, G. Pons-Moll, and M. J. Black, "SMPL: a skinned multi-person linear model," ACM Trans. Graph., vol. 34, no. 6, pp. 1–16, 2015, doi: 10.1145/2816795.2818013.
