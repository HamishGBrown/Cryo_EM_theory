# Cryo-EM Theory

A series of interactive Jupyter notebooks that build intuition for the theory behind cryo-electron microscopy (cryo-EM), from the basics of Fourier transforms through to image formation and 2D classification.

## Notebooks

| Notebook | Topic |
|----------|-------|
| `00 - Intro to Fourier transforms.ipynb` | Fourier transforms and frequency-space representations |
| `01 - Image formation.ipynb` | Contrast transfer functions and how cryo-EM images are formed |
| `02 - 2D classification.ipynb` | Aligning and averaging particle images |

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/HamishGBrown/Cryo_EM_theory.git
cd Cryo_EM_theory
```

### 2. Create and activate a conda environment

```bash
conda create -n cryo-em-theory python=3.11
conda activate cryo-em-theory
```

### 3. Install dependencies

```bash
pip install -e .
```

### 4. Launch Jupyter

```bash
jupyter notebook
```
