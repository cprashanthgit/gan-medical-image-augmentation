# GAN-based Data Augmentation for Medical Image Classification

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.x](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)


## Overview

This project implements **GAN-based Data Augmentation** techniques for improving medical image classification on imbalanced datasets. We use the **HAM10000 dataset** (Human Against Machine with 10,000 dermoscopic images) for skin lesion classification.

### Key Features

- **DCGAN Implementation**: Deep Convolutional GAN for synthetic image generation
- **StyleGAN2 Implementation**: Style-based GAN with mapping network and AdaIN
- **CNN Classifier**: 3-block CNN for 7-class skin lesion classification
- **Comprehensive Evaluation**: Confusion matrices, ROC curves, classification reports
- **Modular Architecture**: Clean, reusable code structure for easy extension

## Project Structure

```
ANN Project/
├── src/
│   ├── __init__.py
│   ├── config.py              # Configuration and hyperparameters
│   ├── data/
│   │   ├── __init__.py
│   │   └── data_loader.py     # Data loading and preprocessing
│   ├── models/
│   │   ├── __init__.py
│   │   ├── dcgan.py           # DCGAN architecture
│   │   ├── stylegan2.py       # StyleGAN2 architecture
│   │   └── classifier.py      # CNN classifier
│   ├── training/
│   │   ├── __init__.py
│   │   ├── train_gan.py       # GAN training utilities
│   │   └── train_classifier.py # Classifier training
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── metrics.py         # Evaluation metrics
│   └── utils/
│       ├── __init__.py
│       └── visualization.py   # Plotting utilities
├── notebooks/
│   ├── ANN_Project2_DCGAN 1.ipynb
│   └── ANN_Project2_StyleGAN2 1.ipynb
├── train_dcgan.py             # DCGAN training pipeline
├── train_stylegan2.py         # StyleGAN2 training pipeline
├── evaluate.py                # Model evaluation script
├── requirements.txt
└── README.md
```

## Installation

### Prerequisites

- Python 3.8 or higher
- NVIDIA GPU with CUDA support (recommended)

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/gan-medical-augmentation.git
cd gan-medical-augmentation
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Download HAM10000 dataset**
   - Download from [Kaggle](https://www.kaggle.com/kmader/skin-cancer-mnist-ham10000)
   - Extract to `data/HAM10000_images/`
   - Place `HAM10000_metadata.csv` in `data/`

## Usage

### Training DCGAN

```bash
python train_dcgan.py --data_dir ./data --gan_epochs 100 --num_synthetic 2000
```

### Training StyleGAN2

```bash
python train_stylegan2.py --data_dir ./data --gan_epochs 100 --dlatent_dim 256
```

### Evaluating Models

```bash
python evaluate.py --model_path ./outputs/classifier/dcgan_augmented.h5 --data_dir ./data
```

### Command Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--data_dir` | Path to data directory | `./data` |
| `--output_dir` | Path for outputs | `./outputs` |
| `--gan_epochs` | GAN training epochs | 100 |
| `--classifier_epochs` | Classifier epochs | 50 |
| `--batch_size` | Training batch size | 64 |
| `--num_synthetic` | Synthetic images to generate | 2000 |
| `--seed` | Random seed | 42 |

## Dataset

### HAM10000 Dataset

The HAM10000 dataset contains 10,015 dermoscopic images of 7 types of skin lesions:

| Class | Description | Count |
|-------|-------------|-------|
| `nv` | Melanocytic Nevi | 6,705 |
| `mel` | Melanoma | 1,113 |
| `bkl` | Benign Keratosis | 1,099 |
| `bcc` | Basal Cell Carcinoma | 514 |
| `akiec` | Actinic Keratoses | 327 |
| `vasc` | Vascular Lesions | 142 |
| `df` | Dermatofibroma | 115 |

**Note**: The dataset is highly imbalanced, which is why GAN-based augmentation is beneficial.

## Models

### DCGAN Architecture

**Generator**:
- Input: Latent vector (100-dim)
- Dense → Reshape (8×8×256)
- ConvTranspose2D blocks: 8→16→32→64
- Output: 64×64×3 image (tanh activation)

**Discriminator**:
- Input: 64×64×3 image
- Conv2D blocks: 64→32→16→8
- Output: Single logit (real/fake)

### StyleGAN2 Architecture

**Mapping Network**:
- 4-layer MLP: z (100-dim) → w (256-dim)
- Latent normalization

**Synthesis Network**:
- Style blocks with AdaIN
- Progressive upsampling: 4→8→16→32→64
- Style modulation at each resolution

### CNN Classifier

- 3 Conv blocks (32→64→128 filters)
- MaxPooling after each block
- Dense (128) with dropout (0.5)
- Softmax output (7 classes)

## Results

### Training Progress

The GAN generates increasingly realistic skin lesion images during training:

| Epoch 1 | Epoch 50 | Epoch 100 |
|---------|----------|-----------|
| Noise | Texture emerging | Realistic lesions |

### Classification Performance

| Model | Accuracy | Macro F1 |
|-------|----------|----------|
| Baseline CNN | ~70% | ~0.45 |
| DCGAN Augmented | ~73% | ~0.50 |
| StyleGAN2 Augmented | ~75% | ~0.52 |

*Note: Actual results may vary based on hyperparameters and training duration.*

## Future Work: End-to-End Multimodal ML Pipeline

This project serves as a foundation for building a comprehensive multimodal ML pipeline:

1. **Data Pipeline**
   - Automated data ingestion and preprocessing
   - Multi-source data fusion (images, metadata, clinical notes)

2. **Model Pipeline**
   - Ensemble of GANs for different classes
   - Transfer learning with pre-trained models (ResNet, EfficientNet)
   - Multimodal fusion (image + tabular features)

3. **Deployment Pipeline**
   - Model serving with TensorFlow Serving
   - REST API for predictions
   - Real-time monitoring and logging

4. **MLOps Integration**
   - Experiment tracking (MLflow, Weights & Biases)
   - Model versioning and registry
   - CI/CD for model deployment

## References

1. Radford, A., Metz, L., & Chintala, S. (2015). *Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks*. arXiv:1511.06434

2. Karras, T., Laine, S., Aittala, M., et al. (2020). *Analyzing and Improving the Image Quality of StyleGAN*. CVPR 2020.

3. Tschandl, P., Rosendahl, C., & Kittler, H. (2018). *The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions*. Scientific Data.


## Author

**Prashanth Chitturi**

## Report

Detailed report can be found [here](https://github.com/cprashanthgit/gan-medical-image-augmentation/blob/main/ANN_Project2_Report.pdf)

## Acknowledgments

- HAM10000 dataset creators
- TensorFlow team for excellent deep learning framework
- The GAN research community for foundational work
