# CSMAW: Client-Specific Masked Adversarial Watermarking

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.8](https://img.shields.io/badge/pytorch-2.8.0-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Official implementation of **"Client-Specific Masked Adversarial Watermarking for Federated Model Ownership Verification"**.

## Overview

CSMAW is a novel **non-invasive, black-box verifiable** watermarking mechanism for federated learning (FL) that enables secure ownership attribution without modifying model parameters. Unlike existing methods that suffer from watermark dilution during model aggregation, CSMAW maintains high verification accuracy (98.8%–100.0%) even with 60 clients.

### Key Features

- **Non-intrusive**: No model parameter modification or retraining required
- **Black-box Verifiable**: Ownership verification through query access only
- **Scalable**: Performance improves as client numbers increase
- **Robust**: Maintains 96%–99% accuracy under various attacks (fine-tuning, pruning, quantization, backdoor, data poisoning)

### Method Overview

CSMAW comprises three main components:

1. **Federated ClientNet Training**: Collaborative model training using FedAvg
2. **Client-Specific Masked Adversarial Watermark Generation**: 
   - Semantic Derangement: Unique label transformation for each client
   - Non-overlapping Spatial Masks: Exclusive pixel regions per client
   - PGD-based adversarial perturbation generation
3. **Client Ownership Verification**: Black-box query-based verification

## Project Structure

```
CSMAW/
├── Step1_FLtraining.py          # Federated learning training
├── Step2_gen_watermarks.py      # Adversarial watermark generation
├── Step3_traceability.py        # Ownership verification & tracing
├── run.sh                       # Automated experiment runner
├── requirements.txt             # Python dependencies
├── utils/                       # Utility modules
│   ├── dataset_utils.py         # Dataset loading utilities
│   └── model_utils.py           # Model architecture definitions
├── data/                        # Dataset storage (auto-downloaded)
├── pretrained_dir/              # Pre-trained model weights
└── artifacts/                   # Experiment outputs
```

### Datasets
The following publicly available datasets are used in this project:
- **MNIST**: The MNIST dataset is a collection of handwritten digits, widely used for training image processing systems. It contains 60,000 training images and 10,000 testing images of digits 0 through 9.
- **CIFAR-10**: The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 different classes, with 6,000 images per class. It is commonly used for benchmarking machine learning algorithms in image classification tasks.
- **GTSRB**: The German Traffic Sign Recognition Benchmark (GTSRB) dataset contains over 50,000 images of traffic signs. It is designed for the task of recognizing traffic signs and is widely used in computer vision research.
  
All these datasets are publicly available and can be accessed from their respective repositories.

## Installation

### Prerequisites

- Python 3.10+
- CUDA 12.4+ (for GPU acceleration)
- 32GB+ GPU memory recommended (Tesla V100 or equivalent)

### Setup

```bash
# Clone the repository
git clone https://github.com/your-username/CSMAW.git
cd CSMAW

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

```
numpy==2.2.6
torch==2.8.0
torchvision==0.23.0
```

## Quick Start

### Run Complete Pipeline

Execute all three steps automatically:

```bash
bash run.sh
```

This will run experiments on MNIST, CIFAR-10, and GTSRB datasets with LeNet, ResNet18, TailNet, and VGG16 architectures.

### Run Individual Steps

#### Step 1: Federated Learning Training

Train client models using FedAvg:

```bash
python Step1_FLtraining.py \
    --num_clients 5 \
    --dataset mnist \
    --model lenet \
    --artifacts_dir ./artifacts/mnist/lenet \
    --pretrained \
    --num_rounds 200 \
    --local_epoch 1 \
    --learning_rate 1e-4 \
    --batch_size 64
```

**Parameters:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--num_clients` | Required | Number of participating FL clients |
| `--dataset` | Required | Dataset name (mnist/cifar10/gtsrb) |
| `--model` | Required | Model architecture (lenet/resnet18/tailnet/vgg16) |
| `--pretrained` | False | Retrain the model, if True, denote load the pretrained model in ./pretrained_dir|
| `--num_rounds` | 1 | Number of FL communication rounds |
| `--local_epoch` | 1 | Local training epochs per round |
| `--learning_rate` | 1e-4 | SGD learning rate |
| `--batch_size` | 64 | Training batch size |

**Outputs:**
- `global_tail.pt`: Aggregated global model
- `client_{i}_tail.pt`: Individual client models (i = 0 to N-1)

#### Step 2: Adversarial Watermark Generation

Generate client-specific masked adversarial watermarks:

```bash
python Step2_gen_watermarks.py \
    --num_clients 5 \
    --dataset mnist \
    --model lenet \
    --artifacts_dir ./artifacts/mnist/lenet \
    --cleanset_max 300 \
    --pgd_eps 0.8 \
    --pgd_alpha 0.04 \
    --pgd_steps 200 \
    --adv_target_count 200
```

**Parameters:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--cleanset_max` | 300 | Maximum clean reference samples |
| `--pgd_eps` | 0.8 | PGD perturbation budget (ε) |
| `--pgd_alpha` | 0.04 | PGD step size (α) |
| `--pgd_steps` | 200 | PGD iteration count |
| `--adv_target_count` | 200 | Successful adversarial examples to save per client |

**Outputs:**
- `clean_set.pt`: Clean reference images and labels
- `masks.pt`: Non-overlapping spatial masks for all clients
- `target_seqs.pt`: Semantic derangement label mappings
- `adv_examples.pt`: Generated adversarial watermarks with indices

#### Step 3: Traceability Verification

Verify ownership and trace adversarial examples:

```bash
python Step3_traceability.py \
    --num_clients 5 \
    --dataset mnist \
    --model lenet \
    --artifacts_dir ./artifacts/mnist/lenet \
    --pgd_eps 0.8
```

**Outputs:**
- Targeted success rate matrix (detector × source)
- Mask delta tracing results with fusion scores

## Experimental Configuration

Based on the paper's configuration:

### Hardware Requirements

| Component | Specification |
|-----------|--------------|
| CPU | Intel Xeon Gold 6154 @ 3.00GHz (72 cores) |
| GPU | NVIDIA Tesla V100-SXM3-32GB |
| RAM | 1TB DDR4 |
| Storage | 8TB SSD |

### Software Environment

| Software | Version |
|----------|---------|
| OS | Ubuntu 22.04.4 |
| NVIDIA Driver | 560.35.05 |
| CUDA | 12.4 |
| Python | 3.10.12 |
| PyTorch | 2.8.0 |

### Default Hyperparameters

| Stage | Parameter | Value |
|-------|-----------|-------|
| **FL Training** | Batch Size | 64 |
| | Communication Rounds | 10 |
| | Local Epochs | 1 |
| | Learning Rate | 1e-4 |
| | Optimizer | SGD (momentum=0.99) |
| **Watermark Generation** | PGD Epsilon (ε) | 0.8 |
| | PGD Alpha (α) | 0.04 |
| | PGD Steps | 200 |
| | Clean Set Max | 300 |
| | Target Count | 200 |
| **Traceability** | PGD Epsilon (ε) | 0.8 |
