# MediScopeDiffusion: A 3D Diffusion Network for Medical Image Classification

## Project Overview

**MediScopeDiffusion** is an advanced deep learning framework that leverages diffusion probabilistic models for robust 3D medical image classification. This project demonstrates how diffusion models, traditionally used for image generation, can be adapted for classification tasks by learning to denoise label distributions rather than images themselves.

The system is specifically designed for analyzing volumetric CT scans to classify lung pathologies (Normal vs COVID-19 affected), combining global contextual understanding with fine-grained local region analysis through a novel dual-granularity approach.

---

## Table of Contents

1. [Problem Statement](#problem-statement)
2. [Methodology](#methodology)
3. [Architecture Overview](#architecture-overview)
4. [Implementation Details](#implementation-details)
5. [Dataset](#dataset)
6. [Training Pipeline](#training-pipeline)
7. [Results](#results)
8. [Installation & Usage](#installation--usage)
9. [Project Structure](#project-structure)
10. [Technical Contributions](#technical-contributions)
11. [Limitations & Future Work](#limitations--future-work)

---

## Problem Statement

### Challenge
Automated classification of 3D medical scans (CT, MRI) faces several critical challenges:

1. **High Inter-Class Similarity**: Different pathologies can appear visually similar in medical images
2. **Noise and Artifacts**: Medical scans often contain acquisition noise and imaging artifacts
3. **Volumetric Complexity**: 3D medical data requires processing spatial relationships across all three dimensions
4. **Limited Data**: Medical datasets are typically small due to privacy concerns and annotation costs
5. **Localized vs Global Features**: Both overall anatomy (global) and specific lesions (local) are important for diagnosis

### Our Solution
We propose **MediScopeDiffusion**, a diffusion-based classification framework that:
- Learns robust representations through iterative denoising
- Combines global contextual information with localized region analysis
- Processes true 3D volumetric data without losing spatial coherence
- Uses heterogeneous noise schedules for improved feature learning

---

## Methodology

<img width="1244" height="447" alt="image" src="https://github.com/user-attachments/assets/6079562e-a62a-450a-bba5-5e5d2f90ba40" />
<img width="1241" height="616" alt="image" src="https://github.com/user-attachments/assets/28f7f500-3764-4970-ab61-cc371859a247" />
<img width="1256" height="658" alt="image" src="https://github.com/user-attachments/assets/361e2e2f-e2b8-47f6-8e2e-606e818a865a" />



Our approach consists of four core innovations:

### 1. **Dual-Granularity Conditional Guidance (DCG)**

We process medical images at two complementary scales:

**Global Stream**:
- Processes the entire 3D volume
- Captures overall anatomical structure and context
- Generates global prediction prior (ŷᵍ)
- Uses 3D ResNet-style encoder

**Local Stream**:
- Extracts top-K Regions of Interest (ROIs) based on saliency
- Focuses on suspicious/abnormal regions
- Generates local prediction prior (ŷˡ)
- Uses gated attention to fuse multiple ROI features

**Why Dual-Granularity?**
Medical diagnosis requires both "seeing the forest" (global anatomy) and "seeing the trees" (specific lesions). Our dual approach mimics radiologist workflow.

### 2. **Dense Guidance Map (M)**

We create a 3D interpolation map between global and local priors:

```
M[i,j,k] = (1 - d[i,j,k]) × ŷᵍ + d[i,j,k] × ŷˡ
```

Where:
- `d[i,j,k]` is a normalized 3D distance matrix
- Each point in the 3D grid gets a weighted combination of global and local information
- Creates smooth transitions between global context and local details

**Purpose**: Provides spatially-varying guidance for the diffusion process, allowing different regions to emphasize global or local features.

### 3. **Image Feature Prior (F)**

We extract deep semantic features using dual encoders:

**Global Feature Encoder**:
- CNN-based architecture (memory-efficient alternative to Transformers)
- Processes entire 3D volume
- Captures hierarchical spatial patterns

**Local Feature Encoder**:
- Processes each extracted ROI
- Captures fine-grained lesion characteristics
- Multiple ROI features fused via learnable attention

**Fusion Strategy**:
```
F = Σ Q ⊙ [F_global, F_roi1, F_roi2, ..., F_roiK]
```

Where Q is a learnable query matrix for gated attention.

### 4. **Heterologous Diffusion Process**

Unlike standard diffusion models that use uniform noise:

**Forward Process (Training)**:
- Broadcast ground truth labels to 3D grid: (C) → (C × Nₚ × Nₚ × Nₚ)
- Each point in grid receives **different timestep** t[i,j,k]
- Add spatially-varying Gaussian noise
- Creates heterogeneous noisy label distribution

**Reverse Process (Inference)**:
- Start from random 3D noise
- Iteratively denoise using 3D U-Net conditioned on:
  - Dense guidance map M
  - Image feature prior F
  - Timestep embeddings t
- Average predictions across all noise points

**Advantages**:
- Different spatial regions learn at different noise levels
- More robust feature learning
- Exploits spatial structure of medical data

### 5. **3D U-Net Denoising Network**

Architecture details:
- **Input**: Concatenated noisy labels + dense guidance map
- **Conditioning**: Time embeddings + image feature prior at each layer
- **Structure**: Encoder-decoder with skip connections
- **Output**: Predicted noise distribution

**Training Objective**:
```
L = MSE(ε_predicted, ε_true) + λ × CrossEntropy(classifier, y_true)
```

Where:
- First term: Standard diffusion denoising loss
- Second term: Auxiliary classification loss (helps guide learning)

---

## Architecture Overview

### Complete Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                    INPUT: 3D CT Scan (128×128×64)               │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                  DUAL-GRANULARITY CONDITIONAL GUIDANCE          │
├─────────────────────────────────────────────────────────────────┤
│  Global Encoder → Saliency Map → Global Prior (ŷᵍ)             │
│  ROI Extractor → Local Encoders → Local Prior (ŷˡ)             │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                     PRIOR GENERATION                            │
├─────────────────────────────────────────────────────────────────┤
│  Dense Guidance Map: M = Interpolate(ŷᵍ, ŷˡ)                   │
│  Feature Prior: F = Fuse(Global_Features, ROI_Features)        │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                  DIFFUSION FORWARD PROCESS (Training)           │
├─────────────────────────────────────────────────────────────────┤
│  y₀ → Expand to 3D grid → Add heterogeneous noise → yₜ         │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                     3D U-NET DENOISING                          │
├─────────────────────────────────────────────────────────────────┤
│  Input: [yₜ, M]                                                 │
│  Conditioning: t (time), F (features)                           │
│  Output: ε̂ (predicted noise)                                    │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│              DIFFUSION REVERSE PROCESS (Inference)              │
├─────────────────────────────────────────────────────────────────┤
│  Random Noise → Iterative Denoising → Clean Labels → Prediction│
└─────────────────────────────────────────────────────────────────┘
```

### Component Breakdown

| Component | Input Shape | Output Shape | Parameters | Purpose |
|-----------|-------------|--------------|------------|---------|
| Global Encoder | (B, 1, 128, 128, 64) | (B, 256, 8, 8, 4) | ~2M | Extract global features |
| Local Encoder | (B, 1, 16, 16, 8) | (B, 128) | ~500K | Extract ROI features |
| Dense Map Generator | (B, 2) × 2 | (B, 2, 8, 8, 8) | 0 | Create guidance |
| Feature Prior Fusion | (B, 256, 1, 1, 1) × K | (B, 256, 1, 1, 1) | ~130K | Fuse features |
| 3D U-Net | (B, 4, 8, 8, 8) | (B, 2, 8, 8, 8) | ~18M | Denoise labels |
| **Total** | - | - | **~22.5M** | - |

---

## Implementation Details

### Technology Stack

- **Framework**: PyTorch 2.0+
- **Hardware**: NVIDIA T4 GPU (15GB VRAM)
- **Platform**: Kaggle Notebooks
- **Key Libraries**:
  - `torch`, `torchvision`
  - `nibabel` (NIfTI medical image loading)
  - `scipy` (volume resampling)
  - `sklearn` (metrics)
  - `matplotlib`, `seaborn` (visualization)

### Key Hyperparameters

```python
# Model Configuration
NUM_CLASSES = 2
NUM_POINTS = 8          # Noise grid resolution (Nₚ)
NUM_ROIS = 5            # Top-K ROIs to extract
EMBED_DIM = 256         # Feature embedding dimension
TIMESTEPS = 500         # Diffusion timesteps

# Data Configuration
INPUT_SIZE = (128, 128, 64)  # Resized volume dimensions
BATCH_SIZE = 4               # Limited by GPU memory

# Training Configuration
DCG_EPOCHS = 15              # DCG pre-training epochs
DIFFUSION_EPOCHS = 30        # Diffusion training epochs
DCG_LR = 1e-3               # DCG learning rate
DIFFUSION_LR = 5e-5         # Diffusion learning rate
```

### Memory Optimizations

To fit on 15GB GPU:

1. **Reduced Spatial Resolution**: 512×512×42 → 128×128×64
2. **Small Batch Size**: 4 samples per batch
3. **CNN instead of Transformer**: Avoids O(n²) attention memory
4. **Gradient Checkpointing**: Trades compute for memory
5. **Regular Memory Clearing**: Explicit CUDA cache clearing
6. **Mixed Precision Training**: FP16 where possible

---

## Dataset

### Source
**MosMedData**: Chest CT Scans with COVID-19 Related Findings

### Statistics

| Split | Normal | COVID | Total | Percentage |
|-------|--------|-------|-------|------------|
| Train | 64 | 64 | 128 | 64% |
| Val | 16 | 16 | 32 | 16% |
| Test | 20 | 20 | 40 | 20% |
| **Total** | **100** | **100** | **200** | **100%** |

### Preprocessing Pipeline

1. **Loading**: Read NIfTI (.nii.gz) files using `nibabel`
2. **Normalization**: 
   - Clip HU values: [-1000, 400] (lung CT range)
   - Scale to [0, 1]
3. **Resampling**: Resize to 128×128×64 using trilinear interpolation
4. **Augmentation** (training only):
   - Random horizontal flip (50%)
   - Random vertical flip (50%)
5. **Tensor Conversion**: Add channel dimension (1, 128, 128, 64)

### Data Characteristics

```
Original Scan Properties:
- Average shape: 512 × 512 × 42 slices
- Intensity range: [-2048, 1771] HU
- Data type: float64
- Memory per scan: ~42 MB

Preprocessed Properties:
- Fixed shape: 128 × 128 × 64
- Intensity range: [0, 1]
- Data type: float32
- Memory per scan: ~4 MB
```

---

## Training Pipeline

### Phase 1: DCG Pre-training (15 epochs)

**Objective**: Train global and local encoders to predict class labels

```python
Loss = CrossEntropy(global_pred, labels) + CrossEntropy(local_pred, labels)
```

**Configuration**:
- Optimizer: AdamW
- Learning Rate: 1e-3 with ReduceLROnPlateau
- Weight Decay: 0.01
- Label Smoothing: 0.1
- Early Stopping: Patience = 7 epochs

**Expected Progress**:
- Epoch 1-3: Rapid improvement (acc: 50% → 70%)
- Epoch 4-10: Steady gains (acc: 70% → 80%)
- Epoch 11-15: Fine-tuning (acc: 80% → 85%)

### Phase 2: Diffusion Training (30 epochs)

**Objective**: Train U-Net to denoise noisy label distributions

**Freeze**: DCG model parameters (fixed feature extraction)

**Loss Function**:
```python
Total_Loss = Diffusion_Loss + 0.5 × Classification_Loss

Where:
- Diffusion_Loss = MSE(ε_predicted, ε_true)
- Classification_Loss = CrossEntropy(classifier(F), labels)
```

**Configuration**:
- Optimizer: AdamW with parameter groups
  - Prior Generator: lr = 5e-5
  - U-Net: lr = 5e-5
  - Classifier Head: lr = 1e-4 (2× higher)
- Scheduler: CosineAnnealingWarmRestarts
- Gradient Clipping: max_norm = 1.0
- Early Stopping: Patience = 10 epochs

**Expected Progress**:
- Epoch 1-5: Model learns denoising basics
- Epoch 6-15: Diffusion converges
- Epoch 16-30: Fine-tuning and refinement

### Training Time

On Kaggle T4 GPU (15GB):
- DCG Pre-training: ~45 minutes (15 epochs)
- Diffusion Training: ~2 hours (30 epochs)
- **Total**: ~2.75 hours

---

## Results

### Performance Metrics

#### Final Test Set Results

| Metric | Score | Interpretation |
|--------|-------|----------------|
| **Accuracy** | 82.50% | ✅ Excellent |
| **Precision** | 84.21% | High confidence in positive predictions |
| **Recall (Sensitivity)** | 80.00% | Detects 80% of COVID cases |
| **F1-Score** | 82.05% | Balanced precision-recall |
| **Specificity** | 85.00% | Low false alarm rate |
| **ROC-AUC** | 0.8250 | Good discriminative ability |

#### Confusion Matrix (Test Set)

```
                Predicted
              Normal  COVID
Actual Normal    17      3     (85% correct)
       COVID      4     16     (80% correct)
```

**Analysis**:
- **True Negatives**: 17 (Normal correctly identified)
- **False Positives**: 3 (Normal misclassified as COVID)
- **False Negatives**: 4 (COVID misclassified as Normal)
- **True Positives**: 16 (COVID correctly identified)

### Comparison with Baseline

| Model | Accuracy | F1-Score | ROC-AUC | Parameters |
|-------|----------|----------|---------|------------|
| 3D ResNet-18 | 75.0% | 74.2% | 0.751 | 33M |
| 3D DenseNet-121 | 77.5% | 76.8% | 0.778 | 8M |
| **MediScopeDiffusion (Ours)** | **82.5%** | **82.0%** | **0.825** | **22.5M** |

**Key Advantages**:
- ✅ +5-7.5% accuracy over baseline CNNs
- ✅ Robust to noise and artifacts (via denoising)
- ✅ Interpretable (saliency maps show ROIs)
- ✅ Reasonable parameter count

### Learning Curves

**DCG Pre-training**:
```
Epoch 1:  Train Acc: 52.3%, Val Acc: 50.0%
Epoch 5:  Train Acc: 71.8%, Val Acc: 68.8%
Epoch 10: Train Acc: 82.0%, Val Acc: 78.1%
Epoch 15: Train Acc: 87.5%, Val Acc: 84.4%  ← Best
```

**Diffusion Training**:
```
Epoch 1:  Train Loss: 0.0421, Val Loss: 0.0456
Epoch 10: Train Loss: 0.0187, Val Loss: 0.0203
Epoch 20: Train Loss: 0.0089, Val Loss: 0.0112
Epoch 28: Train Loss: 0.0041, Val Loss: 0.0058  ← Best (Early Stop)
```

### Qualitative Results

**Saliency Map Visualization**:
- Model correctly focuses on lung regions
- High activation in areas with ground-glass opacities (COVID hallmark)
- Low activation in healthy lung tissue

**ROI Extraction**:
- Captures bilateral lung infiltrates
- Identifies peripheral consolidations
- Highlights lesions accurately

---

## Installation & Usage

### Prerequisites

```bash
# Required Python version
Python >= 3.8

# Required CUDA
CUDA >= 11.0 (for GPU acceleration)
```

### Setup on Kaggle

1. **Create New Notebook**
   - Go to kaggle.com/code
   - Click "New Notebook"
   - Select GPU accelerator (T4 or better)

2. **Upload Dataset**
   - Download MosMedData
   - Upload to Kaggle Dataset
   - Add to notebook

3. **Install Dependencies**
   ```python
   # Most packages pre-installed on Kaggle
   # Additional installs:
   !pip install nibabel --quiet
   ```

4. **Run Cells Sequentially**
   - Cell 1: Setup & Data Loading
   - Cell 2: Preprocessing
   - Cell 3: DCG Model
   - Cell 4: Prior Generation
   - Cell 5: U-Net
   - Cell 6: Training
   - Cell 7: Evaluation

### Inference on New Data

```python
import torch
import nibabel as nib

# Load trained model
checkpoint = torch.load('diffmic3d_enhanced_model.pth')
model = DiffMIC3D_Enhanced(**checkpoint['model_config'])
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Load and preprocess new scan
scan = nib.load('new_scan.nii.gz').get_fdata()
scan = preprocess_scan(scan, target_shape=(128, 128, 64))
scan = torch.FloatTensor(scan).unsqueeze(0).to(device)

# Run inference
with torch.no_grad():
    probs = model(scan, training=False)
    pred = torch.argmax(probs, dim=1)
    confidence = probs[0, pred].item()

print(f"Prediction: {'COVID' if pred == 1 else 'Normal'}")
print(f"Confidence: {confidence*100:.2f}%")
```

---

## Project Structure

```
mediscopediffusion/
│
├── notebooks/
│   ├── 01_data_loading.ipynb           # Cell 1: Data loading
│   ├── 02_preprocessing.ipynb          # Cell 2: Preprocessing
│   ├── 03_dcg_model.ipynb             # Cell 3: DCG architecture
│   ├── 04_prior_generation.ipynb      # Cell 4: Priors
│   ├── 05_unet.ipynb                  # Cell 5: U-Net
│   ├── 06_training.ipynb              # Cell 6: Training
│   └── 07_evaluation.ipynb            # Cell 7: Evaluation
│
├── models/
│   ├── dcg_model.py                   # DCG architecture
│   ├── prior_generator.py             # Prior generation
│   ├── unet_3d.py                     # 3D U-Net
│   └── diffusion_process.py           # Diffusion handler
│
├── utils/
│   ├── preprocessing.py               # Data preprocessing
│   ├── metrics.py                     # Evaluation metrics
│   └── visualization.py               # Plotting functions
│
├── data/
│   ├── MosMedData/
│   │   ├── Normal/                    # Normal CT scans
│   │   └── Abnormal/                  # COVID CT scans
│   └── processed/                     # Preprocessed data
│
├── outputs/
│   ├── diffmic3d_enhanced_model.pth   # Trained model
│   ├── evaluation_results.csv         # Metrics
│   └── figures/                       # Visualizations
│
├── README.md                          # This file
└── requirements.txt                   # Dependencies
```

---

## Technical Contributions

### 1. **Novel Diffusion Application**
First application of diffusion models to 3D medical image classification with heterogeneous noise schedules.

### 2. **Dual-Granularity Framework**
Integration of global contextual and local lesion-specific information through learnable dense guidance maps.

### 3. **3D Spatial Awareness**
Full 3D volumetric processing without slice-by-slice reduction, preserving spatial coherence.

### 4. **Memory-Efficient Design**
Techniques to fit complex diffusion architecture on limited GPU memory (15GB).

### 5. **Auxiliary Supervision**
Joint optimization of diffusion denoising and direct classification for improved convergence.

### 6. **Clinical Interpretability**
Saliency maps and ROI extraction provide visual explanations for predictions.

---

## Limitations & Future Work

### Current Limitations

1. **Dataset Size**: 200 samples is small for deep learning
   - Solution: Data augmentation, transfer learning
   
2. **Binary Classification**: Only Normal vs COVID
   - Solution: Extend to multi-class (pneumonia, tuberculosis, etc.)

3. **Computational Cost**: 2.75 hours training time
   - Solution: Model distillation, pruning

4. **CNN vs Transformer**: Using CNN for global features
   - Solution: Implement memory-efficient transformers

5. **Single Modality**: Only CT scans
   - Solution: Extend to MRI, X-ray

### Future Directions

#### Short-term (1-3 months)
- [ ] Implement gradient checkpointing for Transformer
- [ ] Expand to multi-class classification
- [ ] Add uncertainty quantification
- [ ] Deploy as web application

#### Medium-term (3-6 months)
- [ ] Multi-modal fusion (CT + clinical data)
- [ ] Federated learning for privacy
- [ ] Active learning for efficient labeling
- [ ] Real-time inference optimization

#### Long-term (6-12 months)
- [ ] Clinical validation study
- [ ] FDA/CE regulatory pathway
- [ ] Integration with hospital PACS systems
- [ ] Longitudinal disease progression tracking

### Potential Applications

1. **Screening Tool**: Rapid COVID-19 detection in emergency departments
2. **Second Opinion**: Assist radiologists in complex cases
3. **Resource Allocation**: Triage patients in resource-limited settings
4. **Research Platform**: Benchmark for new medical AI methods
5. **Educational Tool**: Training medical students with AI assistance

---

## Citation

If you use this work, please cite:

```bibtex
@misc{mediscopediffusion2025,
  title={MediScopeDiffusion: A 3D Diffusion Network for Medical Image Classification},
  author={Fahad Ahmad and Ansh Bharadwaj and Anuj Soni and Yash Kumar Saini},
  year={2025},
  institution={Indian Institute of Information Technology, Nagpur},
  howpublished={GitHub Repository},
  url={https://github.com/username/mediscopediffusion}
}
```

---

## Acknowledgments

- **Supervisors**: Mr. Pravin S. Bhagat, Ms. Nayna Potdukhe
- **Institution**: Indian Institute of Information Technology, Nagpur
- **Dataset**: MosMedData Consortium
- **Platform**: Kaggle Notebooks
- **Framework**: PyTorch Team

---

## Contact

For questions, suggestions, or collaborations:


**Project Repository**: [GitHub Link]

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.

**Note**: Medical AI models should only be used as decision support tools and should not replace professional medical diagnosis. Always consult qualified healthcare providers for medical decisions.

---

**Last Updated**: October 2025  
**Version**: 1.0.0  
**Status**: ✅ Complete & Tested
