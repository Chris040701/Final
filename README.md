# Land Cover and Land Use Classification using Satellite Imagery

![Deep Learning](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![Dataset](https://img.shields.io/badge/Dataset-EuroSAT-blue)

A deep learning project for classifying land cover types (e.g., forest, urban, water) from Sentinel-2 satellite imagery using PyTorch.

## Overview

This project leverages a pre-trained WideResNet50 model to classify RGB satellite image patches from the EuroSAT dataset into 10 land cover classes. The implementation achieves **94.3% validation accuracy** with techniques like transfer learning, early stopping, and learning rate scheduling.

## Problem Description

Land cover classification is critical for environmental monitoring, urban planning, and climate studies. Challenges include spectral similarity between classes, cloud cover, and mixed pixels. This project addresses these using deep learning to automate accurate classification.

## Dataset

The **EuroSAT RGB Dataset** (27,000 labeled 64x64 patches) is used. It includes 10 classes:
- AnnualCrop, Forest, HerbaceousVegetation, Highway, Industrial, Pasture, PermanentCrop, Residential, River, SeaLake.

**Access:**
- Download from [EuroSAT](https://github.com/phelber/eurosat) or [Copernicus Data Space](https://dataspace.copernicus.eu) (registration required).

## Methodology

### Supervised Learning with Transfer Learning
1. **Model Architecture**: Fine-tune a pre-trained `wide_resnet50_2` with a custom classifier head.
2. **Training**:
   - Freeze base layers initially, train only the classifier.
   - Unfreeze all layers for fine-tuning (optional).
3. **Augmentation**: Resize, normalize, and apply PyTorch transforms.
4. **Evaluation**: Confusion matrix, accuracy metrics, and loss plots.

## Implementation

### Key Steps

#### 1. Data Preparation
```python
class EuroSATDataset(Dataset):
    def __getitem__(self, idx):
        img_path = os.path.join(self.base_image_dir, label_str, img_id)
        img = Image.open(img_path).convert('RGB')
        return img, encode_label(label_str)
```

#### 2. Model Definition
```python
class LULC_Model(MulticlassClassifierBase):
    def __init__(self):
        self.network = models.wide_resnet50_2(weights=models.Wide_ResNet50_2_Weights.DEFAULT)
        self.network.fc = nn.Sequential(
            nn.Linear(n_inputs, 256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, NUM_CLASSES), nn.LogSoftmax(dim=1)
        )
```

#### 3. Training Loop
```python
history = fit(
    epochs=10,
    max_lr=1e-4,
    model=model,
    train_loader=train_dl_device,
    valid_loader=valid_dl_device,
    opt_func=torch.optim.Adam
)
```

### Results
- **Validation Accuracy**: 94.3%
- **Confusion Matrix**:
  ![Confusion Matrix](figures\confusion_matrix.jpg)
- **Training Curves**:
  - Loss vs. Epochs: ![Loss Plot](figures\Loss.jpg)
  - Accuracy vs. Epochs: ![Accuracy Plot](figures\acc.jpg)

## Environmental Impact

Training deep learning models has a carbon footprint. Strategies to mitigate this:
- **Transfer Learning**: Reduced training time by leveraging pre-trained weights.
- **Early Stopping**: Halts training if validation loss plateaus.
- **Efficient Batches**: Use batch size 64 for optimal GPU utilization.

## Setup & Usage

### Dependencies
Install with:
```bash
pip install -m requirements.txt
```

## References
- Helber P, Bischke B, Dengel A, et al. Eurosat: A novel dataset and deep learning benchmark for land use and land cover classification[J]. IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, 2019, 12(7): 2217-2226.
- Paszke A. Pytorch: An imperative style, high-performance deep learning library[J]. arXiv preprint arXiv:1912.01703, 2019.
