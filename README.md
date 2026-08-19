# Data Diversity and Model Complexity in Natural Scene Classification

## Overview
This repository contains the code, models, and research findings for the project investigating whether increasing training-data diversity through advanced data augmentation provides a comparable generalization benefit to increasing model architectural complexity. The study evaluates a baseline Convolutional Neural Network (CNN) and an ImageNet-pretrained EfficientNetB0 model on a natural scene classification task[cite: 1].

---

## Dataset
*   The project uses the Intel Image Classification Dataset[cite: 1].
*   The dataset contains 16,900 manually cleaned images[cite: 1].
*   The images are categorized into six natural-scene classes: buildings, forest, glacier, mountain, sea, and street[cite: 1].

---

## Methodology & Architecture
The experiments compare two model architectures evaluated under two different data augmentation pipelines:

### Models
*   **Baseline CNN:** A conventional convolutional neural network trained from scratch, containing approximately 10.99 million trainable parameters[cite: 1].
*   **EfficientNetB0:** An ImageNet-pretrained EfficientNetB0 model with a frozen backbone and a trainable classification head containing 7,686 parameters[cite: 1].

### Data Augmentation Conditions
*   **Basic Augmentation:** A pipeline consisting of basic geometric transformations, specifically horizontal flipping, rotation, and zoom[cite: 1].
*   **Advanced Augmentation:** A pipeline incorporating the basic transformations along with brightness, contrast, saturation, hue, Gaussian noise, and random erasing transformations[cite: 1].

---

## Results & Performance
The results indicate that model architecture had a stronger effect on held-out test performance than the specific advanced augmentation strategy examined[cite: 1].

*   EfficientNetB0 substantially outperformed the baseline CNN under both basic and advanced augmentation conditions[cite: 1].
*   The basic augmentation configuration produced higher test accuracy, weighted F1-score, and ROC-AUC than the advanced augmentation configuration for both model architectures[cite: 1].
*   EfficientNetB0 with basic augmentation achieved the best overall performance, reaching a 92.35% test accuracy and a 99.30% weighted ROC-AUC[cite: 1].
*   The baseline CNN with basic augmentation achieved an 81.58% test accuracy[cite: 1].
*   Applying the advanced augmentation pipeline decreased test accuracy to 90.18% for EfficientNetB0 and to 79.53% for the baseline CNN[cite: 1].
*   The advanced augmentation pipeline also substantially increased the computational training time[cite: 1].

---

## Installation & Setup

### Prerequisites
Because this repository contains large `.keras` model files, you must have **Git LFS** installed before cloning.

1. Install Git LFS on your system (e.g., `git lfs install`).
2. Clone the repository:
   ```bash
   git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
   cd your-repo-name
