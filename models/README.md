# Models Directory

Trained models are stored in Git LFS and available upon request or via GitHub release artifacts.

## Available Models

- **EEGNet v5 LOSO**: Cross-subject P300 classification models
- **Traditional ML Pipeline**: Feature-based classifiers (Random Forest, SVM, XGBoost)
- **Calibration Models**: Subject-specific adaptation models

## Model Performance

- **EEGNet LOSO AUC**: 0.85 ± 0.12 (20-fold cross-subject validation)
- **Traditional ML AUC**: 0.78 ± 0.15 (baseline comparison)
- **Real-time Latency**: <20ms inference time

## Usage

Models are automatically downloaded when running training scripts or can be requested via GitHub issues.