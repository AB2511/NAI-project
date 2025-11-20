# Environment Setup Guide

## System Requirements

- **OS**: Windows 11 (tested), Linux/macOS compatible
- **Python**: 3.8+ (recommended: 3.9)
- **GPU**: Optional (CUDA 11.0+ for faster training)
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 5GB for code + results

## Installation

### 1. Clone Repository
```bash
git clone https://github.com/AB2511/NAI-project.git
cd NAI-project
```

### 2. Create Virtual Environment
```bash
python -m venv nai_env
# Windows
nai_env\Scripts\activate
# Linux/macOS  
source nai_env/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Download Dataset
Download ERP CORE P3 dataset from:
- **OSF**: https://osf.io/etdkz/ 
- **OpenNeuro**: https://openneuro.org/datasets/ds003061/

Place files in `data/raw_p3b/`

### 5. Preprocess Data
```bash
python src/preprocess_p3b.py
```

## Running the System

### EEGNet LOSO Training
```bash
python src/eegnet_v5_loso.py --data-dir data/processed_p3b --out-dir results/eegnet_loso
```

### Real-time Dashboard
```bash
# Terminal 1: Start EEG acquisition
python src/acquisition/lsl_acquire.py

# Terminal 2: Start inference server  
python src/inference/infer_server.py

# Terminal 3: Launch dashboard
streamlit run src/dashboard/app.py
```

### Calibration Notebook
```bash
jupyter notebook notebooks/calibration_p300.ipynb
```

## Troubleshooting

### Common Issues
- **CUDA errors**: Use `--device cpu` flag
- **Memory errors**: Reduce batch size in training scripts
- **LSL connection**: Check EEG device compatibility

### GPU Setup (Optional)
```bash
# Install PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Verification
```bash
python test_dataset.py  # Verify dataset loading
python quick_demo.py    # Test system components
```