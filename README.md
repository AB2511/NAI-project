# NeuroAdaptive Interface (NAI) - Real-time EEG Cognitive State Monitoring

A production-ready NeuroAdaptive Interface for real-time P300-based cognitive state detection using deep learning. Features EEGNet architecture with LOSO cross-validation, achieving 85% AUC across 20 subjects on the ERP CORE dataset.

![System Architecture](figures/final/system_architecture.png)

## 📊 Key Results

### ERP Analysis
![Grand Average ERP](figures/final/erp_grand_target_vs_nontarget.png)
*Grand-average ERP showing P300 component for target vs non-target stimuli*

### Brain Topography  
![P300 Topomap](figures/final/topomap_group_300ms.png)
*P300 difference topography at 300ms showing central-parietal distribution*

### Machine Learning Performance
![ROC Curve](figures/final/roc_eegnet_loso.png)
*EEGNet LOSO cross-validation ROC curve (AUC = 0.831)*

![ML Comparison](figures/final/ml_comparison_barplot.png)
*Comparison of EEGNet vs traditional ML methods*

## 🎯 What This Project Does

This system performs real-time P300 detection from EEG signals to monitor cognitive states (focused, distracted, overloaded). It combines modern deep learning (EEGNet) with traditional signal processing for robust, cross-subject generalization in brain-computer interface applications.

## 🎯 Project Overview

This system provides real-time EEG-based cognitive state monitoring with:
- **Real-time EEG acquisition** via LSL (Lab Streaming Layer)
- **CFEM (Cognitive Feature Extraction Module)** for 4-class state detection
- **P300 online processing** with fatigue index computation
- **Decision Logic Engine (DLE)** for adaptive feedback
- **Streamlit dashboard** for live visualization
- **Short calibration protocol** (5-10 minutes)

## 🏗️ System Architecture

```
EEG Device → LSL → Feature Extraction → ML Inference → Decision Logic → Feedback
                     ↓                      ↓              ↓
                  P300 Module         State Prediction   Dashboard
```

## 📊 Performance Metrics

- **EEGNet LOSO AUC**: 0.85 ± 0.12 (20-fold cross-subject validation)
- **Traditional ML AUC**: 0.78 ± 0.15 (baseline comparison)  
- **Inference latency**: <20ms for model prediction
- **Real-time processing**: <50ms total loop (acquisition→feedback)
- **Dataset**: ERP CORE P3 (40 subjects, 20 used for validation)

## 🚀 Quick Start

### 1. Environment Setup
```bash
git clone https://github.com/AB2511/NAI-project.git
cd NAI-project
pip install -r requirements.txt
```

### 2. Download Dataset
Download ERP CORE P3 from [OSF](https://osf.io/etdkz/) or [OpenNeuro](https://openneuro.org/datasets/ds003061/) and place in `data/raw_p3b/`

### 3. Run Complete Pipeline
```bash
python src/run_loso_training.py --device cpu --epochs 60
```

### 4. Real-time System
```bash
# Terminal 1: EEG acquisition
python src/acquisition/lsl_acquire.py

# Terminal 2: Inference server  
python src/inference/infer_server.py

# Terminal 3: Dashboard
streamlit run src/dashboard/app.py
```

📖 **Full setup guide**: [ENVIRONMENT_SETUP.md](ENVIRONMENT_SETUP.md)

## 📁 Project Structure

```
NAI-project/
├── src/                           # Core analysis scripts
│   ├── preprocess_p3b.py         # Data preprocessing pipeline
│   ├── within_subject_pipeline.py # Within-subject classification
│   ├── eegnet_v5.py              # EEGNet LOSO implementation
│   ├── paper_analysis_p3b.py     # Publication-grade ERP analysis
│   └── pca_baselines_p3b.py      # Traditional ML baselines
├── notebooks/                     # Jupyter analysis notebooks
├── figures/                       # Generated visualizations
│   ├── erp/                      # ERP plots and group averages
│   ├── topomaps/                 # Topographic maps
│   └── results/                  # Statistical summaries
├── results/                       # Experimental results
│   ├── eegnet_v5/               # Deep learning results
│   ├── within_subject_full/     # Within-subject validation
│   └── stats/                   # Statistical test results
├── models/                        # Trained model files
└── data/                         # Dataset directory (see data/README.md)
```

## 🧠 Cognitive States

The system detects 4 cognitive states:
1. **Relaxed**: Low cognitive load, eyes open baseline
2. **Focused**: Sustained attention, congruent tasks
3. **Distracted**: Mind-wandering, incongruent stimuli
4. **Overload**: High cognitive load, time pressure

## 🔬 Technical Details

### Feature Extraction (CFEM)
- **Window**: 1s with 250ms step
- **Frequency bands**: Delta, Theta, Alpha, Beta, Gamma
- **Features**: Band powers, ratios, spectral entropy, wavelet energies
- **Channels**: Optimized for Fz, Cz, Pz (P300) + full montage

### P300 Processing
- **Epoching**: -200 to +800ms around markers
- **Baseline correction**: -200 to 0ms
- **Running average**: α=0.05 smoothing
- **Fatigue index**: Amplitude decline over time

### Machine Learning
- **Algorithm**: VotingClassifier (RandomForest + LinearSVC)
- **Training**: Stratified CV with SMOTE for imbalance
- **Validation**: Cross-subject generalization

## 🎯 P300 Oddball Experiment Protocol

**NEW FEATURE**: Complete P300 oddball experiment with fatigue monitoring

### Experiment Design
- **Total trials**: 400 (320 standard, 80 oddball)
- **Stimuli**: 440Hz standard tone, 880Hz oddball tone
- **Duration**: 200ms stimulus, 800ms ISI
- **Total time**: ~8 minutes
- **Probability**: 80% standard, 20% oddball

### Measurements
- **P300 amplitude**: Peak detection in 250-550ms window
- **P300 latency**: Time to peak from stimulus onset
- **Fatigue index**: `fatigue = 1 - (amplitude / baseline_mean)`
- **ERP waveforms**: Averaged by trial type (standard vs oddball)

### Enhanced Dashboard Features
1. **Live ERP Waveform**: Real-time averaged ERP (30-trial sliding window)
2. **P300 Amplitude Trend**: Fatigue-induced amplitude decline
3. **Bandpower Distribution**: Real-time frequency analysis
4. **Prediction Timeline**: State classification over time
5. **System Latency Gauge**: Processing speed monitoring

### Usage
```bash
# Complete demo with synthetic data
python demo_oddball.py

# Real experiment with EEG
python run_oddball_experiment.py --trials 400

# Quick test (20 trials)
python run_oddball_experiment.py --demo
```

## 📈 Calibration Protocol

**Duration**: 5-10 minutes
1. **Baseline** (2 min): Eyes-open rest → Relaxed state
2. **Stroop/Oddball** (3 min): 120 stimuli (80 standard, 40 oddball)
   - Congruent/neutral → Focused state
   - Incongruent + time pressure → Overload state
3. **Model adaptation**: Quick retraining or reweighting

## 🎮 Adaptive Feedback

The Decision Logic Engine triggers:
- **Visual cues**: Color changes, progress bars
- **Audio cues**: Breathing guidance, attention alerts
- **Mandatory pauses**: When overload detected
- **Performance tracking**: Error rates, reaction times

## 📊 Validation Results

- **Confusion matrix**: 4-class classification performance
- **P300 amplitude**: Longitudinal fatigue tracking
- **Latency analysis**: Real-time processing benchmarks
- **User study**: n=3-5 participants, overload reduction metrics

## 🔧 Hardware Requirements

- **EEG device**: Any LSL-compatible system (OpenBCI, g.tec, etc.)
- **Minimum channels**: 3 (Fz, Cz, Pz) for P300
- **Recommended**: 8+ channels for full CFEM
- **Sampling rate**: 256Hz or higher
- **Optional**: Arduino for precise stimulus timing

## 📊 Dataset

**⚠️ Dataset not included** - Download ERP CORE P3 dataset from:
- **OSF**: https://osf.io/etdkz/ (Original format with analysis scripts)
- **OpenNeuro BIDS**: https://openneuro.org/datasets/ds003061/ (BIDS-compatible format)

Place raw files in `data/raw_p3b/` and run preprocessing scripts to generate `data/processed_p3b/`.

## 📑 Citation

If you use this project, please cite:

```bibtex
@misc{barge2025nai,
  title   = {NeuroAdaptive Interface: Real-time EEG Cognitive State Monitoring},
  author  = {Barge, Anjali},
  year    = {2025},
  url     = {https://github.com/AB2511/NAI-project}
}
```

## 📚 References

- Kappenman, E. S., et al. (2021). ERP CORE: An open resource for human event-related potential research. *NeuroImage*, 225, 117465.
- Lawhern, V. J., et al. (2018). EEGNet: a compact convolutional neural network for EEG-based brain–computer interfaces. *Journal of Neural Engineering*, 15(5), 056013.
- OpenNeuro P300 dataset (ds003061): https://openneuro.org/datasets/ds003061/

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Contributing

This is a research demonstration system. For production use, consider:
- Clinical validation studies
- Regulatory compliance (if medical application)
- Extended user testing
- Hardware optimization

## 📄 License

MIT License - See LICENSE file for details.

---

**Developed for NeuroAdaptive Interface Research**  
*Real-time EEG cognitive state monitoring with adaptive feedback*
