# 🚀 NAI System Deployment Guide

## Quick Start (5 minutes)

### 1. System Validation
```bash
python validate_system.py
```
This will test all components and create a demo model if needed.

### 2. Demo with Synthetic Data
```bash
# Terminal 1: Start synthetic data streams
python demo_synthetic.py

# Terminal 2: Start inference server  
python src/inference/infer_server.py

# Terminal 3: Launch dashboard
streamlit run src/dashboard/app.py
```

### 3. Complete System Launcher
```bash
# All-in-one system launcher
python run_nai_system.py --demo

# With Arduino timing module
python run_nai_system.py --arduino-port COM3
```

## Real EEG Hardware Setup

### 1. EEG Device Configuration
- **Supported**: Any LSL-compatible EEG system
- **Minimum channels**: 3 (Fz, Cz, Pz) for P300
- **Recommended**: 8+ channels for full CFEM
- **Sampling rate**: 256 Hz or higher
- **Impedance**: <10kΩ (dry), <5kΩ (gel)

### 2. Calibration Protocol (5-10 minutes)
```bash
jupyter notebook notebooks/calibration_and_eval.ipynb
```

**Protocol Steps:**
1. **Baseline** (2 min): Eyes-open resting
2. **Focused Task** (3 min): Mental arithmetic
3. **Cognitive Load** (3 min): Stroop/oddball
4. **Distraction** (2 min): Multitasking

### 3. Model Training
The calibration notebook will:
- Extract CFEM features from EEG windows
- Train VotingClassifier (RandomForest + SVM)
- Validate with cross-validation
- Save model to `models/nai_voting_model.pkl`

## System Components

### Core Modules
- **Acquisition** (`src/acquisition/`): LSL EEG streaming
- **Feature Extraction** (`src/feature_extraction/`): CFEM processing
- **P300 Detection** (`src/p300/`): ERP analysis & fatigue
- **ML Inference** (`src/inference/`): Real-time classification
- **Decision Engine** (`src/decision_engine/`): Adaptive feedback
- **Dashboard** (`src/dashboard/`): Live visualization

### Optional Components
- **Arduino Timing** (`src/atm/`): Millisecond-precise markers
- **Synthetic Demo** (`demo_synthetic.py`): Hardware-free testing

## Performance Targets

| Metric | Target | Typical |
|--------|--------|---------|
| Classification F1-Macro | >70% | 75-85% |
| Inference Latency | <20ms | 12-18ms |
| Total Pipeline Latency | <50ms | 35-45ms |
| P300 Detection Rate | >80% | 85-92% |

## Troubleshooting

### Common Issues

**1. No LSL streams found**
```bash
# Check LSL streams
python -c "from pylsl import resolve_streams; print(resolve_streams())"
```

**2. Model not found**
```bash
# Create demo model
python validate_system.py
```

**3. Dashboard won't start**
```bash
# Check Streamlit installation
streamlit --version

# Try different port
streamlit run src/dashboard/app.py --server.port 8502
```

**4. Arduino connection issues**
```bash
# List serial ports (Windows)
python -c "import serial.tools.list_ports; print([p.device for p in serial.tools.list_ports.comports()])"

# Test Arduino bridge
python src/atm/arduino_lsl_bridge.py --port COM3 --verbose
```

### Performance Optimization

**1. Reduce latency**
- Use faster CPU
- Reduce feature vector size
- Optimize model (fewer estimators)
- Use multiprocessing

**2. Improve accuracy**
- Collect more calibration data
- Use subject-specific models
- Implement online learning
- Better artifact rejection

## Production Deployment

### 1. Hardware Requirements
- **CPU**: Multi-core (4+ cores recommended)
- **RAM**: 8GB minimum, 16GB recommended  
- **Storage**: SSD for faster model loading
- **EEG**: Professional-grade device with LSL support

### 2. Software Environment
```bash
# Create virtual environment
python -m venv nai_env
source nai_env/bin/activate  # Linux/Mac
# nai_env\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

### 3. Configuration
- Adjust thresholds in `DecisionLogicEngine`
- Customize feedback in dashboard
- Set appropriate intervention cooldowns
- Configure audio/visual cues

### 4. Monitoring & Logging
```bash
# Enable verbose logging
export NAI_LOG_LEVEL=DEBUG

# Monitor system resources
python run_nai_system.py --monitor
```

## Research Applications

### User Study Protocol
1. **Baseline session**: No feedback (20 min)
2. **Adaptive session**: With NAI feedback (20 min)
3. **Metrics**: Error rates, reaction times, subjective workload
4. **Analysis**: Compare overload episodes, performance metrics

### Data Collection
- EEG data saved automatically
- Intervention logs in dashboard
- Performance metrics exported
- P300 amplitude/latency trends

### Validation Metrics
- Confusion matrices for 4-class classification
- P300 fatigue detection accuracy
- Intervention effectiveness (overload reduction)
- Real-time performance (latency, throughput)

## Support & Development

### Getting Help
- **Issues**: GitHub Issues for bug reports
- **Discussions**: GitHub Discussions for questions
- **Documentation**: Wiki for detailed guides
- **Email**: Contact maintainers directly

### Contributing
1. Fork repository
2. Create feature branch
3. Add tests for new features
4. Submit pull request
5. Follow code style guidelines

### Testing
```bash
# Run system validation
python validate_system.py

# Test individual components
python src/inference/infer_server.py --test
python src/p300/p300_online.py
```

---

**🧠 NeuroAdaptive Interface - Real-time EEG Cognitive State Monitoring**  
*Developed for research and educational purposes*