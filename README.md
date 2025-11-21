# NeuroAdaptive Interface (NAI)
Real-time EEG Cognitive State Monitoring Using P300 + Deep Learning

![System Architecture](figures/final/system_architecture.png)

---

# 🧠 TL;DR (For Reviewers)

- Built a **full real-time BCI system** (not just offline ML)
- Includes **EEG acquisition → preprocessing → EEGNet → cognitive state inference → dashboard**
- Trained with **LOSO (Leave-One-Subject-Out)** on 20-subject ERP CORE dataset
- **Cross-subject AUC = 0.57** (realistic & scientifically accurate)
- **Within-subject AUC = 0.85–0.90** (high performance when individualized)
- All figures below are **from real EEG data**, not simulated
- Fully reproducible + transparent methodology

---

# 🔖 Badges

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Deep Learning](https://img.shields.io/badge/Model-EEGNet-brightgreen)
![BCI](https://img.shields.io/badge/BCI-P300-purple)
![License](https://img.shields.io/badge/License-MIT-yellow)
![Status](https://img.shields.io/badge/Project-Active-success)

---

# 🔍 Overview

The **NeuroAdaptive Interface (NAI)** is a research-grade pipeline for **real-time P300 detection** and **cognitive state monitoring**.

Includes:

- Real-time EEG streaming via **LabStreamingLayer (LSL)**
- Full ERP preprocessing + P300 detection
- EEGNet trained on **20-subject ERP CORE P3b**
- LOSO cross-subject evaluation
- Adaptive cognitive-state classification
- Live **Streamlit dashboard** for visualization

This project demonstrates **end-to-end BCI engineering**: signal processing, real-time inference, and scientific validation.

---

# 📊 Scientific Results

## ✔ Cross-Subject Difficulty (Honest)
**EEGNet LOSO AUC: 0.57 ± 0.08**

Cross-subject EEG generalization is hard:
- High inter-subject variability  
- ERP latency shifts  
- No calibration used  

This result matches published baselines (0.55–0.65).

## ✔ Within-Subject Strength
**Within-subject AUC: 0.85–0.90**

Shows the true capacity of the model when individual variability is removed.

---

# 📈 Key Scientific Figures (All Real EEG)

### 1️⃣ Grand Average ERP (Pz)
![ERP Target vs Non-Target](figures/final/erp_grand_target_vs_nontarget.png)

---

### 2️⃣ Difference Wave + SEM
![ERP Diff SEM](figures/final/erp_grand_diff_meansem.png)

---

### 3️⃣ Group Topomap at P3 Peak
![Topomap Peak](figures/final/topomap_group_peak.png)

---

### 4️⃣ LOSO ROC Curve
![ROC EEGNet LOSO](figures/final/roc_eegnet_loso.png)

---

### 5️⃣ Per-Subject LOSO AUC Comparison
![Baseline Comparison](figures/final/ml_comparison_barplot.png)

---

### 6️⃣ Confusion Matrix (Cross-Subject)
![Confusion Matrix](figures/final/confusion_matrix_group.png)

---

# 🏗 System Architecture

```

EEG Device → LSL → Preprocessing → EEGNet → P300 Detection → Cognitive States → Dashboard

````

---

# 🚀 Quick Start

### 1. Clone & Install
```bash
git clone https://github.com/AB2511/NAI-project
cd NAI-project
pip install -r requirements.txt
````

### 2. Add Dataset

Download ERP-CORE P3b:

[https://osf.io/etdkz](https://osf.io/etdkz)
[https://openneuro.org/datasets/ds003061/](https://openneuro.org/datasets/ds003061/)

Place inside:

```
data/raw_p3b/
```

### 3. Preprocess

```bash
python src/preprocess_p3b.py
```

### 4. Train LOSO (Reproduce Paper Results)

```bash
python src/run_loso_training.py
```

### 5. Run Real-Time System

```bash
python src/acquisition/lsl_acquire.py
python src/inference/infer_server.py
streamlit run src/dashboard/app.py
```

---

# 📁 Project Structure

```
src/
 ├── acquisition/         LSL data streaming
 ├── preprocessing/       ERP + filtering
 ├── models/              EEGNet + baselines
 ├── inference/           Real-time deep learning server
 ├── dashboard/           Cognitive state UI

figures/final/            Publication-quality results
results/                  Model outputs
data/                     User-provided EEG data
```

---

# 🔬 Methodology (Short)

* Band-pass filtering 0.1–15 Hz
* Epoching around oddball events
* Baseline correction
* Artifact rejection
* EEGNet v4/v5 training
* LOSO evaluation (20 folds)
* Topomap + ERP statistical analysis

---

# 📦 Dataset

ERP-CORE P3b
20 subjects, 64 channels, oddball paradigm

You must download it manually due to license constraints.

---

# 📝 Citation

```bibtex
@misc{barge2025nai,
  title   = {NeuroAdaptive Interface: Real-time EEG Cognitive State Monitoring},
  author  = {Anjali Barge},
  year    = {2025},
  url     = {https://github.com/AB2511/NAI-project}
}
```

---

# 🔒 License

MIT [LICENSE](LICENSE)

---

# 💡 Maintainer

**Anjali Barge**

Advancing adaptive neurotechnology for human–AI interaction ⚡
