# TFG – Comparación de Modelos Supervisados y Métodos Difusos para HAR

**Comparación de modelos supervisados y métodos difusos para el reconocimiento de actividades cotidianas ambiguas a partir de datos de acelerómetros.**

> Trabajo Fin de Grado – Pablo Anel Rancaño – Universidad de Granada

---

## 📋 Overview

This project compares different approaches for Human Activity Recognition (HAR) using the **UCI HAR Dataset** (30 subjects, 6 activities, 50 Hz smartphone accelerometer/gyroscope data):

1. **Baseline (561 features):** Original UCI HAR feature set with classic supervised models.
2. **Interpretable features:** ~220 hand-crafted time/frequency domain features extracted from raw inertial signals.
3. **Brute-force features (tsfresh):** Thousands of automatically generated features.
4. **Fuzzy / uncertainty methods** *(future phase).*

All evaluations follow an anti-leakage protocol: the official train/test split is by subject, and cross-validation uses `GroupKFold` by subject ID.

---

## 📁 Project Structure

```
project/
├── config.yaml                  # Configuration (paths, parameters)
├── requirements.txt             # Python dependencies
├── README.md                    # This file
│
├── src/
│   ├── config.py                # Configuration management
│   ├── dataset_loader.py        # UCI HAR tabular data loader + legacy eval
│   ├── inertial_loader.py       # Raw inertial signals loader (9 channels)
│   ├── models.py                # Model registry (RF, LR, SVM, KNN)
│   ├── evaluation.py            # Unified evaluation framework
│   ├── feature_extraction_interpretable.py   # Interpretable feature pipeline
│   ├── feature_extraction_tsfresh.py         # tsfresh feature pipeline
│   ├── run_experiments.py       # ★ Unified experiment runner (CLI)
│   ├── generate_report.py       # Cross-pipeline comparison report
│   ├── fuzzy_placeholder.py     # Phase 3 hooks (fuzzy/uncertainty)
│   │
│   ├── Random_Forest_Baseline.py      # Legacy baseline scripts
│   ├── Logistic_Regression_Baseline.py
│   ├── Linear_SVM_Baseline.py
│   └── KNN_Baseline.py
│
├── data/processed/              # Cached feature matrices (git-ignored)
└── results/                     # Experiment outputs (git-ignored)
    ├── baseline_561/
    ├── interpretable/
    ├── tsfresh/
    └── comparison_all_pipelines.{csv,md}
```

---

## 🔧 Setup

### 1. Create virtual environment

```bash
cd project/
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

For tsfresh (optional, large dependency):

```bash
pip install tsfresh
```

### 3. Dataset

The project expects the **UCI HAR Dataset** at the path specified in `config.yaml`:

```yaml
dataset_path: "../DataSets/human+activity+recognition+using+smartphones/UCI HAR Dataset/UCI HAR Dataset"
```

The directory must contain:
```
UCI HAR Dataset/
├── train/
│   ├── X_train.txt          # 561 features
│   ├── y_train.txt           # Activity labels (1-6)
│   ├── subject_train.txt     # Subject IDs (1-30)
│   └── Inertial Signals/     # 9 raw signal files
├── test/
│   ├── X_test.txt
│   ├── y_test.txt
│   ├── subject_test.txt
│   └── Inertial Signals/
├── activity_labels.txt
└── features.txt
```

Override the path via CLI if needed:

```bash
python src/run_experiments.py --dataset-path /path/to/UCI_HAR_Dataset
```

---

## 🚀 Running Experiments

### Quick start: Run everything

```bash
# 1. Extract interpretable features (one-time, ~2–5 min)
python src/feature_extraction_interpretable.py

# 2. Extract tsfresh features (one-time, ~10–30 min, requires tsfresh)
python src/feature_extraction_tsfresh.py

# 3. Run all models on all pipelines
python src/run_experiments.py

# 4. Generate comparison report
python src/generate_report.py
```

### Run specific pipelines and models

```bash
# Only baseline 561 features
python src/run_experiments.py --pipelines baseline_561

# Only interpretable features with Random Forest and KNN
python src/run_experiments.py --pipelines interpretable --models random_forest knn

# All pipelines with normalization
python src/run_experiments.py --normalize

# Custom number of CV folds
python src/run_experiments.py --n-splits 10
```

### Run legacy baseline scripts (backward compatible)

```bash
cd src/
python Random_Forest_Baseline.py
python Logistic_Regression_Baseline.py
python Linear_SVM_Baseline.py
python KNN_Baseline.py
```

---

## 📊 Outputs

Each experiment produces:
- **Metrics file** (`*_metrics.txt`): accuracy, F1 scores, classification report, CV results
- **Confusion matrix** (`*_confusion_test.png`): visual confusion matrix
- **Summary table** (`summary_*.csv` / `summary_*.md`): per-pipeline model comparison

The global comparison report includes:
- `results/comparison_all_pipelines.csv` — machine-readable
- `results/comparison_all_pipelines.md` — report-ready with highlights

---

## 🔬 Feature Extraction Details

### Interpretable Features (~220 features)

Per channel (9 channels × ~18 features):
| Domain | Features |
|--------|----------|
| Time | mean, std, min, max, median, range, IQR, skewness, kurtosis, RMS, MAD, ZCR, energy |
| Frequency | dominant frequency, spectral energy, spectral entropy, mean frequency, spectral bandwidth |

Plus cross-channel features:
- 3 magnitude vectors (total_acc, body_acc, body_gyro) × same features
- 9 axis-pair correlations

### tsfresh Features (hundreds to thousands)

Three extraction levels via `--tsfresh-settings`:
| Level | Approx. Features | Speed |
|-------|------------------|-------|
| `minimal` | ~100 | Fast |
| `efficient` (default) | ~800 | Moderate |
| `comprehensive` | ~4000+ | Slow |

NaN/Inf handling is automatic: columns with >50% NaN are dropped, remaining NaNs imputed with column median.

---

## ⚙️ Configuration

All settings live in `config.yaml` and can be overridden via CLI flags:

| Setting | CLI Flag | Default |
|---------|----------|---------|
| Dataset path | `--dataset-path` | `../DataSets/.../UCI HAR Dataset` |
| Results dir | `--results-dir` | `results` |
| Feature cache dir | `--processed-dir` | `data/processed` |
| CV folds | `--n-splits` | `5` |
| Config file | `--config` | auto-detect `config.yaml` |

---

## 🔒 Anti-Leakage Guarantee

The official UCI HAR split separates subjects into train (21 subjects) and test (9 subjects). This project:
- **Never** creates random splits across windows.
- Uses `GroupKFold` with subject IDs for cross-validation within training.
- All evaluation functions enforce this protocol by design.

---

## 📌 Models

| Key | Model | Hyperparameters |
|-----|-------|----------------|
| `random_forest` | RandomForestClassifier | 100 trees, random_state=42 |
| `logistic_regression` | LogisticRegression | lbfgs, max_iter=3000 |
| `linear_svm` | LinearSVC | random_state=42, max_iter=5000 |
| `knn` | KNeighborsClassifier | k=5 |

---

## 🗺️ Roadmap

- [x] Phase 1: Baseline models on 561 features
- [x] Phase 2a: Interpretable feature extraction from raw signals
- [x] Phase 2b: Brute-force feature extraction (tsfresh)
- [x] Phase 2c: Cross-pipeline comparison framework
- [ ] Phase 3: Fuzzy membership functions on interpretable features
- [ ] Phase 3: Confidence calibration and ambiguity detection
- [ ] Phase 3: Focused analysis on confusable pairs (SITTING/STANDING, walking variants)

---

## 📄 License

Academic project – Universidad de Granada.