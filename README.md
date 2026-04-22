# TFG — Human Activity Recognition (HAR)

Supervised comparison of per-window sensor representations on the UCI HAR benchmark, with subject-aware evaluation and reproducible code.

**TFG** — Pablo Anel Rancano — University of Granada

---

## Project overview

This repository contains the code to load the **UCI HAR Dataset**, build per-window feature matrices (50 Hz, 128-sample windows), and evaluate four classical classifiers:

- Random Forest  
- Logistic Regression  
- Linear SVM  
- k-NN  

Models are trained and scored on the **official train/test split**, with **GroupKFold cross-validation by subject** on the training set so that subjects are not mixed between train and test folds.

Configuration lives in `config.yaml`. The unified experiment entry point is `src/run_experiments.py`.

---

## Initial study: three pipelines

The main, default study uses **only** these three pipelines:

| Pipeline | Description |
|----------|-------------|
| **baseline_561** | 561 precomputed features provided by the UCI HAR dataset. |
| **interpretable** | 225 hand-crafted time- and frequency-domain features extracted in Python from raw inertial windows (see cached metadata under `data/processed/` when present). |
| **tsfresh** | High-volume automated extraction with **tsfresh**; about **5724** columns after aligning train/test columns and dropping columns with a high fraction of NaNs. |

The default `pipelines:` list in `config.yaml` contains **only** these three names. Running `python src/run_experiments.py` without `--pipelines` evaluates that set.

---

## Optional extension: `tsfeatures_r`

`tsfeatures_r` is an **additive** experiment (an attempted improvement on a separate branch). It is **not** part of the default three-pipeline core.

It uses **R** and the CRAN package **`tsfeatures`**. For each of the **nine** inertial channels, the driver applies the **default** `tsfeatures()` call to a list where **each window is one univariate series of length 128**; no extra feature families are added in the R script. The first matrix-based call was corrected so that R receives **one time series per window**, not a layout that swapped the window axis with the time axis.

`tsfeatures_r` is **not** listed under default `pipelines:` in `config.yaml`; invoke it explicitly, for example:

`python src/run_experiments.py --pipelines tsfeatures_r`

**Code:** `src/feature_extraction_tsfeatures_r.py`, `src/r/tsfeatures_extract.R`  
**Feature cache:** `data/processed/X_train_tsfeatures_r.parquet`, `data/processed/X_test_tsfeatures_r.parquet`  
**Run outputs:** `results/tsfeatures_r/`

**Requirements:** `Rscript` on your PATH or passed with `--rscript` (or `tsfeatures_r.rscript_path` in `config.yaml`), and the CRAN package `tsfeatures` installed for that R interpreter.

---

## Global comparison files

When **more than one pipeline** is evaluated in a single `run_experiments.py` run, or when `generate_report.py` rebuilds a table from multiple pipeline summaries under `results/`:

- If the pipeline set is **exactly** `baseline_561`, `interpretable`, and `tsfresh`, aggregated tables are written as `results/comparison_all_pipelines.csv` and `results/comparison_all_pipelines.md`.
- In **any other** case—including runs that include `tsfeatures_r`, any future extension, or a strict subset of the core trio—aggregated tables use `results/comparison_pipelines_<sorted_pipeline_names>.csv` and `.md`.

This naming rule avoids overwriting the core three-pipeline comparison with mixed or extension runs. **Single-pipeline** runs do **not** produce a multi-pipeline comparison file.

---

## Repository layout

```
project/
  config.yaml
  requirements.txt
  README.md
  src/
    config.py
    dataset_loader.py
    inertial_loader.py
    models.py
    evaluation.py
    feature_extraction_interpretable.py
    feature_extraction_tsfresh.py
    feature_extraction_tsfeatures_r.py   # tsfeatures_r extension
    r/tsfeatures_extract.R               # R driver for tsfeatures_r
    run_experiments.py
    generate_report.py
    *_Baseline.py                         # legacy per-model scripts (pre-unified runner)
  data/processed/                        # feature caches (gitignored)
  results/                               # experiment outputs (gitignored)
    baseline_561/
    interpretable/
    tsfresh/
    tsfeatures_r/                        # present only if the extension is run
```

`results/`, `data/`, and `.venv/` are listed in `.gitignore`; generated artifacts stay local unless you change ignore rules.

---

## Dataset

Download the **UCI HAR Dataset** and set `dataset_path` in `config.yaml`. Each split should include `X_*.txt`, `y_*.txt`, `subject_*.txt`, and the `Inertial Signals/` tree.

Override the dataset root from the CLI if needed:

```bash
python src/run_experiments.py --dataset-path /path/to/UCI HAR Dataset
```

---

## Environment setup

Create a virtual environment, activate it, and install Python dependencies:

```bash
cd project/
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows, activate with `.venv\Scripts\activate`.

The **tsfresh** pipeline requires the **tsfresh** package (not installed by default in `requirements.txt`):

```bash
pip install tsfresh
```

For **tsfeatures_r**, install **R**, ensure **`Rscript`** matches the path you pass to extraction, and install the CRAN package **`tsfeatures`** for that R.

---

## Main commands

Feature extraction steps are **only** required when the corresponding Parquet files are missing under `data/processed/` (for example `X_train_interpretable.parquet` or `X_train_tsfresh.parquet`). If caches already exist, skip straight to `run_experiments.py`.

**Core three pipelines** — typical sequence:

```bash
python src/feature_extraction_interpretable.py
python src/feature_extraction_tsfresh.py
python src/run_experiments.py
python src/generate_report.py
```

**Subset runs** (examples):

```bash
python src/run_experiments.py --pipelines baseline_561
python src/run_experiments.py --pipelines interpretable --models random_forest knn
python src/run_experiments.py --normalize
```

**Extension `tsfeatures_r`** — extract once, then evaluate:

```bash
python src/feature_extraction_tsfeatures_r.py --rscript /usr/bin/Rscript
python src/run_experiments.py --pipelines tsfeatures_r
```

**Mixed run** (writes a slugged comparison file; see *Global comparison files*):

```bash
python src/run_experiments.py --pipelines baseline_561 interpretable tsfresh tsfeatures_r
```

---

## Outputs

**Per pipeline** — after `run_experiments.py`, each pipeline has a folder `results/<pipeline>/` containing test metrics, confusion matrices, and a short summary table, for example:

- `*_metrics.txt`
- `*_confusion_test.png`
- `summary_<pipeline>.csv` and `summary_<pipeline>.md`

**Cached features** — extracted representations are stored under `data/processed/` as Parquet files (and optional JSON sidecars for metadata). These caches are gitignored.

**Global comparisons** — multi-pipeline CSV and Markdown tables are written only when more than one pipeline is run together or regenerated via `generate_report.py`; filenames follow the rule in *Global comparison files* above (`comparison_all_pipelines` vs `comparison_pipelines_<sorted_names>`).
