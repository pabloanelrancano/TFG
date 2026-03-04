# TFG -- Reconocimiento de Actividades Humanas (HAR)

Comparacion de modelos supervisados para el reconocimiento de actividades cotidianas a partir de datos de acelerometros.

Trabajo Fin de Grado -- Pablo Anel Rancano -- Universidad de Granada

---

## Que hace este proyecto

Compara tres formas de representar datos de sensores inerciales (acelerometro y giroscopio) del dataset UCI HAR (30 sujetos, 6 actividades, 50 Hz) y las evalua con cuatro clasificadores clasicos.

- **Baseline (561 features):** las features originales del dataset UCI HAR.
- **Interpretable (~225 features):** features de tiempo y frecuencia extraidas a mano desde las senales crudas.
- **tsfresh (~5700 features):** extraccion automatica masiva con la libreria tsfresh.
- Cuatro modelos: Random Forest, Logistic Regression, Linear SVM y k-NN.
- Evaluacion sin data leakage: split oficial por sujeto + GroupKFold CV por sujeto.

---

## Estructura del proyecto

```
project/
  config.yaml                 # Configuracion (rutas, parametros)
  requirements.txt            # Dependencias Python
  src/
    config.py                 # Carga de configuracion
    dataset_loader.py         # Carga de datos tabulares UCI HAR (561 features)
    inertial_loader.py        # Carga de senales inerciales crudas (9 canales)
    models.py                 # Registro de modelos (RF, LR, SVM, KNN)
    evaluation.py             # Evaluacion unificada (test + CV)
    feature_extraction_interpretable.py
    feature_extraction_tsfresh.py
    run_experiments.py        # Ejecutor principal de experimentos
    generate_report.py        # Genera informe comparativo global
    fuzzy_placeholder.py      # Stubs para fase futura (metodos difusos)
    *_Baseline.py             # Scripts legacy (uno por modelo)
  data/processed/             # Cache de features extraidas (gitignored)
  results/                    # Resultados de experimentos (gitignored)
    baseline_561/
    interpretable/
    tsfresh/
    comparison_all_pipelines.csv
    comparison_all_pipelines.md
```

---

## Dataset

El proyecto necesita el **UCI HAR Dataset**. La ruta se configura en `config.yaml`:

```yaml
dataset_path: "../DataSets/human+activity+recognition+using+smartphones/UCI HAR Dataset/UCI HAR Dataset"
```

La carpeta debe contener `train/` y `test/`, cada una con `X_*.txt`, `y_*.txt`, `subject_*.txt` e `Inertial Signals/`.

Se puede cambiar la ruta por CLI: `python src/run_experiments.py --dataset-path /otra/ruta`

---

## Setup

```bash
cd project/
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

tsfresh es una dependencia opcional (solo necesaria para el pipeline tsfresh):

```bash
pip install tsfresh
```

---

## Como ejecutar

```bash
# 1. Extraer features interpretables (una vez, ~2-5 min)
python src/feature_extraction_interpretable.py

# 2. Extraer features tsfresh (una vez, ~10-30 min, requiere tsfresh)
python src/feature_extraction_tsfresh.py

# 3. Ejecutar todos los modelos en todos los pipelines
python src/run_experiments.py

# 4. Generar informe comparativo
python src/generate_report.py
```

Para ejecutar solo un pipeline o modelo concreto:

```bash
python src/run_experiments.py --pipelines baseline_561
python src/run_experiments.py --pipelines interpretable --models random_forest knn
python src/run_experiments.py --normalize
```

---

## Outputs

Cada experimento genera en `results/<pipeline>/`:

- `*_metrics.txt` -- accuracy, F1, classification report, resultados de CV
- `*_confusion_test.png` -- matriz de confusion
- `summary_*.csv` / `summary_*.md` -- tabla resumen del pipeline

El informe global esta en:

- `results/comparison_all_pipelines.csv`
- `results/comparison_all_pipelines.md`

Los datos procesados (features extraidas) se guardan en `data/processed/` como archivos parquet. Tanto `results/` como `data/processed/` estan en `.gitignore` y se generan localmente.
