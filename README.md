# TFG — Reconocimiento de actividades humanas (HAR)

Comparación supervisada de representaciones de ventanas inerciales (UCI HAR) con clasificadores clásicos, evaluación por sujeto y documentación reproducible del código.

**Trabajo Fin de Grado** — Pablo Anel Rancano — Universidad de Granada

---

## Visión general del proyecto

El repositorio contiene el código para cargar el **UCI HAR Dataset**, construir matrices de características por ventana (50 Hz, ventanas de 128 muestras), y evaluar **cuatro modelos** (Random Forest, regresión logística, SVM lineal, k-NN) con el **split oficial** test y **validación cruzada GroupKFold por sujeto** en entrenamiento, sin mezclar sujetos entre train y test.

La configuración vive en `config.yaml`; el punto de entrada unificado de experimentos es `src/run_experiments.py`.

---

## Estudio núcleo (congelado): tres pipelines

El **estudio principal del repositorio** (núcleo experimental del TFG) se basa **únicamente** en estos tres pipelines:

| Pipeline | Descripción breve |
|----------|-------------------|
| **baseline_561** | 561 características precomputadas del propio UCI HAR. |
| **interpretable** | Características de dominio (~225) extraídas en Python desde las señales crudas. |
| **tsfresh** | Extracción automática de alto volumen (**~5724** columnas finales tras alinear train/test y depurar columnas con muchos NaN) con la librería **tsfresh**. |

En `config.yaml`, la lista `pipelines:` por defecto incluye **solo** estos tres nombres. Ejecutar `python src/run_experiments.py` sin `--pipelines` recorre ese núcleo.

---

## Extensión opcional: `tsfeatures_r`

`tsfeatures_r` es un **experimento aditivo** (rama de trabajo), no parte del trío congelado:

- Usa **R** con el paquete CRAN **`tsfeatures`**: llamada por defecto a `tsfeatures()` **por canal** (9 canales inerciales), una serie univariante de longitud 128 por ventana, sin familias extra de características en el driver R.
- **No** está en la lista por defecto `pipelines:` de `config.yaml`; hay que invocarlo **explícitamente**:  
  `python src/run_experiments.py --pipelines tsfeatures_r`
- Código: `src/feature_extraction_tsfeatures_r.py` y `src/r/tsfeatures_extract.R`. Caché: `data/processed/X_{train,test}_tsfeatures_r.parquet`. Resultados: `results/tsfeatures_r/`.
- Requisitos: **`Rscript`** accesible (PATH o ruta con `--rscript` / `tsfeatures_r.rscript_path` en `config.yaml`) y el paquete CRAN **`tsfeatures`** instalado para ese intérprete R.

---

## Política de ficheros de comparación global

Al agrupar **varios** pipelines en una misma ejecución de `run_experiments.py` o al regenerar tablas con `generate_report.py`:

- Si el conjunto de pipelines es **exactamente** `{baseline_561, interpretable, tsfresh}`, los agregados se escriben como  
  **`results/comparison_all_pipelines.csv`** y **`results/comparison_all_pipelines.md`**.
- Si interviene **`tsfeatures_r`** u otro pipeline de extensión, o el conjunto no coincide con ese trío, los agregados van a  
  **`results/comparison_pipelines_<nombres_ordenados>.csv`** y **`.md`** (mismo criterio de nombres ordenados), para no sobrescribir el comparativo del núcleo con mezclas accidentales.

Con un solo pipeline no se genera comparativa global multi-pipeline.

---

## Estructura del repositorio

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
    feature_extraction_tsfeatures_r.py   # extensión R tsfeatures_r
    r/tsfeatures_extract.R                # driver R (tsfeatures por ventana/canal)
    run_experiments.py
    generate_report.py
    *_Baseline.py                         # evaluación legacy, un script por modelo (previos al runner unificado)
  data/processed/                        # cachés parquet (gitignored)
  results/                               # salidas por pipeline (gitignored)
    baseline_561/
    interpretable/
    tsfresh/
    tsfeatures_r/                        # solo si se ejecuta la extensión
```

`results/`, `data/` y `.venv/` están en `.gitignore`; las salidas son locales.

---

## Dataset

Descargar el **UCI HAR Dataset** y fijar la ruta en `config.yaml` (`dataset_path`). Cada split debe incluir `X_*.txt`, `y_*.txt`, `subject_*.txt` e `Inertial Signals/`.

Sobrescritura por CLI:  
`python src/run_experiments.py --dataset-path /ruta/al/UCI HAR Dataset`

---

## Configuración (entorno Python)

```bash
cd project/
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**tsfresh** (solo para el pipeline `tsfresh`):  
`pip install tsfresh`

**Extensión `tsfeatures_r`:** además, **R** con **`tsfeatures`** instalado para el mismo **`Rscript`** que usará la extracción.

---

## Comandos principales

**Núcleo (tres pipelines)** — los dos primeros comandos solo hacen falta **si no existen ya** los parquet en `data/processed/` (p. ej. `X_train_interpretable.parquet`, `X_train_tsfresh.parquet`); si las cachés están presentes, puede omitirse la extracción y ejecutarse directamente `run_experiments.py`.

```bash
python src/feature_extraction_interpretable.py
python src/feature_extraction_tsfresh.py
python src/run_experiments.py
python src/generate_report.py
```

**Solo un pipeline o modelo** (ejemplos):

```bash
python src/run_experiments.py --pipelines baseline_561
python src/run_experiments.py --pipelines interpretable --models random_forest knn
python src/run_experiments.py --normalize
```

**Extensión `tsfeatures_r`** (extracción una vez; entrenamiento cuando proceda):

```bash
python src/feature_extraction_tsfeatures_r.py --rscript /usr/bin/Rscript
python src/run_experiments.py --pipelines tsfeatures_r
```

**Varios pipelines incluyendo extensión** (comparativa con nombre slug, ver sección anterior):

```bash
python src/run_experiments.py --pipelines baseline_561 interpretable tsfresh tsfeatures_r
```

---

## Salidas

Por pipeline, en `results/<nombre_pipeline>/`:

- `*_metrics.txt` — métricas en test, informe por clase, CV por sujeto
- `*_confusion_test.png` — matriz de confusión
- `summary_<pipeline>.csv` / `.md` — tabla resumen del pipeline

Comparativas globales: ver **Política de ficheros de comparación global** arriba. Las características en caché viven en `data/processed/` (parquet + metadatos JSON cuando aplica).
