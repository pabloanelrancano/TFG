# TFG - Human Activity Recognition (UCI HAR)

Repositorio del TFG para el reconocimiento de actividades humanas (HAR) usando el dataset **UCI HAR**.  
Incluye baselines con **scikit-learn** y utilidades comunes de carga/evaluación (métricas, matrices de confusión y validación cruzada por sujeto).

## Estructura del proyecto

- `src/` : código fuente (baselines + utilidades)
- `results/` : salidas generadas (`.txt` y `.png`)
- `requirements.txt` : dependencias del proyecto
- `.venv/` : entorno virtual

> Nota: los datasets se mantienen fuera del repo o ignorados por `.gitignore` para no subir datos pesados.

## Requisitos

- Python 3.10+

## Instalación

Desde la carpeta del proyecto:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

```