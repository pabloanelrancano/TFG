# src/dataset_loader.py - Pablo Anel Rancaño

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GroupKFold, cross_val_score


# Dataset
DATASET_PATH = Path(
    "../DataSets/human+activity+recognition+using+smartphones/UCI HAR Dataset/UCI HAR Dataset"
)


# Etiquetas del dataset - UCI HAR
ACTIVITY_NAMES: Dict[int, str] = {
    1: "WALKING",
    2: "WALKING_UPSTAIRS",
    3: "WALKING_DOWNSTAIRS",
    4: "SITTING",
    5: "STANDING",
    6: "LAYING",
}
CLASS_IDS: List[int] = [1, 2, 3, 4, 5, 6]
CLASS_LABELS: List[str] = [ACTIVITY_NAMES[i] for i in CLASS_IDS]


def assert_dataset_exists(dataset_path: Path = DATASET_PATH) -> None:
    if not dataset_path.exists():
        raise FileNotFoundError(f"No existe el dataset:\n{dataset_path.resolve()}")


def load_split(split_name: str, dataset_path: Path = DATASET_PATH) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    X = np.loadtxt(dataset_path / split_name / f"X_{split_name}.txt")
    y = np.loadtxt(dataset_path / split_name / f"y_{split_name}.txt").astype(int)
    subjects = np.loadtxt(dataset_path / split_name / f"subject_{split_name}.txt").astype(int)

    return X, y, subjects


# Guarda una matriz de confusión como PNG
def save_confusion_matrix(cm: np.ndarray, output_file: Path, title: str) -> None:
    
    plt.figure(figsize=(8, 6))
    plt.imshow(cm)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.colorbar()

    plt.xticks(range(len(CLASS_LABELS)), CLASS_LABELS, rotation=45, ha="right")
    plt.yticks(range(len(CLASS_LABELS)), CLASS_LABELS)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")

    plt.tight_layout()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=200)
    plt.close()


# Guarda un texto en un fichero
def write_text(out_path: Path, text: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")


# A) Entrena en train y evalúa en test oficial
# B) Validación cruzada por sujeto en train (GroupKFold)
def evaluate_model_baseline(
    model,
    model_tag: str,
    results_prefix: str,
    title: str,
    dataset_path: Path = DATASET_PATH,
    n_splits: int = 5,
) -> None:

    assert_dataset_exists(dataset_path)

    X_train, y_train, subjects_train = load_split("train", dataset_path)
    X_test, y_test, subjects_test = load_split("test", dataset_path)

    # A) Test
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc_test = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, digits=4)

    cm = confusion_matrix(y_test, y_pred, labels=CLASS_IDS)
    cm_path = Path(f"results/{results_prefix}_confusion_test.png")
    save_confusion_matrix(cm, cm_path, title)

    # B) CV por sujeto
    gkf = GroupKFold(n_splits=n_splits)
    scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=gkf,
        groups=subjects_train,
        scoring="accuracy",
        n_jobs=-1,
    )

    text = []
    text.append(f"MODEL: {model_tag}")
    text.append(f"Test accuracy: {acc_test:.6f}")
    text.append("")
    text.append("Classification report (test):")
    text.append(report)
    text.append("")
    text.append(f"CV by subject (train, GroupKFold={n_splits}):")
    text.append(f"Fold accuracies: {scores}")
    text.append(f"Mean: {scores.mean():.6f}")
    text.append(f"Std:  {scores.std():.6f}")
    text.append("")
    text.append(f"Confusion matrix png: {cm_path.as_posix()}")

    metrics_path = Path(f"results/{results_prefix}_metrics.txt")
    write_text(metrics_path, "\n".join(text))

    print(f"{model_tag} DONE")
    print("Saved:", metrics_path.as_posix(), "and", cm_path.as_posix())
