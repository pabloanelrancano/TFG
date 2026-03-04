# KNN Baseline - Pablo Anel Rancano

# src/KNN_Baseline.py

from sklearn.neighbors import KNeighborsClassifier

from dataset_loader import evaluate_model_baseline


def main():
    model = KNeighborsClassifier(n_neighbors=5)

    evaluate_model_baseline(
        model=model,
        model_tag="k-NN (k=5)",
        results_prefix="04_knn",
        title="k-NN (k=5) - Test oficial",
    )


if __name__ == "__main__":
    main()
