# Linear SVM Baseline - Pablo Anel Rancaño

# src/Linear_SVM_Baseline.py

from sklearn.svm import LinearSVC

from dataset_loader import evaluate_model_baseline


def main():
    model = LinearSVC(random_state=42)

    evaluate_model_baseline(
        model=model,
        model_tag="Linear SVM (LinearSVC)",
        results_prefix="03_lsvm",
        title="Linear SVM (LinearSVC) - Test oficial",
    )


if __name__ == "__main__":
    main()
