# Logistic Regression Baseline - Pablo Anel Rancano

from sklearn.linear_model import LogisticRegression

from dataset_loader import evaluate_model_baseline


def main():
    model = LogisticRegression(max_iter=3000, solver="lbfgs", n_jobs=-1)

    evaluate_model_baseline(
        model=model,
        model_tag="Logistic Regression (linear)",
        results_prefix="02_lr",
        title="Logistic Regression - Test oficial",
    )


if __name__ == "__main__":
    main()
