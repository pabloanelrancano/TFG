# Random Forest Baseline - Pablo Anel Rancaño

# src/Random_Forest_Baseline.py

from sklearn.ensemble import RandomForestClassifier

from dataset_loader import evaluate_model_baseline


def main():
    model = RandomForestClassifier(random_state=42, n_jobs=-1)

    evaluate_model_baseline(
        model=model,
        model_tag="Random Forest",
        results_prefix="01_rf",
        title="Random Forest - Test oficial",
    )


if __name__ == "__main__":
    main()
