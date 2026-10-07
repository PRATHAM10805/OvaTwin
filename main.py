import joblib

from src.ML.model_comparison import compare_models

from src.ML.data_loader import (
    load_data,
    split_features_target
)

from src.ML.preprocessing import prepare_data

from src.ML.feature_selection import select_features

from src.ML.evaluate import evaluate_model

from src.ML.explain import explain_model

from src.ML.config import MODEL_DIR, MODEL_PATH


def main():

    print("\n")
    print("=" * 60)
    print("             OVATWIN - PHASE 1")
    print("       PCOS Prediction Pipeline")
    print("=" * 60)

    # --------------------------------------------------
    # 1. Load Dataset
    # --------------------------------------------------

    df = load_data()

    # --------------------------------------------------
    # 2. Separate Features and Target
    # --------------------------------------------------

    X, y = split_features_target(df)

    print("\nInput features:", X.shape)
    print("Target:", y.shape)

    # --------------------------------------------------
    # 3. Preprocessing + Train/Test Split + SMOTE
    # --------------------------------------------------

    (
        X_train,
        X_test,
        y_train,
        y_test
    ) = prepare_data(X, y)

    # --------------------------------------------------
    # 4. Feature Selection
    # --------------------------------------------------

    (
        X_train_selected,
        X_test_selected,
        selected_features,
        selector
    ) = select_features(
        X_train,
        y_train,
        X_test,
        k=15
    )

    # --------------------------------------------------
    # 5. MODEL COMPARISON
    # --------------------------------------------------

    xgb_model, rf_model, comparison_results = compare_models(
        X_train_selected,
        y_train,
        X_test_selected,
        y_test
    )

    # --------------------------------------------------
    # 6. Evaluate XGBoost
    # --------------------------------------------------

    print("\n")
    print("=" * 60)
    print("             XGBOOST DETAILED RESULTS")
    print("=" * 60)

    y_pred, y_prob, metrics = evaluate_model(
        xgb_model,
        X_test_selected,
        y_test
    )

    # --------------------------------------------------
    # 7. SHAP Explanation for XGBoost
    # --------------------------------------------------

    print("\nGenerating SHAP explanation...")

    explain_model(
        xgb_model,
        X_test_selected
    )

    print("SHAP explanation generated.")

    # --------------------------------------------------
    # 8. Save XGBoost Model
    # --------------------------------------------------

    MODEL_DIR.mkdir(exist_ok=True)

    joblib.dump(
        {
            "model": xgb_model,
            "selector": selector,
            "selected_features": selected_features
        },
        MODEL_PATH
    )

    print("\nXGBoost model saved to:")
    print(MODEL_PATH)

    # --------------------------------------------------
    # 9. Final Summary
    # --------------------------------------------------

    print("\n")
    print("=" * 60)
    print("             MODEL COMPARISON SUMMARY")
    print("=" * 60)

    for model_name, result in comparison_results.items():

        print(f"\n{model_name}")
        print("-" * 30)

        print(
            f"Accuracy     : {result['accuracy']:.4f}"
        )

        print(
            f"Precision    : {result['precision']:.4f}"
        )

        print(
            f"Recall       : {result['recall']:.4f}"
        )

        print(
            f"F1 Score     : {result['f1_score']:.4f}"
        )

        print(
            f"ROC-AUC      : {result['roc_auc']:.4f}"
        )

        print(
            f"False Pos.   : {result['false_positives']}"
        )

        print(
            f"False Neg.   : {result['false_negatives']}"
        )

    # --------------------------------------------------
    # 10. Completion
    # --------------------------------------------------

    print("\n")
    print("=" * 60)
    print("             PHASE 1 COMPLETE")
    print("=" * 60)

    print("\nGenerated files:")

    print("[OK] XGBoost model")
    print("[OK] Model comparison results")
    print("[OK] Evaluation metrics")
    print("[OK] SHAP explanation")

    print("\nCheck:")
    print("results/model_comparison.json")
    print("results/metrics.json")
    print("results/shap_summary.png")

    print("\n")


if __name__ == "__main__":
    main()