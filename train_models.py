import os
import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, classification_report, mean_absolute_error, mean_squared_error, r2_score
from sklearn.svm import SVR

from data_utils import DEFAULT_DATASET, get_stress_level, prepare_dataset

warnings.filterwarnings("ignore")

# ─── File Paths ───────────────────────────────────────────────────────────────
OUTPUT_DIR = "outputs"
RESULTS_CSV = os.path.join(OUTPUT_DIR, "model_results.csv")
PREDICTIONS_CSV = os.path.join(OUTPUT_DIR, "predictions.csv")
FEATURE_IMPORTANCE_CSV = os.path.join(OUTPUT_DIR, "feature_importance.csv")


def ensure_output_dir() -> None:
    """Create outputs/ folder if it doesn't exist."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("Output directory ready: '{}'".format(OUTPUT_DIR))


# ─── Console Helpers ──────────────────────────────────────────────────────────
def section(title: str) -> None:
    print("\n{}".format(title))
    print("-" * len(title))


def print_results_table(results_df: pd.DataFrame) -> None:
    print("Model                     R2 Score    MAE      RMSE    CatAcc")
    print("--------------------------------------------------------------")
    for _, row in results_df.iterrows():
        print("{:<24} {:<10.4f} {:<8.4f} {:<8.4f} {:.2f}%".format(
            row["Model"], row["R2"], row["MAE"], row["RMSE"], row["CatAcc"]
        ))


def print_feature_importance(model: RandomForestRegressor, feature_names: list[str]) -> pd.DataFrame:
    importance_df = pd.DataFrame(
        {"Feature": feature_names, "Importance": model.feature_importances_}
    ).sort_values("Importance", ascending=False)

    print("Feature                                  Importance")
    print("---------------------------------------------------")
    for _, row in importance_df.iterrows():
        pct = row["Importance"] * 100
        print("{:<40} {:>9.2f}%".format(row["Feature"], pct))

    return importance_df


# ─── Save Functions ───────────────────────────────────────────────────────────
def save_results_csv(results_df: pd.DataFrame) -> None:
    """Save model comparison table to CSV."""
    results_df.to_csv(RESULTS_CSV, index=False)
    print("Results saved -> {}".format(RESULTS_CSV))


def save_predictions_csv(y_test: pd.Series, preds: dict[str, np.ndarray]) -> None:
    """Save actual vs predicted values for every model to CSV."""
    pred_df = pd.DataFrame({"Actual": y_test.values})
    pred_df["Actual_Level"] = pred_df["Actual"].apply(get_stress_level)

    for model_name, y_pred in preds.items():
        col_score = "{}_Predicted".format(model_name)
        col_level = "{}_Level".format(model_name)
        pred_df[col_score] = y_pred
        pred_df[col_level] = pd.Series(y_pred).apply(get_stress_level).values

    pred_df.reset_index(drop=True, inplace=True)
    pred_df.to_csv(PREDICTIONS_CSV, index=False)
    print("Predictions saved -> {}".format(PREDICTIONS_CSV))


def save_feature_importance_csv(importance_df: pd.DataFrame) -> None:
    """Save Random Forest feature importances to CSV."""
    importance_df["Importance_Pct"] = (importance_df["Importance"] * 100).round(2)
    importance_df.to_csv(FEATURE_IMPORTANCE_CSV, index=False)
    print("Feature importance saved -> {}".format(FEATURE_IMPORTANCE_CSV))


# ─── Core ML Functions ────────────────────────────────────────────────────────
def evaluate_models(X_train, X_test, y_train, y_test, models: dict[str, object]) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, object]]:
    rows: list[dict[str, float | str]] = []
    preds: dict[str, np.ndarray] = {}
    trained: dict[str, object] = {}

    y_test_level = y_test.apply(get_stress_level)

    for name, model in models.items():
        print("Training: {}".format(name))
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        trained[name] = model
        preds[name] = y_pred

        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        y_pred_level = pd.Series(y_pred).apply(get_stress_level)
        cat_acc = accuracy_score(y_test_level, y_pred_level) * 100

        rows.append(
            {
                "Model": name,
                "R2": r2,
                "MAE": mae,
                "RMSE": rmse,
                "CatAcc": cat_acc,
            }
        )

        print("  Metrics -> R2: {:.4f}, MAE: {:.4f}, RMSE: {:.4f}, CatAcc: {:.2f}%".format(
            r2, mae, rmse, cat_acc
        ))

    results_df = pd.DataFrame(rows).sort_values("R2", ascending=False).reset_index(drop=True)
    return results_df, preds, trained


def run_reference_prediction(prepared: dict, rf_model: RandomForestRegressor) -> tuple[float, str]:
    sample_raw = pd.DataFrame([prepared["X"].median()]).reset_index(drop=True)

    if prepared.get("scaler") is not None:
        sample = pd.DataFrame(
            prepared["scaler"].transform(sample_raw),
            columns=sample_raw.columns,
        )
    else:
        sample = sample_raw

    pred_score = float(rf_model.predict(sample)[0])
    stress_level = get_stress_level(pred_score)
    return pred_score, stress_level


# ─── Main Pipeline ────────────────────────────────────────────────────────────
def main() -> None:
    print("Stress Score Modeling Pipeline (Console Edition)")

    # Setup output folder first
    section("File Handling: Output Directory Setup")
    ensure_output_dir()

    section("Step 1: Data Loading and Preparation")
    # Dataset is loaded from file via data_utils (DEFAULT_DATASET path defined there)
    prepared = prepare_dataset(DEFAULT_DATASET, normalize=True)
    df = prepared["raw_df"]
    target_col = prepared["target_col"]

    print("Dataset loaded from: {}".format(DEFAULT_DATASET))
    print("Rows: {}, Columns: {}".format(df.shape[0], df.shape[1]))
    print("Task: Regression -> target = {}".format(target_col))

    X = prepared["X"]
    y = prepared["y"]
    X_train = prepared["X_train"]
    X_test = prepared["X_test"]
    y_train = prepared["y_train"]
    y_test = prepared["y_test"]

    print("Feature matrix shape: {}".format(X.shape))
    print("Target vector shape : {}".format(y.shape))
    print("Train size: {} | Test size: {}".format(len(X_train), len(X_test)))
    print("Target range: {} to {} (mean={:.2f})".format(y.min(), y.max(), y.mean()))

    section("Step 2: Model Training and Evaluation")
    models: dict[str, object] = {
        "Linear Regression": LinearRegression(),
        "SVR (RBF)": SVR(kernel="rbf", C=1.0),
        "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
    }
    results_df, preds, trained = evaluate_models(X_train, X_test, y_train, y_test, models)

    section("Step 3: Model Comparison")
    print_results_table(results_df)

    section("Step 4: Best Model Analysis")
    best_name = str(results_df.iloc[0]["Model"])
    best_pred = preds[best_name]
    print("Best model: {}".format(best_name))
    print("R2   : {:.4f}".format(r2_score(y_test, best_pred)))
    print("MAE  : {:.4f}".format(mean_absolute_error(y_test, best_pred)))
    print("RMSE : {:.4f}".format(np.sqrt(mean_squared_error(y_test, best_pred))))

    y_test_level = y_test.apply(get_stress_level)
    y_pred_level = pd.Series(best_pred).apply(get_stress_level)
    cat_acc_best = accuracy_score(y_test_level, y_pred_level) * 100
    print("Category Accuracy (Low/Medium/High): {:.2f}%".format(cat_acc_best))

    print("\nClassification Report:")
    print(classification_report(y_test_level, y_pred_level, target_names=["Low", "Medium", "High"]))

    section("Step 5: Random Forest Feature Importance")
    rf_model = trained["Random Forest"]
    importance_df = print_feature_importance(rf_model, list(X.columns))

    section("Step 6: Reference Profile Prediction")
    pred_score, stress_level = run_reference_prediction(prepared, rf_model)
    print("Predicted Stress Score: {:.2f}".format(pred_score))
    print("Predicted Stress Level: {}".format(stress_level))

    # ── File Handling: Save All Outputs ──────────────────────────────────────
    section("Step 7: Saving Results to Files")
    save_results_csv(results_df)                    # outputs/model_results.csv
    save_predictions_csv(y_test, preds)             # outputs/predictions.csv
    save_feature_importance_csv(importance_df)      # outputs/feature_importance.csv

    section("Final Summary")
    print("Task: Regression (Stress_Score prediction)")
    print("Leakage handled: Stress_Level removed from training features")
    print("Features used: {}".format(len(X.columns)))
    print("Ranking by R2:")
    for i, (_, row) in enumerate(results_df.iterrows(), 1):
        print("  {}. {:<20} R2={:.4f} MAE={:.4f} RMSE={:.4f} CatAcc={:.2f}%".format(
            i, row["Model"], row["R2"], row["MAE"], row["RMSE"], row["CatAcc"]
        ))

    print("Top feature (Random Forest): {}".format(importance_df.iloc[0]["Feature"]))
    print("\nOutput files saved:")
    print("  - {}".format(RESULTS_CSV))
    print("  - {}".format(PREDICTIONS_CSV))
    print("  - {}".format(FEATURE_IMPORTANCE_CSV))
    print("Run status: Success")


if __name__ == "__main__":
    main()