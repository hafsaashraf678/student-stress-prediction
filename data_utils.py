import os
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

DEFAULT_DATASET = os.path.join(os.path.dirname(__file__), "university_student_stress_dataset.csv")

# Target column —predict (continuous number = Regression)
TARGET_COLUMN = "Stress_Score"
# Yeh features target se directly linked hain — data leakage rokne ke liye hataya hain
#  Stress_Level ab yahan nahi hata — hum ise predicted Stress_Score se if/elif se derive karenge
LEAKAGE_FEATURES = ["Anxiety_Level"]

# Yeh features informative nahi  — remove 
# Age: sirf 6 unique values (19-24), sab students same range mein
DROP_FEATURES = ["Age"]

def get_stress_level(score: float) -> str:
    """
    Predicted Stress_Score se Stress_Level derive karo (if/elif conditions).

    Thresholds:
        score <= 10  ->  'Low'
        10 < score <= 20  ->  'Medium'
        score >  20  ->  'High'
    """
    if score <= 10:
        return "Low"
    elif score <= 20:
        return "Medium"
    else:
        return "High"


def load_raw_dataset(dataset_path: str = DEFAULT_DATASET) -> pd.DataFrame:
    """Load the dataset from disk."""
    return pd.read_csv(dataset_path)


def encode_categorical_columns(
    data: pd.DataFrame, target_column: str
) -> tuple[pd.DataFrame, dict[str, LabelEncoder]]:
    """
    Encode all text/categorical columns.
    
    Encoding strategy:
    - Gender, Tuition, Physical_Exercise  -> LabelEncoder (2 values)
    - Family_Income_Level                 -> Ordinal (Low=0, Medium=1, High=2)
    - University_Type                     -> One-Hot Encoding (3 categories)
    """
    encoded = data.copy()
    encoders: dict[str, LabelEncoder] = {}

    # --- Ordinal encoding for Family_Income_Level ---
    if "Family_Income_Level" in encoded.columns:
        income_map = {"Low": 0, "Medium": 1, "High": 2}
        encoded["Family_Income_Level"] = encoded["Family_Income_Level"].map(income_map)
        print("[data_utils] Family_Income_Level -> Ordinal encoded (Low=0, Medium=1, High=2)")

    # --- One-Hot encoding for University_Type ---
    if "University_Type" in encoded.columns:
        dummies = pd.get_dummies(encoded["University_Type"], prefix="UniType")
        encoded = pd.concat([encoded.drop(columns=["University_Type"]), dummies], axis=1)
        print(f"[data_utils] University_Type -> One-Hot encoded: {list(dummies.columns)}")

    # --- Label encoding for remaining categorical columns ---
    obj_cols = encoded.select_dtypes(include=["object", "category", "string"]).columns
    # Skip target column if it somehow ended up as object
    obj_cols = [c for c in obj_cols if c != target_column]
    for col in obj_cols:
        le = LabelEncoder()
        encoded[col] = le.fit_transform(encoded[col].astype(str))
        encoders[col] = le
        print(f"[data_utils] {col} -> Label encoded")

    return encoded, encoders


def prepare_dataset(
    dataset_path: str = DEFAULT_DATASET,
    target_column: str | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
    normalize: bool = True,
) -> dict[str, Any]:
    """
    Load, encode, split, and optionally normalize the dataset for REGRESSION.
    
    Target  : Stress_Score  (continuous number — regression task)
    NOTE    : Stress_Level dataset mein hai lekin hum ise drop karte hain training se
              aur baad mein predicted score se if/elif se derive karte hain:
                  score <= 10  ->  Low Stress
                  score <= 20  ->  Medium Stress
                  score >  20  ->  High Stress
    Removed : Age           (sirf 6 unique values — model ke liye useless)
    """
    raw_df = load_raw_dataset(dataset_path)

    # Use provided target or default
    target_col = target_column if target_column and target_column in raw_df.columns else TARGET_COLUMN

    if target_col not in raw_df.columns:
        raise ValueError(
            f"Target column '{target_col}' not found. Available: {list(raw_df.columns)}"
        )

    print(f"[data_utils] Target column: {target_col} (REGRESSION)")
    print(f"[data_utils] Total records: {len(raw_df)}")

    data = raw_df.dropna().copy()
    data, encoders = encode_categorical_columns(data, target_col)

    # --- Remove leakage + useless features ---
    # Stress_Level explicitly drop karo — yeh Stress_Score se bani category hai (leakage)
    STRESS_LEVEL_COL = "Stress_Level"
    all_to_drop = [target_col] + \
                  [f for f in LEAKAGE_FEATURES if f in data.columns] + \
                  [f for f in DROP_FEATURES if f in data.columns] + \
                  ([STRESS_LEVEL_COL] if STRESS_LEVEL_COL in data.columns else [])

    removed_leakage = [f for f in LEAKAGE_FEATURES if f in data.columns]
    if STRESS_LEVEL_COL in data.columns:
        removed_leakage.append(STRESS_LEVEL_COL)
    removed_drop    = [f for f in DROP_FEATURES if f in data.columns]

    X = data.drop(columns=all_to_drop)
    y = data[target_col]  # Continuous values — regression target

    print(f"\n[data_utils]  Leakage features removed : {removed_leakage}")
    print(f"[data_utils]  Low-info features removed : {removed_drop}")
    print(f"[data_utils]  Features used for training: {list(X.columns)}")
    print(f"[data_utils]  Target (y) range           : {y.min()} to {y.max()}")
    print()

    # Train/Test split — no stratify for regression
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
    )

    scaler = None
    X_train_used = X_train.copy()
    X_test_used  = X_test.copy()

    if normalize:
        scaler = StandardScaler()
        X_train_used = pd.DataFrame(
            scaler.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index,
        )
        X_test_used = pd.DataFrame(
            scaler.transform(X_test),
            columns=X_test.columns,
            index=X_test.index,
        )
        print("[data_utils]  StandardScaler applied (fit on train only)")

    return {
        "raw_df"      : raw_df,
        "encoded_df"  : data,
        "target_col"  : target_col,
        "encoders"    : encoders,
        "X"           : X,
        "y"           : y,
        "X_train"     : X_train_used,
        "X_test"      : X_test_used,
        "y_train"     : y_train,
        "y_test"      : y_test,
        "scaler"      : scaler,
        "dataset_path": dataset_path,
        "normalized"  : normalize,
    }
