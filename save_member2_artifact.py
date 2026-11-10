"""
Member 2 -- Save the classification model artifacts.

Re-trains the FINAL classification model from Classification.ipynb
(Logistic Regression, C=0.5, SMOTE inside an imblearn pipeline) using the
EXACT same steps, then saves it with joblib using THIS environment's
scikit-learn version, so the API can load it without version mismatch.

Run from the repo root:
    python save_member2_artifact.py

Creates (in model/):
    distress_risk_pipeline.joblib    -> preprocessor + SMOTE + LogisticRegression
    distress_label_encoder.joblib    -> maps 0/1/2 back to Distressed/Healthy/Watch
    distress_model_metadata.joblib   -> expected input fields + allowed categories
"""
import os
import urllib.request

import imblearn
import joblib
import numpy as np
import pandas as pd
import sklearn
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder, StandardScaler

RANDOM_STATE = 42
TARGET = "distress_risk_category"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "model")
DATASET_PATH = os.path.join(DATA_DIR, "corporate_financial_distress.csv")
DATASET_URL = ("https://raw.githubusercontent.com/course-files/RegressionAndClassification/"
               "refs/heads/main/data/corporate_financial_distress.csv")

# ---------------------------------------------------------------------------
# 1. Load data (same source as the notebook)
# ---------------------------------------------------------------------------
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
if not os.path.exists(DATASET_PATH):
    print("Downloading dataset...")
    urllib.request.urlretrieve(DATASET_URL, DATASET_PATH)

data = pd.read_csv(DATASET_PATH)
X = data.drop(columns=[TARGET])
y = data[TARGET]

# ---------------------------------------------------------------------------
# 2. Same stratified 80/20 split as the notebook
# ---------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

# ---------------------------------------------------------------------------
# 3. Same upstream missing-value handling as the notebook
# ---------------------------------------------------------------------------
# audit_going_concern_flag: structurally missing for Private companies
X_train["audit_going_concern_flag"] = X_train["audit_going_concern_flag"].fillna("Not Flagged")
X_test["audit_going_concern_flag"] = X_test["audit_going_concern_flag"].fillna("Not Flagged")

# net_profit_margin_pct: missing at random -> train median (no leakage)
net_margin_median = X_train["net_profit_margin_pct"].median()
X_train["net_profit_margin_pct"] = X_train["net_profit_margin_pct"].fillna(net_margin_median)
X_test["net_profit_margin_pct"] = X_test["net_profit_margin_pct"].fillna(net_margin_median)

# ---------------------------------------------------------------------------
# 4. Encode the target (model predicts 0/1/2)
# ---------------------------------------------------------------------------
target_encoder = LabelEncoder()
y_train_encoded = target_encoder.fit_transform(y_train)
y_test_encoded = target_encoder.transform(y_test)

# ---------------------------------------------------------------------------
# 5. Same preprocessor as the notebook
# ---------------------------------------------------------------------------
nominal_cols = ["industry_sector", "listing_status", "audit_going_concern_flag"]
ordinal_col = ["credit_rating_band"]
credit_rating_rank = [["Investment Grade", "Speculative", "High Risk"]]
numeric_cols = [c for c in X.select_dtypes(include=[np.number]).columns if c != "company_id"]

preprocessor = ColumnTransformer(transformers=[
    ("numeric", Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]), numeric_cols),
    ("nominal", Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ]), nominal_cols),
    ("ordinal", Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ordinal", OrdinalEncoder(categories=credit_rating_rank,
                                   handle_unknown="use_encoded_value", unknown_value=-1)),
    ]), ordinal_col),
])

# ---------------------------------------------------------------------------
# 6. Final selected model: untuned Logistic Regression (C=0.5) + SMOTE
# ---------------------------------------------------------------------------
final_pipeline = ImbPipeline(steps=[
    ("preprocessor", preprocessor),
    ("oversampler", SMOTE(random_state=RANDOM_STATE)),
    ("estimator", LogisticRegression(max_iter=3000, C=0.5, random_state=RANDOM_STATE)),
])
final_pipeline.fit(X_train, y_train_encoded)

# ---------------------------------------------------------------------------
# 7. Sanity check -- should match the notebook (Test F1-macro = 0.4950)
# ---------------------------------------------------------------------------
y_pred = final_pipeline.predict(X_test)
print(f"Test F1-macro: {f1_score(y_test_encoded, y_pred, average='macro'):.4f}  (notebook: 0.4950)")
print(classification_report(y_test_encoded, y_pred, target_names=target_encoder.classes_))

# ---------------------------------------------------------------------------
# 8. Save artifacts
# ---------------------------------------------------------------------------
joblib.dump(final_pipeline, os.path.join(MODEL_DIR, "distress_risk_pipeline.joblib"))
joblib.dump(target_encoder, os.path.join(MODEL_DIR, "distress_label_encoder.joblib"))
joblib.dump({
    "numeric_features": numeric_cols,
    "categorical_features": nominal_cols + ordinal_col,
    "class_labels": list(target_encoder.classes_),
    "categories": {
        "industry_sector": sorted(X_train["industry_sector"].unique().tolist()),
        "listing_status": sorted(X_train["listing_status"].unique().tolist()),
        "audit_going_concern_flag": sorted(X_train["audit_going_concern_flag"].unique().tolist()),
        "credit_rating_band": credit_rating_rank[0],
    },
    "net_profit_margin_median": float(net_margin_median),
    "sklearn_version": sklearn.__version__,
    "imblearn_version": imblearn.__version__,
}, os.path.join(MODEL_DIR, "distress_model_metadata.joblib"))

print(f"Saved 3 artifacts to {MODEL_DIR}")
print(f"scikit-learn {sklearn.__version__}, imbalanced-learn {imblearn.__version__}")
