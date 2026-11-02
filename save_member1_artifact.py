"""
save_member1_artifact.py
=======================
Member 1 (retail daily-sales regression) — persist the FULLY FITTED pipeline to
model/ridge_regressor_for_retail_sales.joblib, then prove it reloads and predicts.

WHY THIS SCRIPT EXISTS
----------------------
The Colab notebook (Retail_Daily_Sales_Regression.ipynb) only ever prints its
fit into memory; it is not a runnable script, so the exact pipeline object cannot
be recovered outside that Colab session. Rather than hand-copying notebook cells
(where a single missed transformation silently changes every prediction), this
script rebuilds the pipeline with the SAME structure the notebook uses, fits it on
the training split, and saves it.

If you would rather export the object straight out of Colab (which is the
original fitted model, weights and all), skip this file and follow the block in
the docstring at the bottom -- then just re-run the verification section.

Where the design decisions come from (Retail_Daily_Sales_Regression.ipynb):
  * 10 features kept after correlation filtering -> VIF -> RFE -> Lasso:
        store_size_sqm, loyalty_program_members, promotions_active,
        avg_basket_size_kes, store_age_years, online_order_count,
        location_type, delivery_available, store_size_category, store_format
  * store_format / location_type / delivery_available  -> OHE (nominal)
  * store_size_category                                -> ORDINAL Small<Medium<Large
  * online_order_count: 564/900 missing because a store with no delivery simply
    has no online orders -> STRUCTURAL missingness -> constant 0 + missingness
    indicator (never a median, which would invent orders that never happened)
  * all other numerics -> median imputation (missing at random)
  * target daily_sales_revenue_kes is right-skewed (6.56) -> log1p
  * Ridge (alpha tuned) selected by GridSearchCV over the final pipeline

Run:
    python save_member1_artifact.py
"""

import os

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

RANDOM_STATE = 42
DATA_PATH = os.path.join(
    "..", "Module_4_Regression_and_Classification", "data", "retail_daily_sales.csv"
)
MODEL_PATH = os.path.join("model", "ridge_regressor_for_retail_sales.joblib")

TARGET = "daily_sales_revenue_kes"

# The exact 10 columns the pipeline is fitted on, in this order.
NUMERIC_FEATURES = [
    "store_size_sqm",
    "loyalty_program_members",
    "promotions_active",
    "avg_basket_size_kes",
    "store_age_years",
    "online_order_count",
]
NOMINAL_FEATURES = ["location_type", "delivery_available", "store_format"]
ORDINAL_FEATURES = ["store_size_category"]
FEATURES = NUMERIC_FEATURES + NOMINAL_FEATURES + ORDINAL_FEATURES


def build_pipeline() -> Pipeline:
    """The same ColumnTransformer + Ridge the notebook builds."""

    # online_order_count is handled on its own: structural missingness.
    structural_numeric = ["online_order_count"]
    plain_numeric = [f for f in NUMERIC_FEATURES if f not in structural_numeric]

    numeric_plain = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    numeric_structural = Pipeline([
        # constant 0: "no delivery" means zero online orders, not "unknown"
        ("imputer", SimpleImputer(strategy="constant", fill_value=0, add_indicator=True)),
        ("scaler", StandardScaler()),
    ])

    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    ordinal = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ordinal", OrdinalEncoder(categories=[["Small", "Medium", "Large"]])),
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_plain, plain_numeric),
        ("num_structural", numeric_structural, structural_numeric),
        ("nom", categorical, NOMINAL_FEATURES),
        ("ord", ordinal, ORDINAL_FEATURES),
    ])

    return Pipeline([("preprocessor", preprocessor), ("model", Ridge(random_state=RANDOM_STATE))])


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {df.shape[0]} rows, {df.shape[1]} columns from {DATA_PATH}")

    X = df[FEATURES]
    y = df[TARGET]

    # --- The notebook's split: STRATIFIED on revenue quintiles, not a plain
    # random split. Bin the target into five equal-count groups and take the
    # first fold as the 20% test slice, so every revenue band is represented in
    # both halves. This is what keeps the test R2 comparable to the notebook's.
    percentile_edges = np.percentile(y, [0, 20, 40, 60, 80, 100])
    percentile_edges[0] -= 1e-5
    revenue_bins = np.digitize(y, percentile_edges) - 1
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    train_idx, test_idx = next(splitter.split(X, revenue_bins))

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    y_train_log = np.log1p(y_train)       # right-skewed target -> log1p
    y_test_log = np.log1p(y_test)
    print(f"Stratified split -> train {len(X_train)} / test {len(X_test)} rows")

    pipeline = build_pipeline()

    print("Tuning Ridge alpha with GridSearchCV ...")
    search = GridSearchCV(
        pipeline,
        param_grid={"model__alpha": [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]},
        cv=5,
        scoring="neg_root_mean_squared_error",
    )
    search.fit(X_train, y_train_log)
    print(f"  best alpha = {search.best_params_['model__alpha']}")

    final_pipeline = search.best_estimator_

    # ---- evaluate in KES, on the held-out set only ----
    pred_kes = np.expm1(final_pipeline.predict(X_test))
    actual_kes = np.expm1(y_test_log)
    mae = float(np.mean(np.abs(pred_kes - actual_kes)))
    rmse = float(np.sqrt(np.mean((pred_kes - actual_kes) ** 2)))
    mape = float(np.mean(np.abs((pred_kes - actual_kes) / actual_kes)) * 100)
    ss_res = float(np.sum((actual_kes - pred_kes) ** 2))
    ss_tot = float(np.sum((actual_kes - actual_kes.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot
    print(f"  test MAE  {mae:,.2f} KES")
    print(f"  test RMSE {rmse:,.2f} KES")
    print(f"  test MAPE {mape:.2f}%")
    print(f"  test R2   {r2:.4f}")

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(final_pipeline, MODEL_PATH)
    print(f"Saved: {MODEL_PATH}")

    # ---- verification: reload from disk and predict the two demo rows ----
    loaded = joblib.load(MODEL_PATH)
    print("Reloaded OK. steps:", list(loaded.named_steps))
    print("feature order:", list(loaded.feature_names_in_))

    demo_rows = {
        "delivery store": {
            "store_size_sqm": 180.0, "promotions_active": 2, "avg_basket_size_kes": 1350.0,
            "store_age_years": 5.2, "online_order_count": 45.0, "loyalty_program_members": 60,
            "store_format": "Supermarket", "location_type": "Mall",
            "delivery_available": "Yes", "store_size_category": "Medium",
        },
        "no-delivery store (null online orders)": {
            "store_size_sqm": 95.0, "promotions_active": 0, "avg_basket_size_kes": 900.0,
            "store_age_years": 2.0, "online_order_count": None, "loyalty_program_members": 25,
            "store_format": "Kiosk", "location_type": "Street",
            "delivery_available": "No", "store_size_category": "Small",
        },
    }
    for label, row in demo_rows.items():
        frame = pd.DataFrame([row])
        # A JSON null makes this column dtype 'object'; coerce it back to a
        # float with NaN so the pipeline's imputer actually sees a missing value.
        for col in NUMERIC_FEATURES:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
        pred = float(np.expm1(loaded.predict(frame)[0]))
        print(f"  {label}: {pred:,.2f} KES")

    # Structural-missingness demo row must survive a JSON-null round trip.
    # NOTE: a one-row DataFrame built from JSON gives online_order_count dtype
    # object, and SimpleImputer silently passes object columns through -- so the
    # API must coerce to numeric before predict(). See api.py's route.


if __name__ == "__main__":
    main()
