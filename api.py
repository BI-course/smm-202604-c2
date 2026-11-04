"""
api.py

A RESTish API, built with Flask, that serves FOUR machine learning artifacts
produced in earlier labs:

    1. Regression       -- SME monthly revenue prediction
    2. Classification   -- SME Corporate financial distress
    3. Clustering       -- Mall customer segment assignment
    4. Association Rule -- Product recommendation from a cart
       Mining (ARM)

The point of this file is to show you that a fitted scikit-learn /
imbalanced-learn Pipeline is a SINGLE object that already knows how to impute
missing values, encode categorical columns, scale numeric columns, and (for
the classifier) apply SMOTE only during training -- never during inference.
Because all of that preprocessing lives inside the Pipeline itself, this file
never manually label-encodes anything, never manually engineers date features,
and never manually reorders columns before calling .predict(). It builds a
DataFrame straight from the incoming JSON and hands it to the pipeline as-is.
This is the main lesson of the lab: "the pipeline IS the deployment artifact."

Run this file directly to start a local development server:

    python api.py
"""

import os

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

app = Flask(__name__)

# -----------------------------------------------------------------------
# CORS (Cross-Origin Resource Sharing)
# -----------------------------------------------------------------------
# By default, a browser blocks JavaScript on one origin (e.g., a page served
# from http://127.0.0.1:5500) from calling an API on a different origin
# (e.g., http://127.0.0.1:5000). `flask_cors` relaxes that restriction for the
# specific origins we trust during local development.
#
# A few alternative configurations are shown below (commented out) so you can
# see how the restriction can be loosened or tightened depending on your
# deployment target. Only ONE of these should be active at a time.

# Option A (the most permissive. This is fine for a quick classroom demo or
# local development, *NEVER* for production):
# CORS(app)

# Option B (allow only a specific frontend origin):
# CORS(app, resources={r"/api/*": {"origins": "http://127.0.0.1:5500"}})

# Option C (the one actually in use here -- allow localhost/127.0.0.1 on the
# ports our lab's frontend files are typically served from):
CORS(app, resources={r"/api/*": {"origins": [
    "http://localhost:5000", "http://127.0.0.1:5000",
    "http://localhost:5500", "http://127.0.0.1:5500",
    "http://localhost:8000", "http://127.0.0.1:8000",  # python -m http.server 8000
    "http://localhost:8080", "http://127.0.0.1:8080",
    # Through the Nginx reverse proxy there is no port in the origin: the page
    # and the API share https://localhost, so no CORS entry is needed at all.
    # That is exactly why the frontend pages are switched to a RELATIVE
    # API path -- same origin means the browser never asks about CORS.
    "https://localhost:443", "https://127.0.0.1:443",
    "http://localhost:63342",  # VS Code Live Server extension
    "null",  # allows opening a frontend .html file directly (file://) with no web server
]}})

# -----------------------------------------------------------------------
# Load the persisted artifacts ONCE, at startup, not per-request. Loading
# inside a route function would re-read the file from disk on every call,
# which is slow and pointless -- the fitted model never changes while the
# server is running.
# -----------------------------------------------------------------------
FRONTEND_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend")
MODEL_DIR = "./model"

# Member 1: retail daily-sales regression (Ridge pipeline). Loads the full
# pipeline, preprocessing and all, so the route below does no manual encoding
# or scaling of its own.
regression_pipeline = joblib.load(
    os.path.join(MODEL_DIR, "ridge_regressor_for_retail_sales.joblib")
)

# Member 2: corporate financial distress classification (Logistic Regression
# inside an imblearn Pipeline: preprocessor -> SMOTE -> LogisticRegression).
# Created by save_member2_artifact.py. The pipeline predicts 0/1/2, so the
# LabelEncoder is loaded too, to turn those back into class names.
classification_pipeline = joblib.load(
    os.path.join(MODEL_DIR, "distress_risk_pipeline.joblib")
)
classification_label_encoder = joblib.load(
    os.path.join(MODEL_DIR, "distress_label_encoder.joblib")
)
classification_metadata = joblib.load(
    os.path.join(MODEL_DIR, "distress_model_metadata.joblib")
)


clustering_artifact = joblib.load(
    os.path.join(
        MODEL_DIR,
        "20261102_102110_kmeans_ecommerce_customer_segmentation.joblib"
    )
)

clustering_imputer = clustering_artifact["imputer"]
clustering_scaler = clustering_artifact["scaler"]
clustering_kmeans = clustering_artifact["kmeans"]
clustering_features = clustering_artifact["selected_features"]


# Member 4: streaming-show recommender. The association rules were mined and
# cleaned in the ARM lab (Apriori, min_support = 105/3500 sessions,
# confidence >= 0.50, lift >= 1.50, redundant rules removed -> 11 rules) and
# saved as one bundle. They are only LOADED here, never re-mined.
arm_artifact = joblib.load(
    os.path.join(MODEL_DIR, "apriori_recommendation_rules.joblib")
)
arm_rules = arm_artifact["cleaned_rules"]                          # DataFrame of the 11 final rules
recommendation_lookup = arm_artifact["recommendation_lookup"]      # {frozenset(antecedent show_ids): [rule, ...]}
show_lookup = arm_artifact["item_lookup"]                          # show_id, item_name, category, ...

show_id_to_name = dict(zip(show_lookup["show_id"], show_lookup["item_name"]))
show_id_to_category = dict(zip(show_lookup["show_id"], show_lookup["category"]))
# Case-insensitive lookup so "Crime City", "crime city" and "S01" all resolve to S01
show_key_to_id = {name.strip().lower(): sid for sid, name in show_id_to_name.items()}
show_key_to_id.update({sid.lower(): sid for sid in show_id_to_name})

print("All the four artifacts have been loaded. The API is ready.")


# 0. STATIC FRONTEND -- serve the demo HTML pages from this same Flask app.
# That keeps the page and the API on one origin during the local demo, so the
# pages can use a relative API path and the browser never needs a CORS check.
# In the Docker setup Nginx takes this job over instead.
@app.route("/")
def index():
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.route("/<path:filename>")
def serve_frontend_file(filename):
    return send_from_directory(FRONTEND_DIR, filename)


# 1. REGRESSION -- predict DAILY sales revenue (KES) for a retail store
#
# Member 1's model: Ridge on the retail daily-sales dataset (900 stores),
# target daily_sales_revenue_kes. Trained on the log of revenue because the raw
# target is heavily right-skewed, so the route converts back with expm1.
# Saved as model/ridge_regressor_for_retail_sales.joblib.
#
# Sample request (cURL):
#   curl -X POST http://127.0.0.1:5000/api/v1/models/store-daily-sales-regressor/predictions \
#        -H "Content-Type: application/json" \
#        -d '{"store_size_sqm": 180.0, "promotions_active": 2,
#             "avg_basket_size_kes": 1350.0, "store_age_years": 5.2,
#             "online_order_count": 45.0, "loyalty_program_members": 60,
#             "store_format": "Supermarket", "location_type": "Mall",
#             "delivery_available": "Yes", "store_size_category": "Medium"}'
#   -> {"predicted_daily_sales_revenue_kes": 208681.74, ...}
#
# Sample request (PowerShell):
#   $body = @{store_size_sqm=180.0; promotions_active=2; avg_basket_size_kes=1350.0;
#             store_age_years=5.2; online_order_count=45.0; loyalty_program_members=60;
#             store_format="Supermarket"; location_type="Mall";
#             delivery_available="Yes"; store_size_category="Medium"} | ConvertTo-Json
#   Invoke-RestMethod -Method Post -ContentType "application/json" `
#     -Uri "http://127.0.0.1:5000/api/v1/models/store-daily-sales-regressor/predictions" -Body $body
#
# Structural missingness: online_order_count is null for a store with
# delivery_available="No" -- it has no online orders to count, which is not the
# same as an unknown value. The pipeline handles this itself (constant 0 + a
# missingness indicator), so nothing here special-cases it.
NUMERIC_INPUT_FIELDS = [
    "store_size_sqm", "loyalty_program_members", "promotions_active",
    "avg_basket_size_kes", "store_age_years", "online_order_count",
]


@app.route("/api/v1/models/store-daily-sales-regressor/predictions", methods=["POST"])
def predict_daily_sales():
    payload = request.get_json()

    # Build a one-row DataFrame straight from the JSON body. No manual
    # encoding, no manual scaling, no manual column reordering -- the pipeline's
    # ColumnTransformer already knows which columns it expects and how to
    # preprocess each one.
    input_row = pd.DataFrame([payload])

    # Coerce the numeric fields to float64 before the pipeline sees them.
    # Hit this first time I tested a null through the route: pandas makes the
    # column dtype "object" when the JSON has a null in it, and SimpleImputer
    # then silently passes the column through unimputed, so Ridge raises a
    # cryptic NaN error. The same thing happens with whole numbers (180 rather
    # than 180.0 -> int64, which the fitted imputer's float fill can't be cast
    # into). Casting everything to float64 up front avoids both.
    for column in NUMERIC_INPUT_FIELDS:
        if column in input_row.columns:
            input_row[column] = pd.to_numeric(
                input_row[column], errors="coerce"
            ).astype("float64")

    predicted_log_revenue = regression_pipeline.predict(input_row)[0]
    # The pipeline was trained on log1p(daily_sales_revenue_kes); expm1 is its
    # exact inverse and puts the answer back into KES.
    predicted_revenue_kes = float(np.expm1(predicted_log_revenue))

    return jsonify({
        "predicted_daily_sales_revenue_kes": round(predicted_revenue_kes, 2),
        "model": "Ridge Regressor",
    })


# 2. CLASSIFICATION -- predict a company's financial distress risk category
#    (Distressed / Healthy / Watch)
#
# Member 2's model: Logistic Regression (C=0.5) on the corporate financial
# distress dataset (1,400 companies), target distress_risk_category. SMOTE
# lives inside the pipeline and only runs during .fit(), never at prediction
# time. Saved as model/distress_risk_pipeline.joblib.
#
# Sample request (cURL):
#   curl -X POST http://127.0.0.1:5000/api/v1/models/corporate-distress-classifier/predictions \
#        -H "Content-Type: application/json" \
#        -d '{"industry_sector": "Retail", "listing_status": "Private",
#             "credit_rating_band": "High Risk", "audit_going_concern_flag": null,
#             "years_in_operation": 3, "num_employees": 12,
#             "debt_to_equity_ratio": 4.8, "current_ratio": 0.6,
#             "interest_coverage_ratio": 0.9, "revenue_growth_pct": -18.0,
#             "net_profit_margin_pct": -5.2, "working_capital_to_assets_ratio": -0.15,
#             "macroeconomic_interest_rate_pct": 12.5}'
#   -> {"predicted_distress_risk_category": "Distressed", ...}


# Structural missingness: audit_going_concern_flag is null for Private
# companies -- they are not subject to going-concern audit disclosure. In
# training this was filled with "Not Flagged" BEFORE the pipeline, so the
# route must do the same fill to process inputs exactly as in training.
# net_profit_margin_pct may also be null; the pipeline's median imputer
# handles that one internally.
CLASSIFICATION_NUMERIC_FIELDS = classification_metadata["numeric_features"]
CLASSIFICATION_CATEGORICAL_FIELDS = classification_metadata["categorical_features"]
CLASSIFICATION_ALLOWED_CATEGORIES = classification_metadata["categories"]


@app.route("/api/v1/models/corporate-distress-classifier/predictions", methods=["POST"])
def predict_distress_risk():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "Send a JSON object with Content-Type: application/json."}), 400

    # Every field the pipeline was trained on must be present (company_id is
    # not needed -- it was excluded from the model as an identifier).
    required_fields = CLASSIFICATION_NUMERIC_FIELDS + CLASSIFICATION_CATEGORICAL_FIELDS
    missing_fields = [field for field in required_fields if field not in payload]
    if missing_fields:
        return jsonify({"error": "Missing required fields.", "missing_fields": missing_fields}), 400

    input_row = pd.DataFrame([payload])[required_fields]

    # Same upstream fill the notebook did before the pipeline.
    input_row["audit_going_concern_flag"] = input_row["audit_going_concern_flag"].fillna("Not Flagged")

    # Reject categories the model never saw. The pipeline's OneHotEncoder
    # would otherwise silently ignore them and still return a prediction.
    invalid_values = {}
    for field in CLASSIFICATION_CATEGORICAL_FIELDS:
        value = input_row.at[0, field]
        if value not in CLASSIFICATION_ALLOWED_CATEGORIES[field]:
            invalid_values[field] = {"received": value, "allowed": CLASSIFICATION_ALLOWED_CATEGORIES[field]}
    if invalid_values:
        return jsonify({"error": "Invalid category values.", "details": invalid_values}), 400

    # Same float64 coercion as the regression route: a null or a whole number
    # would otherwise give the column the wrong dtype for the imputer/scaler.
    for column in CLASSIFICATION_NUMERIC_FIELDS:
        input_row[column] = pd.to_numeric(input_row[column], errors="coerce").astype("float64")

    # The pipeline was trained on LabelEncoded targets, so it returns 0/1/2;
    # the label encoder turns those back into Distressed/Healthy/Watch.
    predicted_code = classification_pipeline.predict(input_row)[0]
    predicted_class = classification_label_encoder.inverse_transform([predicted_code])[0]

    predicted_probabilities = classification_pipeline.predict_proba(input_row)[0]
    class_codes = classification_pipeline.named_steps["estimator"].classes_
    class_labels = classification_label_encoder.inverse_transform(class_codes)

    return jsonify({
        "predicted_distress_risk_category": str(predicted_class),
        "class_probabilities": {
            str(label): round(float(prob), 4)
            for label, prob in zip(class_labels, predicted_probabilities)
        },
        "model": "Logistic Regression (C=0.5, SMOTE-resampled at training only)",
    })



# -----------------------------------------------------------------------
# 3. CLUSTERING -- assign an e-commerce customer to a segment
# -----------------------------------------------------------------------

# Sample request:
# POST /api/v1/models/ecommerce-customer-segmenter/predictions
#
# JSON:
# {
#     "recency_days": 12,
#     "frequency_per_year": 14,
#     "avg_order_value_kes": 6200,
#     "num_categories_purchased": 5
# }

@app.route(
    "/api/v1/models/ecommerce-customer-segmenter/predictions",
    methods=["POST"]
)
def predict_customer_segment():

    payload = request.get_json(silent=True)

    if not isinstance(payload, dict):
        return jsonify({"error": "Please provide customer data as JSON."}), 400

    # Check that all required features are provided
    missing_features = [
        feature for feature in clustering_features
        if feature not in payload
    ]

    if missing_features:
        return jsonify({
            "error": "Missing required customer features.",
            "missing_features": missing_features
        }), 400

    try:
        # Arrange the features in the exact order used during training
        input_row = pd.DataFrame(
            [[payload[feature] for feature in clustering_features]],
            columns=clustering_features
        )

        # Convert values to numeric
        input_row = input_row.apply(pd.to_numeric, errors="raise")

        # Apply the saved preprocessing steps
        imputed_input = clustering_imputer.transform(input_row)
        scaled_input = clustering_scaler.transform(imputed_input)

        # Predict the customer's cluster
        assigned_cluster = int(
            clustering_kmeans.predict(scaled_input)[0]
        )

        return jsonify({
            "assigned_cluster": assigned_cluster,
            "model": "K-Means E-commerce Customer Segmentation",
            "number_of_clusters": 3
        }), 200

    except (TypeError, ValueError) as error:
        return jsonify({
            "error": "Customer feature values must be numeric or null.",
            "details": str(error)
        }), 400



# -----------------------------------------------------------------------
# 4. ASSOCIATION RULE MINING -- recommend TV shows from a viewing session
# -----------------------------------------------------------------------
#
# Member 4's model: Apriori association rules mined from the streaming
# "series watched" dataset (3,500 viewing sessions, 15 shows). Saved as
# model/apriori_recommendation_rules.joblib.
#
# Send the shows a viewer has already watched (names in any case, or show IDs
# such as "S06"). Every rule whose antecedent shows are ALL in that list is
# applied, so a 2-show session can still trigger a 1-show rule. Shows already
# watched are never recommended; if two rules suggest the same show, the
# higher-confidence rule wins. Results are ranked by confidence, then lift.
#
# Sample request (cURL) -- a rule matches:
#   curl -X POST http://127.0.0.1:5000/api/v1/models/streaming-show-recommender/predictions \
#        -H "Content-Type: application/json" \
#        -d '{"watched": ["Space Frontier", "Time Anomaly"], "top_n": 3}'
#   -> recommends "Galactic Wars" (confidence 0.774, lift 5.0)
#
# Sample request (cURL) -- no rule matches:
#   curl -X POST http://127.0.0.1:5000/api/v1/models/streaming-show-recommender/predictions \
#        -H "Content-Type: application/json" \
#        -d '{"watched": ["Crime City"]}'
#   -> "recommendations": [] with a message explaining why
#
# Sample request (PowerShell):
#   $body = @{watched=@("Space Frontier","Time Anomaly"); top_n=3} | ConvertTo-Json
#   Invoke-RestMethod -Method Post -ContentType "application/json" `
#     -Uri "http://127.0.0.1:5000/api/v1/models/streaming-show-recommender/predictions" -Body $body
#
# Note: there is no scikit-learn ".predict()" here -- the mined rules were
# precomputed into a lookup dictionary during the notebook's persistence stage,
# and this endpoint only does the lookup + rank step on each request.
def recommend_shows(watched_ids, top_n=3):
    """Find the saved rules that apply to watched_ids and return ranked recommendations."""
    candidates = {}
    matched_rules = []
    for antecedent_ids, rule_entries in recommendation_lookup.items():
        if not antecedent_ids.issubset(watched_ids):
            continue
        for rule in rule_entries:
            matched_rules.append({
                "if_watched": sorted(show_id_to_name[s] for s in antecedent_ids),
                "then_recommend": sorted(show_id_to_name[s] for s in rule["consequents"]),
                "confidence": round(float(rule["confidence"]), 3),
                "lift": round(float(rule["lift"]), 2),
            })
            for show_id in rule["consequents"]:
                if show_id in watched_ids:
                    continue  # never recommend something already watched
                best = candidates.get(show_id)
                if best is None or rule["confidence"] > best[0]["confidence"]:
                    candidates[show_id] = (rule, antecedent_ids)

    ranked = sorted(
        candidates.items(),
        key=lambda item: (item[1][0]["confidence"], item[1][0]["lift"]),
        reverse=True,
    )[:top_n]

    recommendations = [
        {
            "show_id": str(show_id),
            "show_name": show_id_to_name[show_id],
            "category": show_id_to_category[show_id],
            "because_you_watched": sorted(show_id_to_name[s] for s in antecedent_ids),
            "confidence": round(float(rule["confidence"]), 3),
            "lift": round(float(rule["lift"]), 2),
            "support": round(float(rule["support"]), 3),
        }
        for show_id, (rule, antecedent_ids) in ranked
    ]
    return recommendations, matched_rules


@app.route("/api/v1/models/streaming-show-recommender/predictions", methods=["POST"])
def recommend_streaming_shows():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "Send a JSON object with Content-Type: application/json."}), 400

    watched = payload.get("watched", payload.get("cart"))
    if isinstance(watched, str):
        watched = [watched]
    if not isinstance(watched, list) or len(watched) == 0:
        return jsonify({
            "error": "Provide 'watched': a non-empty list of shows the viewer has already watched.",
            "example": {"watched": ["Space Frontier", "Time Anomaly"], "top_n": 3},
            "valid_shows": sorted(show_id_to_name.values()),
        }), 400

    try:
        top_n = int(payload.get("top_n", 3))
        if top_n < 1:
            raise ValueError
    except (TypeError, ValueError):
        return jsonify({"error": "'top_n' must be a whole number of 1 or more."}), 400

    unknown_shows = [s for s in watched if str(s).strip().lower() not in show_key_to_id]
    if len(unknown_shows) == len(watched):
        return jsonify({
            "error": "None of the shows sent are in the catalogue.",
            "unknown_shows": unknown_shows,
            "valid_shows": sorted(show_id_to_name.values()),
        }), 400

    watched_ids = frozenset(
        show_key_to_id[str(s).strip().lower()] for s in watched
        if str(s).strip().lower() in show_key_to_id
    )
    recommendations, matched_rules = recommend_shows(watched_ids, top_n)

    if recommendations:
        message = f"{len(recommendations)} recommendation(s) found."
    elif matched_rules:
        message = "Rules matched, but they only suggest shows already watched, so there is no new recommendation."
    else:
        message = "No association rule matches these shows, so no recommendation is available."

    response = {
        "watched": sorted(show_id_to_name[s] for s in watched_ids),
        "recommendations": recommendations,
        "matched_rules": matched_rules,
        "message": message,
        "model": "Apriori association rules (11 cleaned rules), precomputed lookup",
    }
    if unknown_shows:
        response["ignored_unknown_shows"] = unknown_shows
    return jsonify(response), 200


@app.route("/api/v1/models/streaming-show-recommender/shows", methods=["GET"])
def list_streaming_shows():
    """List the shows the recommender knows (useful for a frontend dropdown)."""
    return jsonify({
        "shows": show_lookup[["show_id", "item_name", "category"]].to_dict(orient="records")
    }), 200

# -----------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------
if __name__ == "__main__":
    # Development server. debug=True gives auto-reload and an interactive
    # debugger on errors -- convenient in class, but NEVER use debug=True in
    # a deployed/public environment (it can expose a remote code execution
    # console to anyone who can reach the server).
    app.run(debug=True)

    # Production-style alternative (no debugger, no auto-reload):
    # app.run(debug=False, host="0.0.0.0", port=5000)

    # HTTPS/TLS alternative for local development (requires a self-signed
    # certificate; see app_server_reverse_proxy_server_setup.md):
    # app.run(debug=False, ssl_context=("cert.pem", "key.pem"))