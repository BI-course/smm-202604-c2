import os

import gradio as gr
import joblib
import numpy as np
import pandas as pd
import spaces

# -----------------------------------------------------------------------
# Hugging Face's free tier now provisions new Gradio Spaces on ZeroGPU
# hardware only (see the note in Step 2 below). None of this app's four
# models needs a GPU -- every prediction below runs on plain scikit-learn
# arrays -- but some free-tier Spaces fail to start reliably unless at
# least one function is decorated with @spaces.GPU. This function does
# nothing GPU-related; it exists only to satisfy that platform check.
# Calling it once, here, registers it before Gradio launches below.
@spaces.GPU
def _warm_zerogpu():
    return None

_warm_zerogpu()

# -----------------------------------------------------------------------
# Load all four persisted artifacts ONCE, at import time -- the same
# principle api.py uses: load once at startup, reuse across every request
# (here, across every tab and every submit click), never inside a function.
#
# These are Group C2's own models, the same files api.py and the Docker
# image serve. On the Space they sit next to this file; locally they are in
# the repo's model/ folder, so both locations are tried.
# -----------------------------------------------------------------------
MODEL_DIR = "model" if os.path.isdir("model") else "."

regression_pipeline = joblib.load(os.path.join(MODEL_DIR, "ridge_regressor_for_retail_sales.joblib"))

# Member 2's classifier predicts 0/1/2; the LabelEncoder turns those codes
# back into Distressed / Healthy / Watch.
classification_pipeline = joblib.load(os.path.join(MODEL_DIR, "distress_risk_pipeline.joblib"))
classification_label_encoder = joblib.load(os.path.join(MODEL_DIR, "distress_label_encoder.joblib"))
classification_metadata = joblib.load(os.path.join(MODEL_DIR, "distress_model_metadata.joblib"))
CLASSIFICATION_REQUIRED_FIELDS = (
    classification_metadata["numeric_features"] + classification_metadata["categorical_features"]
)

# Member 3's clusterer needs four objects, applied in this order:
# imputer -> scaler -> kmeans.predict.
clustering_artifact = joblib.load(os.path.join(MODEL_DIR, "kmeans_ecommerce_customer_segmentation.joblib"))
clustering_imputer = clustering_artifact["imputer"]
clustering_scaler = clustering_artifact["scaler"]
clustering_kmeans = clustering_artifact["kmeans"]
clustering_features = clustering_artifact["selected_features"]

# Member 4's rules. The lookup key is item_lookup (show_id / item_name /
# category), not sku_lookup.
arm_artifact = joblib.load(os.path.join(MODEL_DIR, "apriori_recommendation_rules.joblib"))
recommendation_lookup = arm_artifact["recommendation_lookup"]
show_lookup = arm_artifact["item_lookup"]
show_id_to_name = dict(zip(show_lookup["show_id"], show_lookup["item_name"]))
show_key_to_id = {name.strip().lower(): sid for sid, name in show_id_to_name.items()}
show_key_to_id.update({sid.lower(): sid for sid in show_id_to_name})

# Same coercion api.py applies: a Gradio Number can arrive as an int, and a
# blank field as None, neither of which the fitted imputers expect.
NUMERIC_RETAIL_FIELDS = [
    "store_size_sqm", "loyalty_program_members", "promotions_active",
    "avg_basket_size_kes", "store_age_years", "online_order_count",
]


# -----------------------------------------------------------------------
# 1. Regression -- predict DAILY sales revenue (KES) for a retail store
# -----------------------------------------------------------------------
def predict_daily_sales(store_size_sqm, promotions_active, avg_basket_size_kes,
                        store_age_years, online_order_count, loyalty_program_members,
                        store_format, location_type, delivery_available,
                        store_size_category):
    payload = {
        "store_size_sqm": store_size_sqm, "promotions_active": promotions_active,
        "avg_basket_size_kes": avg_basket_size_kes, "store_age_years": store_age_years,
        "online_order_count": online_order_count,
        "loyalty_program_members": loyalty_program_members,
        "store_format": store_format, "location_type": location_type,
        "delivery_available": delivery_available,
        "store_size_category": store_size_category,
    }
    input_row = pd.DataFrame([payload])
    for column in NUMERIC_RETAIL_FIELDS:
        input_row[column] = pd.to_numeric(input_row[column], errors="coerce").astype("float64")

    # The pipeline was trained on log1p(daily_sales_revenue_kes); expm1 is
    # its exact inverse and puts the answer back into KES.
    predicted_revenue_kes = float(np.expm1(regression_pipeline.predict(input_row)[0]))
    return f"KES {predicted_revenue_kes:,.2f} / day"


# -----------------------------------------------------------------------
# 2. Classification -- predict a company's financial distress category
# -----------------------------------------------------------------------
def predict_distress(industry_sector, listing_status, credit_rating_band,
                     audit_going_concern_flag, years_in_operation, num_employees,
                     debt_to_equity_ratio, current_ratio, interest_coverage_ratio,
                     revenue_growth_pct, net_profit_margin_pct,
                     working_capital_to_assets_ratio, macroeconomic_interest_rate_pct):
    payload = {
        "industry_sector": industry_sector, "listing_status": listing_status,
        "credit_rating_band": credit_rating_band,
        "audit_going_concern_flag": audit_going_concern_flag,
        "years_in_operation": years_in_operation, "num_employees": num_employees,
        "debt_to_equity_ratio": debt_to_equity_ratio, "current_ratio": current_ratio,
        "interest_coverage_ratio": interest_coverage_ratio,
        "revenue_growth_pct": revenue_growth_pct,
        "net_profit_margin_pct": net_profit_margin_pct,
        "working_capital_to_assets_ratio": working_capital_to_assets_ratio,
        "macroeconomic_interest_rate_pct": macroeconomic_interest_rate_pct,
    }
    input_row = pd.DataFrame([payload])[CLASSIFICATION_REQUIRED_FIELDS]

    # Same upstream fill the notebook used: a Private company has no
    # going-concern disclosure, so a blank means "Not Flagged".
    input_row["audit_going_concern_flag"] = input_row["audit_going_concern_flag"].fillna("Not Flagged")

    for column in classification_metadata["numeric_features"]:
        input_row[column] = pd.to_numeric(input_row[column], errors="coerce").astype("float64")

    predicted_code = classification_pipeline.predict(input_row)[0]
    predicted_class = classification_label_encoder.inverse_transform([predicted_code])[0]
    probabilities = classification_pipeline.predict_proba(input_row)[0]
    labels = classification_label_encoder.inverse_transform(
        classification_pipeline.named_steps["estimator"].classes_
    )
    probability_lines = "\n".join(
        f"  {label}: {prob:.1%}" for label, prob in zip(labels, probabilities)
    )
    return f"Predicted category: {predicted_class}\n\n{probability_lines}"


# -----------------------------------------------------------------------
# 3. Clustering -- assign an e-commerce customer to one of three segments
# -----------------------------------------------------------------------
def predict_segment(recency_days, frequency_per_year, avg_order_value_kes,
                    num_categories_purchased):
    input_row = pd.DataFrame(
        [[recency_days, frequency_per_year, avg_order_value_kes, num_categories_purchased]],
        columns=clustering_features,
    ).apply(pd.to_numeric, errors="raise")

    imputed_input = clustering_imputer.transform(input_row)
    scaled_input = clustering_scaler.transform(imputed_input)
    cluster = int(clustering_kmeans.predict(scaled_input)[0])
    return f"Assigned segment: Cluster {cluster} of {clustering_kmeans.n_clusters}"


# -----------------------------------------------------------------------
# 4. Association Rule Mining -- recommend shows from a viewing session
# -----------------------------------------------------------------------
def recommend_shows(watched_text, top_n):
    watched = [name.strip() for name in watched_text.split(",") if name.strip()]
    if not watched:
        return "Enter at least one show name."

    unknown = [name for name in watched if name.lower() not in show_key_to_id]
    watched_ids = frozenset(
        show_key_to_id[name.lower()] for name in watched if name.lower() in show_key_to_id
    )
    if not watched_ids:
        return f"None of those shows are in the catalogue. Try: {', '.join(sorted(show_id_to_name.values())[:5])} ..."

    candidates = {}
    for antecedent_ids, rule_entries in recommendation_lookup.items():
        if not antecedent_ids.issubset(watched_ids):
            continue
        for rule in rule_entries:
            for show_id in rule["consequents"]:
                if show_id in watched_ids:
                    continue  # never recommend something already watched
                best = candidates.get(show_id)
                if best is None or rule["confidence"] > best["confidence"]:
                    candidates[show_id] = rule

    ranked = sorted(
        candidates.items(),
        key=lambda item: (item[1]["confidence"], item[1]["lift"]),
        reverse=True,
    )[:int(top_n)]

    if not ranked:
        return "No association rule matches these shows, so there is no recommendation."

    lines = [
        f"{show_id_to_name[show_id]} ({show_id})  --  "
        f"confidence {rule['confidence']:.1%}, lift {rule['lift']:.2f}, "
        f"support {rule['support']:.3f}"
        for show_id, rule in ranked
    ]
    if unknown:
        lines.append(f"\n(Ignored, not in the catalogue: {', '.join(unknown)})")
    return "\n".join(lines)


# -----------------------------------------------------------------------
# One gr.Blocks() app, four gr.Tab()s -- one shared Space URL, no linking
# between separately-hosted demos required.
# -----------------------------------------------------------------------
with gr.Blocks(title="BBT 4206 Model Validation Demos") as demo:
    gr.Markdown("# BBT 4206 -- Group C2 Model Validation Demos\nSwitch tabs to try a different model.")

    with gr.Tab("Retail Daily Sales Regressor"):
        gr.Interface(
            fn=predict_daily_sales,
            inputs=[
                gr.Number(label="Store Size (sqm)", value=180.0),
                gr.Number(label="Promotions Active", value=2),
                gr.Number(label="Average Basket Size (KES)", value=1350.0),
                gr.Number(label="Store Age (years)", value=5.2),
                gr.Number(label="Online Order Count (blank if no delivery)", value=45.0),
                gr.Number(label="Loyalty Program Members", value=60),
                gr.Dropdown(label="Store Format",
                            choices=["Convenience", "Hypermarket", "Kiosk", "Supermarket"],
                            value="Supermarket"),
                gr.Dropdown(label="Location Type",
                            choices=["CBD", "Mall", "Residential", "Street"], value="Mall"),
                gr.Dropdown(label="Delivery Available", choices=["Yes", "No"], value="Yes"),
                gr.Dropdown(label="Store Size Category",
                            choices=["Small", "Medium", "Large"], value="Medium"),
            ],
            outputs=gr.Textbox(label="Predicted Daily Sales Revenue"),
        )

    with gr.Tab("Corporate Distress Classifier"):
        gr.Interface(
            fn=predict_distress,
            inputs=[
                gr.Dropdown(label="Industry Sector",
                            choices=classification_metadata["categories"]["industry_sector"],
                            value="Retail"),
                gr.Dropdown(label="Listing Status",
                            choices=classification_metadata["categories"]["listing_status"],
                            value="Private"),
                gr.Dropdown(label="Credit Rating Band",
                            choices=classification_metadata["categories"]["credit_rating_band"],
                            value="High Risk"),
                # Blank = Private company, not subject to going-concern disclosure.
                gr.Dropdown(label="Audit Going Concern Flag (blank = not applicable)",
                            choices=["Not Flagged", "No", "Yes"], value=None),
                gr.Number(label="Years in Operation", value=3),
                gr.Number(label="Number of Employees", value=12),
                gr.Number(label="Debt-to-Equity Ratio", value=4.8),
                gr.Number(label="Current Ratio", value=0.6),
                gr.Number(label="Interest Coverage Ratio", value=0.9),
                gr.Number(label="Revenue Growth (%)", value=-18.0),
                gr.Number(label="Net Profit Margin (%) (blank if unknown)", value=-5.2),
                gr.Number(label="Working Capital to Assets Ratio", value=-0.15),
                gr.Number(label="Macroeconomic Interest Rate (%)", value=12.5),
            ],
            outputs=gr.Textbox(label="Predicted Distress Risk Category"),
        )

    with gr.Tab("E-commerce Customer Segmenter"):
        gr.Interface(
            fn=predict_segment,
            inputs=[
                gr.Number(label="Recency (days since last order)", value=12),
                gr.Number(label="Frequency (orders per year)", value=14),
                gr.Number(label="Average Order Value (KES)", value=6200),
                gr.Number(label="Number of Categories Purchased", value=5),
            ],
            outputs=gr.Textbox(label="Prediction"),
        )

    with gr.Tab("Streaming Show Recommender"):
        gr.Interface(
            fn=recommend_shows,
            inputs=[
                gr.Textbox(label="Shows already watched (comma-separated names or IDs)",
                           value="Space Frontier, Time Anomaly"),
                gr.Number(label="Top N Recommendations", value=3),
            ],
            outputs=gr.Textbox(label="Recommendations"),
        )

if __name__ == "__main__":
    demo.launch()
