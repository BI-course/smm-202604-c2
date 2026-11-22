import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os

# -----------------------------------------------------------------------
# Load all four persisted artifacts ONCE, at module import time. Streamlit
# re-runs this whole script top-to-bottom on every widget interaction, so
# @st.cache_resource keeps joblib.load() from re-reading the files from
# disk on every single click -- without it, every tab switch and every
# form submission would reload all four artifacts again.
#
# These are Group C2's own models, the same files api.py, the Docker image
# and the Hugging Face Space serve. Streamlit Community Cloud sets the
# working directory to the REPOSITORY ROOT, so `model/` must be a sibling
# of this folder -- which is why MODEL_DIR is "./model" and not "../model".
# Running locally: `streamlit run streamlit-sharing-using-streamlit/app.py`
# from the repo root, so local paths behave exactly like the cloud.
# -----------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    MODEL_DIR = "./model"

    # Member 1: the regression Pipeline already holds its own imputer,
    # encoders and scaler, and was trained on log1p(target).
    regression_pipeline = joblib.load(
        os.path.join(MODEL_DIR, "ridge_regressor_for_retail_sales.joblib")
    )

    # Member 2: the pipeline predicts 0/1/2; the LabelEncoder turns those
    # back into Distressed / Healthy / Watch. Both were saved separately.
    classification_pipeline = joblib.load(
        os.path.join(MODEL_DIR, "distress_risk_pipeline.joblib")
    )
    classification_label_encoder = joblib.load(
        os.path.join(MODEL_DIR, "distress_label_encoder.joblib")
    )
    classification_metadata = joblib.load(
        os.path.join(MODEL_DIR, "distress_model_metadata.joblib")
    )

    # Member 3: one dictionary holding four objects that must be applied
    # in this order -- imputer -> scaler -> kmeans.predict.
    clustering_artifact = joblib.load(
        os.path.join(MODEL_DIR, "kmeans_ecommerce_customer_segmentation.joblib")
    )

    # Member 4: the lookup key is item_lookup, NOT sku_lookup.
    arm_artifact = joblib.load(
        os.path.join(MODEL_DIR, "apriori_recommendation_rules.joblib")
    )

    return (regression_pipeline, classification_pipeline,
            classification_label_encoder, classification_metadata,
            clustering_artifact, arm_artifact)


(regression_pipeline, classification_pipeline, classification_label_encoder,
 classification_metadata, clustering_artifact, arm_artifact) = load_artifacts()

clustering_imputer = clustering_artifact["imputer"]
clustering_scaler = clustering_artifact["scaler"]
clustering_kmeans = clustering_artifact["kmeans"]
clustering_features = clustering_artifact["selected_features"]

recommendation_lookup = arm_artifact["recommendation_lookup"]
show_lookup = arm_artifact["item_lookup"]
show_id_to_name = dict(zip(show_lookup["show_id"], show_lookup["item_name"]))
show_key_to_id = {name.strip().lower(): sid for sid, name in show_id_to_name.items()}
show_key_to_id.update({sid.lower(): sid for sid in show_id_to_name})

CLASSIFICATION_REQUIRED_FIELDS = (
    classification_metadata["numeric_features"] + classification_metadata["categorical_features"]
)

# Same coercion api.py applies: a blank field must become NaN, not dtype
# object, or the fitted imputer silently skips it.
NUMERIC_RETAIL_FIELDS = [
    "store_size_sqm", "loyalty_program_members", "promotions_active",
    "avg_basket_size_kes", "store_age_years", "online_order_count",
]

st.set_page_config(page_title="BBT 4206 Model Validation Demos", page_icon="\U0001F4CA", layout="centered")
st.title("BBT 4206 -- Group C2 Model Validation Demos")
st.caption("Same four saved artifacts as the API, the Docker image and the Hugging Face Space. "
           "Switch tabs to try a different model.")

tab_regression, tab_classification, tab_clustering, tab_arm = st.tabs([
    "Retail Daily Sales Regressor", "Corporate Distress Classifier",
    "E-commerce Customer Segmenter", "Streaming Show Recommender",
])

# -----------------------------------------------------------------------
# 1. Regression -- DAILY sales revenue (KES) for a retail store
# -----------------------------------------------------------------------
with tab_regression:
    with st.form("regression_form"):
        store_size_sqm = st.number_input("Store Size (sqm)", value=180.0)
        promotions_active = st.number_input("Promotions Active", value=2)
        avg_basket_size_kes = st.number_input("Average Basket Size (KES)", value=1350.0)
        store_age_years = st.number_input("Store Age (years)", value=5.2)
        # Structural missingness: a fully online/kiosk store has no foot
        # traffic equivalent. Leave it blank rather than typing 0.
        online_order_count = st.number_input(
            "Online Order Count (leave blank if not applicable)", value=None, placeholder="blank = no delivery")
        loyalty_program_members = st.number_input("Loyalty Program Members", value=60)
        store_format = st.selectbox(
            "Store Format", ["Convenience", "Hypermarket", "Kiosk", "Supermarket"], index=3)
        location_type = st.selectbox(
            "Location Type", ["CBD", "Mall", "Residential", "Street"], index=1)
        delivery_available = st.selectbox("Delivery Available", ["Yes", "No"])
        store_size_category = st.selectbox("Store Size Category", ["Small", "Medium", "Large"], index=1)
        submitted = st.form_submit_button("Predict Daily Sales")

    if submitted:
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

        # The pipeline was trained on log1p(daily_sales_revenue_kes); expm1
        # is its exact inverse and puts the answer back into KES.
        predicted_revenue_kes = float(np.expm1(regression_pipeline.predict(input_row)[0]))
        st.success(f"Predicted daily sales revenue: KES {predicted_revenue_kes:,.2f}")

# -----------------------------------------------------------------------
# 2. Classification -- a company's financial distress category
# -----------------------------------------------------------------------
with tab_classification:
    with st.form("classification_form"):
        industry_sector = st.selectbox(
            "Industry Sector", classification_metadata["categories"]["industry_sector"], index=3)
        listing_status = st.selectbox(
            "Listing Status", classification_metadata["categories"]["listing_status"])
        credit_rating_band = st.selectbox(
            "Credit Rating Band", classification_metadata["categories"]["credit_rating_band"], index=2)
        # Blank = a Private company, which has no going-concern disclosure.
        audit_going_concern_flag = st.selectbox(
            "Audit Going Concern Flag (blank = not applicable)", ["Not Flagged", "No", "Yes"], index=None)
        years_in_operation = st.number_input("Years in Operation", value=3)
        num_employees = st.number_input("Number of Employees", value=12)
        debt_to_equity_ratio = st.number_input("Debt-to-Equity Ratio", value=4.8)
        current_ratio = st.number_input("Current Ratio", value=0.6)
        interest_coverage_ratio = st.number_input("Interest Coverage Ratio", value=0.9)
        revenue_growth_pct = st.number_input("Revenue Growth (%)", value=-18.0)
        net_profit_margin_pct = st.number_input("Net Profit Margin (%)", value=-5.2)
        working_capital_to_assets_ratio = st.number_input("Working Capital to Assets Ratio", value=-0.15)
        macroeconomic_interest_rate_pct = st.number_input("Macroeconomic Interest Rate (%)", value=12.5)
        submitted2 = st.form_submit_button("Predict Distress Category")

    if submitted2:
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

        # Same upstream fill the notebook used: a blank means "Not Flagged".
        input_row["audit_going_concern_flag"] = input_row["audit_going_concern_flag"].fillna("Not Flagged")
        for column in classification_metadata["numeric_features"]:
            input_row[column] = pd.to_numeric(input_row[column], errors="coerce").astype("float64")

        predicted_code = classification_pipeline.predict(input_row)[0]
        predicted_class = classification_label_encoder.inverse_transform([predicted_code])[0]
        probabilities = classification_pipeline.predict_proba(input_row)[0]
        labels = classification_label_encoder.inverse_transform(
            classification_pipeline.named_steps["estimator"].classes_
        )
        st.success(f"Predicted distress risk category: {predicted_class}")
        st.write({str(label): round(float(prob), 4)
                  for label, prob in zip(labels, probabilities)})

# -----------------------------------------------------------------------
# 3. Clustering -- an e-commerce customer's segment
# -----------------------------------------------------------------------
with tab_clustering:
    with st.form("clustering_form"):
        recency_days = st.number_input("Recency (days since last order)", min_value=0, value=12)
        frequency_per_year = st.number_input("Frequency (orders per year)", min_value=0, value=14)
        avg_order_value_kes = st.number_input("Average Order Value (KES)", min_value=0.0, value=6200.0)
        num_categories_purchased = st.number_input("Number of Categories Purchased", min_value=0, value=5)
        submitted3 = st.form_submit_button("Predict Segment")

    if submitted3:
        input_row = pd.DataFrame(
            [[recency_days, frequency_per_year, avg_order_value_kes, num_categories_purchased]],
            columns=clustering_features,
        ).apply(pd.to_numeric, errors="raise")

        # Order matters, and matches CONTRACT.md: imputer -> scaler -> kmeans.
        imputed_input = clustering_imputer.transform(input_row)
        scaled_input = clustering_scaler.transform(imputed_input)
        cluster = int(clustering_kmeans.predict(scaled_input)[0])
        st.success(f"Assigned segment: Cluster {cluster} of {clustering_kmeans.n_clusters}")

# -----------------------------------------------------------------------
# 4. Association Rule Mining -- recommend shows from a viewing session
# -----------------------------------------------------------------------
with tab_arm:
    with st.form("arm_form"):
        watched_text = st.text_input("Shows watched (comma-separated names or IDs)",
                                     "Space Frontier, Time Anomaly")
        top_n = st.number_input("Top N Recommendations", min_value=1, value=3)
        submitted4 = st.form_submit_button("Get Recommendations")

    if submitted4:
        watched = [name.strip() for name in watched_text.split(",") if name.strip()]
        unknown = [name for name in watched if name.lower() not in show_key_to_id]
        watched_ids = frozenset(
            show_key_to_id[name.lower()] for name in watched if name.lower() in show_key_to_id
        )

        if not watched:
            st.warning("Enter at least one show name.")
        elif not watched_ids:
            st.warning("None of those shows are in the catalogue. Try: "
                       + ", ".join(sorted(show_id_to_name.values())[:5]) + " ...")
        else:
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
                st.warning("No association rule matches these shows, so there is no recommendation.")
            else:
                for show_id, rule in ranked:
                    st.write(f"**{show_id_to_name[show_id]}** ({show_id}) "
                             f"-- confidence {rule['confidence']:.1%}, "
                             f"lift {rule['lift']:.2f}, support {rule['support']:.3f}")
            if unknown:
                st.caption("Ignored, not in the catalogue: " + ", ".join(unknown))
