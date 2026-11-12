"""
api_consumer_from_dev_flask.py

A frontend-style client that calls each of the four prediction endpoints
exposed by api.py, running on the local Flask development server. Start
api.py first (python api.py), then run this script separately.

This mirrors api.py's four artifacts one-to-one:

    1. Regression        -- /api/v1/models/store-daily-sales-regressor/predictions
    2. Classification     -- /api/v1/models/corporate-distress-classifier/predictions
    3. Clustering         -- /api/v1/models/ecommerce-customer-segmenter/predictions
    4. Association Rule   -- /api/v1/models/streaming-show-recommender/predictions
       Mining (ARM)

Each call uses the same requests.post(..., json=payload, headers=headers)
pattern, the same header set, and the same try/except structure as the
single-endpoint version this file replaces -- only the URL, payload, and the
key read back out of the JSON response differ per endpoint, because each
endpoint returns a different shape (a regression figure, a class + class
probabilities, a cluster id, or a list of recommendations).
"""

import requests

BASE_URL = "http://127.0.0.1:5000"

HEADERS = {
    'Content-Type': 'application/json',  # This header tells the server that the payload is in JSON format
    'Accept': 'application/json',        # This header tells the server that the client expects a JSON response
    'User-Agent': 'ServingMLModels-Client/1.0',  # Identifies the client application (useful for backend logging/debugging)
    # 'Authorization': 'Bearer <token>',  # This can be used for token-based authentication if the API requires it
}


def call_endpoint(path, payload):
    """Send one POST request to the given API path and return the parsed
    JSON response, or None if the request failed. Centralizing the
    try/except here means each of the four calls below gets identical
    error handling without repeating it four times."""
    url = f"{BASE_URL}{path}"
    try:
        response = requests.post(url, json=payload, headers=HEADERS, timeout=10)
        response.raise_for_status()  # Raises an HTTPError if the status code is 4xx or 5xx
        return response.json()

    except requests.exceptions.HTTPError as e:
        print("HTTP error:", e)
        if e.response is not None:
            print("Response body:", e.response.text)

    except requests.exceptions.RequestException as e:
        print("Request failed:", e)

    return None


# 1. REGRESSION -- predict DAILY sales revenue (KES) for a retail store

regression_payload = {
    "store_size_sqm": 180.0, "promotions_active": 2,
    "avg_basket_size_kes": 1350.0, "store_age_years": 5.2,
    "online_order_count": 45.0, "loyalty_program_members": 60,
    "store_format": "Supermarket", "location_type": "Mall",
    "delivery_available": "Yes", "store_size_category": "Medium",
}

result = call_endpoint("/api/v1/models/store-daily-sales-regressor/predictions", regression_payload)
if result is not None:
    print("Predicted daily sales revenue (KES) =", result["predicted_daily_sales_revenue_kes"])

# Second row: a store with NO delivery. online_order_count is null on purpose --
# structural missingness, not missing at random. The pipeline handles it.
no_delivery_payload = {
    "store_size_sqm": 95.0, "promotions_active": 0,
    "avg_basket_size_kes": 900.0, "store_age_years": 2.0,
    "online_order_count": None, "loyalty_program_members": 25,
    "store_format": "Kiosk", "location_type": "Street",
    "delivery_available": "No", "store_size_category": "Small",
}

result = call_endpoint("/api/v1/models/store-daily-sales-regressor/predictions", no_delivery_payload)
if result is not None:
    print("Predicted daily sales revenue, no-delivery store (KES) =",
          result["predicted_daily_sales_revenue_kes"])

print("-" * 60)



# 2. CLASSIFICATION -- predict corporate financial distress (Distressed/Healthy/Watch)

classification_payload = {
    "industry_sector": "Retail", "listing_status": "Private",
    "credit_rating_band": "High Risk",
    "audit_going_concern_flag": None,  # Private companies: structural missingness
    "years_in_operation": 3, "num_employees": 12,
    "debt_to_equity_ratio": 4.8, "current_ratio": 0.6,
    "interest_coverage_ratio": 0.9, "revenue_growth_pct": -18.0,
    "net_profit_margin_pct": -5.2, "working_capital_to_assets_ratio": -0.15,
    "macroeconomic_interest_rate_pct": 12.5,
}

result = call_endpoint("/api/v1/models/corporate-distress-classifier/predictions", classification_payload)
if result is not None:
    print("Predicted distress risk category =", result["predicted_distress_risk_category"])
    print("Class probabilities =", result["class_probabilities"])

print("-" * 60)



# 3. CLUSTERING -- assign an e-commerce customer to an existing segment

clustering_payload = {
    "recency_days": 12,
    "frequency_per_year": 14,
    "avg_order_value_kes": 6200,
    "num_categories_purchased": 5,
}

result = call_endpoint("/api/v1/models/ecommerce-customer-segmenter/predictions", clustering_payload)
if result is not None:
    print("Assigned cluster =", result["assigned_cluster"],
          "(of", result["number_of_clusters"], "clusters)")

print("-" * 60)



# 4. ASSOCIATION RULE MINING -- recommend shows from a viewing session

recommendation_payload = {
    "watched": ["Space Frontier", "Time Anomaly"],
    "top_n": 3,
}

result = call_endpoint("/api/v1/models/streaming-show-recommender/predictions", recommendation_payload)
if result is not None:
    print("Watched =", result["watched"])
    print("Recommendations =", result["recommendations"])