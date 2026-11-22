# Render deployment — Group C2 (Member 6)

The group's **unmodified `api.py`** and the six saved artifacts in `model/` are deployed to Render
as a Web Service built directly from the `main` branch. Nothing is retrained on Render: the models are
loaded once at startup with `joblib.load`.

**Public URL:** https://smm-202604-c2-nicole.onrender.com

Verified: 22 Nov 2026 — all four endpoints return the same answers on Render as on the local Flask
server (`results_local.txt` vs `results_render.txt`).

---

## 1. Service configuration

| Setting               | Value                                                                                        |
|:----------------------|:---------------------------------------------------------------------------------------------|
| Service type          | Web Service (Free instance, 512 MB RAM)                                                      |
| Repository / branch   | `BI-course/smm-202604-c2` / `main`                                                           |
| Region                | Oregon (US West)                                                                             |
| Root directory        | *(blank — repository root, so `./model` resolves)*                                           |
| Runtime               | Python 3, `PYTHON_VERSION=3.12.10`                                                           |
| Build command         | `pip install -r requirements/base.txt -r requirements/prod.txt -c requirements/constraints.txt` |
| Start command         | `gunicorn -w 4 -b 0.0.0.0:$PORT api:app`                                                     |
| Health check path     | `/` (serves `frontend/index.html`)                                                           |

**Why the `-c requirements/constraints.txt` flag matters:** it pins `scikit-learn==1.9.0`,
`imbalanced-learn==0.14.2`, `joblib==1.6.0` and `mlxtend==0.25.0` — the versions the artifacts were
saved with. Without it Render may install a newer scikit-learn, and unpickling the saved models then
fails or warns (`InconsistentVersionWarning`).

**Why Gunicorn, not `python api.py`:** `api.py` ends with `app.run(debug=True)`, Flask's development
server. Render runs Gunicorn instead, importing `api:app` and binding to the port Render assigns in
`$PORT`. The same `api.py` file is therefore used unchanged.

A successful deploy log ends with:

```
All the four artifacts have been loaded. The API is ready.
[INFO] Listening at: http://0.0.0.0:10000
==> Your service is live
```

---

## 2. Endpoints

All prediction endpoints take `POST` with header `Content-Type: application/json`.
Request bodies are in `tests/payloads/`.

| # | Model                    | Method | Path                                                         | Demo body                         | Expected answer                          |
|:--|:-------------------------|:-------|:-------------------------------------------------------------|:----------------------------------|:-----------------------------------------|
| 1 | Regression (Ridge)       | POST   | `/api/v1/models/store-daily-sales-regressor/predictions`     | `tests/payloads/regressor.json`   | 208,681.74 KES per day                   |
| 2 | Classification (LogReg)  | POST   | `/api/v1/models/corporate-distress-classifier/predictions`   | `tests/payloads/classifier.json`  | Distressed, P = 0.9783                   |
| 3 | Clustering (K-Means)     | POST   | `/api/v1/models/ecommerce-customer-segmenter/predictions`    | `tests/payloads/segmenter.json`   | Cluster 0 of 3                           |
| 4 | Recommender (Apriori)    | POST   | `/api/v1/models/streaming-show-recommender/predictions`      | `tests/payloads/recommender.json` | Galactic Wars, confidence 0.774, lift 5.0 |
| – | Show catalogue           | GET    | `/api/v1/models/streaming-show-recommender/shows`            | –                                 | list of 15 shows                         |
| – | Demo web pages           | GET    | `/`                                                          | –                                 | HTML index with four model cards         |

---

## 3. Example requests and responses (from Render, 22 Nov 2026)

### 3.1 Regression

```bash
curl -s -X POST https://smm-202604-c2-nicole.onrender.com/api/v1/models/store-daily-sales-regressor/predictions \
  -H "Content-Type: application/json" -d @tests/payloads/regressor.json
```

Request body:

```json
{"store_size_sqm":180.0,"promotions_active":2,"avg_basket_size_kes":1350.0,"store_age_years":5.2,
 "online_order_count":45.0,"loyalty_program_members":60,"store_format":"Supermarket",
 "location_type":"Mall","delivery_available":"Yes","store_size_category":"Medium"}
```

Response (200):

```json
{"model":"Ridge Regressor","predicted_daily_sales_revenue_kes":208681.74}
```

### 3.2 Classification

```bash
curl -s -X POST https://smm-202604-c2-nicole.onrender.com/api/v1/models/corporate-distress-classifier/predictions \
  -H "Content-Type: application/json" -d @tests/payloads/classifier.json
```

Request body:

```json
{"industry_sector":"Retail","listing_status":"Private","credit_rating_band":"High Risk",
 "audit_going_concern_flag":null,"years_in_operation":3,"num_employees":12,
 "debt_to_equity_ratio":4.8,"current_ratio":0.6,"interest_coverage_ratio":0.9,
 "revenue_growth_pct":-18.0,"net_profit_margin_pct":-5.2,
 "working_capital_to_assets_ratio":-0.15,"macroeconomic_interest_rate_pct":12.5}
```

Response (200):

```json
{"class_probabilities":{"Distressed":0.9783,"Healthy":0.0,"Watch":0.0217},
 "model":"Logistic Regression (C=0.5, SMOTE-resampled at training only)",
 "predicted_distress_risk_category":"Distressed"}
```

### 3.3 Clustering

```bash
curl -s -X POST https://smm-202604-c2-nicole.onrender.com/api/v1/models/ecommerce-customer-segmenter/predictions \
  -H "Content-Type: application/json" -d @tests/payloads/segmenter.json
```

Request body:

```json
{"recency_days":12,"frequency_per_year":14,"avg_order_value_kes":6200,"num_categories_purchased":5}
```

Response (200):

```json
{"assigned_cluster":0,"model":"K-Means E-commerce Customer Segmentation","number_of_clusters":3}
```

### 3.4 Recommender

```bash
curl -s -X POST https://smm-202604-c2-nicole.onrender.com/api/v1/models/streaming-show-recommender/predictions \
  -H "Content-Type: application/json" -d @tests/payloads/recommender.json
```

Request body:

```json
{"watched":["Space Frontier","Time Anomaly"]}
```

Response (200):

```json
{"matched_rules":[{"confidence":0.774,"if_watched":["Space Frontier","Time Anomaly"],"lift":5.0,
  "then_recommend":["Galactic Wars"]}],
 "message":"1 recommendation(s) found.",
 "model":"Apriori association rules (11 cleaned rules), precomputed lookup",
 "recommendations":[{"because_you_watched":["Space Frontier","Time Anomaly"],"category":"Sci-Fi",
  "confidence":0.774,"lift":5.0,"show_id":"S07","show_name":"Galactic Wars","support":0.03}],
 "watched":["Space Frontier","Time Anomaly"]}
```

### 3.5 Error handling — missing input

```bash
curl -s -X POST https://smm-202604-c2-nicole.onrender.com/api/v1/models/ecommerce-customer-segmenter/predictions \
  -H "Content-Type: application/json" -d '{}'
```

Response (400):

```json
{"error":"Missing required customer features.",
 "missing_features":["recency_days","frequency_per_year","avg_order_value_kes","num_categories_purchased"]}
```

---

## 4. Evidence files

| File                                     | What it shows                                              |
|:-----------------------------------------|:-----------------------------------------------------------|
| `tests/results_local.txt`                | All five requests against `python api.py` on 127.0.0.1:5000 |
| `tests/results_render.txt`               | The same five requests against the Render URL — identical answers |
| `tests/smm-202604-c2-nicole.postman_collection.json` | Postman collection with the five saved requests      |
| `tests/evidence/render_*.png`            | Postman screenshots: URL, request body and response for each endpoint |

To re-run the Render check yourself:

```bash
BASE=https://smm-202604-c2-nicole.onrender.com
curl -s -X POST $BASE/api/v1/models/ecommerce-customer-segmenter/predictions \
  -H "Content-Type: application/json" -d @tests/payloads/segmenter.json
```

---

## 5. Cold start (free tier)

Render's free instances **sleep after 15 minutes without traffic**. The first request after that
wakes the service and can take **30–60 seconds** (the first Postman request on 22 Nov took 48 s);
every request after that answers in about 0.5–1 s. This is a hosting limit, not an API error.
Before a demo, send one request about a minute beforehand to wake the service.
