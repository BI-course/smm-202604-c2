# Member 1 (lead) — what is done, and what only you can do

Written for the handover to Member 6 (Render). Everything below was run on this machine
with the project venv (Python 3.14.3, scikit-learn 1.9.0, pandas 3.0.5, numpy 2.5.3,
imbalanced-learn 0.14.2, mlxtend 0.25.0).

---

## 1. Done — merge order followed, `main` is now complete

| Step | What happened                                                                                                                                              | Commit on `main` |
| :--- | :--------------------------------------------------------------------------------------------------------------------------------------------------------- | :--------------- |
| 1    | PR #2 (Member 2, classification) reviewed and merged first                                                                                                 | `d7af007`        |
| 2    | Member 3 (clustering) merged — artifact re-saved for sklearn 1.9.0, renamed to `kmeans_ecommerce_customer_segmentation.joblib`, `joblib.load` line updated | `0d3e0dd`        |
| 3    | Member 4 (Apriori) merged — no conflicts; all four routes kept                                                                                             | `6d7cd92`        |
| 4    | Table B completed (classification / clustering / recommender rows) + dev consumer fixed                                                                    | `5dc9019`        |
| 5    | `frontend/index.html` cards point at the four real endpoints                                                                                               | `2f9ebb8`        |

`git ls-files model/` on `main` now contains all four artifacts:

```
model/ridge_regressor_for_retail_sales.joblib
model/distress_risk_pipeline.joblib
model/distress_label_encoder.joblib
model/distress_model_metadata.joblib
model/kmeans_ecommerce_customer_segmentation.joblib
model/apriori_recommendation_rules.joblib
```

(The old `svc_classifier_for_sme_credit_risk.joblib`, `kmeans_mall_customer_segmentation.joblib`
and `lasso_regressor_for_sme_revenue.joblib` are still in `model/` but nothing loads them any more.)

## 2. Done — the four answers, verified end to end

`python api.py` in one terminal, then `python frontend\api_consumer_from_dev_flask.py`:

```
Predicted daily sales revenue (KES) = 208681.74
Predicted daily sales revenue, no-delivery store (KES) = 38894.2
------------------------------------------------------------
Predicted distress risk category = Distressed
Class probabilities = {'Distressed': 0.9783, 'Healthy': 0.0, 'Watch': 0.0217}
------------------------------------------------------------
Assigned cluster = 0 (of 3 clusters)
------------------------------------------------------------
Watched = ['Space Frontier', 'Time Anomaly']
Recommendations = [{'because_you_watched': ['Space Frontier', 'Time Anomaly'], 'category': 'Sci-Fi',
                    'confidence': 0.774, 'lift': 5.0, 'show_id': 'S07', 'show_name': 'Galactic Wars',
                    'support': 0.03}]
```

`python test_member4_recommender.py` → **10/10 tests passed**.

These are the numbers Member 6 should see on Render. They are also in `CONTRACT.md` Table B.

---

## 3. Things I could not do from here (and what to do)

### 3.1 Close the Pull Requests on GitHub (2 minutes, browser only)

I merged the branches locally and pushed to `main`, so the code is correct — but GitHub still
shows the PRs as open. Do this so the history reads properly:

1. Open <https://github.com/BI-course/smm-202604-c2/pulls>.
2. For each open PR (Member 2's, Member 3's, Member 4's): open it, check the green
   _"This branch has no conflicts with the base branch"_ / _"Merged"_ banner, then click
   **Merge pull request → Confirm merge**.
   If GitHub says _"already merged"_ or the button is greyed out, just click **Close pull request**
   and add the comment: _"Merged into main by the lead as `<commit sha>`."_
3. Paste the test output from section 2 into the group chat as the merge evidence (CONTRACT rule 5).

### 3.2 Render deployment (Member 6's job — this is the guide she needs)

Render builds whatever is on `main`, so nothing else has to be pushed.

1. **New → Web Service → connect the GitHub repo** `BI-course/smm-202604-c2`.
2. Settings:
   - **Branch:** `main`
   - **Runtime:** Python 3
   - **Build command:** `pip install -r requirements/prod.txt -c requirements/constraints.txt`
   - **Start command:** `gunicorn -w 4 -b 0.0.0.0:$PORT api:app`
     (this is exactly what `Procfile` says, so leaving the start command blank also works)
   - **Health check path:** `/`
3. **Environment variable:** `PYTHON_VERSION = 3.12.10` (matches `runtime.txt` / `.python-version`).
4. Deploy, then check the log for `All the four artifacts have been loaded. The API is ready.`
   If a model fails to load, the log names the file — that means the artifact was saved with a
   different scikit-learn version than the one Render installed.
5. Test the public URL with the four requests in section 4 and compare with section 2.

> **Watch out:** `requirements/prod.txt` pulls `base.txt` from the course GitHub URL, which does
> **not** pin scikit-learn. `requirements/constraints.txt` pins `scikit-learn==1.9.0`,
> `imbalanced-learn==0.14.2`, `joblib==1.6.0`, `mlxtend==0.25.0` — that is why the `-c` flag is in
> the build command. Without it Render may install a different scikit-learn and the clustering
> endpoint will 500 again.

### 3.3 The HTML demo pages (Member 5 / bonus marks)

`frontend/index.html` now links to the right endpoints, but three of the four pages still post the
**old** payloads, so they will show an error against the new API:

| Page                               | Status                                                                          |
| :--------------------------------- | :------------------------------------------------------------------------------ |
| `store_daily_sales_regressor.html` | ✅ correct (Member 1's model)                                                   |
| `sme_credit_risk_classifier.html`  | ❌ posts the old SME loan fields → needs the 13 distress fields                 |
| `mall_customer_segmenter.html`     | ❌ posts `Age / Annual Income / Spending Score` → needs the 4 e-commerce fields |
| `ecommerce_recommender.html`       | ❌ posts `{"cart": [...]}` → needs `{"watched": [...]}`                         |

The exact field lists are in `CONTRACT.md` Table B (sections _Classification_, _Clustering_,
_Recommender_). Whoever owns the frontend pages should copy the payloads from
`frontend/api_consumer_from_dev_flask.py`, which is already correct.

### 3.4 Two small clean-ups for Members 2–4 (their own lanes, not mine)

- **Member 2:** the route is `/api/v1/models/corporate-distress-classifier/predictions`, but the
  branch is `member2-classification-endpoint` (not `feature/classification-distress` as Table A
  says). Either rename the branch or update Table A — the contribution table now links to the
  branch that actually exists.
- **Member 3:** the clustering route emits a harmless warning
  `X does not have valid feature names, but StandardScaler was fitted with feature names`
  because `imputer.transform()` returns a numpy array. It does not change the answer; if she wants
  it gone, wrap the imputer output back into a DataFrame with `clustering_features` as columns.
- **Member 4:** `api.py` no longer ends with a newline (her branch removed it). Cosmetic only.

---

## 4. The four requests to run against the Render URL

Replace `https://<your-app>.onrender.com` with the real URL.

```powershell
# 1. Regression -> 208681.74
$body = @{store_size_sqm=180.0; promotions_active=2; avg_basket_size_kes=1350.0;
          store_age_years=5.2; online_order_count=45.0; loyalty_program_members=60;
          store_format="Supermarket"; location_type="Mall";
          delivery_available="Yes"; store_size_category="Medium"} | ConvertTo-Json
Invoke-RestMethod -Method Post -ContentType "application/json" `
  -Uri "https://<your-app>.onrender.com/api/v1/models/store-daily-sales-regressor/predictions" -Body $body

# 2. Classification -> Distressed (0.9783)
$body = @{industry_sector="Retail"; listing_status="Private"; credit_rating_band="High Risk";
          audit_going_concern_flag=$null; years_in_operation=3; num_employees=12;
          debt_to_equity_ratio=4.8; current_ratio=0.6; interest_coverage_ratio=0.9;
          revenue_growth_pct=-18.0; net_profit_margin_pct=-5.2;
          working_capital_to_assets_ratio=-0.15; macroeconomic_interest_rate_pct=12.5} | ConvertTo-Json
Invoke-RestMethod -Method Post -ContentType "application/json" `
  -Uri "https://<your-app>.onrender.com/api/v1/models/corporate-distress-classifier/predictions" -Body $body

# 3. Clustering -> assigned_cluster 0
$body = @{recency_days=12; frequency_per_year=14; avg_order_value_kes=6200;
          num_categories_purchased=5} | ConvertTo-Json
Invoke-RestMethod -Method Post -ContentType "application/json" `
  -Uri "https://<your-app>.onrender.com/api/v1/models/ecommerce-customer-segmenter/predictions" -Body $body

# 4. Recommender -> Galactic Wars (0.774, 5.0)
$body = @{watched=@("Space Frontier","Time Anomaly"); top_n=3} | ConvertTo-Json
Invoke-RestMethod -Method Post -ContentType "application/json" `
  -Uri "https://<your-app>.onrender.com/api/v1/models/streaming-show-recommender/predictions" -Body $body
```

---

## 5. Local commands (for the video)

```powershell
cd c:\Users\Catriona\Downloads\BIII\Module_6_Flask_ML_API
..\.venv\Scripts\python.exe api.py                                   # terminal 1
..\.venv\Scripts\python.exe frontend\api_consumer_from_dev_flask.py  # terminal 2
..\.venv\Scripts\python.exe test_member4_recommender.py              # optional: 10/10
```

Then open <http://127.0.0.1:5000/> for the four cards.
