# Group C2 — Lab Contract (Serving ML Models)

> **This file is the single source of truth for this lab.** If your work disagrees with this file, this file
> wins. If you need to change something in it, message the group lead first, then the lead edits this file.

Repo: `https://github.com/BI-course/smm-202604-c2` · Lead: Member 1 · Tier: **Advanced**

---

## Table A — Owners and branches

| Member | Model / task                                | Branch                            | GitHub username |
| :----- | :------------------------------------------ | :-------------------------------- | :-------------- |
| 1      | Retail daily-sales regression               | `feature/regression-retail`       | _(lead)_        |
| 2      | Corporate financial distress classification | `feature/classification-distress` | _TBC_           |
| 3      | E-commerce customer clustering              | `feature/clustering-ecommerce`    | _TBC_           |
| 4      | Streaming show recommender (Apriori)        | `feature/apriori-streaming`       | _TBC_           |
| 5      | Docker + Gunicorn + Nginx                   | `feature/docker-gunicorn-nginx`   | _TBC_           |
| 6      | Render deployment + video                   | `feature/render-deploy`           | _TBC_           |

---

## Table B — Artifact contract

Fill your own row. **Every row must be complete before Phase 2 (API work) begins.**

| Model          | File in `model/`                          | Loaded as         | Demo input               | Expected output    |
| :------------- | :---------------------------------------- | :---------------- | :----------------------- | :----------------- |
| Regression     | `ridge_regressor_for_retail_sales.joblib` | fitted `Pipeline` | 10 retail fields (below) | **208,681.74 KES** |
| Classification | _TBC_                                     | _TBC_             | _TBC_                    | _TBC_              |
| Clustering     | _TBC_                                     | _TBC_             | _TBC_                    | _TBC_              |
| Recommender    | _TBC_                                     | _TBC_             | _TBC_                    | _TBC_              |

### Regression (complete — reference row)

Input keys, exactly these ten, no more required:
`store_size_sqm, promotions_active, avg_basket_size_kes, store_age_years, online_order_count,
loyalty_program_members, store_format, location_type, delivery_available, store_size_category`

Valid categorical values: `store_format` ∈ {Convenience, Hypermarket, Kiosk, Supermarket} ·
`location_type` ∈ {CBD, Mall, Residential, Street} · `delivery_available` ∈ {Yes, No} ·
`store_size_category` ∈ {Small, Medium, Large}

Demo row 1 (delivery store): `180.0, 2, 1350.0, 5.2, 45.0, 60, Supermarket, Mall, Yes, Medium`
Demo row 2 (structural missingness): `95.0, 0, 900.0, 2.0, null, 25, Kiosk, Street, No, Small`

---

## Table C — Decisions

| Decision             | Value                                                                                       |
| :------------------- | :------------------------------------------------------------------------------------------ |
| Tier                 | **Advanced** (all 4 models + Docker/Nginx + Render + HF Space + Streamlit + HTML pages +10) |
| Framework            | **Flask** (`api.py`). FastAPI is a later, optional extra — not part of this submission      |
| Model filenames      | stable, lowercase, no timestamps                                                            |
| Response field names | agreed per Table B; nobody renames after Phase 2 starts                                     |
| Demo machine         | _TBC — one person, one machine, used for Docker and the video_                              |

**Why Flask and not FastAPI:** the marking guide never mentions FastAPI, and `requirements/`, the Dockerfiles,
the `Procfile`, all four frontend pages and the Gradio/Streamlit apps are all built around the Flask app object.
FastAPI is an alternative _track_, not an extra requirement.

---

## Table D — Rules of engagement

1. **`main` is never pushed to directly.** Every change arrives by Pull Request and is merged by the lead.
2. **Stay in your lane.** You may edit only:
   - your own model file in `Module_6_Flask_ML_API/model/`
   - your own `joblib.load(...)` line
   - your own route function
   - your own docstring / cURL example
     **Off-limits:** the import block at the top, the other three routes, `if __name__ == "__main__"`.
3. **Always `git pull origin main` before you start work**, then switch to your branch.
4. **Test before you push.** If it does not return a real answer on your machine, it does not get pushed.
5. **Paste your test output in the group chat** when you open a PR — the lead merges on evidence, not on "done".
6. **No timestamped filenames.** Copy the file into `model/` under the name in Table B.
7. **Never commit** `.env`, `.venv`, certificates, or notebooks you did not write.
8. **Model artifacts freeze** once all four are on `main`. After that, changing a model means rebuilding Docker
   and redeploying Render — ask the lead first.

### Read this before you save your artifact (Members 2 & 3)

Your notebook's own save cell will produce an artifact the API **cannot fully use**. This is not a mistake you
made — it is how the notebooks were written. Fix it when you re-save:

**Member 2 — the label encoder lives outside the pipeline.**
`Classification.ipynb` fits `LabelEncoder()` separately and saves only the pipeline, so the file predicts
`0 / 1 / 2` instead of `Healthy / Watch / Distressed`. Save **both** objects:

```python
import joblib
joblib.dump(
    {"pipeline": final_pipeline, "target_encoder": target_encoder},
    "distress_classifier_for_group_c2.joblib",
)
```

**Member 3 — the clusterer needs four objects, in order.**
Save all four, exactly as the notebook builds them:

```python
joblib.dump(
    {"imputer": imputer, "scaler": scaler, "kmeans": kmeans_final,
     "selected_features": selected_features},
    "kmeans_ecommerce_customer_segmentation.joblib",
)
```

The API route must then apply them **in this order**: `imputer.transform` → `scaler.transform` →
`kmeans.predict`. Miss the imputer and predictions will be wrong or will throw.

**Member 4 — the lookup key is `item_lookup`, not `sku_lookup`.**
Save `{"cleaned_rules", "item_lookup", "recommendation_lookup"}` as
`apriori_recommendation_rules.joblib`, and note that the API's request uses **show names**, not IDs.

**Member 1 — the regression pipeline is complete as saved.** It already contains preprocessing and the
`log1p` target transform, so the API applies `np.expm1()` on the way out. Name it
`ridge_regressor_for_retail_sales.joblib`.

---

## What "done" means for each phase

| Phase            | Done when                                                                                             | Blocked by                            |
| :--------------- | :---------------------------------------------------------------------------------------------------- | :------------------------------------ |
| 1 — Artifacts    | All four files committed to `main`, `git ls-files model/` shows them, each returns its Table B output | Everyone re-saving their own artifact |
| 2 — API          | `python frontend\api_consumer_from_dev_flask.py` prints four correct answers                          | Phase 1                               |
| 3 — Docker/Nginx | A request to `https://localhost/api/...` returns the same number as `http://127.0.0.1:5000`           | Phase 2                               |
| 4 — Render       | The public URL answers a request from a different network                                             | Phase 2 + `model/` committed          |
| 5 — Bonus        | HF Space and Streamlit each show four working tabs; HTML page rejects blank input                     | Phase 2                               |
| 6 — Submission   | Contribution table has six rows with branch links; video shows every surface                          | All of the above                      |

---

## Contribution table (copy into the submission)

Each member fills their own row. **A missing row or branch link is −5 marks for that member.**

| Member | Student ID | Name | Contribution                                                       | Branch link                                                                 | What I learned |
| :----- | :--------- | :--- | :----------------------------------------------------------------- | :-------------------------------------------------------------------------- | :------------- |
| 1      |            |      | Retail regression model + API integration (merged all four routes) | `https://github.com/BI-course/smm-202604-c2/tree/feature/regression-retail` |                |
| 2      |            |      | Corporate distress classifier + route                              | `.../feature/classification-distress`                                       |                |
| 3      |            |      | E-commerce clustering + route                                      | `.../feature/clustering-ecommerce`                                          |                |
| 4      |            |      | Apriori recommender + route                                        | `.../feature/apriori-streaming`                                             |                |
| 5      |            |      | Docker, Gunicorn, Nginx reverse proxy                              | `.../feature/docker-gunicorn-nginx`                                         |                |
| 6      |            |      | Render deployment, testing, video                                  | `.../feature/render-deploy`                                                 |                |
