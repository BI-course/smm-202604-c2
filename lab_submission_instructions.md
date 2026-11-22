# Lab Submission Instructions

## Student Details and Individual Member Contributions

**Name of the team on GitHub Classroom:** smm-202604-c2 (Group C2) — repo: https://github.com/BI-course/smm-202604-c2

**Member 1:**

| **Details**                                                                                                                           | **Comment** |
|:--------------------------------------------------------------------------------------------------------------------------------------|:------------|
| **Student ID**                                                                                                                        | 155499      |
| **Name**                                                                                                                              | Benvictor Mwangi |
| **What part of the lab did you personally<br/>contribute to (provide a link to the<br/>branch(es)), and what did you learn from it?** | I built the retail daily-sales **regression** model and its endpoint — `save_member1_artifact.py` rebuilds and fits the Ridge pipeline (structural-missingness handling for `online_order_count`, `log1p` target) and saves it as `model/ridge_regressor_for_retail_sales.joblib`; the `POST /api/v1/models/store-daily-sales-regressor/predictions` route in `api.py` serves it and returns 208,681.74 KES for the demo row. As group lead I reviewed and merged the PRs for the classification, clustering and recommender routes, completed Table B, updated `frontend/api_consumer_from_dev_flask.py` to call all four new endpoints, and re-ran the full test suite on the merged `main`. Branch: https://github.com/BI-course/smm-202604-c2/tree/feature/regression-retail — What I learned: the pipeline is the deployment artifact; a JSON `null` silently makes a column dtype `object` so the imputer skips it (fixed by coercing to `float64` before `predict`), and branch protection + PRs were what let four people add routes to one `api.py` without overwriting each other. |

**Member 2:**

| **Details**                                                                                                                           | **Comment** |
|:--------------------------------------------------------------------------------------------------------------------------------------|:------------|
| **Student ID**                                                                                                                        | 168662 |
| **Name**                                                                                                                              | Lewa Kadzo |
| **What part of the lab did you personally<br/>contribute to (provide a link to the<br/>branch(es)), and what did you learn from it?** | I built the corporate financial distress **classification** model and its endpoint (`POST /api/v1/models/corporate-distress-classifier/predictions`), saved as `model/distress_risk_pipeline.joblib` with its label encoder and metadata; the route validates categories and returns the class probabilities. Branch: https://github.com/BI-course/smm-202604-c2/tree/member2-classification-endpoint — What I learned: training the classifier is only half the work: the API has to process inputs exactly as training did. A saved pipeline only carries the steps inside it, so the label encoder and the "Not Flagged" fill that lived outside it had to be handled in the route; a model must be saved with the same scikit-learn version the API runs; and unknown categories should be rejected instead of silently ignored. |

**Member 3:**

| **Details**                                                                                                                           | **Comment**  |
|:--------------------------------------------------------------------------------------------------------------------------------------|:-------------|
| **Student ID**                                                                                                                        | 169389       |
| **Name**                                                                                                                              | Juliet Mumbi |
| **What part of the lab did you personally<br/>contribute to (provide a link to the<br/>branch(es)), and what did you learn from it?** |  I worked on the K-Means clustering part of the lab. I prepared the e-commerce customer data by handling missing values and scaling the selected features before training a K-Means model to group customers into three clusters. I also saved the trained model and integrated it into the Flask API so that users could submit customer details and receive a cluster prediction. I tested the endpoint using cURL. Through this work, I learned how to build a clustering model, save and reuse model artifacts, integrate machine learning models into an API, and collaborate using Git branches and pull requests.            |

**Member 4:**

| **Details**                                                                                                                           | **Comment** |
|:--------------------------------------------------------------------------------------------------------------------------------------|:------------|
| **Student ID**                                                                                                                        | 154683 |
| **Name**                                                                                                                              | Whitney Karimi |
| **What part of the lab did you personally<br/>contribute to (provide a link to the<br/>branch(es)), and what did you learn from it?** | I built the Apriori streaming-show **recommender** endpoint: loaded the saved association rules (`model/apriori_recommendation_rules.joblib`) into `api.py` and built `POST /api/v1/models/streaming-show-recommender/predictions`, plus the recommender HTML page. Branch: https://github.com/BI-course/smm-202604-c2/tree/feature/apriori-streaming — What I learned: mining the association rules is only half the work: the API is what makes them usable. I learned to load saved rules from disk instead of re-mining them on every request, how to match a viewer's shows against rule antecedents, and why ranking by confidence and lift matters. |

**Member 5:**

| **Details**                                                                                                                           | **Comment** |
|:--------------------------------------------------------------------------------------------------------------------------------------|:------------|
| **Student ID**                                                                                                                        | 129086 |
| **Name**                                                                                                                              | Faith Njung'e |
| **What part of the lab did you personally<br/>contribute to (provide a link to the<br/>branch(es)), and what did you learn from it?** | I set up **Docker, Gunicorn and the Nginx reverse proxy**: generated the self-signed HTTPS certificate with OpenSSL and deployed it to Nginx, built and ran the stack with two Gunicorn replicas (5 workers each) behind Nginx with an HTTP-to-HTTPS redirect, verified all four models through Nginx against their expected answers, and wrote `tests/smoke_nginx.sh`, `tests/manual_requests.txt` and `DOCKER.md`. Branch: https://github.com/BI-course/smm-202604-c2/tree/feature/docker-gunicorn-nginx — What I learned: the same `api.py` behaves differently depending on what serves it. `api.py` and `model/` are copied into the image at build time, so a teammate's merged change needs a rebuild, whereas `nginx.conf` and the certificates are mounted volumes that only need an Nginx restart. Nginx also looks up the replicas only once when it starts, so it must be restarted after scaling. |

**Member 6:**

| **Details**                                                                                                                           | **Comment** |
|:--------------------------------------------------------------------------------------------------------------------------------------|:------------|
| **Student ID**                                                                                                                        | 167317 |
| **Name**                                                                                                                              | Nicole Mokua |
| **What part of the lab did you personally<br/>contribute to (provide a link to the<br/>branch(es)), and what did you learn from it?** | I deployed the group's unmodified `api.py` and the six saved artifacts to **Render** (https://smm-202604-c2-nicole.onrender.com) as a Web Service built from `main`, started with Gunicorn on `$PORT`, with scikit-learn pinned through `requirements/constraints.txt`. I tested all four public endpoints with curl and a Postman collection (plus the missing-input error case) and documented the request/response formats in `tests/RENDER.md`, with local-vs-Render results and screenshots in `tests/`. I deployed the same artifacts to a **Hugging Face Space** (Gradio, four tabs), verified the group's **Streamlit** app, coordinated end-to-end testing across all surfaces and assembled the demo video. Branch: https://github.com/BI-course/smm-202604-c2/tree/feature/render-deploy — What I learned: a pickled model only loads reliably when the server installs the same library versions it was saved with, which is why the constraints file matters more than the code; free-tier hosts sleep and the first request can take close to a minute; and the same artifacts must give identical answers on every surface, which is the real proof that nothing was retrained. |


---

## Chosen Level of Difficulty

**Specify the chosen level of difficulty** (baseline, intermediate, or advanced): **Advanced** (all four models served from saved artifacts, plus Docker/Gunicorn/Nginx, Render, HTML pages, Hugging Face Space and Streamlit Community Cloud)

## Video Demonstration

Submit the link to a short video (**not more than 10 minutes**) demonstrating your complete solution at whichever difficulty level you chose. Please ensure that the lecturer has rights to view the video.

At minimum, the video must show, in one continuous walkthrough:

1. `api.py` running **locally**, with a live request/response against each model endpoint you implemented.
2. If you attempted the Intermediate or Advanced tier: the same API running inside its **Gunicorn + Nginx Docker container**, reached through the reverse proxy, not the Flask development server.
3. If you attempted the Advanced tier: your **Hugging Face Space** and your **Streamlit Community Cloud app**, each opened live in a browser, with at least one prediction made on each tab, and your **Render**-hosted API answering a request from Postman or `curl`.

**Why this matters for grading:** free-tier public deployments are not guaranteed to still exist by the time this is graded. The video is treated as the authoritative record of what was actually working at submission time; a broken link at grading time is not penalized if the video clearly shows that deployment working.

Note that you are required to submit the link to the video and NOT the video itself. The video should NOT be uploaded to your repository—that would be a misuse of GitHub.

**Link to the video:** https://drive.google.com/file/d/1d6OEFAEgpzEk2CQ9-oW3vnCsPiT7z53r/view?usp=sharing

## Public URLs

Fill in only the rows that apply to your chosen difficulty level (see Part C below). Leave a row blank if your tier does not require that deployment target.

| **Deployment target**                             | **URL** |
|:--------------------------------------------------|:--------|
| Render (the full `api.py`, all endpoints)         | https://smm-202604-c2-nicole.onrender.com |
| Hugging Face Space (Gradio, one tab per model)    | https://huggingface.co/spaces/nimoks/smm-202604-c2-nicole |
| Streamlit Community Cloud app (one tab per model) | https://smm-202604-c2-lewa.streamlit.app |

---

## Scenario

Across earlier labs, **your group** was assigned its own dataset and trained your own models for four distinct tasks:

1. A **regression** task
2. A **classification** task
3. A **k-Means clustering** task
4. An **Apriori association-rule mining** task

Different groups worked with different datasets and, in some cases, different algorithms for each task -- there is no single shared reference repository or model list for the whole class. Wherever this document refers to "your regressor," "your classifier," "your clustering model," or "your recommender," it means **whichever model your own group already trained and saved for that task**, not a specific named algorithm.

You are now required to build a single Flask API (`api.py`) that serves **your own group's four models**, loaded from disk (never retrained inside `api.py`), and to demonstrate that API running correctly across a growing number of deployment surfaces depending on the difficulty tier you choose.

### Part A: Regression and Classification

- Locate the trained regression and classification model files your group produced and saved in your earlier labs on Regression and Classification.
- Update [api.py](api.py) to include end-points that load these two models from disk and serve predictions from them.

### Part B: Clustering and Association Rule Mining

- Locate the trained clustering model and the association rules your group produced and saved in your earlier labs on Clustering and Association Rule Mining.
- Update [api.py](api.py) to include end-points that:
  - Load the clustering model from disk and predict which cluster a new client belongs to
  - Load the association rules from disk and recommend products based on them

**Note 1:** **`api.py` is NOT production-grade as it is.** It is only meant for demonstration purposes. Scalability and security must be taken into consideration before deploying an API in a production environment.

**Note 2:** Some students often treat the API as an afterthought, focusing only on training ML models. In practice, **the API is the product** -- it is how others interact with your model. The "hidden" learning here is that the delivery mechanism (API design, usability, error handling, and even documentation) often matters more to stakeholders in the industry than the models themselves.

### Part C: Deployment Breadth (Read Before Choosing Your Tier)

All three tiers below ask you to serve **the same four models** (your regressor, your classifier, your clustering model, your recommender) from **one, unmodified `api.py`**. The model files and the Flask routes do not change between tiers -- only how many different places that one working API is shown to run correctly.

**Baseline (Required):**

- Update `api.py` to serve **at least three of your four** models, loaded from disk.
- Run and demonstrate `api.py` **locally** with Flask's development server, with a successful request against each implemented endpoint.

**Intermediate (Recommended):**

- Update `api.py` to serve **all four** of your models, loaded from disk.
- Dockerize your Flask API using **Gunicorn**, and place it behind an **Nginx** reverse proxy (see the reverse-proxy lab notes for the full Docker Compose setup).
- Deploy the same, unmodified `api.py` to **Render**, and demonstrate a real request against it from outside your own machine (cURL or Postman).

**Advanced (Optional):**

- Everything required for Intermediate, plus:
  - Create a web page (or pages) using Basic HTML, CSS, and Vanilla JavaScript that demonstrates the use of the API
  - Implement basic error handling (e.g., missing inputs)
  - Build **one Gradio app with a tab per model** (all four of your models) and publish it as a **Hugging Face Space**
  - Build **one Streamlit app with a tab per model**, covering the same four models, and publish it on **Streamlit Community Cloud**
  - All deployment targets -- local, Dockerized/Nginx, Render, Hugging Face, and Streamlit -- must serve models loaded **from the same trained artifacts**; retraining different models per platform does not satisfy this requirement

Why this is production-relevant beyond the model itself:

- **Portability:** the same container image and the same trained artifacts run unmodified on your laptop, inside Docker, and on a managed cloud host.
- **Consistency:** no "but it works on my machine" gap between what you demo locally and what a domain expert sees on a public URL.
- **Separation of concerns:** a REST API (Render), a no-code demo UI (Hugging Face/Streamlit), and your own frontend (the Advanced-tier HTML pages) are three different ways of exposing the *same* underlying models to three different kinds of user -- a developer integrating your API, a non-technical domain expert validating it, and an end customer.

---

## Marking Guide

Refer to the marking guide available here: [lab_submission_marking_guide.md](lab_submission_marking_guide.md)