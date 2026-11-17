# Demo video — Group C2, BBT 4206 Module 6 (Advanced tier)

> **Read this on a second screen/phone while you record. Do not read it on camera.**
> Total target **9:00**, hard stop **10:00**. One continuous take per "take" below.
> The lecturer's rule: **real request and real response on screen for every endpoint** — if the
> screen does not show it, it did not happen.

---

## 0. Why this is in two takes (decided: 17 Nov)

Hugging Face Spaces and Streamlit Community Cloud do not activate accounts younger than 30 days.
The accounts reach 30 days on the **20th**, so those two surfaces cannot be recorded yet.

The submission rules do not require one single file, but they do require that the lecturer can
**see the deployment working**. So:

| Take       | Recorded on     | Shows                                                     | Maps to requirement  |
| :--------- | :-------------- | :-------------------------------------------------------- | :------------------- |
| **Take A** | now → 19 Nov    | local `api.py`, HTML pages, Docker/Gunicorn/Nginx, Render | Requirements 1 and 2 |
| **Take B** | on/after 20 Nov | HF Space and Streamlit, same four models                  | Requirement 3        |

**Take B opens by saying it is the continuation of Take A** and shows the same demo row returning
the same number. That is what makes the two clips legible as one walkthrough.

**Do not** leave Take A until the 20th. Render's free tier sleeps and a free-tier URL can die;
the recording is your insurance. Take A alone already banks **90 of the 100 marks**.

---

## 1. Three things to fix before you press record

1. **`store_daily_sales_regressor.html` is the odd page out.** It is the only page that uses a
   relative path (`/api/...`). The other three build `hostname:5000` on HTTP and `hostname:443` on
   HTTPS, so they work on the Flask dev server but will **fail behind Nginx** (from
   `https://localhost` they call `https://localhost:443`, which Nginx is not listening on).
   Two options — pick one and write it down:
   - **Option 1 (recommended, 4 × 30 s):** make all four pages use the relative path, exactly as
     `store_daily_sales_regressor.html` does: `const API_URL = "/api/v1/models/<name>/predictions";`
     Then the pages work on :5000 **and** behind Nginx **and** behind Render, no edit per platform.
   - **Option 2:** demo the HTML pages only on the Flask dev server (take A, segment 2) and never
     open them through Nginx.
2. **`frontend/sme_revenue_regressor.html` is stale** — it still calls
   `/api/v1/models/sme-revenue-regressor/predictions`, a route that no longer exists (it is the old
   Lasso model). Do not open it on camera and do not link it. `index.html` no longer does.
   Same for `mall_customer_segmenter.html` and `sme_credit_risk_classifier.html` if they are still
   on disk — `api.py` serves **every** file under `frontend/` at `http://127.0.0.1:5000/<name>.html`,
   so an old page is still reachable even after it is unlinked.
3. **Decide the demo machine** (Table C is still `_TBC_`). It must be the machine that has Docker
   running _and_ the `.venv` that returns the numbers in §2. Every number below was verified on
   this machine on 17 Nov.

---

## 2. The four answers — memorise these, say them out loud

These are from `main` with the merge branch applied, run on 17 Nov, Python 3.14.3 / sklearn 1.9.0:

| Model                | Demo input                                                                                                                                           | Say this answer                                            |
| :------------------- | :--------------------------------------------------------------------------------------------------------------------------------------------------- | :--------------------------------------------------------- |
| Regression           | 180 sqm, 2 promos, 1350 basket, 5.2 yrs, 45 online orders, 60 loyalty, Supermarket/Mall/Yes/Medium                                                   | **KES 208,681.74 per day**                                 |
| Regression (2nd row) | 95 sqm, kiosk, `online_order_count: null`                                                                                                            | **KES 38,894.20**                                          |
| Classification       | Retail, Private, High Risk, going-concern `null`, 3 yrs, 12 emp, D/E 4.8, current 0.6, ICR 0.9, rev growth −18, NPM −5.2, WC/assets −0.15, rate 12.5 | **Distressed**, P = **0.9783** (Watch 0.0217, Healthy 0.0) |
| Clustering           | recency 12, frequency 14, AOV 6200, 5 categories                                                                                                     | **Cluster 0 of 3**                                         |
| Recommender          | watched Space Frontier + Time Anomaly                                                                                                                | **Galactic Wars**, confidence 0.774, lift 5.0              |

If a number on screen differs from this table, **stop**. Something was rebuilt or a file changed.
Fix it before recording, not after.

---

## 3. Pre-flight (30 min before, in order — do not skip)

```powershell
cd c:\Users\Catriona\Downloads\BIII\Module_6_Flask_ML_API

# 0. Build from scratch -- DO NOT trust cached layers (see the warning below)
docker compose -f docker-compose.yaml -f docker-compose-dev.yaml build --no-cache flask-gunicorn-app

# 1. Every answer correct, right now
..\.venv\Scripts\python.exe api.py                                    # terminal 1, leave running
..\.venv\Scripts\python.exe frontend\api_consumer_from_dev_flask.py   # terminal 2 -> all four
..\.venv\Scripts\python.exe test_member4_recommender.py               # 10/10

# 2. Docker stack up (terminal 3, Git Bash) — needs Docker Desktop running
docker compose -f docker-compose.yaml -f docker-compose-dev.yaml up -d --build --scale flask-gunicorn-app=2
docker compose -f docker-compose.yaml -f docker-compose-dev.yaml restart nginx
bash tests/smoke_nginx.sh

# 3. Render awake (free tier sleeps after 15 min)
$ProgressPreference='SilentlyContinue'
Invoke-WebRequest https://smm-202604-c2-apill.onrender.com/ -UseBasicParsing | Select-Object StatusCode
```

Then:

- [ ] `docker compose ps` shows **2** `flask-gunicorn-app` + `nginx`, all healthy
- [ ] `https://localhost/` loads (accept the self-signed warning once)
- [ ] Render answered once (so it is warm) — then let it sleep so the wake-up on camera is honest
- [ ] Tabs already open, in this order: `http://127.0.0.1:5000/`, `store_daily_sales_regressor.html`,
      `corporate_distress_classifier.html`, `ecommerce-customer-segmenter.html`,
      `ecommerce_recommender.html`, `https://localhost/`, Render dashboard, Render service log
- [ ] Postman open with the four saved requests (or the PowerShell snippets below in a file you paste from)
- [ ] Which browser/tool you will use for the Nginx requests, chosen: **Git Bash + `tests/manual_requests.txt`**
- [ ] Notifications off, 1080p, terminal font ≥ 16 pt, `-k` reminder for the self-signed cert
- [ ] Contribution table filled for all six (see §6) — a `TBC` costs 5 marks, and it is easier to fix now

---

## 4. TAKE A — record now (target 9:00)

### 0:00–0:40 · Intro (no code)

**On screen:** `model/` folder in Explorer, then the repo root (`api.py`, `Procfile`,
`docker-compose.yaml`, `docker-compose-dev.yaml`, `tests/`).

**Say:** _"BBT 4206 Module 6 — Group C2, Advanced tier. We serve our group's four trained models
through one Flask API: a Ridge regressor on retail daily sales, a logistic-regression distress
classifier, a K-Means customer segmenter, and Apriori association rules for show recommendations.
Everything you will see loads those saved files. Nothing is retrained anywhere."_

### 0:40–1:00 · "Loaded, not trained"

**On screen:** scroll `api.py` to the four `joblib.load(...)` lines (~lines 73–95).

**Say:** _"Four `joblib.load` calls, once at startup, plus one print line when they are ready. Not
per request — and never re-fitted."_ (This is the −10 "retrained inside `api.py`" penalty.)

### 1:00–5:00 · Local Flask — four endpoints, ~60 s each

**On screen:** press up-arrow on `python api.py` in terminal 1 so the _start_ is visible, keep the
log pane beside the terminal. For each request: show the body → send → show the response → **say
the expected number**.

Paste from these (they are already in `Module_6_Flask_ML_API`; do not type JSON live):

```powershell
# 1. Regression -> 208681.74
$b = Get-Content tests\payloads\regressor.json -Raw
Invoke-RestMethod -Method Post -ContentType "application/json" -Body $b `
  -Uri "http://127.0.0.1:5000/api/v1/models/store-daily-sales-regressor/predictions"

# 1b. Regression, structural missingness -> 38894.2
$b = '{"store_size_sqm":95.0,"promotions_active":0,"avg_basket_size_kes":900.0,"store_age_years":2.0,"online_order_count":null,"loyalty_program_members":25,"store_format":"Kiosk","location_type":"Street","delivery_available":"No","store_size_category":"Small"}'
Invoke-RestMethod -Method Post -ContentType "application/json" -Body $b `
  -Uri "http://127.0.0.1:5000/api/v1/models/store-daily-sales-regressor/predictions"

# 2. Classification -> Distressed, 0.9783
$b = Get-Content tests\payloads\classifier.json -Raw
Invoke-RestMethod -Method Post -ContentType "application/json" -Body $b `
  -Uri "http://127.0.0.1:5000/api/v1/models/corporate-distress-classifier/predictions"

# 3. Clustering -> cluster 0 of 3
$b = Get-Content tests\payloads\segmenter.json -Raw
Invoke-RestMethod -Method Post -ContentType "application/json" -Body $b `
  -Uri "http://127.0.0.1:5000/api/v1/models/ecommerce-customer-segmenter/predictions"

# 4. Recommender -> Galactic Wars, 0.774, lift 5.0
$b = Get-Content tests\payloads\recommender.json -Raw
Invoke-RestMethod -Method Post -ContentType "application/json" -Body $b `
  -Uri "http://127.0.0.1:5000/api/v1/models/streaming-show-recommender/predictions"
```

Say one sentence per model, in this order:

1. _"A fitted pipeline is the deployment artifact — it carries its own imputer, encoders and scaler.
   It was trained on log1p of revenue, so the route applies expm1 on the way out. Skipping that is
   the classic silent bug."_ → then 1b: _"A null online-order count is structural, not missing. The
   pipeline has a dedicated branch for that one column — its own imputer plus a missing-indicator —
   so the model can tell 'does not deliver' apart from 'delivers, but zero orders'. That branch is
   why a non-delivering kiosk comes out at 38,894."_

   **Do not say** "the imputer handles a NaN on any numeric column." It does not. `online_order_count`
   is the only column with that structural branch. Any *other* numeric field sent as `null` fails the
   same way whichever surface you use: JSON `null` makes pandas store the value as `None`, the column
   becomes dtype `object`, the imputer passes it through untouched, and Ridge raises
   `Input X contains NaN`. Verified 17 Nov on all four surfaces. `api.py` guards only `online_order_count`
   (every other numeric going through `to_numeric` can still yield NaN), so demo with the payloads in §2
   and do not improvise a null anywhere else.
2. _"The classifier was SMOTE-balanced at training only; at inference the resampler is a pass-through.
   The label encoder was saved separately, so the route turns 0/1/2 back into
   Distressed/Healthy/Watch."_
3. _"K-Means does not bundle its scaler. The route applies imputer → scaler → predict, in that order —
   the four objects the notebook saved together."_
4. _"No `predict()` here. The rules were precomputed into a lookup, and the route does the same
   match-and-rank a production recommender would. Confidence and lift order the results."_

**Close with** `frontend\api_consumer_from_dev_flask.py` printing all four (~15 s) — that is your
"tested fresh immediately before recording" evidence.

### 5:00–5:40 · HTML frontend (+2)

**On screen:** `http://127.0.0.1:5000/` → the four cards → open each card in turn.

1. Regression → submit → **KES 208,681.74**.
2. **Blank a required field and submit** → the error message appears on camera. _This is the
   explicitly marked "basic error handling for missing inputs". Do not skip it._
3. Clustering → submit → **Cluster 0 of 3**, and the matching segment row is highlighted. _These four
   cards, one form each, are our HTML/CSS/Vanilla-JS front end; they talk to the same API._

### 5:40–7:00 · Docker + Gunicorn + Nginx (10 marks)

All of this in **Git Bash** (the commands are bash).

```bash
docker compose -f docker-compose.yaml -f docker-compose-dev.yaml ps
docker compose -f docker-compose.yaml -f docker-compose-dev.yaml logs flask-gunicorn-app | head -20
bash tests/smoke_nginx.sh
```

> **If any endpoint 404s through Nginx, it is a stale image, not your code.** Verified on 17 Nov:
> `docker compose build` reported every step `CACHED` and the image shipped the **original reference**
> `api.py` (0 of the four group routes). After `build --no-cache` the same command gave 6/6. So before
> recording, either build with `--no-cache`, or prove the image is right with:
> ```bash
> docker compose ... exec -T flask-gunicorn-app sh -c "grep -c 'streaming-show-recommender' /opt/app-smm/api.py"
> # must print 5, not 0
> ```

1. `ps` → **2** `flask-gunicorn-app` replicas + `nginx`.
2. `logs` → show `Starting gunicorn` and `Listening at: http://0.0.0.0:8000`, and say: _"Flask's
   development server is not running anywhere in this stack — this is Gunicorn with 5 workers per
   replica."_
3. Open `docker-compose.yaml` and point at the `expose: "8000"` line: _"That port is on the internal
   Docker network only, never published to the host. `http://localhost:8000` refuses — Nginx is the
   only way in."_ (`smoke_nginx.sh` checks this for you: "Gunicorn must NOT be reachable directly".)
4. Send all four through `https://localhost` — accept the self-signed certificate warning **on
   camera** so nobody thinks it is being hidden. Responses must match §2 exactly.
5. One line on the proxy: _"`proxy_pass http://flask_backend;` has no trailing slash on purpose —
   adding one strips the `/api/` prefix, so Flask receives `/v1/models/...` instead of
   `/api/v1/models/...` and the route 404s."_ (The reasoning is written out in `nginx.conf` lines 63–77.)
6. 10 s: `http://localhost/` → 301 → `https://localhost/`.

### 7:00–8:00 · Render (10 marks)

**On screen:** the Render dashboard service, its URL, and the live log. Then terminal — run **one**
request, wake the service if it slept, and show the wake-up rather than hiding it:

```powershell
$ProgressPreference='SilentlyContinue'
$b = Get-Content tests\payloads\classifier.json -Raw
Invoke-RestMethod -Method Post -ContentType "application/json" -Body $b `
  -Uri "https://smm-202604-c2-apill.onrender.com/api/v1/models/corporate-distress-classifier/predictions"
```

**Say:** _"Same `api.py`, same model files, unmodified — Render builds it from `main` and starts it
with Gunicorn on `$PORT`. That took a moment because the free tier had gone to sleep; the first
request woke it."_ Then one more endpoint (10 s) so "all four reachable" is unambiguous — clustering
is the quickest. Mention `/` returns 404 on Render by design (it is an API, not a website).

**End of Take A.** Stop recording. Do not delete the deployments — Take B needs the same models.

---

## 5. TAKE B — record on or after 20 Nov (target 2:00)

### Before you record Take B — two Streamlit traps, both verified 17 Nov

1. **Root directory is `Module_6_Flask_ML_API`, not the outer `BIII` folder.** The Streamlit `app.py` loads from
   `./model`, and the *runner* must be started with the repo folder as the root directory. Deployed
   from `BIII` the app dies immediately with
   `FileNotFoundError: './model\\ridge_regressor_for_retail_sales.joblib'`. `Module_6_Flask_ML_API`
   contains `model/`, `requirements/` and the Streamlit folder — that is the repo to point at.
2. **Put the HF files at the repo root, not in a subfolder.** `huggingface-spaces-using-gradio/app.py`
   loads by bare filename (`MODEL_DIR = "model" if os.path.isdir("model") else "."`). On a Space the
   files must sit **flat** next to `app.py`: `app.py`, `requirements.txt`, and the six `.joblib` files.
   Do **not** nest them in a `model/` folder there.
3. **`requirements.txt` for the Space** must include `spaces` (the ZeroGPU import) alongside gradio,
   pandas, scikit-learn and joblib. The sample `app.py` calls a no-op `@spaces.GPU` function purely to
   pass the platform's startup check; if the build fails with
   *"No @spaces.GPU function detected during startup"*, use **Settings → Factory rebuild** before
   touching the code.

Sanity-check locally before deploying — this reproduced the exact answers on 17 Nov:

```powershell
cd c:\Users\Catriona\Downloads\BIII\Module_6_Flask_ML_API
..\.venv\Scripts\python.exe -m streamlit run streamlit-sharing-using-streamlit/app.py
# clustering tab -> "Assigned segment: Cluster 0 of 3"
# recommender tab -> "Galactic Wars (S07) — confidence 77.4%, lift 5.00, support 0.030"
```

Open with the sentence that stitches the two takes together:

> _"This continues the Group C2 demo. Same four saved model files you saw in the local API and the
> Docker stack — now on the two managed demo surfaces."_

1. **Hugging Face Space** — open the Space URL. Submit the clustering tab with 12 / 14 / 6200 / 5 →
   **Cluster 0 of 3**, _same answer as the API_. Then one more tab (recommender → Galactic Wars).
2. **Streamlit Community Cloud** — same, at least one tab, two if time. Clustering then recommender.
3. Close with: _"Same four `.joblib` files, the same answers, three different delivery surfaces — a
   REST API for developers, these no-code demos for domain experts, and the HTML pages for end users."_

**Upload order:** export both files as `smm-202604-c2-demo-partA` and `smm-202604-c2-demo-partB`,
upload both to the same place (unlisted, or "anyone with the link"), and put **both links** in the
submission, labelled. Put part A first.

---

## 6. Submission edits to make before recording the intro

- [ ] `lab_submission_instructions.md` → **Link to the video:** both links, labelled "Part A
      (local + Docker + Render)" and "Part B (HF Space + Streamlit, recorded after accounts unlocked)".
- [ ] **Public URLs** table → Render row: `https://smm-202604-c2-apill.onrender.com`; HF and
      Streamlit rows filled in on the 20th.
- [ ] Contribution table → **Member 6** row still empty (`-5`). Member 6 owns Render, the smoke
      tests and the video coordination — that is a real contribution, write it up.
- [ ] CONTRACT.md Table A → only Member 6's username is still `_TBC_`.

---

## 7. If time is running out, cut in this order

1. Streamlit second tab
2. HTML segment (but keep the **blank-field error** if you can — it is worth 2 marks)
3. Render second endpoint

**Never cut:** the four local endpoints, or the through-Nginx demo. Those are 90 of the 100.
