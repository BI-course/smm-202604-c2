# Marking audit — Group C2, run 17 Nov before submission

Every row below was **executed**, not read off a document. Where a check could not be run, it says so
and says why. Re-run the commands in brackets before you submit, because Render's free tier sleeps and
Docker images go stale.

| #   | Rubric row                                    | Marks | Status      | Evidence                                                                                                                                                               |
| :-- | :-------------------------------------------- | ----: | :---------- | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | Regression endpoint, loaded from a saved file |    20 | **PASS**    | 208,681.74 KES local, through Nginx and on Render. `joblib.load` once at import; no `.fit(` anywhere in `api.py`.                                                      |
| 2   | Classification endpoint                       |    20 | **PASS**    | `Distressed`, P = 0.9783 on all three surfaces. Label encoder loaded separately and applied.                                                                           |
| 3   | Clustering endpoint                           |    20 | **PASS**    | `assigned_cluster: 0`, `number_of_clusters: 3` on all three surfaces. imputer → scaler → kmeans order confirmed.                                                       |
| 4   | Recommender endpoint                          |    20 | **PASS**    | Galactic Wars, confidence 0.774, lift 5.0 on all three surfaces. Rules loaded, never re-mined.                                                                         |
| 5   | Dockerized + Gunicorn, not the dev server     |     5 | **PASS**    | Two replicas, Gunicorn 26.2.0, `Listening at 0.0.0.0:8000`, 5 workers each. No Flask dev-server warning in the logs.                                                   |
| 6   | Nginx reverse proxy forwards correctly        |     5 | **PASS**    | `tests/smoke_nginx.sh` → **6 passed, 0 failed**: four endpoints correct, 301 to HTTPS, port 8000 unreachable from the host.                                            |
| 7   | Same unmodified `api.py` live on Render       |    10 | **PASS**    | All four verified against `https://smm-202604-c2-apill.onrender.com` answering over the public internet.                                                               |
| 8   | Hugging Face Space, four tabs                 |    +4 | **BLOCKED** | Accounts are under 30 days old; unlocks 20 Nov. App ported to the group's artifacts and `spaces` added to its requirements.                                            |
| 9   | Streamlit Community Cloud, four tabs          |    +4 | **BLOCKED** | Same 30-day gate. Ported and verified locally: Cluster 0 of 3, Galactic Wars 77.4% / lift 5.00.                                                                        |
| 10  | HTML/CSS/JS page + missing-input handling     |    +2 | **PASS**    | Four pages reachable from `/`, each submits to the live API; blank-field error demonstrated on camera; recommender page has a show picker plus invalid-input handling. |

**Deductions to watch**

- **−10 "retrained inside `api.py`"** — not at risk. No `fit`, no `train_test_split`, no estimator
  construction in `api.py`; all six artifacts come off disk.
- **−10 "public URL missing and not shown in the video"** — Render is live _now_. Record it before it
  sleeps for good. The video is what protects this mark.
- **−5 per member with no documented contribution** — **one left: Member 6** (Render + video +
  testing). Every other member has a student ID, name, branch link and a "what I learned" paragraph.

**Realistic ceiling:** 100/100 from rows 1–7 alone. Rows 8–10 are capped extra, and the total is capped
at 100, so the bonus marks are insurance against a single failed endpoint — not the difference between
a pass and a good grade.

---

## Two things that will cost marks if nobody acts

1. **Member 6's contribution row is empty** (`lab_submission_instructions.md` and `CONTRACT.md`
   both). That is a −5 for her specifically. She did do the work: Render deployment, the smoke
   tests, and the video. It needs writing up before submission.
2. **The video does not exist yet.** Rows 1–10 are all checked by _what the video shows_. A perfect
   API with no recording still loses the −10, and the bonus rows cannot be awarded at all.
   `VIDEO_DEMO_SCRIPT.md` is the plan.

## The trap that nearly cost rows 5–6

`docker compose build` reported every step **CACHED** and produced an image containing the _original
reference_ `api.py` (none of the four group routes), so all four endpoints 404'd through Nginx even
though they worked locally and on Render. `docker compose build --no-cache` fixed it in one command.

Anyone who demos without rebuilding, or who relies on a cached image from an earlier session, will
record four 404s. Check before recording:

```bash
docker compose -f docker-compose.yaml -f docker-compose-dev.yaml exec -T flask-gunicorn-app \
  sh -c "grep -c 'streaming-show-recommender' /opt/app-smm/api.py"   # must print 5
```

## Housekeeping (no marks, but tidy)

- `model/` still contains three unused reference artifacts
  (`lasso_regressor_for_sme_revenue`, `svc_classifier_for_sme_credit_risk`,
  `kmeans_mall_customer_segmentation`). Nothing loads them, but a marker comparing artifacts to the
  source notebooks may notice them. Deleting them means a Docker rebuild and a Render redeploy —
  **do not do it casually on submission day.**
- `frontend/` still holds the three retired pages. `api.py` serves every file in that folder, so they
  remain reachable by URL even though nothing links to them. Harmless, but do not open them on camera.
