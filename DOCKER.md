# Running the API with Docker, Gunicorn and Nginx

*Owner: Member 5 (Docker / Gunicorn / Nginx), branch `feature/docker-gunicorn-nginx`*

## Architecture

```
Browser / curl / Postman
        │  https://localhost  (port 443; port 80 redirects to 443)
        ▼
┌──────────────────────┐
│ nginx                │  SSL termination, serves frontend/ at "/",
│ (servingmlmodels-    │  proxies /api/ to the app replicas
│  nginx:1.0.0)        │  (round-robin load balancing)
└─────────┬────────────┘
          │ http://flask-gunicorn-app:8000  (internal Docker network only)
   ┌──────┴───────┐
   ▼              ▼
┌────────┐   ┌────────┐
│ app-1  │   │ app-2  │   Gunicorn, 5 sync workers each,
│        │   │        │   running api:app (the same, unmodified api.py)
└────────┘   └────────┘   Models are loaded once from model/ at import time.
```

Port 8000 is `expose`d to the Docker network but **not published** to the host,
so the only way to reach the API is through Nginx.

## Prerequisites

- Docker Desktop running
- `.env` created from `env.example`
- `container-volumes/nginx/certs/selfsigned.crt` and `selfsigned.key` present
  (see the lecturer's Step 1.1; neither file is committed)

## Start the stack

```bash
docker compose -f docker-compose.yaml -f docker-compose-dev.yaml \
  up -d --build --scale flask-gunicorn-app=2

# Always restart Nginx AFTER scaling, so it resolves both replicas
docker compose -f docker-compose.yaml -f docker-compose-dev.yaml restart nginx

docker compose -f docker-compose.yaml -f docker-compose-dev.yaml ps
```

## Verify

```bash
bash tests/smoke_nginx.sh
```

This sends one request per model endpoint through `https://localhost`, then checks
that HTTP redirects to HTTPS and that port 8000 is not reachable directly.

To see that it's Gunicorn and not the Flask development server:

```bash
docker compose -f docker-compose.yaml -f docker-compose-dev.yaml top flask-gunicorn-app
docker compose -f docker-compose.yaml -f docker-compose-dev.yaml logs flask-gunicorn-app | head
```

The output should show a Gunicorn master and worker processes, `Starting gunicorn`, and
`Listening at: http://0.0.0.0:8000`. It should not contain
`WARNING: This is a development server`.

Load balancing: Nginx sends requests to `flask-gunicorn-app-1` and `flask-gunicorn-app-2` in turn.
Both replicas are listed by `ps` and `top` above, and Nginx is restarted after scaling so it knows about both.

## When to rebuild versus restart

| What changed | What to run |
|---|---|
| `api.py`, `model/`, `requirements/`, the Dockerfile | `up -d --build --scale flask-gunicorn-app=2`, then `restart nginx` |
| `nginx.conf` or the certificates (mounted volumes) | `restart nginx` only |
| Number of replicas | `up -d --scale flask-gunicorn-app=N`, then `restart nginx` |

`api.py` and `model/` are copied into the image at build time, so changes merged
from `main` don't take effect until you rebuild.

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `ReadTimeoutError ... files.pythonhosted.org` during build | Slow network, not a code problem. Rerun the same `up -d --build` command; layers that already finished are cached. |
| `No matching distribution found for numpy` | Usually also a network failure, not a real version problem. Rerun the build. |
| `502 Bad Gateway` from Nginx | The app containers are down or crashed. Check `logs flask-gunicorn-app`; usually a missing package in `requirements/base.txt` or a missing file in `model/`. |
| Only one replica ever gets traffic | Nginx started before the second replica. Run `restart nginx`. |
| The browser shows an old certificate or config | Run `restart nginx`. The files are mounted, but Nginx only reads them at start. |
| A frontend page fails behind Nginx but worked on Flask | The page calls `http://127.0.0.1:5000/...`. Use a relative `/api/...` path. |
