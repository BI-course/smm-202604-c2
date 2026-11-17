#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Smoke test for the Dockerised API, reached ONLY through the Nginx reverse proxy.
# Author: Member 5 (Docker / Gunicorn / Nginx)
#
# Proves CONTRACT.md Phase 3: a request to https://localhost/api/... returns the
# same answer as the Flask dev server, i.e. the Table B expected output.
#
# Run from the repository root, with the stack already up:
#   bash tests/smoke_nginx.sh                  # defaults to https://localhost
#   bash tests/smoke_nginx.sh https://127.0.0.1
#
# Demo inputs (from CONTRACT.md Table B) live in tests/payloads/<name>.json.
# -----------------------------------------------------------------------------
set -u
BASE="${1:-https://localhost}"
PAYLOADS="tests/payloads"
HDR="$(mktemp)"; BODY="$(mktemp)"
trap 'rm -f "$HDR" "$BODY"' EXIT
pass=0; fail=0; skip=0

ok()  { echo "  PASS  $*"; pass=$((pass+1)); }
bad() { echo "  FAIL  $*"; fail=$((fail+1)); }

# The clustering route is declared over several lines in api.py, so its path is
# read from api.py instead of being hard-coded. Override with CLUSTER_PATH=...
CLUSTER_PATH="${CLUSTER_PATH:-$(grep -A3 '@app.route($' api.py 2>/dev/null \
  | grep -o '/api/v1/models/[^"]*' | head -1)}"

check_model() {   # $1 label, $2 URL path, $3 payload file, $4 expected (regex), $5 expected (readable)
  local label="$1" path="$2" file="$PAYLOADS/$3" expect="$4" readable="$5"
  if [ -z "$path" ]; then echo "  SKIP  $label  (route path not found)"; skip=$((skip+1)); return; fi
  if [ ! -s "$file" ]; then echo "  SKIP  $label  (no payload $file)"; skip=$((skip+1)); return; fi
  local code server
  code=$(curl -sk -D "$HDR" -o "$BODY" -w "%{http_code}" \
           -X POST "$BASE$path" -H "Content-Type: application/json" --data @"$file")
  server=$(grep -i '^server:' "$HDR" | tr -d '\r' | cut -d' ' -f2-)
  if [ "$code" != "200" ]; then
    bad "$label  [HTTP $code]  $path"
  elif ! echo "$server" | grep -qi nginx; then
    bad "$label  [Server: ${server:-none}; request did not go through Nginx]"
  elif ! grep -Eq "$expect" "$BODY"; then
    bad "$label  [HTTP 200 via Nginx, but expected $readable]"
  else
    ok "$label  [HTTP 200, Server: $server, got $readable]"
  fi
  echo "        -> $(head -c 400 "$BODY")"
}

echo "== Model endpoints through Nginx ($BASE), checked against CONTRACT.md Table B =="
check_model "Regression    " /api/v1/models/store-daily-sales-regressor/predictions \
  regressor.json   '208681\.74'                         "208,681.74 KES"
check_model "Classification" /api/v1/models/corporate-distress-classifier/predictions \
  classifier.json  '"Distressed"'                       "Distressed"
check_model "Clustering    " "$CLUSTER_PATH" \
  segmenter.json   '"assigned_cluster": ?0[,} ]'         "cluster 0"
check_model "Recommender   " /api/v1/models/streaming-show-recommender/predictions \
  recommender.json 'Galactic Wars'                      "Galactic Wars"

echo "== Reverse-proxy behaviour =="
# 1. Plain HTTP on port 80 must redirect to HTTPS.
redir=$(curl -s -o /dev/null -w "%{http_code} %{redirect_url}" http://localhost/)
case "$redir" in
  30[1278]\ https://*) ok "http://localhost/ redirects to HTTPS  [$redir]" ;;
  *)                   bad "http://localhost/ did not redirect to HTTPS  [$redir]" ;;
esac

# 2. Gunicorn's port must NOT be reachable from the host; Nginx is the only way in.
direct=$(curl -s -o /dev/null --max-time 3 -w "%{http_code}" http://localhost:8000/ 2>/dev/null)
if [ "$direct" = "000" ]; then
  ok "Port 8000 (Gunicorn) is not exposed to the host"
else
  bad "Port 8000 answered with HTTP $direct; the app is reachable without Nginx"
fi

echo "== Summary: $pass passed, $fail failed, $skip skipped =="
[ "$fail" -eq 0 ] && [ "$skip" -eq 0 ]
