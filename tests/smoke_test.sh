#!/usr/bin/env bash
# Usage: bash tests/smoke_test.sh http://127.0.0.1:5000
#        bash tests/smoke_test.sh https://<your-service>.onrender.com
BASE="${1:-http://127.0.0.1:5000}"
H="Content-Type: application/json"
echo "Testing $BASE"
echo; echo "1. Regression (expect 208681.74)"
curl -s -X POST "$BASE/api/v1/models/store-daily-sales-regressor/predictions" -H "$H" -d @tests/regression.json
echo; echo; echo "2. Classification (expect Distressed)"
curl -s -X POST "$BASE/api/v1/models/corporate-distress-classifier/predictions" -H "$H" -d @tests/classification.json
echo; echo; echo "3. Clustering (expect cluster 0)"
curl -s -X POST "$BASE/api/v1/models/ecommerce-customer-segmenter/predictions" -H "$H" -d @tests/clustering.json
echo; echo; echo "4. Recommender (expect Galactic Wars)"
curl -s -X POST "$BASE/api/v1/models/streaming-show-recommender/predictions" -H "$H" -d @tests/recommender.json
echo; echo; echo "5. Recommender, no match (expect empty list)"
curl -s -X POST "$BASE/api/v1/models/streaming-show-recommender/predictions" -H "$H" -d @tests/recommender_no_match.json
echo; echo; echo "6. Error handling, empty input (expect missing-fields error)"
curl -s -X POST "$BASE/api/v1/models/ecommerce-customer-segmenter/predictions" -H "$H" -d '{}'
echo
