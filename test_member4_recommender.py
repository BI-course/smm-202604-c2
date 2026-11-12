"""
test_member4_recommender.py -- Member 4: tests for the Apriori streaming-show recommender.

Run from the project root (same folder as api.py):
    python test_member4_recommender.py

Imports the real api.py, so all four model files must be in ./model.
Covers: rules that match, cases with no recommendation, and bad input.
"""
from api import app

client = app.test_client()
URL = "/api/v1/models/streaming-show-recommender/predictions"

# (description, JSON body, expected HTTP status, expect recommendations?)
cases = [
    ("Rule matches -- 2-show rule",                {"watched": ["Space Frontier", "Time Anomaly"]},      200, True),
    ("Rule matches -- 2-show rule",                {"watched": ["Crime City", "Silent Witness Redux"]},  200, True),
    ("Rule matches -- 1-show rule",                {"watched": ["Love Island Nairobi"]},                 200, True),
    ("Rule matches -- lower case + show ID",       {"watched": ["laugh track", "S02"]},                  200, True),
    ("No rule matches",                            {"watched": ["Crime City"]},                          200, False),
    ("No rule matches",                            {"watched": ["Cooking Masters"]},                     200, False),
    ("Rule matches but show already watched",      {"watched": ["Laugh Track", "Office Antics"]},        200, False),
    ("Error -- missing 'watched'",                 {},                                                   400, None),
    ("Error -- show not in catalogue",             {"watched": ["Game of Thrones"]},                     400, None),
    ("Error -- invalid top_n",                     {"watched": ["Laugh Track"], "top_n": "abc"},         400, None),
]

passed = 0
for description, body, want_status, want_recs in cases:
    response = client.post(URL, json=body)
    data = response.get_json()
    ok = response.status_code == want_status
    if want_recs is not None:
        ok = ok and (len(data["recommendations"]) > 0) == want_recs
    passed += ok

    print(f"[{'PASS' if ok else 'FAIL'}] {description}  ->  HTTP {response.status_code}")
    print(f"    request : {body}")
    if response.status_code == 200:
        recs = [f"{r['show_name']} (conf={r['confidence']}, lift={r['lift']})" for r in data["recommendations"]]
        print(f"    result  : {', '.join(recs) if recs else 'no recommendation'} | {data['message']}")
    else:
        print(f"    error   : {data['error']}")

print(f"\n{passed}/{len(cases)} tests passed")
