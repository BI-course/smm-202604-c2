# Member 1 notes - merging + checks for Member 6's part

Jotting this down so I don't lose track of where we are (10 Nov).

## Where main is at

Good news first: main already has all four models on it, merged in the order
Member 6 asked for (classification first, then clustering, then association):

- d7af007 - PR #2, Member 2's distress classifier endpoint
- 0d3e0dd - Member 3's e-commerce clustering
- 6d7cd92 - Member 4's Apriori streaming recommender

I also did some tidy-up commits on top of that: finished off the Table B rows in
CONTRACT.md, pointed the dev consumer at the four new endpoints, fixed the index
page cards so they hit the merged routes, and left handover notes for Member 6.

## About the sklearn thing Member 3 flagged

No action needed. I checked the clustering file on main - it's already named
kmeans_ecommerce_customer_segmentation.joblib (no timestamp, as the contract
says), and it opens fine in my 1.9.0 venv, so it was clearly re-saved after the
1.6.1 problem. The route also does imputer -> scaler -> kmeans.predict in the
right order, which is what the contract warned about.

## What I actually ran before signing off on this

Ran everything against my own venv (sklearn 1.9.0, Python 3.14):

- My quick 5-case smoke test: all passed, including the null
  online_order_count regression row (came back 38,894.20 KES, sensible for a
  small kiosk with no delivery).
- Member 4's test_member4_recommender.py: 10/10. Nice work Member 4, the
  lower-case/ID handling test was a good catch to include.
- The real thing: started api.py, ran
  python frontend\api_consumer_from_dev_flask.py against it. All four answers
  line up with Table B:
  - Regression: 208,681.74 KES
  - Classification: Distressed (P = 0.9783)
  - Clustering: cluster 0 of 3
  - Recommender: Galactic Wars, conf 0.774, lift 5.0

## Merging on GitHub

One thing I can't do from my terminal: the repo blocks direct pushes to main on
purpose (we tested that earlier, see commit a482091), so merges have to be
clicked on the website by me. I checked the PR list and the three merges above
are already in main, so there's nothing sitting in an open PR right now. If any
of you reopens one later, the order stands: classification, then clustering,
then association, and the test output above is your evidence to attach.

## For Member 6

You're unblocked. main is in the state you asked for - the test output above is
yours to reuse in your deploy notes. Ping me if Render complains about
anything, my first guess would be the Python version on Render (we're on
3.12.10 per .python-version).

## Small stuff still open (doesn't block the deploy)

- Table A usernames for members 2, 3, 5, 6 are still TBC - send them to me.
- Contribution table: most of us haven't put our student IDs/names in yet
  (Member 4 did hers - thank you). Worth marks, don't forget.
- We still haven't picked the demo machine for the video.

---

Chat message I'll post (draft, tweak before sending):

Hey team. Finally sat down and went through everything Member 6 raised. Short
version: we're good, she can start.

The three merges went in the order she wanted - Member 2's classifier first,
then Member 3's clustering, then Member 4's recommender - and main has all four
models now. Member 3, the sklearn issue you flagged: I checked and your file on
main is already named kmeans_ecommerce_customer_segmentation.joblib with no
timestamp, and it loads fine in my 1.9.0 setup, so whoever re-saved it got there
before me. Nothing to redo.

I ran the full check myself before posting this: my smoke test across all four
endpoints passed, Member 4's test file went 10/10 (nice one, especially the
case-insensitive show test), and python frontend\api_consumer_from_dev_flask.py
against a live server gave all four Table B answers:

- Regression: 208,681.74 KES
- Classification: Distressed (P = 0.9783)
- Clustering: cluster 0 of 3
- Recommender: Galactic Wars, confidence 0.774, lift 5.0

Those numbers are the proof Member 6 asked for - feel free to copy them into
your deploy notes.

One favour from me: Table A usernames for 2, 3, 5 and 6 are still TBC, and most
of us haven't filled our student ID and name into the contribution table (only
Member 4 has). It costs us marks per empty row, so send them over please. Also
we still need to pick the demo machine for the video.

Member 6 - go ahead, main is ready for you.
