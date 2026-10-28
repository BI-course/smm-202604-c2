# Member 1 -> Member 2 handover notes

Working branch: `feature/regression-retail`
Everything below assumes you have pulled the latest on that branch, or that you
branched off it freshly.

## What already works

`api.py` starts cleanly and all four demo routes answer. Regression has been
switched over to our group's own model:

- loads `model/ridge_regressor_for_retail_sales.joblib`
- endpoint: `POST /api/v1/models/store-daily-sales-regressor/predictions`
- returns the prediction in KES, e.g. 208,681.74 for a Mall supermarket with
  delivery, 38,894.20 for a Kiosk with no delivery
- the other three routes still serve the reference artifacts untouched, so
  nothing Member 2/3/4 needs has been broken or pre-empted

Test it yourself before writing any code:

    python api.py            (terminal 1)
    python frontend\api_consumer_from_dev_flask.py    (terminal 2)

You should get all four sections printing. Save that output, it doubles as the
"screenshots for the report" evidence.

## Things I hit that you will also hit

1.  **JSON nulls turn columns into dtype `object`.** The pipeline's imputer
    silently skips object columns, and then Ridge blows up with an unrelated
    NaN error. Look at how `predict_daily_sales()` coerces every numeric field
    to float64 before predict() and do the same in your route.
2.  **JSON whole numbers become int64.** `{"x": 5}` arrives as int64 even if
    your model was fitted on float64. Same fix: cast before predict.
3.  **Load the artifact dict, not just the pipeline.** If you saved
    `{"pipeline": ..., "target_encoder": ...}`, you must pull both out of the
    dict, otherwise you cannot map predictions back to label names. Confirm
    which keys are actually in your file before writing the route:

        import joblib
        print(list(joblib.load("model/<your file>").keys()))

4.  **Copy my route's comment style.** The route docstring explains the request
    shape with a cURL and PowerShell example. When Members 2-4 add theirs the
    file stays readable if all four routes look alike.

## Git etiquette from here on

`main` is protected and everything we push straight there gets flagged. Do all
of your work on your own feature branch and open a pull request when your route
returns a real prediction. If your branch falls behind:

    git fetch origin
    git merge origin/main     # or rebase, whichever you are comfortable with

Merge order will be regression -> classification -> clustering -> recommender,
so there will be conflicts in `api.py` as each person adds their route. That is
expected; resolve them by keeping the shared loader block intact and adding
your own lines, not by deleting other people's.

## What I still owe the group

- merging PRs in order as they come in
- keeping `CONTRACT.md` Table B updated once each member confirms their
  artifact filename and expected demo output
