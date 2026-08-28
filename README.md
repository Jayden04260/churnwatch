# churnwatch

A customer churn prediction API that monitors its own predictions for
drift, alerts when they degrade, and can be retrained and redeployed on
demand - the MLOps piece (a model watching itself in production) that
none of my other projects demonstrate.

Built as a companion to [emotisense](https://github.com/Jayden04260/emotisense)'s
Docker/CI-CD/Terraform/AWS Lambda retrofit: same deployment patterns,
deliberately different ML domain (tabular scikit-learn, not NLP/transformers)
to show range.

## The real-drift story

Most drift-detection demos inject a fake distribution shift to show the
detector "working." This one doesn't need to: it's trained on
[UCI's Online Retail II dataset](https://archive.ics.uci.edu/dataset/502/online+retail+ii)
(Daqing Chen, CC BY 4.0) - two genuine calendar years (Dec 2009-Dec 2011)
of real e-commerce transactions. The model trains on Year 1 behaviour;
Year 2 traffic is replayed against the live API as "incoming" requests.

The drift that shows up is real and was measured, not designed in: comparing
Year 2 reference-date snapshots against the Year 1 training baseline (via
Population Stability Index - see `drift.py`), `recency_days` goes from
~87 days to ~204 days by September 2011, and the true churn rate itself
shifts from 46.7% (training) to 64.7% (Year 2) - genuine behavioural
change in the customer base over the two years this dataset spans, not
holiday seasonality specifically, and not anything synthetically injected.

## Architecture

```
Online Retail II (UCI, auto-downloaded by build_dataset.py)
        |
        v
  build_dataset.py -> RFM features (features.py) + churn labels
        |             (90-day-forward-no-purchase rule)
        v
     train.py -> models/v{n}/ (model, metrics, PSI baseline stats)
        |
        v
  api/main.py (FastAPI) -> Docker + Lambda Web Adapter -> AWS Lambda
     - GET  /health
     - POST /predict  (logs each request to S3 - prediction_log.py)
        |
        v
  drift_check.py (separate Lambda, AWS's own Python base image, EventBridge
  schedule) -> reads recent S3 prediction logs, computes PSI per feature
  (drift.py) vs. the current model's baseline -> writes a report to S3,
  publishes to SNS (email) if any feature's PSI > 0.2
        |
        v
  .github/workflows/retrain.yml (workflow_dispatch only - manually
  triggered after an alert) -> retrain.py: retrains, evaluates against the
  same holdout train.py used, promotes only if registry.is_better() says
  the candidate actually wins
        |
        v
  dashboard/app.py (Streamlit) -> prediction volume, PSI trend per
  feature, model version/metric history
```

## Local usage

```
pip install -r requirements.txt
python build_dataset.py       # downloads the raw dataset if not present, builds the 3 parquet files
python train.py                # trains v1, writes models/v1/
uvicorn api.main:app --reload  # serve locally
pytest -v                      # 35 tests: features, drift math (PSI), training, API, S3 logging, drift Lambda
```

Try it:

```
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"recency_days": 200, "frequency": 1, "monetary": 30.0}'
```

## Deployment

`Dockerfile` (API, Lambda Web Adapter pattern - see emotisense for the
full writeup of that pattern and the AWS gotchas it works around) and
`Dockerfile.drift_check` (the scheduled job, AWS's own Lambda Python base
image - a plain handler-style function doesn't need the Web Adapter at
all) both build and run correctly locally, verified against real data.

**AWS deployment status: infrastructure-as-code complete, not yet applied
live.** `terraform/` defines the full stack (ECR x2, both Lambdas, the S3
data bucket, the EventBridge schedule, the SNS alerts topic, and a scoped
`churnwatch-deploy` IAM policy for the deploy user - a separate identity
from `emotisense-deploy`, not a widened version of it). Applying it needs
an admin bootstrap step (see emotisense's terraform/README.md for why: the
scoped policy has to be proven sufficient by actually removing broader
access and re-running `terraform plan`, which needs a one-time admin
session to get started).

## Known real gotchas hit building this (see repo commit history / PR for the full story)

- **numpy/pandas wouldn't install on AWS's Lambda Python base image**:
  that image is still Amazon Linux 2 (glibc 2.26), but current numpy/pandas
  releases only ship wheels requiring glibc 2.28+, and AL2's own available
  gcc (7.3.1) is too old to build them from source as a fallback. Fixed by
  pinning `numpy==1.26.4` / `pandas==2.1.4` in `requirements-drift.txt` -
  the last versions still shipping manylinux2014-compatible wheels.
- **Low-cardinality features break naive quantile bucketing**: `frequency`
  has many customers sharing the same small integer value, so 10 quantile
  cutoffs can produce duplicate bucket edges - `pd.cut` rejects that
  outright. Fixed in `train.py`'s `compute_baseline_stats` by deduplicating
  edges (fewer, wider buckets for a low-cardinality feature is correct
  behaviour, not a bug to paper over) - and then storing the *actual*
  per-bucket training proportions instead of assuming a uniform 1/N split,
  since deduplication breaks that assumption.
- Same Lambda Function URL two-permission-statement requirement, and the
  same Windows `local-exec` single-line-command requirement, as emotisense.

## What's genuinely new here vs. emotisense

- Statistical drift detection (Population Stability Index) with a cited,
  industry-standard threshold.
- A lightweight model registry with an explicit, legible promotion gate
  (`registry.is_better()`) rather than auto-promoting anything that trains.
- Multi-resource Terraform (EventBridge, SNS, S3, two Lambdas with two
  different packaging strategies) vs. emotisense's single Lambda.
- A scheduled/event-driven Lambda alongside a request-driven one.
- A different ML stack entirely (scikit-learn/tabular vs. transformers/NLP).
