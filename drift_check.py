"""
drift_check.py

Lambda handler triggered on a schedule (EventBridge - see terraform/main.tf)
to check recent prediction traffic for drift against the current model's
training baseline. Reads recent prediction logs from S3, computes PSI per
feature (see drift.py), writes a report to S3, and publishes an SNS alert
if any feature crosses DRIFT_THRESHOLD.

check_drift() is the pure(ish) core - given already-loaded predictions and
baseline stats, it just does the comparison. run_drift_check() handles the
S3/SNS I/O around it, so the actual drift logic is testable without touching
AWS at all (see tests/test_drift_check.py, which does still exercise the
I/O path too, with moto).
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone

import boto3
import pandas as pd

from drift import compute_feature_drift
from features import FEATURE_COLUMNS

DATA_BUCKET = os.environ.get("DATA_BUCKET")
SNS_TOPIC_ARN = os.environ.get("SNS_TOPIC_ARN")
LOOKBACK_HOURS = int(os.environ.get("DRIFT_LOOKBACK_HOURS", "24"))


def check_drift(predictions: pd.DataFrame, baseline: dict) -> dict:
    results = {}
    for feature, stats in baseline.items():
        if feature not in predictions.columns or predictions.empty:
            continue
        results[feature] = compute_feature_drift(
            predictions[feature], stats["bucket_edges"], stats["reference_proportions"]
        )

    drifted_features = [name for name, result in results.items() if result["drifted"]]
    return {
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "rows_checked": len(predictions),
        "features": results,
        "drifted_features": drifted_features,
        "any_drift": len(drifted_features) > 0,
    }


def _load_current_baseline(s3) -> tuple[int, dict]:
    current = json.loads(s3.get_object(Bucket=DATA_BUCKET, Key="models/current.json")["Body"].read())
    version = current["version"]
    baseline = json.loads(
        s3.get_object(Bucket=DATA_BUCKET, Key=f"models/v{version}/baseline_stats.json")["Body"].read()
    )
    return version, baseline


def _load_recent_predictions(s3, lookback_hours: int) -> pd.DataFrame:
    """
    Reads logs/{date}.jsonl for every date the lookback window could touch
    (just today's, unless the window spans midnight UTC), then filters down
    to rows actually inside the window.
    """
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(hours=lookback_hours)
    dates = {now.strftime("%Y-%m-%d"), cutoff.strftime("%Y-%m-%d")}

    records = []
    for date in dates:
        try:
            body = s3.get_object(Bucket=DATA_BUCKET, Key=f"logs/{date}.jsonl")["Body"].read().decode("utf-8")
        except s3.exceptions.NoSuchKey:
            continue
        records.extend(json.loads(line) for line in body.splitlines() if line.strip())

    if not records:
        return pd.DataFrame(columns=[*FEATURE_COLUMNS, "timestamp"])

    df = pd.DataFrame(records)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df[df["timestamp"] >= cutoff]


def run_drift_check() -> dict:
    s3 = boto3.client("s3")
    model_version, baseline = _load_current_baseline(s3)
    predictions = _load_recent_predictions(s3, LOOKBACK_HOURS)

    report = check_drift(predictions, baseline)
    report["model_version"] = model_version

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    s3.put_object(
        Bucket=DATA_BUCKET,
        Key=f"drift_reports/{timestamp}.json",
        Body=json.dumps(report, indent=2).encode("utf-8"),
    )

    if report["any_drift"] and SNS_TOPIC_ARN:
        sns = boto3.client("sns")
        sns.publish(
            TopicArn=SNS_TOPIC_ARN,
            Subject="churnwatch: drift detected",
            Message=(
                f"Drift detected in: {', '.join(report['drifted_features'])}\n\n"
                f"Full report: {json.dumps(report, indent=2)}"
            ),
        )

    return report


def lambda_handler(event, context):
    report = run_drift_check()
    return {
        "statusCode": 200,
        "body": json.dumps({"any_drift": report["any_drift"], "drifted_features": report["drifted_features"]}),
    }
