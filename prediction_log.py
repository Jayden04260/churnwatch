"""
prediction_log.py

Logs each /predict request+response to S3 as JSONL, partitioned by date
(logs/YYYY-MM-DD.jsonl) - this is what drift_check.py (Phase C) reads to
compute live feature distributions against the training baseline.

Kept separate from api/main.py so the S3-specific logic is testable in
isolation (with moto - see tests/test_prediction_log.py) without needing
real AWS credentials, and so api/main.py itself stays free of boto3 details.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import boto3


def log_prediction(bucket: str, features: dict, churn_probability: float, model_version: int) -> None:
    """
    Appends one JSON line to logs/{today}.jsonl in the given S3 bucket.
    S3 has no native "append" operation, so this reads the existing
    object (if any), adds the new line, and writes it back - fine at this
    project's traffic volume; a high-throughput version would batch
    writes instead (e.g. via Kinesis Firehose), noted here rather than
    silently pretending this scales further than it does.
    """
    s3 = boto3.client("s3")
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    key = f"logs/{today}.jsonl"

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_version": model_version,
        "churn_probability": churn_probability,
        **features,
    }
    line = json.dumps(record)

    try:
        existing = s3.get_object(Bucket=bucket, Key=key)["Body"].read().decode("utf-8")
    except s3.exceptions.NoSuchKey:
        existing = ""

    s3.put_object(Bucket=bucket, Key=key, Body=(existing + line + "\n").encode("utf-8"))
