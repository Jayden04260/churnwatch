import json
from datetime import datetime, timezone

import boto3
from moto import mock_aws

from prediction_log import log_prediction

BUCKET = "test-churnwatch-bucket"


@mock_aws
def _make_bucket():
    s3 = boto3.client("s3", region_name="eu-north-1")
    s3.create_bucket(
        Bucket=BUCKET, CreateBucketConfiguration={"LocationConstraint": "eu-north-1"}
    )
    return s3


@mock_aws
def test_log_prediction_writes_a_jsonl_line_for_todays_date():
    s3 = _make_bucket()
    log_prediction(BUCKET, {"recency_days": 10, "frequency": 2, "monetary": 50.0}, 0.42, model_version=1)

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    body = s3.get_object(Bucket=BUCKET, Key=f"logs/{today}.jsonl")["Body"].read().decode("utf-8")
    lines = [line for line in body.splitlines() if line]

    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["recency_days"] == 10
    assert record["frequency"] == 2
    assert record["monetary"] == 50.0
    assert record["churn_probability"] == 0.42
    assert record["model_version"] == 1
    assert "timestamp" in record


@mock_aws
def test_log_prediction_appends_rather_than_overwrites():
    s3 = _make_bucket()
    log_prediction(BUCKET, {"recency_days": 1, "frequency": 1, "monetary": 1.0}, 0.1, model_version=1)
    log_prediction(BUCKET, {"recency_days": 2, "frequency": 2, "monetary": 2.0}, 0.2, model_version=1)

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    body = s3.get_object(Bucket=BUCKET, Key=f"logs/{today}.jsonl")["Body"].read().decode("utf-8")
    lines = [line for line in body.splitlines() if line]

    assert len(lines) == 2
    assert json.loads(lines[0])["recency_days"] == 1
    assert json.loads(lines[1])["recency_days"] == 2
