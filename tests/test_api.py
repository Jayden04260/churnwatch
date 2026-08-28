import json
from datetime import datetime, timezone

import boto3
from fastapi.testclient import TestClient
from moto import mock_aws

import api.main as main_module
from api.main import app

BUCKET = "test-churnwatch-bucket"


def test_health_returns_ok_and_model_version():
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert isinstance(body["model_version"], int)


def test_predict_returns_a_probability_between_zero_and_one():
    with TestClient(app) as client:
        response = client.post(
            "/predict", json={"recency_days": 30, "frequency": 5, "monetary": 500.0}
        )
        assert response.status_code == 200
        body = response.json()
        assert 0.0 <= body["churn_probability"] <= 1.0
        assert isinstance(body["model_version"], int)


def test_predict_a_long_absent_low_activity_customer_scores_higher_churn_risk():
    # Not a strict correctness guarantee of the model's internals, but a
    # sane end-to-end check that the API is really calling the model and
    # not e.g. returning a constant: a customer who hasn't bought in a
    # year with low historical activity should score as more likely to
    # churn than a customer who bought yesterday and buys often.
    with TestClient(app) as client:
        at_risk = client.post(
            "/predict", json={"recency_days": 365, "frequency": 1, "monetary": 20.0}
        ).json()
        engaged = client.post(
            "/predict", json={"recency_days": 1, "frequency": 20, "monetary": 5000.0}
        ).json()

    assert at_risk["churn_probability"] > engaged["churn_probability"]


@mock_aws
def test_predict_logs_to_s3_when_data_bucket_is_configured(monkeypatch):
    s3 = boto3.client("s3", region_name="eu-north-1")
    s3.create_bucket(Bucket=BUCKET, CreateBucketConfiguration={"LocationConstraint": "eu-north-1"})
    monkeypatch.setattr(main_module, "DATA_BUCKET", BUCKET)

    with TestClient(app) as client:
        response = client.post(
            "/predict", json={"recency_days": 42, "frequency": 3, "monetary": 250.0}
        )
    assert response.status_code == 200

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    body = s3.get_object(Bucket=BUCKET, Key=f"logs/{today}.jsonl")["Body"].read().decode("utf-8")
    record = json.loads(body.splitlines()[0])
    assert record["recency_days"] == 42
    assert record["churn_probability"] == response.json()["churn_probability"]


@mock_aws
def test_predict_still_succeeds_if_logging_fails(monkeypatch):
    # DATA_BUCKET points at a bucket that was never created in this mocked
    # AWS environment, so log_prediction() will fail with a real
    # botocore error - predict() must still return a normal response, per
    # the try/except around logging in api/main.py.
    monkeypatch.setattr(main_module, "DATA_BUCKET", "this-bucket-does-not-exist-at-all")

    with TestClient(app) as client:
        response = client.post(
            "/predict", json={"recency_days": 10, "frequency": 1, "monetary": 10.0}
        )
    assert response.status_code == 200
    assert 0.0 <= response.json()["churn_probability"] <= 1.0
