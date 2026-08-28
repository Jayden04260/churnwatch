import json
from datetime import datetime, timedelta, timezone

import boto3
import pandas as pd
from moto import mock_aws

from drift_check import check_drift, run_drift_check

BUCKET = "test-churnwatch-bucket"

BASELINE = {
    "recency_days": {
        "bucket_edges": [float("-inf"), 25, 50, 75, float("inf")],
        "reference_proportions": [0.25, 0.25, 0.25, 0.25],
    }
}


def test_check_drift_reports_no_drift_for_matching_traffic():
    predictions = pd.DataFrame({"recency_days": [10, 15, 35, 40, 60, 65, 85, 90]})
    report = check_drift(predictions, BASELINE)

    assert report["any_drift"] is False
    assert report["drifted_features"] == []
    assert report["rows_checked"] == 8


def test_check_drift_flags_a_real_shift():
    predictions = pd.DataFrame({"recency_days": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
    report = check_drift(predictions, BASELINE)

    assert report["any_drift"] is True
    assert "recency_days" in report["drifted_features"]


def test_check_drift_handles_empty_predictions_without_crashing():
    predictions = pd.DataFrame(columns=["recency_days"])
    report = check_drift(predictions, BASELINE)

    assert report["rows_checked"] == 0
    assert report["any_drift"] is False
    assert report["features"] == {}


@mock_aws
def test_run_drift_check_end_to_end_writes_report_and_alerts_on_drift(monkeypatch):
    monkeypatch.setattr("drift_check.DATA_BUCKET", BUCKET)

    s3 = boto3.client("s3", region_name="eu-north-1")
    s3.create_bucket(Bucket=BUCKET, CreateBucketConfiguration={"LocationConstraint": "eu-north-1"})
    s3.put_object(Bucket=BUCKET, Key="models/current.json", Body=json.dumps({"version": 1}))
    s3.put_object(Bucket=BUCKET, Key="models/v1/baseline_stats.json", Body=json.dumps(BASELINE))

    # Log lines all within the lookback window, entirely in the lowest
    # bucket - a real, detectable shift against the uniform baseline.
    now = datetime.now(timezone.utc)
    lines = [
        json.dumps({"timestamp": (now - timedelta(minutes=i)).isoformat(), "recency_days": 1 + i})
        for i in range(10)
    ]
    s3.put_object(Bucket=BUCKET, Key=f"logs/{now.strftime('%Y-%m-%d')}.jsonl", Body="\n".join(lines))

    sqs = boto3.client("sqs", region_name="eu-north-1")
    queue_url = sqs.create_queue(QueueName="drift-alerts-test-queue")["QueueUrl"]
    queue_arn = sqs.get_queue_attributes(QueueUrl=queue_url, AttributeNames=["QueueArn"])["Attributes"][
        "QueueArn"
    ]

    sns = boto3.client("sns", region_name="eu-north-1")
    topic_arn = sns.create_topic(Name="churnwatch-drift-alerts")["TopicArn"]
    sns.subscribe(TopicArn=topic_arn, Protocol="sqs", Endpoint=queue_arn)
    monkeypatch.setattr("drift_check.SNS_TOPIC_ARN", topic_arn)

    report = run_drift_check()

    assert report["any_drift"] is True
    assert "recency_days" in report["drifted_features"]

    # A report object should have been written to S3.
    listing = s3.list_objects_v2(Bucket=BUCKET, Prefix="drift_reports/")
    assert listing["KeyCount"] == 1

    # And a real alert should have actually been delivered, not just implied
    # by the report's own fields.
    messages = sqs.receive_message(QueueUrl=queue_url, MaxNumberOfMessages=10).get("Messages", [])
    assert len(messages) == 1
    assert "recency_days" in messages[0]["Body"]


@mock_aws
def test_run_drift_check_does_not_publish_when_no_drift(monkeypatch):
    monkeypatch.setattr("drift_check.DATA_BUCKET", BUCKET)

    s3 = boto3.client("s3", region_name="eu-north-1")
    s3.create_bucket(Bucket=BUCKET, CreateBucketConfiguration={"LocationConstraint": "eu-north-1"})
    s3.put_object(Bucket=BUCKET, Key="models/current.json", Body=json.dumps({"version": 1}))
    s3.put_object(Bucket=BUCKET, Key="models/v1/baseline_stats.json", Body=json.dumps(BASELINE))

    now = datetime.now(timezone.utc)
    # Matches the uniform baseline - roughly a quarter of values per bucket.
    values = [10, 15, 35, 40, 60, 65, 85, 90]
    lines = [
        json.dumps({"timestamp": (now - timedelta(minutes=i)).isoformat(), "recency_days": v})
        for i, v in enumerate(values)
    ]
    s3.put_object(Bucket=BUCKET, Key=f"logs/{now.strftime('%Y-%m-%d')}.jsonl", Body="\n".join(lines))

    # Subscribe a real (moto-mocked) SQS queue to the SNS topic, so we can
    # rigorously check whether a message actually arrived - not just infer
    # it from the report's own fields, which would be redundant with the
    # pure check_drift() tests above.
    sqs = boto3.client("sqs", region_name="eu-north-1")
    queue_url = sqs.create_queue(QueueName="drift-alerts-test-queue")["QueueUrl"]
    queue_arn = sqs.get_queue_attributes(QueueUrl=queue_url, AttributeNames=["QueueArn"])["Attributes"][
        "QueueArn"
    ]

    sns = boto3.client("sns", region_name="eu-north-1")
    topic_arn = sns.create_topic(Name="churnwatch-drift-alerts")["TopicArn"]
    sns.subscribe(TopicArn=topic_arn, Protocol="sqs", Endpoint=queue_arn)
    monkeypatch.setattr("drift_check.SNS_TOPIC_ARN", topic_arn)

    report = run_drift_check()
    assert report["any_drift"] is False

    messages = sqs.receive_message(QueueUrl=queue_url, MaxNumberOfMessages=10).get("Messages", [])
    assert messages == []
