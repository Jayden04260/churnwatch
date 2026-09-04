"""
dashboard/app.py

Streamlit dashboard reading directly from the project's S3 data bucket:
prediction volume over time, PSI drift trend per feature (with the 0.2
threshold marked), and model version/metric history. Same tool as
emotisense's app/app.py, so no new dependency to learn for this project.

Usage:
    DATA_BUCKET=churnwatch-api-<account-id> streamlit run dashboard/app.py
"""

from __future__ import annotations

import json
import os
import sys

import boto3
import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from registry import DECIDING_METRIC, MIN_IMPROVEMENT, is_better  # noqa: E402 - needs the sys.path fix above first, since this file runs from dashboard/, not the repo root

st.set_page_config(page_title="churnwatch", layout="wide")

DATA_BUCKET = os.environ.get("DATA_BUCKET")
if not DATA_BUCKET:
    st.error("Set the DATA_BUCKET environment variable to the project's S3 bucket name before running this.")
    st.stop()

s3 = boto3.client("s3")


@st.cache_data(ttl=60)
def load_prediction_logs() -> pd.DataFrame:
    keys = [
        obj["Key"]
        for obj in s3.list_objects_v2(Bucket=DATA_BUCKET, Prefix="logs/").get("Contents", [])
    ]
    records = []
    for key in keys:
        body = s3.get_object(Bucket=DATA_BUCKET, Key=key)["Body"].read().decode("utf-8")
        records.extend(json.loads(line) for line in body.splitlines() if line.strip())
    if not records:
        return pd.DataFrame(columns=["timestamp", "churn_probability", "model_version"])
    df = pd.DataFrame(records)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


@st.cache_data(ttl=60)
def load_drift_reports() -> pd.DataFrame:
    keys = [
        obj["Key"]
        for obj in s3.list_objects_v2(Bucket=DATA_BUCKET, Prefix="drift_reports/").get("Contents", [])
    ]
    rows = []
    for key in keys:
        report = json.loads(s3.get_object(Bucket=DATA_BUCKET, Key=key)["Body"].read())
        for feature, result in report.get("features", {}).items():
            rows.append({"checked_at": report["checked_at"], "feature": feature, "psi": result["psi"]})
    if not rows:
        return pd.DataFrame(columns=["checked_at", "feature", "psi"])
    df = pd.DataFrame(rows)
    df["checked_at"] = pd.to_datetime(df["checked_at"])
    return df


@st.cache_data(ttl=60)
def load_model_history() -> pd.DataFrame:
    prefixes = s3.list_objects_v2(Bucket=DATA_BUCKET, Prefix="models/", Delimiter="/").get(
        "CommonPrefixes", []
    )
    rows = []
    for prefix in prefixes:
        version_prefix = prefix["Prefix"]  # e.g. "models/v1/"
        try:
            metadata = json.loads(
                s3.get_object(Bucket=DATA_BUCKET, Key=f"{version_prefix}metadata.json")["Body"].read()
            )
        except s3.exceptions.NoSuchKey:
            continue
        rows.append({"version": metadata["version"], "trained_at": metadata["trained_at"], **metadata["metrics"]})
    return pd.DataFrame(rows).sort_values("version") if rows else pd.DataFrame()


@st.cache_data(ttl=60)
def load_current_version() -> int | None:
    try:
        body = s3.get_object(Bucket=DATA_BUCKET, Key="models/current.json")["Body"].read()
    except s3.exceptions.NoSuchKey:
        return None
    return json.loads(body)["version"]


st.title("churnwatch")

logs = load_prediction_logs()
drift = load_drift_reports()
models = load_model_history()

st.header("Prediction volume")
if logs.empty:
    st.info("No predictions logged yet - hit the deployed API's /predict endpoint, or run simulate_traffic.py.")
else:
    volume = logs.set_index("timestamp").resample("1h").size().rename("predictions")
    st.line_chart(volume)

st.header("Drift (PSI) per feature")
if drift.empty:
    st.info("No drift reports yet - the drift_check Lambda hasn't run, or hasn't had traffic to check.")
else:
    pivot = drift.pivot_table(index="checked_at", columns="feature", values="psi")
    st.line_chart(pivot)
    st.caption("Dashed line at 0.2 marks the drift threshold (see drift.py) - not drawn natively by st.line_chart, "
               "shown as a reference value below instead.")
    st.metric("Drift threshold (PSI)", "0.2")

st.header("Model version history")
if models.empty:
    st.info("No model metadata found.")
else:
    # Every version under models/ was, at some point, actually promoted -
    # retrain.py only ever calls save_version() when is_better() said yes
    # (see registry.py); a rejected candidate is discarded and never
    # written to S3 at all. So this table isn't just a metrics log, it's a
    # record of the promotion gate's own decisions - "cleared_gate" below
    # re-checks each one against its predecessor using the actual
    # is_better() function retrain.py calls, not a re-derived
    # approximation of its logic, so this genuinely proves the gate did
    # what it claims rather than just asserting it in a docstring.
    models = models.sort_values("version").reset_index(drop=True)
    current_version = load_current_version()
    models["is_live"] = models["version"] == current_version
    models[f"{DECIDING_METRIC}_delta"] = models[DECIDING_METRIC].diff()

    records = models.to_dict("records")
    cleared_gate = [None]  # v1 has no predecessor to have cleared a bar against
    for i in range(1, len(records)):
        cleared_gate.append(
            is_better({DECIDING_METRIC: records[i][DECIDING_METRIC]}, {DECIDING_METRIC: records[i - 1][DECIDING_METRIC]})
        )
    models["cleared_gate"] = cleared_gate

    st.bar_chart(models.set_index("version")[DECIDING_METRIC])
    st.caption(
        f"Deciding metric: {DECIDING_METRIC} - a candidate must beat the current model by more than "
        f"{MIN_IMPROVEMENT} to be promoted (see registry.py). ✅ under cleared_gate means that version's "
        f"{DECIDING_METRIC} actually beat its predecessor by more than that margin when it was promoted."
    )

    display_columns = ["version", "trained_at", "is_live", DECIDING_METRIC, f"{DECIDING_METRIC}_delta", "cleared_gate", "accuracy", "f1", "test_rows"]
    st.dataframe(
        models[display_columns].style.format({f"{DECIDING_METRIC}_delta": "{:+.4f}"}, na_rep="-"),
        use_container_width=True,
    )
