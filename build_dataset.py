"""
build_dataset.py

Turns the raw Online Retail II transaction log (data/online_retail_II.xlsx)
into two labeled/feature datasets for the churn model:

  data/train.parquet      - RFM snapshots taken at several reference dates
                             through Year 1 (2010), each with a real
                             forward-looking churn label.
  data/live_2011.parquet  - RFM snapshots taken at reference dates through
                             Year 2 (2011), features only. This is the
                             "incoming scoring traffic" simulate_traffic.py
                             replays against the deployed API in Phase C -
                             the genuine Nov/Dec 2010 and 2011 holiday
                             spending spike shows up here as real feature
                             drift relative to the Year 1 training
                             distribution, not anything synthetically
                             injected.

Dataset: Online Retail II, UCI Machine Learning Repository
(https://archive.ics.uci.edu/dataset/502/online+retail+ii), Daqing Chen,
licensed CC BY 4.0.
"""

from __future__ import annotations

import zipfile
from io import BytesIO
from pathlib import Path

import pandas as pd
import requests

from features import CHURN_WINDOW_DAYS, compute_rfm, label_churn

RAW_URL = "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip"
RAW_PATH = Path("data/online_retail_II.xlsx")
TRAIN_OUT = Path("data/train.parquet")
LIVE_OUT = Path("data/live_2011.parquet")
LIVE_LABELED_OUT = Path("data/live_2011_labeled.parquet")


def ensure_raw_data() -> None:
    """Downloads and extracts the raw dataset if it isn't already present -
    not committed to the repo (it's 44MB, and freely re-downloadable, so
    there's no reason to bloat the repo with it), but this keeps a fresh
    clone fully reproducible with just `python build_dataset.py`."""
    if RAW_PATH.exists():
        return
    print(f"{RAW_PATH} not found - downloading from {RAW_URL}...")
    response = requests.get(RAW_URL, timeout=120)
    response.raise_for_status()
    with zipfile.ZipFile(BytesIO(response.content)) as archive:
        archive.extractall(RAW_PATH.parent)
    print(f"Downloaded and extracted to {RAW_PATH}")

# Monthly reference dates through Year 1. Starting in March 2010 gives at
# least 3 months of transaction history to compute RFM features from
# (the dataset begins 2009-12-01); ending in August 2010 leaves a full
# CHURN_WINDOW_DAYS of real forward data to label against before the
# Year 2 traffic begins in earnest (Dec 2010).
TRAIN_REFERENCE_DATES = pd.date_range("2010-03-01", "2010-08-01", freq="MS")

# Monthly reference dates through Year 2, deliberately spanning the real
# Nov/Dec holiday season (both the 2010 and 2011 ones) - this is what
# should show up as genuine drift against the Year 1 training baseline.
LIVE_REFERENCE_DATES = pd.date_range("2010-12-01", "2011-09-01", freq="MS")


def load_transactions() -> pd.DataFrame:
    xl = pd.ExcelFile(RAW_PATH)
    sheets = [xl.parse(name) for name in xl.sheet_names]
    transactions = pd.concat(sheets, ignore_index=True)

    before = len(transactions)
    transactions = transactions.dropna(subset=["Customer ID"])
    dropped_no_customer = before - len(transactions)

    returns = (transactions["Quantity"] < 0).sum()

    print(f"Loaded {before} raw rows from {len(xl.sheet_names)} sheet(s).")
    print(f"Dropped {dropped_no_customer} rows with no Customer ID (can't attribute to a customer).")
    print(f"{returns} rows are returns (negative quantity) - kept in the log, excluded from RFM purchase math.")

    transactions["Customer ID"] = transactions["Customer ID"].astype(int)
    return transactions.sort_values("InvoiceDate").reset_index(drop=True)


def build_snapshot(transactions: pd.DataFrame, reference_date: pd.Timestamp, with_labels: bool) -> pd.DataFrame:
    history = transactions[transactions["InvoiceDate"] <= reference_date]
    snapshot = compute_rfm(history, reference_date)
    snapshot["reference_date"] = reference_date

    if with_labels:
        window_end = reference_date + pd.Timedelta(days=CHURN_WINDOW_DAYS)
        future = transactions[
            (transactions["InvoiceDate"] > reference_date) & (transactions["InvoiceDate"] <= window_end)
        ]
        snapshot = label_churn(snapshot, future, reference_date)

    return snapshot


def main() -> None:
    ensure_raw_data()
    transactions = load_transactions()

    train_frames = [build_snapshot(transactions, ref, with_labels=True) for ref in TRAIN_REFERENCE_DATES]
    train = pd.concat(train_frames, ignore_index=True)
    train.to_parquet(TRAIN_OUT, index=False)
    print(f"\nWrote {len(train)} labeled training rows ({len(TRAIN_REFERENCE_DATES)} reference dates) to {TRAIN_OUT}")
    print(f"Churn rate in training data: {train['churned'].mean():.1%}")

    live_frames = [build_snapshot(transactions, ref, with_labels=False) for ref in LIVE_REFERENCE_DATES]
    live = pd.concat(live_frames, ignore_index=True)
    live.to_parquet(LIVE_OUT, index=False)
    print(f"Wrote {len(live)} unlabeled live rows ({len(LIVE_REFERENCE_DATES)} reference dates) to {LIVE_OUT}")

    # Same reference dates, but WITH labels - simulates the real-world
    # "churn labels arrive late" constraint retrain.py (Phase D) works
    # under: at scoring time, only live_2011.parquet (features only) is
    # available; this labeled version represents what becomes available
    # once each reference date's 90-day forward window has actually
    # elapsed in the dataset's own timeline, and is what retraining
    # evaluates a candidate model against.
    live_labeled_frames = [build_snapshot(transactions, ref, with_labels=True) for ref in LIVE_REFERENCE_DATES]
    live_labeled = pd.concat(live_labeled_frames, ignore_index=True)
    live_labeled.to_parquet(LIVE_LABELED_OUT, index=False)
    print(f"Wrote {len(live_labeled)} labeled live rows to {LIVE_LABELED_OUT} (for retrain.py)")
    print(f"Churn rate in live 2011 data: {live_labeled['churned'].mean():.1%}")


if __name__ == "__main__":
    main()
