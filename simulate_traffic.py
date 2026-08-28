"""
simulate_traffic.py

Replays data/live_2011.parquet rows against a deployed /predict endpoint,
at a controlled rate. This is how the project actually demonstrates real
drift: the feature values themselves are genuine 2011 customer behaviour
(see build_dataset.py) - only the replay speed is compressed for demo
convenience (real requests wouldn't arrive months apart in a live demo).
Nothing about the drift itself is synthetic.

Usage:
    python simulate_traffic.py <api-url> [--reference-date 2011-06-01] [--rate 5] [--limit 200]
"""

from __future__ import annotations

import argparse
import time

import pandas as pd
import requests

from features import FEATURE_COLUMNS


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay live 2011 traffic against a deployed churnwatch API.")
    parser.add_argument("api_url", help="Base URL of the deployed API, e.g. https://xyz.lambda-url.eu-north-1.on.aws")
    parser.add_argument(
        "--reference-date",
        default=None,
        help="Only replay rows from this reference date (e.g. 2011-06-01). Default: all reference dates.",
    )
    parser.add_argument("--rate", type=float, default=5.0, help="Requests per second (default: 5)")
    parser.add_argument("--limit", type=int, default=None, help="Max rows to replay (default: all)")
    args = parser.parse_args()

    live = pd.read_parquet("data/live_2011.parquet")
    if args.reference_date:
        live = live[live["reference_date"] == pd.Timestamp(args.reference_date)]
    if args.limit:
        live = live.head(args.limit)

    predict_url = args.api_url.rstrip("/") + "/predict"
    delay = 1.0 / args.rate

    sent, errors = 0, 0
    for _, row in live.iterrows():
        payload = {col: float(row[col]) for col in FEATURE_COLUMNS}
        try:
            response = requests.post(predict_url, json=payload, timeout=10)
            response.raise_for_status()
        except requests.RequestException as exc:
            errors += 1
            print(f"Request failed: {exc}")
        else:
            sent += 1
        time.sleep(delay)

    print(f"\nSent {sent} requests ({errors} failed) to {predict_url}")


if __name__ == "__main__":
    main()
