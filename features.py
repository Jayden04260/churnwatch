"""
features.py

Pure RFM (Recency, Frequency, Monetary) feature engineering for the churn
model - no file I/O here, so these are easy to unit test in isolation from
the dataset-loading/labeling logic in build_dataset.py.

"Churn" here means: this customer does NOT make another purchase within
CHURN_WINDOW_DAYS after the reference date. That's the standard definition
for non-subscription retail (there's no natural "cancellation" event to
key off, unlike a subscription product) - see README for the citation.
"""

from __future__ import annotations

import pandas as pd

CHURN_WINDOW_DAYS = 90

# Frequency/monetary look back over a trailing window, not all history since
# the dataset began - a lifetime-cumulative total only ever grows, so two
# customers scored a year apart would differ mainly because more time had
# passed, not because of any real change in behaviour. A trailing window
# instead captures "how active has this customer been lately", which is what
# actually moves with real seasonal effects (e.g. a Christmas spending
# spike) - the thing this project's drift detector is meant to catch.
# Recency stays unbounded (full history): a customer who hasn't bought
# anything within the window is exactly the kind of at-risk customer a
# churn model needs to be able to look at, not exclude.
RFM_LOOKBACK_DAYS = 180


def compute_rfm(
    transactions: pd.DataFrame,
    reference_date: pd.Timestamp,
    lookback_days: int = RFM_LOOKBACK_DAYS,
) -> pd.DataFrame:
    """
    transactions: rows with columns "Customer ID", "Invoice", "InvoiceDate",
    "Quantity", "Price". Must already be restricted to transactions ON OR
    BEFORE reference_date - that slicing is the caller's job (build_dataset.py),
    which keeps this function pure and trivial to test with small hand-built
    frames instead of needing a real reference-date cutoff inside the test.

    Returns one row per customer with at least one purchase on or before
    reference_date: Customer ID, recency_days, frequency, monetary.
    """
    purchases = transactions[transactions["Quantity"] > 0].copy()
    purchases["line_total"] = purchases["Quantity"] * purchases["Price"]

    # Recency: search all history, so an inactive customer is still scoreable.
    recency = purchases.groupby("Customer ID")["InvoiceDate"].max().rename("last_purchase")

    # Frequency/monetary: only the trailing window.
    window_start = reference_date - pd.Timedelta(days=lookback_days)
    recent = purchases[purchases["InvoiceDate"] > window_start]
    activity = recent.groupby("Customer ID").agg(
        frequency=("Invoice", "nunique"),
        monetary=("line_total", "sum"),
    )

    # A customer with a purchase in history but none inside the trailing
    # window is real and important (recent=False everywhere) - keep them via
    # an outer join, filling their window-scoped activity with zero rather
    # than dropping them.
    combined = recency.to_frame().join(activity, how="left")
    combined[["frequency", "monetary"]] = combined[["frequency", "monetary"]].fillna(0)
    combined["frequency"] = combined["frequency"].astype(int)

    combined["recency_days"] = (reference_date - combined["last_purchase"]).dt.days
    return combined[["recency_days", "frequency", "monetary"]].reset_index()


def compute_features(
    transactions: pd.DataFrame,
    reference_date: pd.Timestamp,
    lookback_days: int = RFM_LOOKBACK_DAYS,
) -> pd.DataFrame:
    """
    RFM (see compute_rfm) plus three behavioural features. Same contract as
    compute_rfm - transactions already restricted to on/before
    reference_date - but also needs a "StockCode" column.

    - mean_gap_days: average days between distinct purchase days, over the
      customer's whole history (days from first purchase to reference date,
      divided by number of distinct purchase days). A customer's own
      purchase rhythm is what makes a given recency alarming or not: 60
      days of silence means something very different for a weekly buyer
      than for a twice-a-year one. By far the most predictive feature added
      (see README "Retraining v2").
    - unique_products_180d: distinct StockCodes bought in the trailing
      window - breadth of engagement, not just spend.
    - avg_order_value_180d: monetary / frequency over the trailing window
      (0 if no orders in the window).

    All three are rates or trailing-window counts, deliberately not
    lifetime totals - a lifetime total only ever grows, so it would show up
    as "drift" purely because time passed (see RFM_LOOKBACK_DAYS above).
    A lifetime purchase-day count was tested and scored marginally higher,
    and was left out for exactly that reason.
    """
    features = compute_rfm(transactions, reference_date, lookback_days).set_index("Customer ID")
    purchases = transactions[transactions["Quantity"] > 0]

    first_purchase = purchases.groupby("Customer ID")["InvoiceDate"].min()
    tenure_days = (reference_date - first_purchase.reindex(features.index)).dt.days
    purchase_days = (
        purchases.assign(day=purchases["InvoiceDate"].dt.normalize()).groupby("Customer ID")["day"].nunique()
    )
    features["mean_gap_days"] = tenure_days / purchase_days.reindex(features.index).clip(lower=1)

    window_start = reference_date - pd.Timedelta(days=lookback_days)
    recent = purchases[purchases["InvoiceDate"] > window_start]
    unique_products = recent.groupby("Customer ID")["StockCode"].nunique()
    features["unique_products_180d"] = unique_products.reindex(features.index).fillna(0).astype(int)
    features["avg_order_value_180d"] = features["monetary"] / features["frequency"].clip(lower=1)

    return features[FEATURE_COLUMNS].reset_index()


def label_churn(
    customers: pd.DataFrame,
    future_transactions: pd.DataFrame,
    reference_date: pd.Timestamp,
    window_days: int = CHURN_WINDOW_DAYS,
) -> pd.DataFrame:
    """
    customers: output of compute_rfm (one row per customer as of
    reference_date).
    future_transactions: any transactions after reference_date - this
    function does its own (reference_date, reference_date + window_days]
    filtering, so callers don't have to get that slicing right themselves.

    Adds a `churned` column: 1 if the customer makes no purchase at all in
    that forward window, 0 otherwise.
    """
    window_end = reference_date + pd.Timedelta(days=window_days)
    in_window = future_transactions[
        (future_transactions["InvoiceDate"] > reference_date) & (future_transactions["InvoiceDate"] <= window_end)
    ]
    future_purchasers = set(in_window.loc[in_window["Quantity"] > 0, "Customer ID"])

    customers = customers.copy()
    customers["churned"] = (~customers["Customer ID"].isin(future_purchasers)).astype(int)
    return customers


RFM_COLUMNS = ["recency_days", "frequency", "monetary"]
FEATURE_COLUMNS = [*RFM_COLUMNS, "mean_gap_days", "unique_products_180d", "avg_order_value_180d"]
