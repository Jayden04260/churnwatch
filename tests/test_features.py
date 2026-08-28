import pandas as pd

from features import compute_rfm, label_churn


def _txn(customer_id, invoice, date, qty, price):
    return {
        "Customer ID": customer_id,
        "Invoice": invoice,
        "InvoiceDate": pd.Timestamp(date),
        "Quantity": qty,
        "Price": price,
    }


def test_compute_rfm_basic_recency_frequency_monetary():
    transactions = pd.DataFrame(
        [
            _txn(1, "A1", "2024-01-01", 2, 10.0),  # 20.0
            _txn(1, "A2", "2024-01-10", 1, 5.0),  # 5.0, most recent for customer 1
            _txn(2, "B1", "2024-01-05", 3, 2.0),  # 6.0
        ]
    )
    result = compute_rfm(transactions, reference_date=pd.Timestamp("2024-01-15"))
    result = result.set_index("Customer ID")

    assert result.loc[1, "recency_days"] == 5  # Jan 15 - Jan 10
    assert result.loc[1, "frequency"] == 2  # two distinct invoices
    assert result.loc[1, "monetary"] == 25.0  # 20.0 + 5.0

    assert result.loc[2, "recency_days"] == 10  # Jan 15 - Jan 5
    assert result.loc[2, "frequency"] == 1
    assert result.loc[2, "monetary"] == 6.0


def test_compute_rfm_excludes_returns_from_monetary_and_frequency():
    transactions = pd.DataFrame(
        [
            _txn(1, "A1", "2024-01-01", 2, 10.0),  # a real purchase: 20.0
            _txn(1, "A2", "2024-01-05", -1, 10.0),  # a return: must not count as a purchase
        ]
    )
    result = compute_rfm(transactions, reference_date=pd.Timestamp("2024-01-15")).set_index("Customer ID")

    assert result.loc[1, "frequency"] == 1
    assert result.loc[1, "monetary"] == 20.0
    # recency should be based on the real purchase (Jan 1), not the return (Jan 5)
    assert result.loc[1, "recency_days"] == 14


def test_compute_rfm_frequency_and_monetary_only_count_the_trailing_window():
    transactions = pd.DataFrame(
        [
            # Outside the (short, test-only) 30-day lookback window, but it's
            # still this customer's most recent purchase overall.
            _txn(1, "OLD", "2023-11-01", 5, 100.0),
            # Inside the window.
            _txn(1, "NEW", "2024-01-10", 1, 5.0),
        ]
    )
    result = compute_rfm(
        transactions, reference_date=pd.Timestamp("2024-01-15"), lookback_days=30
    ).set_index("Customer ID")

    assert result.loc[1, "frequency"] == 1  # only the Jan 10 invoice
    assert result.loc[1, "monetary"] == 5.0  # only the Jan 10 line
    assert result.loc[1, "recency_days"] == 5  # still measured from the true last purchase, Jan 10


def test_compute_rfm_customer_active_in_history_but_not_recently_still_appears():
    transactions = pd.DataFrame([_txn(1, "OLD", "2023-11-01", 5, 100.0)])
    result = compute_rfm(
        transactions, reference_date=pd.Timestamp("2024-01-15"), lookback_days=30
    ).set_index("Customer ID")

    # No purchase inside the window at all - frequency/monetary are zero,
    # but the customer must still show up (they're exactly who a churn
    # model needs to be able to flag), with a real recency measured from
    # their actual last purchase.
    assert result.loc[1, "frequency"] == 0
    assert result.loc[1, "monetary"] == 0
    assert result.loc[1, "recency_days"] == 75


def test_compute_rfm_only_includes_customers_with_a_purchase_before_reference_date():
    transactions = pd.DataFrame([_txn(1, "A1", "2024-01-01", 1, 10.0)])
    result = compute_rfm(transactions, reference_date=pd.Timestamp("2024-01-15"))
    assert set(result["Customer ID"]) == {1}


def test_label_churn_marks_no_future_purchase_as_churned():
    customers = pd.DataFrame({"Customer ID": [1, 2]})
    future = pd.DataFrame(
        [
            _txn(2, "F1", "2024-01-20", 1, 5.0),  # customer 2 buys again, within window
        ]
    )
    result = label_churn(
        customers, future, reference_date=pd.Timestamp("2024-01-15"), window_days=90
    ).set_index("Customer ID")

    assert result.loc[1, "churned"] == 1  # no future purchase at all
    assert result.loc[2, "churned"] == 0  # bought again


def test_label_churn_ignores_purchases_outside_the_window():
    customers = pd.DataFrame({"Customer ID": [1]})
    future = pd.DataFrame([_txn(1, "F1", "2024-05-01", 1, 5.0)])  # ~106 days later
    result = label_churn(
        customers, future, reference_date=pd.Timestamp("2024-01-15"), window_days=90
    ).set_index("Customer ID")

    assert result.loc[1, "churned"] == 1


def test_label_churn_ignores_a_return_as_evidence_of_retention():
    customers = pd.DataFrame({"Customer ID": [1]})
    future = pd.DataFrame([_txn(1, "F1", "2024-01-20", -1, 5.0)])  # a return, not a new purchase
    result = label_churn(
        customers, future, reference_date=pd.Timestamp("2024-01-15"), window_days=90
    ).set_index("Customer ID")

    assert result.loc[1, "churned"] == 1
