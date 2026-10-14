
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = ROOT / "data" / "processed" / "online_retail_clean.csv"

CUTOFFS = [
    "2011-03-01",
    "2011-04-01",
    "2011-05-01",
    "2011-06-01",
    "2011-08-01",
    "2011-10-01",
]

HORIZON_DAYS = 60


def main():
    df = pd.read_csv(
        DATA_PATH,
        parse_dates=["InvoiceDate"],
    )

    df["InvoiceNo"] = df["InvoiceNo"].astype(str).str.strip()

    df = df[
        ~df["InvoiceNo"].str.upper().str.startswith("C")
        & (df["Quantity"] > 0)
        & (df["UnitPrice"] > 0)
        & df["CustomerID"].notna()
    ].copy()

    print("DATA LEAKAGE CHECK")
    print("==================")

    for cutoff_string in CUTOFFS:
        cutoff = pd.Timestamp(cutoff_string)
        future_end = cutoff + pd.Timedelta(days=HORIZON_DAYS)

        history = df[df["InvoiceDate"] < cutoff]
        future = df[
            (df["InvoiceDate"] >= cutoff)
            & (df["InvoiceDate"] < future_end)
        ]

        # Historical features must use strictly earlier dates.
        assert not history.empty
        assert history["InvoiceDate"].max() < cutoff

        # The label window must start at the cutoff.
        assert (
            future.empty
            or future["InvoiceDate"].min() >= cutoff
        )

        # No label-window transaction may reach the next boundary.
        assert (
            future.empty
            or future["InvoiceDate"].max() < future_end
        )

        # Historical and future windows must not overlap.
        assert set(history.index).isdisjoint(set(future.index))

        print(
            f"PASS: {cutoff.date()} | "
            f"history ends before cutoff | "
            f"label window ends before {future_end.date()}"
        )

    print("\nAll temporal boundary checks passed.")
    print(
        "Note: this checks the date-window rules. "
        "It does not prove that every possible leakage source "
        "in every model or report has been ruled out."
    )


if __name__ == "__main__":
    main()