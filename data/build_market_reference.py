"""
data/build_market_reference.py

Builds a compact REAL-WORLD market-rate reference table from the property
datasets the user supplied (Pune resale listings, Bangalore resale listings,
and a monthly locality price index), and writes it to
`data/market_reference.csv`.

WHY THIS IS KEPT SEPARATE FROM THE COST *TRAINING* DATA
--------------------------------------------------------
`ml/cost_dataset.csv` (used by ml/train_random_forest.py and
ml/train_cost_ensemble.py) has to stay a construction-cost dataset: its
target is "what it costs to BUILD a house of X sqft", with a formula-
consistent relationship between builtup_area, floors, bathrooms and cost.

The three source files here are RESALE/RENTAL market data for apartments —
`price` there means "what a buyer/tenant pays for an already-built unit",
which bundles in land cost, developer margin and location premium. That is
a different quantity than construction cost per sqft, and the source rows
are apartments/flats, not independent houses on a plot (which is what this
app's layout generator plans for). Silently pasting these prices into the
cost-regression training set would quietly corrupt that model with a
different target variable — so instead this script aggregates them into a
transparent **reference table of real ₹/sqft market rates by city/tier**,
which the app can show alongside its own construction-cost estimate as a
"how does this compare to real listings" sanity check, without touching
the ML training data or its accuracy claims.

Run:
    python data/build_market_reference.py

Output columns:
    source, city, tier, avg_price_per_sqft_inr, n_listings
"""
import os
import re
import pandas as pd

HERE = os.path.dirname(__file__)
SRC = os.path.join(HERE, "real_estate_source")
OUT = os.path.join(HERE, "market_reference.csv")


def _num(series):
    return pd.to_numeric(
        series.astype(str).str.replace(r"[^\d.]", "", regex=True), errors="coerce"
    )


def from_pune():
    df = pd.read_csv(os.path.join(SRC, "Pune_property_data.csv"))
    rate = _num(df["pricepersquare"])
    df = df.assign(rate=rate).dropna(subset=["rate"])
    df = df[(df["rate"] > 500) & (df["rate"] < 100000)]  # drop obvious data-entry outliers
    return pd.DataFrame({
        "source": "Pune_property_data.csv",
        "city": "Pune",
        "tier": "unspecified",
        "avg_price_per_sqft_inr": [round(df["rate"].mean(), 0)],
        "n_listings": [len(df)],
    })


def from_bangalore():
    df = pd.read_csv(os.path.join(SRC, "bangalore_cleaned.csv"))
    df = df.dropna(subset=["price_persqft"])
    df = df[(df["price_persqft"] > 500) & (df["price_persqft"] < 100000)]
    return pd.DataFrame({
        "source": "bangalore_cleaned.csv",
        "city": "Bangalore",
        "tier": "unspecified",
        "avg_price_per_sqft_inr": [round(df["price_persqft"].mean(), 0)],
        "n_listings": [len(df)],
    })


def from_locality_index():
    df = pd.read_csv(os.path.join(SRC, "locality_prices_monthly.csv"))
    latest = df[df["year_month"] == df["year_month"].max()]
    grouped = latest.groupby("tier")["underconstruction_price_per_sqft_carpet_inr"].agg(["mean", "count"])
    grouped = grouped.reset_index()
    return pd.DataFrame({
        "source": "locality_prices_monthly.csv",
        "city": "multi-city (see tier)",
        "tier": grouped["tier"],
        "avg_price_per_sqft_inr": grouped["mean"].round(0),
        "n_listings": grouped["count"],
    })


def main():
    parts = [from_pune(), from_bangalore(), from_locality_index()]
    out = pd.concat(parts, ignore_index=True)
    out.to_csv(OUT, index=False)
    print(f"Wrote {len(out)} rows to {OUT}")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
