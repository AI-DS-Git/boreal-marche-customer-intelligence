import pandas as pd

file = r"data\raw\online_retail_II.xlsx"

df1 = pd.read_excel(file, sheet_name="Year 2009-2010")
df2 = pd.read_excel(file, sheet_name="Year 2010-2011")
df = pd.concat([df1, df2], ignore_index=True)

# Apply the same purchase-cleaning logic established earlier.
df["Invoice"] = df["Invoice"].astype(str)
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")

df = df.drop_duplicates()
df = df[~df["Invoice"].str.startswith("C")]
df = df[df["Quantity"] > 0]
df = df[df["Price"] > 0]
df = df[df["InvoiceDate"].notna()]
df = df[df["Customer ID"].notna()].copy()

customer_orders = (
    df[["Customer ID", "Invoice", "InvoiceDate"]]
    .drop_duplicates()
)

max_date = customer_orders["InvoiceDate"].max()

print("=" * 75)
print("HEAD A - OBSERVATION / LABEL WINDOW ANALYSIS")
print("=" * 75)

print(f"Clean Gold transaction rows: {len(df):,}")
print(f"Unique customers: {df['Customer ID'].nunique():,}")
print(f"Latest transaction date: {max_date}")
print()

candidate_cutoffs = [
    pd.Timestamp("2011-06-30"),
    pd.Timestamp("2011-07-31"),
    pd.Timestamp("2011-08-31"),
]

for cutoff in candidate_cutoffs:
    label_end = cutoff + pd.Timedelta(days=90)

    eligible = customer_orders[
        customer_orders["InvoiceDate"] <= cutoff
    ]

    label_period = customer_orders[
        (customer_orders["InvoiceDate"] > cutoff)
        & (customer_orders["InvoiceDate"] <= label_end)
    ]

    customers_observed = set(eligible["Customer ID"])
    retained_customers = set(label_period["Customer ID"])

    y = pd.Series(
        [customer in retained_customers for customer in customers_observed]
    )

    print("-" * 75)
    print(f"Cutoff:       {cutoff.date()}")
    print(f"Label end:    {label_end.date()}")
    print(f"Observation:  through {cutoff.date()}")
    print(f"Label window: {cutoff.date()} < date <= {label_end.date()}")
    print(f"Eligible customers: {len(customers_observed):,}")
    print(f"Positive label:     {int(y.sum()):,}")
    print(f"Negative label:     {int((~y).sum()):,}")
    print(f"Positive rate:      {y.mean():.2%}")
    print()

cutoff = pd.Timestamp("2011-08-31")
label_end = cutoff + pd.Timedelta(days=90)

print("=" * 75)
print("RECOMMENDED D2 WINDOW")
print("=" * 75)
print(f"Observation window: transaction history through {cutoff.date()}")
print(f"Label window:       90 days after cutoff")
print(f"Label end:          {label_end.date()}")
print()
print("Definition:")
print("y = 1 if the customer places at least one purchase order")
print("    during the 90-day label window.")
print("y = 0 otherwise.")
print()
print("Leakage protection:")
print("- Features use only transactions on or before the cutoff.")
print("- Label activity is never used as a feature.")
print("- The final test period remains chronologically after training data.")
