import pandas as pd

FILE = r"data\raw\online_retail_II.xlsx"
CUTOFF = pd.Timestamp("2011-08-31 23:59:59")

df1 = pd.read_excel(FILE, sheet_name="Year 2009-2010")
df2 = pd.read_excel(FILE, sheet_name="Year 2010-2011")
df = pd.concat([df1, df2], ignore_index=True)

df["Invoice"] = df["Invoice"].astype(str)
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")

df = df.drop_duplicates()
df = df[~df["Invoice"].str.startswith("C")]
df = df[df["Quantity"] > 0]
df = df[df["Price"] > 0]
df = df[df["InvoiceDate"].notna()]
df = df[df["Customer ID"].notna()]

before = set(
    df.loc[df["InvoiceDate"] <= CUTOFF, "Customer ID"]
)

future = set(
    df.loc[
        (df["InvoiceDate"] > CUTOFF) &
        (df["InvoiceDate"] <= CUTOFF + pd.Timedelta(days=90)),
        "Customer ID"
    ]
)

print("=" * 70)
print("CUSTOMER POPULATION CONSISTENCY CHECK")
print("=" * 70)

print(f"Customers with history by cutoff: {len(before):,}")
print(f"Customers with future purchases:   {len(future):,}")
print(f"Customers eligible for Head A:     {len(before):,}")

print()
print("Retained customers among eligible:")
print(len(before & future))

print()
print("Customers appearing only in future:")
future_only = future - before
print(len(future_only))

if future_only:
    print(sorted(future_only))

print()
print("Expected Gold population:")
print(len(before))

print()
print("Expected labels:")
print(f"Positive: {len(before & future):,}")
print(f"Negative: {len(before - future):,}")
print(f"Positive rate: {len(before & future) / len(before):.2%}")
