import pandas as pd

file = r"data\raw\online_retail_II.xlsx"

df1 = pd.read_excel(file, sheet_name="Year 2009-2010")
df2 = pd.read_excel(file, sheet_name="Year 2010-2011")
df = pd.concat([df1, df2], ignore_index=True)

initial = len(df)

# Standardize key fields
df["Invoice"] = df["Invoice"].astype(str)
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")

# Track sequential cleaning
results = []

def record(stage, before, after, rule):
    results.append({
        "Stage": stage,
        "Rows Before": before,
        "Rows Removed": before - after,
        "Rows After": after,
        "Rule": rule
    })

# Bronze
record(
    "Bronze",
    initial,
    initial,
    "Raw data loaded from both UCI workbook sheets"
)

# 1. Remove exact duplicates
before = len(df)
df = df.drop_duplicates()
record(
    "Silver 1",
    before,
    len(df),
    "Remove exact duplicate transaction rows"
)

# 2. Remove cancellation invoices
before = len(df)
df = df[~df["Invoice"].str.startswith("C")]
record(
    "Silver 2",
    before,
    len(df),
    "Remove cancellation invoices from purchase modeling"
)

# 3. Remove non-positive quantities
before = len(df)
df = df[df["Quantity"] > 0]
record(
    "Silver 3",
    before,
    len(df),
    "Keep completed purchases with positive quantity"
)

# 4. Remove non-positive prices
before = len(df)
df = df[df["Price"] > 0]
record(
    "Silver 4",
    before,
    len(df),
    "Remove invalid or zero-price purchase records"
)

# 5. Remove invalid dates
before = len(df)
df = df[df["InvoiceDate"].notna()]
record(
    "Silver 5",
    before,
    len(df),
    "Keep transactions with valid timestamps"
)

print("=" * 90)
print("SEQUENTIAL CLEANING RESULTS")
print("=" * 90)
print()

for r in results:
    print(
        f"{r['Stage']:<10} "
        f"Before: {r['Rows Before']:>10,} | "
        f"Removed: {r['Rows Removed']:>10,} | "
        f"After: {r['Rows After']:>10,}"
    )
    print(f"           Rule: {r['Rule']}")

print()
print("=" * 90)
print("SILVER SUMMARY")
print("=" * 90)
print(f"Bronze rows: {initial:,}")
print(f"Final Silver rows: {len(df):,}")
print(f"Total rows removed: {initial - len(df):,}")
print(f"Silver retention: {len(df) / initial:.2%}")
print()

print("=" * 90)
print("GOLD CUSTOMER POPULATION")
print("=" * 90)

gold = df[df["Customer ID"].notna()].copy()

print(f"Silver rows with Customer ID: {len(gold):,}")
print(f"Unique customers in Gold: {gold['Customer ID'].nunique():,}")
print(f"Rows excluded from Gold due to missing Customer ID: {df['Customer ID'].isna().sum():,}")

print()
print("Silver date range:")
print("Minimum:", df["InvoiceDate"].min())
print("Maximum:", df["InvoiceDate"].max())

print()
print("Gold customers by country:")
print(gold["Country"].nunique())

# Save the sequential results for later D2 documentation
pd.DataFrame(results).to_csv(
    "docs/cleaning_pipeline_results.csv",
    index=False
)

print()
print("Saved: docs/cleaning_pipeline_results.csv")
