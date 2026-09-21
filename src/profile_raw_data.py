import pandas as pd

file = r"data\raw\online_retail_II.xlsx"

print("Loading workbook...")

df_09 = pd.read_excel(file, sheet_name="Year 2009-2010")
df_10 = pd.read_excel(file, sheet_name="Year 2010-2011")

df = pd.concat([df_09, df_10], ignore_index=True)

print()
print("=" * 70)
print("RAW DATA PROFILE")
print("=" * 70)

print(f"Total rows: {len(df):,}")
print(f"Total columns: {len(df.columns)}")
print()

print("Missing values:")
print(df.isna().sum().to_string())
print()

print("Duplicate rows:", df.duplicated().sum())
print()

print("Quantity <= 0:", (df["Quantity"] <= 0).sum())
print("Price <= 0:", (df["Price"] <= 0).sum())
print()

print("Invoices starting with C:",
      df["Invoice"].astype(str).str.startswith("C").sum())
print()

print("Date range:")
print("Minimum:", df["InvoiceDate"].min())
print("Maximum:", df["InvoiceDate"].max())
print()

print("Unique customers:", df["Customer ID"].nunique())
print("Unique invoices:", df["Invoice"].nunique())
print("Unique products:", df["StockCode"].nunique())
print("Unique countries:", df["Country"].nunique())
print()

print("Rows by year:")
print(df["InvoiceDate"].dt.year.value_counts().sort_index().to_string())

print()
print("Top countries:")
print(df["Country"].value_counts().head(10).to_string())

