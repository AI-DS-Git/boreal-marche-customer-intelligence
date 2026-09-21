import pandas as pd

file = r"data\raw\online_retail_II.xlsx"

df1 = pd.read_excel(file, sheet_name="Year 2009-2010")
df2 = pd.read_excel(file, sheet_name="Year 2010-2011")
df = pd.concat([df1, df2], ignore_index=True)

invoice_text = df["Invoice"].astype(str)
cancelled = invoice_text.str.startswith("C")
negative_qty = df["Quantity"] <= 0
zero_price = df["Price"] <= 0

print("=" * 70)
print("CLEANING-RULE OVERLAP ANALYSIS")
print("=" * 70)

print(f"Cancellation invoices: {cancelled.sum():,}")
print(f"Non-positive quantity: {negative_qty.sum():,}")
print(f"Both cancellation AND non-positive quantity: {(cancelled & negative_qty).sum():,}")
print(f"Non-cancellation AND non-positive quantity: {(~cancelled & negative_qty).sum():,}")
print()

print("Cancellation quantity distribution:")
print(df.loc[cancelled, "Quantity"].value_counts().head(15).to_string())
print()

print("Non-positive quantity distribution:")
print(df.loc[negative_qty, "Quantity"].value_counts().head(15).to_string())
print()

print("Missing Customer ID among cancellation rows:",
      df.loc[cancelled, "Customer ID"].isna().sum())
print()

print("Missing Customer ID among positive-quantity rows:",
      df.loc[df["Quantity"] > 0, "Customer ID"].isna().sum())
print()

print("Zero/negative price among positive-quantity rows:",
      ((df["Price"] <= 0) & (df["Quantity"] > 0)).sum())
print()

print("Duplicate rows that also have non-positive quantity:",
      df.loc[df.duplicated(keep=False), "Quantity"].le(0).sum())
