import pandas as pd

file = r"data\raw\online_retail_II.xlsx"

excel = pd.ExcelFile(file)

for sheet in excel.sheet_names:
    print("=" * 60)
    print(f"SHEET: {sheet}")
    print("=" * 60)

    df = pd.read_excel(file, sheet_name=sheet, nrows=5)

    print("Columns:")
    print(df.columns.tolist())
    print()

    print("First 5 rows:")
    print(df.to_string(index=False))
    print()
