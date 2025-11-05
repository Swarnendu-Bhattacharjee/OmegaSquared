import pandas as pd, os, sys

folder = os.path.basename(os.getcwd())
agency = folder.split()[0].lower()

# Locate deduped CSV
file_candidates = [f for f in os.listdir(os.getcwd()) if f.startswith("filteredpromax_kaggle_dedup_") and f.endswith(".csv")]
if not file_candidates:
    sys.exit("No filteredpromax_kaggle_dedup_*.csv file found in this folder.")
input_file = os.path.join(os.getcwd(), file_candidates[0])

df = pd.read_csv(input_file)

# Auto-detect date column
date_col = None
for c in df.columns:
    if any(x in c.lower() for x in ["date", "time", "year", "month"]):
        date_col = c
        break

if not date_col:
    sys.exit("No column found containing date/time info.")

# Convert to datetime safely
df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

# Create new month/year columns
df["month"] = df[date_col].dt.month
df["year"] = df[date_col].dt.year

# Drop the original date column
df = df.drop(columns=[date_col])

# Move month/year to end
cols = [c for c in df.columns if c not in ["month", "year"]] + ["month", "year"]
df = df[cols]

# Save output
agency_short = agency.replace("'", "").replace("&", "").replace(" ", "")
out_file = os.path.join(os.getcwd(), f"filteredpromax_kaggle_with_year_month_{agency_short}.csv")

df.to_csv(out_file, index=False)
print(f"Date column '{date_col}' split → new [month, year] columns added → {out_file}")
