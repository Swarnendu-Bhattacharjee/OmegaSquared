import pandas as pd, os, sys

# Get current working folder (e.g., "Egan-Jones Ratings Company")
folder = os.path.basename(os.getcwd())
agency = folder.split()[0].lower()

# Find the promax file in this folder
file_candidates = [f for f in os.listdir(os.getcwd()) if f.startswith("filteredpromax_kaggle_") and f.endswith(".csv")]
if not file_candidates:
    sys.exit("No filteredpromax_kaggle_*.csv file found in this folder.")
input_file = os.path.join(os.getcwd(), file_candidates[0])

# Read CSV
df = pd.read_csv(input_file)

# Identify date columns (so we can exclude them from dedup check)
date_cols = [c for c in df.columns if "date" in c.lower() or "time" in c.lower()]

# Prepare columns for deduplication (exclude date columns)
cols_for_check = [c for c in df.columns if c not in date_cols]

# Drop duplicates ignoring date
before = len(df)
deduped = df.drop_duplicates(subset=cols_for_check, keep="first")
after = len(deduped)

# Clean agency name for filename
agency_short = agency.replace("'", "").replace("&", "").replace(" ", "")
out_file = os.path.join(os.getcwd(), f"filteredpromax_kaggle_dedup_{agency_short}.csv")

# Save deduplicated file
deduped.to_csv(out_file, index=False)
print(f"Removed {before - after} duplicates (kept {after} rows) → {out_file}")
