import pandas as pd, os, sys

# Get current working folder (e.g., "Egan-Jones Ratings Company")
folder = os.path.basename(os.getcwd())
agency = folder.split()[0].lower()

# Build full path to the input file
input_file = os.path.join(os.getcwd(), "filteredpro_kaggle.csv")
if not os.path.exists(input_file):
    sys.exit(f"Input file not found: {input_file}")

# Read CSV
df = pd.read_csv(input_file)

# Detect column name that contains agency info
col = None
for c in df.columns:
    if "agency" in c.lower():
        col = c
        break
if col is None:
    sys.exit("No 'agency' column found in dataset.")

# Filter rows where rating_agency matches folder
mask = df[col].str.lower().str.contains(agency, na=False)
filtered = df[mask].copy()

# Clean agency name for filename
agency_short = agency.replace("'", "").replace("&", "").replace(" ", "")
out_file = os.path.join(os.getcwd(), f"filteredpromax_kaggle_{agency_short}.csv")
filtered.to_csv(out_file, index=False)
print(f"Saved {len(filtered)} rows → {out_file}")
