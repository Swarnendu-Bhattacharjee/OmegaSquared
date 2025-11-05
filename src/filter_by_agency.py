import pandas as pd, os, sys
folder = os.path.basename(os.getcwd())
mapping = {
    "Egan-Jones Ratings Company": "Egan",
    "Fitch Ratings": "Fitch",
    "Moody's Investors Service": "Moody",
    "Standard & Poor's Ratings Services": "S&P"
}
target = mapping.get(folder)
if not target: sys.exit(f"Unknown folder: {folder}")
base = "../data/" + folder + "/corporate_credit_rating_with_financial_ratios.csv"
df = pd.read_csv(base)
mask = df['rating_agency'].str.contains(target, case=False, na=False)
df = df[mask].copy()
df.to_csv(f"../data/{folder}/filteredpro_kaggle.csv", index=False)
print(f"Saved {len(df)} rows for {target}")
