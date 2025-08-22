import pandas as pd
from pathlib import Path

DATA = Path("data/example_logs.csv")

def main():
    df = pd.read_csv(DATA)
    # Simple normalization example: lowercase, strip codes
    df["clean_message"] = df["message"].str.lower().str.replace(r"alm-\d+", "ALM-CODE", regex=True)
    print(df.head())
    out = Path("data/clean_logs.parquet")
    df.to_parquet(out, index=False)
    print(f"Saved {out}")

if __name__ == "__main__":
    main()