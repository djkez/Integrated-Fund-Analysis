#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Merge FF-style factor CSVs (2–4 files) into one unified table.

Usage examples:
    python "Merge FF Momentum.py" "FF Research Data 5 Factors 2x3.csv" "FF Momentum Factor.csv"
    python "Merge FF Momentum.py" "Emerging_5_Factors.csv" "Emerging_MOM_Factor.csv"
    python "Merge FF Momentum.py" "Emerging_5_Factors.csv" "Emerging_MOM_Factor.csv" "FF Research Data 5 Factors 2x3.csv" "FF Momentum Factor.csv" -o merged_all.csv
"""
import argparse, pandas as pd, re
from pathlib import Path

def read_ff_csv(path: str) -> pd.DataFrame:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    start_idx = None
    for i, line in enumerate(lines):
        parts = [p.strip() for p in line.strip().split(",")]
        if parts and re.fullmatch(r"\d{6}", parts[0] or ""):
            start_idx = i
            break
    if start_idx is None:
        return pd.read_csv(path)

    header_idx = start_idx - 1
    while header_idx >= 0 and (not lines[header_idx].strip()):
        header_idx -= 1
    if header_idx < 0:
        header_idx = start_idx - 1

    nrows = 0
    for j in range(start_idx, len(lines)):
        L = lines[j].strip().lower()
        if not L or L.startswith(("annual", "monthly", "copyright", "notes")):
            break
        nrows += 1

    df = pd.read_csv(path, header=0, skiprows=header_idx, nrows=(nrows if nrows>0 else None), engine="python")
    df.columns = [str(c).strip() for c in df.columns]
    return df

def clean_ff_table(df: pd.DataFrame) -> pd.DataFrame:
    """Standardise a FF table: parse date, convert to decimals if needed."""
    date_col = None
    for c in df.columns:
        if re.fullmatch(r"(date|yyyymm|year|caldt)", str(c).strip().lower()):
            date_col = c
            break
    if date_col is None:
        date_col = df.columns[0]

    s = df[date_col].astype(str).str.strip()
    if s.str.fullmatch(r"\d{6}").all():
        dt = pd.to_datetime(s.str[:4] + "-" + s.str[4:] + "-01") + pd.offsets.MonthEnd(0)
    else:
        dt = pd.to_datetime(s, errors="coerce")

    out = df.copy()
    out["Date"] = dt
    out = out.dropna(subset=["Date"]).drop(columns=[date_col])

    num_cols = [c for c in out.columns if pd.api.types.is_numeric_dtype(out[c])]
    out = out[num_cols + ["Date"]]

    # convert percents to decimals
    for c in num_cols:
        ser = pd.to_numeric(out[c], errors="coerce")
        med = ser.abs().median()
        if pd.notna(med) and med > 0 and med <= 50:
            out[c] = ser / 100.0
        else:
            out[c] = ser
    return out.sort_values("Date").reset_index(drop=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csvs", nargs="+", help="Paths to 2–4 FF CSVs to merge")
    ap.add_argument("-o", "--out", default="merged_factors.csv", help="Output CSV path")
    args = ap.parse_args()

    if len(args.csvs) < 2 or len(args.csvs) > 4:
        raise SystemExit("Please provide between 2 and 4 CSV files.")

    dfs = []
    for path in args.csvs:
        print(f"Reading: {path}")
        df = read_ff_csv(path)
        dfs.append(clean_ff_table(df))

    merged = dfs[0]
    for extra in dfs[1:]:
        merged = pd.merge(merged, extra, on="Date", how="inner")

    # Drop duplicate columns (same name)
    merged = merged.loc[:, ~merged.columns.duplicated()]

    # Reorder Date first
    cols = ["Date"] + [c for c in merged.columns if c != "Date"]
    merged = merged[cols]

    merged.to_csv(args.out, index=False)
    print(f"\nWrote merged file: {args.out}")
    print(merged.head())

if __name__ == "__main__":
    main()
