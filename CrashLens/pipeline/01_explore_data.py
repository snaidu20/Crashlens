"""
CrashLens — Step 1: Data Exploration
Explore CRSS 2020-2023 data: target variable distribution, key features,
missing value patterns, and record counts.
"""
import pandas as pd
import numpy as np
import os, json

RAW_DIR = "/home/user/workspace/crashlens/data/raw"
YEARS = [2020, 2021, 2022, 2023]

# ── 1. Load & combine core tables across years ──────────────────────────
def load_table(table_name):
    """Load a specific table from all years, adding YEAR column."""
    frames = []
    for y in YEARS:
        folder = f"crss_{y}/CRSS{y}CSV"
        fp = os.path.join(RAW_DIR, folder, f"{table_name}.csv")
        df = pd.read_csv(fp, low_memory=False, encoding='latin-1')
        df["DATA_YEAR"] = y
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True)
    return combined

print("Loading person table (all years)...")
person = load_table("person")
print(f"  Person records: {len(person):,}")

print("Loading accident table (all years)...")
accident = load_table("accident")
print(f"  Accident records: {len(accident):,}")

print("Loading vehicle table (all years)...")
vehicle = load_table("vehicle")
print(f"  Vehicle records: {len(vehicle):,}")

# ── 2. Target variable: INJ_SEV ─────────────────────────────────────────
print("\n" + "="*70)
print("TARGET VARIABLE: INJ_SEV (Injury Severity)")
print("="*70)

# Get the severity distribution with labels
sev_dist = person.groupby(["INJ_SEV", "INJ_SEVNAME"]).size().reset_index(name="count")
sev_dist["pct"] = (sev_dist["count"] / sev_dist["count"].sum() * 100).round(2)
sev_dist = sev_dist.sort_values("INJ_SEV")
print(sev_dist.to_string(index=False))

print("\n--- By Year ---")
yearly_sev = person.groupby(["DATA_YEAR", "INJ_SEV"]).size().unstack(fill_value=0)
print(yearly_sev.to_string())

# ── 3. Key crash-level features ─────────────────────────────────────────
print("\n" + "="*70)
print("KEY CRASH-LEVEL FEATURES (accident table)")
print("="*70)

key_accident_cols = [
    "URBANICITY", "DAY_WEEK", "HOUR", "MONTH", 
    "LGT_COND", "WEATHER", "MAN_COLL", "TYP_INT", 
    "REL_ROAD", "WRK_ZONE", "MAX_SEV", "ALCOHOL"
]
for col in key_accident_cols:
    name_col = col + "NAME"
    if name_col in accident.columns:
        vc = accident[[col, name_col]].value_counts().head(8).reset_index(name="count")
        print(f"\n{col}:")
        print(vc.to_string(index=False))

# ── 4. Key vehicle-level features ───────────────────────────────────────
print("\n" + "="*70)
print("KEY VEHICLE-LEVEL FEATURES (vehicle table)")
print("="*70)

key_vehicle_cols = [
    "BODY_TYP", "TRAV_SP", "SPEEDREL", "ROLLOVER", 
    "DEFORMED", "VTRAFWAY", "VSPD_LIM", "VALIGN",
    "VPROFILE", "VSURCOND", "VTRAFCON"
]
for col in key_vehicle_cols:
    name_col = col + "NAME"
    if name_col in vehicle.columns:
        vc = vehicle[[col, name_col]].value_counts().head(8).reset_index(name="count")
        print(f"\n{col}:")
        print(vc.to_string(index=False))

# ── 5. Key person-level features ────────────────────────────────────────
print("\n" + "="*70)
print("KEY PERSON-LEVEL FEATURES (person table)")
print("="*70)

key_person_cols = [
    "AGE", "SEX", "PER_TYP", "SEAT_POS", "REST_USE",
    "HELM_USE", "AIR_BAG", "EJECTION", "DRINKING", "DRUGS"
]
for col in key_person_cols:
    name_col = col + "NAME"
    if name_col in person.columns:
        vc = person[[col, name_col]].value_counts().head(8).reset_index(name="count")
        print(f"\n{col}:")
        print(vc.to_string(index=False))

# ── 6. Missing value overview (coded as 98, 99, 998, 999, etc.) ────────
print("\n" + "="*70)
print("MISSING VALUES / UNKNOWN CODES — Accident Table")
print("="*70)
# In CRSS, missing/unknown values are coded as 98, 99, 998, 999, 9998, 9999 etc.
# Also NaN for true nulls
for col in accident.select_dtypes(include=[np.number]).columns:
    null_count = accident[col].isna().sum()
    # Count common "unknown" codes
    unknown_codes = [8, 9, 98, 99, 998, 999, 9998, 9999, 99999]
    unknown_count = accident[col].isin(unknown_codes).sum()
    if null_count > 0 or unknown_count > 500:
        pct = ((null_count + unknown_count) / len(accident) * 100)
        if pct > 1:
            print(f"  {col}: NaN={null_count}, Unknown-coded={unknown_count} ({pct:.1f}%)")

print("\n--- Person Table ---")
for col in person.select_dtypes(include=[np.number]).columns:
    null_count = person[col].isna().sum()
    unknown_codes = [8, 9, 98, 99, 998, 999, 9998, 9999, 99999]
    unknown_count = person[col].isin(unknown_codes).sum()
    if null_count > 0 or unknown_count > 500:
        pct = ((null_count + unknown_count) / len(person) * 100)
        if pct > 1:
            print(f"  {col}: NaN={null_count}, Unknown-coded={unknown_count} ({pct:.1f}%)")

# ── 7. Summary stats ────────────────────────────────────────────────────
print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"Total crashes (accident records): {len(accident):,}")
print(f"Total vehicles: {len(vehicle):,}")
print(f"Total persons: {len(person):,}")
print(f"Years covered: {YEARS}")
print(f"\nUnique crashes per year:")
for y in YEARS:
    n = accident[accident["DATA_YEAR"] == y]["CASENUM"].nunique()
    print(f"  {y}: {n:,}")

# Map for the 5-class severity target
severity_map = {0: "O_NoInjury", 1: "C_Possible", 2: "B_NonIncap", 3: "A_Incap", 4: "K_Fatal"}
valid_person = person[person["INJ_SEV"].isin([0, 1, 2, 3, 4])].copy()
valid_person["SEVERITY_CLASS"] = valid_person["INJ_SEV"].map(severity_map)

print(f"\nPerson records with valid 5-class severity: {len(valid_person):,} "
      f"({len(valid_person)/len(person)*100:.1f}% of total)")
print("\n5-Class distribution:")
dist = valid_person["SEVERITY_CLASS"].value_counts().sort_index()
for cls, cnt in dist.items():
    print(f"  {cls}: {cnt:,} ({cnt/len(valid_person)*100:.1f}%)")

print("\nDone!")
