"""
CrashLens — Step 2: Merge Tables & Feature Engineering
Merges accident + vehicle + person tables, selects meaningful features,
engineers new features, and handles CRSS-specific coding conventions.
"""
import pandas as pd
import numpy as np
import os, warnings
warnings.filterwarnings("ignore")

RAW_DIR = "/home/user/workspace/crashlens/data/raw"
OUT_DIR = "/home/user/workspace/crashlens/data/processed"
YEARS = [2020, 2021, 2022, 2023]

def load_table(table_name):
    frames = []
    for y in YEARS:
        folder = f"crss_{y}/CRSS{y}CSV"
        fp = os.path.join(RAW_DIR, folder, f"{table_name}.csv")
        df = pd.read_csv(fp, low_memory=False, encoding="latin-1")
        df["DATA_YEAR"] = y
        frames.append(df)
    return pd.concat(frames, ignore_index=True)

# ═══════════════════════════════════════════════════════════════════════
# 1. LOAD CORE TABLES
# ═══════════════════════════════════════════════════════════════════════
print("Loading tables...")
person = load_table("person")
accident = load_table("accident")
vehicle = load_table("vehicle")
print(f"  Person: {len(person):,} | Accident: {len(accident):,} | Vehicle: {len(vehicle):,}")

# ═══════════════════════════════════════════════════════════════════════
# 2. FILTER PERSON TABLE — valid severity + motor vehicle occupants
# ═══════════════════════════════════════════════════════════════════════
print("\nFiltering person records...")
# Keep only valid 5-class severity (0=O, 1=C, 2=B, 3=A, 4=K)
person = person[person["INJ_SEV"].isin([0, 1, 2, 3, 4])].copy()
print(f"  After severity filter: {len(person):,}")

# Keep motor vehicle occupants only (drivers + passengers)
# PER_TYP: 1=Driver, 2=Passenger
person = person[person["PER_TYP"].isin([1, 2])].copy()
print(f"  After MV occupant filter: {len(person):,}")

# ═══════════════════════════════════════════════════════════════════════
# 3. SELECT RELEVANT FEATURES FROM EACH TABLE
# ═══════════════════════════════════════════════════════════════════════
# Strategy: pick meaningful features, avoid redundant NAME columns
# and duplicate fields that appear across tables

# -- Person features (unit of analysis) --
person_cols = [
    "CASENUM", "VEH_NO", "PER_NO", "DATA_YEAR", "WEIGHT",
    # Target
    "INJ_SEV",
    # Demographics
    "AGE", "SEX",
    # Person type & position
    "PER_TYP", "SEAT_POS",
    # Safety equipment
    "REST_USE", "HELM_USE", "AIR_BAG", "EJECTION",
    # Substance involvement
    "DRINKING",
    # Additional
    "ALC_STATUS",
]
person_sel = person[person_cols].copy()

# -- Accident features (crash-level) --
accident_cols = [
    "CASENUM", "DATA_YEAR",
    # Location/environment
    "REGION", "URBANICITY",
    # Time
    "MONTH", "DAY_WEEK", "HOUR",
    # Crash characteristics
    "HARM_EV", "MAN_COLL",
    # Junction/road
    "RELJCT2", "TYP_INT", "REL_ROAD",
    # Conditions
    "LGT_COND", "WEATHER", "WRK_ZONE",
    # Crash severity summary
    "VE_TOTAL", "PEDS", "PERMVIT", "PERNOTMVIT",
    # Alcohol at crash level
    "ALCOHOL",
    # School bus
    "SCH_BUS",
]
accident_sel = accident[accident_cols].copy()

# -- Vehicle features --
vehicle_cols = [
    "CASENUM", "VEH_NO", "DATA_YEAR",
    # Vehicle characteristics
    "BODY_TYP", "MOD_YEAR", "NUMOCCS",
    # Crash dynamics
    "TRAV_SP", "SPEEDREL", "ROLLOVER", "DEFORMED",
    "IMPACT1", "UNDEROVERRIDE",
    # Road characteristics (vehicle-level)
    "VTRAFWAY", "VNUM_LAN", "VSPD_LIM",
    "VALIGN", "VPROFILE", "VSURCOND", "VTRAFCON",
    # Pre-crash
    "P_CRASH1", "P_CRASH2",
    # Vehicle type
    "UNITTYPE", "HIT_RUN",
    # Special use
    "BUS_USE", "SPEC_USE", "EMER_USE",
    # Towing
    "TOWED",
    # Fire
    "FIRE_EXP",
]
vehicle_sel = vehicle[vehicle_cols].copy()

# ═══════════════════════════════════════════════════════════════════════
# 4. MERGE TABLES
# ═══════════════════════════════════════════════════════════════════════
print("\nMerging tables...")

# Person + Accident (on CASENUM + DATA_YEAR)
merged = person_sel.merge(
    accident_sel, on=["CASENUM", "DATA_YEAR"], how="left"
)
print(f"  After person+accident merge: {len(merged):,}")

# + Vehicle (on CASENUM + VEH_NO + DATA_YEAR)
merged = merged.merge(
    vehicle_sel, on=["CASENUM", "VEH_NO", "DATA_YEAR"], how="left"
)
print(f"  After +vehicle merge: {len(merged):,}")

# ═══════════════════════════════════════════════════════════════════════
# 5. LOAD & AGGREGATE SUPPLEMENTARY TABLES
# ═══════════════════════════════════════════════════════════════════════
print("\nAggregating supplementary tables...")

# -- Distraction: aggregate to vehicle level (any distraction flag) --
distract = load_table("distract")
# DRDISTRACT: 0 = Not Distracted, other values = types of distraction
dist_agg = distract.groupby(["CASENUM", "VEH_NO", "DATA_YEAR"]).agg(
    DISTRACTED=("DRDISTRACT", lambda x: int((x != 0).any())),
    NUM_DISTRACTIONS=("DRDISTRACT", lambda x: int((x != 0).sum()))
).reset_index()
merged = merged.merge(dist_agg, on=["CASENUM", "VEH_NO", "DATA_YEAR"], how="left")
print(f"  +distraction: {len(merged):,}")

# -- Driver impairment: aggregate to vehicle level --
drimpair = load_table("drimpair")
# DRIMPAIR: 0 = None, other = types of impairment
imp_agg = drimpair.groupby(["CASENUM", "VEH_NO", "DATA_YEAR"]).agg(
    DRIVER_IMPAIRED=("DRIMPAIR", lambda x: int((x != 0).any())),
).reset_index()
merged = merged.merge(imp_agg, on=["CASENUM", "VEH_NO", "DATA_YEAR"], how="left")
print(f"  +impairment: {len(merged):,}")

# -- Weather: aggregate to crash level (worst weather) --
# Use the accident-level WEATHER already included; skip multi-weather for simplicity

# -- Crash-related factors: aggregate to crash level --
crashrf = load_table("crashrf")
crashrf_agg = crashrf.groupby(["CASENUM", "DATA_YEAR"]).agg(
    NUM_CRASH_FACTORS=("CRASHRF", lambda x: int((x != 0).sum()))
).reset_index()
merged = merged.merge(crashrf_agg, on=["CASENUM", "DATA_YEAR"], how="left")
print(f"  +crash factors: {len(merged):,}")

# -- Pre-crash maneuver: aggregate to vehicle level --
# MANEUVER: 0=Going Straight, 98=Not Reported, 99=Unknown — any other = active maneuver
maneuver = load_table("maneuver")
man_agg = maneuver.groupby(["CASENUM", "VEH_NO", "DATA_YEAR"]).agg(
    HAS_PRE_CRASH_MANEUVER=("MANEUVER", lambda x: int(((x != 0) & (~x.isin([98, 99]))).any()))
).reset_index()
merged = merged.merge(man_agg, on=["CASENUM", "VEH_NO", "DATA_YEAR"], how="left")
print(f"  +maneuver (pre-crash action): {len(merged):,}")

# -- Driver risk factors: aggregate to vehicle level --
# DRIVERRF: 0=None, >0=specific risk behavior (e.g. careless driving, erratic operation)
driverrf_tbl = load_table("driverrf")
driverrf_agg = driverrf_tbl.groupby(["CASENUM", "VEH_NO", "DATA_YEAR"]).agg(
    HAS_DRIVER_RF=("DRIVERRF", lambda x: int((x != 0).any())),
    NUM_DRIVER_RF=("DRIVERRF", lambda x: int((x != 0).sum()))
).reset_index()
merged = merged.merge(driverrf_agg, on=["CASENUM", "VEH_NO", "DATA_YEAR"], how="left")
print(f"  +driver risk factors: {len(merged):,}")

# -- Traffic violations: aggregate to vehicle level --
# VIOLATION: 0=None, >0=specific violation (fail to yield, speeding, DUI, etc.)
violatn_tbl = load_table("violatn")
viol_agg = violatn_tbl.groupby(["CASENUM", "VEH_NO", "DATA_YEAR"]).agg(
    HAS_VIOLATION=("VIOLATION", lambda x: int((x != 0).any())),
    NUM_VIOLATIONS=("VIOLATION", lambda x: int((x != 0).sum()))
).reset_index()
merged = merged.merge(viol_agg, on=["CASENUM", "VEH_NO", "DATA_YEAR"], how="left")
print(f"  +violations: {len(merged):,}")

# ═══════════════════════════════════════════════════════════════════════
# 6. FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════════
print("\nEngineering features...")

# -- Time features --
# Time of day categories
def hour_to_period(h):
    if h in [99, 98]: return "Unknown"
    if 6 <= h < 10: return "Morning_Rush"
    if 10 <= h < 16: return "Midday"
    if 16 <= h < 20: return "Evening_Rush"
    if 20 <= h < 24: return "Night"
    return "Late_Night"  # 0-5

merged["TIME_PERIOD"] = merged["HOUR"].apply(hour_to_period)

# Weekend flag
merged["IS_WEEKEND"] = merged["DAY_WEEK"].isin([1, 7]).astype(int)  # Sun=1, Sat=7

# -- Vehicle age --
merged["VEHICLE_AGE"] = merged["DATA_YEAR"] - merged["MOD_YEAR"]
# Clean impossible values
merged.loc[merged["VEHICLE_AGE"] < 0, "VEHICLE_AGE"] = np.nan
merged.loc[merged["VEHICLE_AGE"] > 50, "VEHICLE_AGE"] = np.nan
merged.loc[merged["MOD_YEAR"].isin([9999, 9998]), "VEHICLE_AGE"] = np.nan

# -- Speed features --
# Clean TRAV_SP: 998=Not Reported, 999=Unknown
merged["TRAV_SP_CLEAN"] = merged["TRAV_SP"].copy()
merged.loc[merged["TRAV_SP"].isin([998, 999, 997]), "TRAV_SP_CLEAN"] = np.nan

# Speed relative to limit
merged["SPEED_OVER_LIMIT"] = np.where(
    (merged["TRAV_SP_CLEAN"].notna()) & (merged["VSPD_LIM"].notna()) & 
    (~merged["VSPD_LIM"].isin([98, 99])),
    merged["TRAV_SP_CLEAN"] - merged["VSPD_LIM"],
    np.nan
)

# -- Body type categories (simplify 100+ codes into meaningful groups) --
def categorize_body(bt):
    if pd.isna(bt): return "Unknown"
    bt = int(bt)
    if bt in [1,2,3,4,5,6,7,8,9]: return "Passenger_Car"
    if bt in [10,11,12,13,14,15,16,17,19]: return "SUV_Crossover"
    if bt in [20,21,22,23,24,25,28,29]: return "Van_Minivan"
    if bt in [30,31,32,33,34,39]: return "Pickup_Truck"
    if bt in [40,41,42,43,44,45,48,49,50,51,52,55,58,59,60,61,62,63,64,
              65,66,67,68,69,71,72,73,78,79]: return "Large_Truck_Bus"
    if bt in [80,81,82,83,88,89]: return "Motorcycle"
    if bt in [90,91,92,93,94,95,97]: return "Other"
    return "Unknown"

merged["BODY_TYPE_CAT"] = merged["BODY_TYP"].apply(categorize_body)

# -- Light condition categories --
def categorize_light(lc):
    if pd.isna(lc): return "Unknown"
    lc = int(lc)
    if lc == 1: return "Daylight"
    if lc in [3]: return "Dark_Lighted"
    if lc in [2, 6]: return "Dark_Unlighted"
    if lc in [4, 5]: return "Dawn_Dusk"
    return "Unknown"

merged["LIGHT_CAT"] = merged["LGT_COND"].apply(categorize_light)

# -- Weather categories --
def categorize_weather(w):
    if pd.isna(w): return "Unknown"
    w = int(w)
    if w == 1: return "Clear"
    if w == 10: return "Cloudy"
    if w == 2: return "Rain"
    if w in [4, 11]: return "Snow_Sleet"
    if w in [5]: return "Fog"
    if w in [6, 7, 8, 12]: return "Severe"
    return "Unknown"

merged["WEATHER_CAT"] = merged["WEATHER"].apply(categorize_weather)

# -- Collision type categories --
def categorize_collision(mc):
    if pd.isna(mc): return "Unknown"
    mc = int(mc)
    if mc == 0: return "Non_Collision"
    if mc == 1: return "Rear_End"
    if mc == 2: return "Head_On"
    if mc == 6: return "Angle"
    if mc in [7, 8]: return "Sideswipe"
    if mc in [9, 10, 11]: return "Other"
    return "Unknown"

merged["COLLISION_TYPE"] = merged["MAN_COLL"].apply(categorize_collision)

# -- Age group --
def age_group(a):
    if pd.isna(a) or a >= 998: return "Unknown"
    a = int(a)
    if a < 16: return "Child"
    if a < 25: return "Young_Adult"
    if a < 35: return "Adult_25_34"
    if a < 45: return "Adult_35_44"
    if a < 55: return "Adult_45_54"
    if a < 65: return "Adult_55_64"
    if a < 75: return "Senior_65_74"
    return "Elderly_75plus"

merged["AGE_GROUP"] = merged["AGE"].apply(age_group)

# -- Clean AGE: replace coded unknowns with NaN --
merged["AGE_CLEAN"] = merged["AGE"].copy()
merged.loc[merged["AGE"].isin([998, 999]), "AGE_CLEAN"] = np.nan

# -- Sex: binary encode (drop unknowns later) --
# 1=Male, 2=Female, 8=Not Reported, 9=Unknown
merged["SEX_CLEAN"] = merged["SEX"].map({1: "Male", 2: "Female"})
merged.loc[merged["SEX"].isin([8, 9]), "SEX_CLEAN"] = "Unknown"

# -- Restraint use simplified --
def categorize_restraint(r):
    if pd.isna(r): return "Unknown"
    r = int(r)
    if r in [3]: return "SeatBelt_Full"
    if r in [1, 2]: return "SeatBelt_Partial"
    if r in [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]: return "Child_Restraint"
    if r in [0, 20]: return "None"
    if r == 96: return "Not_Applicable"
    return "Unknown"

merged["RESTRAINT_CAT"] = merged["REST_USE"].apply(categorize_restraint)

# -- Airbag deployment simplified --
def categorize_airbag(ab):
    if pd.isna(ab): return "Unknown"
    ab = int(ab)
    if ab == 20: return "Not_Deployed"
    if ab in [1, 2, 3, 7, 8, 9]: return "Deployed"
    if ab == 97: return "Not_Applicable"
    return "Unknown"

merged["AIRBAG_CAT"] = merged["AIR_BAG"].apply(categorize_airbag)

# -- Ejection simplified --
def categorize_ejection(e):
    if pd.isna(e): return "Unknown"
    e = int(e)
    if e == 0: return "Not_Ejected"
    if e in [1, 2, 3]: return "Ejected"
    if e == 8: return "Not_Applicable"
    return "Unknown"

merged["EJECTION_CAT"] = merged["EJECTION"].apply(categorize_ejection)

# -- Person type --
merged["IS_DRIVER"] = (merged["PER_TYP"] == 1).astype(int)

# -- Multi-vehicle crash --
merged["MULTI_VEHICLE"] = (merged["VE_TOTAL"] > 1).astype(int)

# -- Rollover binary --
merged["ROLLOVER_FLAG"] = merged["ROLLOVER"].apply(
    lambda x: 0 if x == 0 else (1 if x in [1, 2, 3, 9] else 0)
)

# -- Speed-related binary --
merged["SPEED_RELATED"] = merged["SPEEDREL"].apply(
    lambda x: 1 if x in [2, 3, 4, 5] else 0
)

# -- Deformation severity --
def categorize_deformation(d):
    if pd.isna(d): return "Unknown"
    d = int(d)
    if d == 0: return "None"
    if d == 2: return "Minor"
    if d == 4: return "Functional"
    if d == 6: return "Disabling"
    return "Unknown"

merged["DEFORMATION_CAT"] = merged["DEFORMED"].apply(categorize_deformation)

# -- Surface condition --
def categorize_surface(s):
    if pd.isna(s): return "Unknown"
    s = int(s)
    if s == 1: return "Dry"
    if s == 2: return "Wet"
    if s in [3, 4, 10, 11]: return "Snow_Ice"
    if s == 0: return "Not_Trafficway"
    return "Unknown"

merged["SURFACE_CAT"] = merged["VSURCOND"].apply(categorize_surface)

# -- Work zone flag --
merged["IN_WORK_ZONE"] = merged["WRK_ZONE"].apply(
    lambda x: 1 if x in [1, 2, 3, 4] else 0
)

# -- Junction flag --
merged["AT_JUNCTION"] = merged["TYP_INT"].apply(
    lambda x: 0 if x == 1 else (1 if x in [2, 3, 4, 5, 6, 7] else 0)
)

# -- Drinking flag (simplified) --
merged["DRINKING_FLAG"] = merged["DRINKING"].apply(
    lambda x: 1 if x == 1 else (0 if x == 0 else -1)  # -1 = Unknown
)

# -- Driver impaired (fill NaN from non-driver passengers) --
merged["DRIVER_IMPAIRED"] = merged["DRIVER_IMPAIRED"].fillna(0).astype(int)
merged["DISTRACTED"] = merged["DISTRACTED"].fillna(0).astype(int)
merged["NUM_DISTRACTIONS"] = merged["NUM_DISTRACTIONS"].fillna(0).astype(int)
merged["NUM_CRASH_FACTORS"] = merged["NUM_CRASH_FACTORS"].fillna(0).astype(int)

# -- Hit and run binary (from vehicle table) --
# HIT_RUN: 0=No, 1=Yes (driver fled scene)
merged["HIT_RUN_FLAG"] = merged["HIT_RUN"].apply(
    lambda x: 1 if x == 1 else 0
)

# -- Traffic control device at scene (from vehicle table) --
# VTRAFCON CRSS coding: 0=No Controls, 1-6=Signal types, 7-10=Sign types,
#   20-28=RR Crossing devices, 40=School Zone, 97-99=Other/Unknown
def categorize_vtrafcon(v):
    if pd.isna(v): return "Unknown"
    v = int(v)
    if v == 0: return "No_Control"
    if 1 <= v <= 6: return "Traffic_Signal"
    if 7 <= v <= 10: return "Sign_Control"
    if 20 <= v <= 28: return "RR_Crossing"
    if v == 40: return "School_Zone"
    return "Unknown"

merged["VTRAFCON_CAT"] = merged["VTRAFCON"].apply(categorize_vtrafcon)

# -- Fill NaN for new aggregated columns (non-linked vehicles get 0) --
for col in ["HAS_PRE_CRASH_MANEUVER", "HAS_DRIVER_RF", "NUM_DRIVER_RF",
            "HAS_VIOLATION", "NUM_VIOLATIONS"]:
    merged[col] = merged[col].fillna(0).astype(int)

# -- Speed limit categories --
def categorize_speed_limit(sl):
    if pd.isna(sl) or sl in [98, 99]: return "Unknown"
    sl = int(sl)
    if sl <= 25: return "Low_25orLess"
    if sl <= 35: return "Medium_30_35"
    if sl <= 45: return "Medium_40_45"
    if sl <= 55: return "High_50_55"
    return "Very_High_60plus"

merged["SPEED_LIMIT_CAT"] = merged["VSPD_LIM"].apply(categorize_speed_limit)

# ═══════════════════════════════════════════════════════════════════════
# 7. DEFINE TARGET VARIABLE
# ═══════════════════════════════════════════════════════════════════════
severity_map = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4}
severity_labels = {0: "O_NoInjury", 1: "C_Possible", 2: "B_NonIncap", 3: "A_Incap", 4: "K_Fatal"}
merged["SEVERITY"] = merged["INJ_SEV"].map(severity_map)
merged["SEVERITY_LABEL"] = merged["INJ_SEV"].map(severity_labels)

# ═══════════════════════════════════════════════════════════════════════
# 8. SAVE MERGED DATASET
# ═══════════════════════════════════════════════════════════════════════
os.makedirs(OUT_DIR, exist_ok=True)
out_path = os.path.join(OUT_DIR, "crashlens_merged.parquet")
merged.to_parquet(out_path, index=False)
print(f"\nSaved merged dataset: {out_path}")
print(f"  Shape: {merged.shape}")
print(f"  Columns: {len(merged.columns)}")

# Summary
print(f"\n{'='*70}")
print("MERGED DATASET SUMMARY")
print(f"{'='*70}")
print(f"Records: {len(merged):,}")
print(f"Columns: {len(merged.columns)}")
print(f"\nSeverity distribution:")
for sev, label in sorted(severity_labels.items()):
    cnt = (merged["SEVERITY"] == sev).sum()
    pct = cnt / len(merged) * 100
    print(f"  {sev} ({label}): {cnt:,} ({pct:.1f}%)")

print(f"\nNew features added (vs. previous run):")
for col in ["HAS_PRE_CRASH_MANEUVER", "HAS_DRIVER_RF", "NUM_DRIVER_RF",
            "HAS_VIOLATION", "NUM_VIOLATIONS", "HIT_RUN_FLAG", "VTRAFCON_CAT"]:
    if col in merged.columns:
        if merged[col].dtype == object or str(merged[col].dtype) in ['string', 'category']:
            print(f"  {col}: {merged[col].value_counts().to_dict()}")
        else:
            print(f"  {col}: mean={merged[col].mean():.3f}, sum={int(merged[col].sum())}")

print(f"\nEngineered feature categories:")
for col in ["BODY_TYPE_CAT", "LIGHT_CAT", "WEATHER_CAT", "COLLISION_TYPE",
            "AGE_GROUP", "RESTRAINT_CAT", "AIRBAG_CAT", "EJECTION_CAT",
            "DEFORMATION_CAT", "SURFACE_CAT", "SPEED_LIMIT_CAT", "TIME_PERIOD", "VTRAFCON_CAT"]:
    vc = merged[col].value_counts()
    print(f"\n  {col}:")
    for val, cnt in vc.items():
        print(f"    {val}: {cnt:,} ({cnt/len(merged)*100:.1f}%)")

print("\nDone!")
