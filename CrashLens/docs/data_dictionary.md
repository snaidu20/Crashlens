# CrashLens — Data Dictionary

## Source
NHTSA Crash Report Sampling System (CRSS), 2020–2023  
https://www.nhtsa.gov/file-downloads?p=nhtsa/downloads/CRSS/

## Target Variable

| Column | Type | Description |
|--------|------|-------------|
| `SEVERITY` | Integer (0–4) | Injury severity class |

### Severity Classes
| Code | KABCO | Label | Description |
|------|-------|-------|-------------|
| 0 | O | No Injury | No apparent injury to the person |
| 1 | C | Possible | Possible injury (complaint of pain, limping, etc.) |
| 2 | B | Non-Incapacitating | Suspected minor injury (visible bruise, swelling) |
| 3 | A | Incapacitating | Suspected serious injury (broken bones, severe lacerations) |
| 4 | K | Fatal | Fatal injury (death within 30 days of crash) |

---

## Numeric Features (9)

| Feature | Source Table | Description | Range | Imputation |
|---------|-------------|-------------|-------|------------|
| `AGE_CLEAN` | Person | Person's age in years | 0–100 | Median (34) |
| `TRAV_SP_CLEAN` | Vehicle | Travel speed in MPH | 0–150 | Median (20) |
| `SPEED_OVER_LIMIT` | Derived | Travel speed minus posted speed limit | -50–80 | Median (-15) |
| `VEHICLE_AGE` | Derived | Data year minus model year | 0–50 | Median (7) |
| `VE_TOTAL` | Accident | Total vehicles involved in crash | 1–15 | None needed |
| `NUMOCCS` | Vehicle | Number of occupants in the vehicle | 1–10 | Median (1) |
| `VSPD_LIM` | Vehicle | Posted speed limit (MPH) | 0–90 | Median (40) |
| `VNUM_LAN` | Vehicle | Number of travel lanes | 0–7 | Median (3) |
| `NUM_CRASH_FACTORS` | CrashRF | Count of crash-related factors | 0–5 | 0 (filled) |

---

## Categorical Features (13)

### BODY_TYPE_CAT — Vehicle Body Type
| Category | CRSS Codes | Examples |
|----------|------------|---------|
| Passenger_Car | 1–9 | Sedan, coupe, hatchback, station wagon |
| SUV_Crossover | 10–19 | Compact/large utility vehicles |
| Pickup_Truck | 30–39 | Light/standard pickup trucks |
| Van_Minivan | 20–29 | Passenger van, minivan, cargo van |
| Large_Truck_Bus | 40–79 | Semi, bus, medium/heavy truck |
| Motorcycle | 80–89 | Two-wheel motorcycle, moped |
| Other | 90–97 | ATV, snowmobile, other |
| Unknown | 99, NaN | Unknown/not reported |

### LIGHT_CAT — Lighting Condition
| Category | CRSS Code | Description |
|----------|-----------|-------------|
| Daylight | 1 | Normal daylight conditions |
| Dark_Lighted | 3 | Darkness with street lighting |
| Dark_Unlighted | 2, 6 | Darkness without lighting / unknown lighting |
| Dawn_Dusk | 4, 5 | Dawn or dusk (transitional) |
| Unknown | 8, 9 | Not reported / unknown |

### WEATHER_CAT — Weather Condition
| Category | CRSS Code | Description |
|----------|-----------|-------------|
| Clear | 1 | Clear skies |
| Cloudy | 10 | Overcast/cloudy |
| Rain | 2 | Rain |
| Snow_Sleet | 4, 11 | Snow, sleet, freezing rain |
| Fog | 5 | Fog, smog, smoke |
| Severe | 6, 7, 8, 12 | Severe crosswinds, blowing sand, other |
| Unknown | 98, 99 | Not reported / unknown |

### COLLISION_TYPE — Manner of Collision
| Category | CRSS Code | Description |
|----------|-----------|-------------|
| Rear_End | 1 | Front-to-rear collision |
| Angle | 6 | Angle (side-impact) collision |
| Head_On | 2 | Front-to-front collision |
| Sideswipe | 7, 8 | Same or opposite direction sideswipe |
| Non_Collision | 0 | Not a collision with another vehicle (rollover, fixed object) |
| Other | 9, 10, 11 | Rear-to-side, rear-to-rear, other |
| Unknown | — | Unknown/not reported |

### AGE_GROUP — Age Category
| Category | Age Range |
|----------|-----------|
| Child | 0–15 |
| Young_Adult | 16–24 |
| Adult_25_34 | 25–34 |
| Adult_35_44 | 35–44 |
| Adult_45_54 | 45–54 |
| Adult_55_64 | 55–64 |
| Senior_65_74 | 65–74 |
| Elderly_75plus | 75+ |
| Unknown | Not reported (998/999) |

### SEX_CLEAN — Sex
| Category | CRSS Code |
|----------|-----------|
| Male | 1 |
| Female | 2 |
| Unknown | 8, 9 |

### RESTRAINT_CAT — Restraint Use
| Category | Description |
|----------|-------------|
| SeatBelt_Full | Shoulder + lap belt |
| SeatBelt_Partial | Lap or shoulder only |
| Child_Restraint | Child seat (forward/rear-facing, booster) |
| None | No restraint used |
| Unknown | Not reported / unknown |

### AIRBAG_CAT — Airbag Deployment
| Category | Description |
|----------|-------------|
| Deployed | Any airbag deployed (front, side, combination) |
| Not_Deployed | No airbag deployment |
| Unknown | Not reported / unknown |

### EJECTION_CAT — Ejection Status
| Category | Description |
|----------|-------------|
| Not_Ejected | Occupant remained in vehicle |
| Ejected | Totally or partially ejected |
| Not_Applicable | Non-occupant |
| Unknown | Not reported / unknown |

### DEFORMATION_CAT — Vehicle Deformation
| Category | CRSS Code | Description |
|----------|-----------|-------------|
| None | 0 | No visible damage |
| Minor | 2 | Cosmetic damage only |
| Functional | 4 | Vehicle drivable but damaged |
| Disabling | 6 | Vehicle not drivable |
| Unknown | 7, 8, 9 | Not reported / unknown |

### SURFACE_CAT — Road Surface Condition
| Category | Description |
|----------|-------------|
| Dry | Dry pavement |
| Wet | Wet pavement |
| Snow_Ice | Snow, ice, slush, or frost |
| Not_Trafficway | Not on a trafficway |
| Unknown | Not reported / unknown |

### SPEED_LIMIT_CAT — Speed Limit Category
| Category | Range |
|----------|-------|
| Low_25orLess | ≤ 25 MPH |
| Medium_30_35 | 30–35 MPH |
| Medium_40_45 | 40–45 MPH |
| High_50_55 | 50–55 MPH |
| Very_High_60plus | ≥ 60 MPH |
| Unknown | Not reported |

### TIME_PERIOD — Time of Day
| Category | Hours |
|----------|-------|
| Late_Night | 00:00–05:59 |
| Morning_Rush | 06:00–09:59 |
| Midday | 10:00–15:59 |
| Evening_Rush | 16:00–19:59 |
| Night | 20:00–23:59 |
| Unknown | Not reported |

---

## Binary Features (10)

| Feature | Description | 1 = Yes |
|---------|-------------|---------|
| `IS_WEEKEND` | Crash on weekend | Saturday or Sunday |
| `IS_DRIVER` | Person is driver | PER_TYP = 1 |
| `MULTI_VEHICLE` | Multi-vehicle crash | VE_TOTAL > 1 |
| `ROLLOVER_FLAG` | Rollover occurred | ROLLOVER ∈ {1,2,3,9} |
| `SPEED_RELATED` | Speed-related crash | SPEEDREL ∈ {2,3,4,5} |
| `IN_WORK_ZONE` | Crash in work zone | WRK_ZONE ∈ {1,2,3,4} |
| `AT_JUNCTION` | Crash at junction | TYP_INT ∈ {2,3,4,5,6,7} |
| `DISTRACTED` | Driver was distracted | Any DRDISTRACT ≠ 0 |
| `DRIVER_IMPAIRED` | Driver was impaired | Any DRIMPAIR ≠ 0 |
| `DRINKING_FLAG` | Alcohol involvement | 1=Yes, 0=No, -1=Unknown |

---

## Ordinal Features (5)

| Feature | Description | Values |
|---------|-------------|--------|
| `URBANICITY` | Urban vs rural | 1=Urban, 2=Rural |
| `REGION` | U.S. Census region | 1=NE, 2=MW, 3=South, 4=West |
| `DAY_WEEK` | Day of week | 1=Sun, 2=Mon, ..., 7=Sat |
| `HOUR` | Hour of crash | 0–23 |
| `MONTH` | Month of crash | 1–12 |

---

## Meta Columns (not model features)

| Column | Description |
|--------|-------------|
| `CASENUM` | CRSS case number (crash ID) |
| `DATA_YEAR` | Year of data (2020–2023) |
| `WEIGHT` | CRSS sampling weight (for national estimates) |
| `SEVERITY_LABEL` | Human-readable severity label |

---

## Data Processing Notes

1. **Population:** Motor vehicle occupants only (drivers + passengers). Pedestrians and cyclists excluded — their injury mechanisms differ fundamentally and would need a separate model.
2. **Missing Values:** CRSS encodes unknowns as 98, 99, 998, 999, 9998, 9999 depending on the field. All decoded to NaN then imputed (numeric → median, categorical → "Unknown" category).
3. **Group-Aware Splitting:** All persons from the same crash (CASENUM) are in the same split to prevent data leakage.
4. **SMOTE:** Applied only to training set. Validation and test sets retain original distribution.
5. **Encoding:** Label encoding for categoricals (integer indices for embedding layers).
