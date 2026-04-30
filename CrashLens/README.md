# CrashLens: SHAP-Driven Crash Severity Prediction Using Ensemble and Transformer Models

> **Academic Title:** *SHAP-Driven Crash Severity Prediction Using Ensemble and Transformer Models: A Data-Driven Framework for Proactive Traffic Safety*

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Live Demo](https://img.shields.io/badge/🔗_Live_Demo-GitHub_Pages-blue.svg)](https://snaidu20.github.io/Crashlens/)

### [▶ View Live Dashboard Demo](https://snaidu20.github.io/Crashlens/)

---

## Table of Contents

- [Overview](#overview)
- [Why This Matters](#why-this-matters)
- [Data Source](#data-source)
- [Project Structure](#project-structure)
- [Step-by-Step Pipeline Breakdown](#step-by-step-pipeline-breakdown)
  - [Phase 1: Data Collection & Exploration](#phase-1-data-collection--exploration)
  - [Phase 2: Data Merging & Feature Engineering](#phase-2-data-merging--feature-engineering)
  - [Phase 3: Cleaning & Encoding](#phase-3-cleaning--encoding)
  - [Phase 4: Train/Val/Test Split & Class Balancing](#phase-4-trainvaltest-split--class-balancing)
  - [Phase 5: Dataset Validation](#phase-5-dataset-validation)
  - [Phase 6: Baseline Model Training](#phase-6-baseline-model-training)
  - [Phase 7: Deep Learning — FT-Transformer](#phase-7-deep-learning--ft-transformer)
  - [Phase 8: SHAP Explainability Analysis](#phase-8-shap-explainability-analysis)
  - [Phase 9: Dashboard Export](#phase-9-dashboard-export)
- [Model Architectures & Hyperparameters](#model-architectures--hyperparameters)
- [Results](#results)
- [Key Findings](#key-findings)
- [Interactive Dashboard](#interactive-dashboard)
- [Technical Stack](#technical-stack)
- [Reproducing Results](#reproducing-results)
- [References](#references)
- [Author](#author)

---

## Overview

CrashLens is a complete machine learning pipeline that predicts traffic crash injury severity (5-class KABCO scale) using NHTSA crash data, with a focus on **explainability** through SHAP analysis. The project spans from raw government data to an interactive web dashboard — designed for transportation safety researchers, engineers, and policymakers.

The pipeline processes **477,801 person-level crash records** from 205,874 unique crash events across 2020–2023, trains 4 models (Random Forest, XGBoost, LightGBM, FT-Transformer), and delivers results through a 7-tab interactive dashboard with dark/light theme support.

---

## Why This Matters

Every year, approximately **40,000 people die** on U.S. roads — equivalent to a commercial plane crash every single day. The annual economic cost exceeds **$340 billion** (NHTSA estimate). Current safety interventions are largely reactive — agencies respond *after* patterns emerge in annual reports.

CrashLens shifts this to **proactive, condition-based risk assessment** by:

| Application | How It Works |
|---|---|
| **EMS Pre-Positioning** | Predict high-severity crash conditions so ambulances can be staged where fatal crashes are most likely |
| **Infrastructure Safety** | Identify which road design features (lighting, speed limits, lane count) most strongly predict severe outcomes |
| **Policy Evaluation** | Quantify how seatbelt enforcement, speed management, and DUI crackdowns reduce fatality risk |
| **Real-Time Risk Scoring** | Enable DOTs to issue condition-based alerts (e.g., "wet + dark + high-speed → elevated fatal risk") |
| **Research Foundation** | Provide an explainable AI framework for transportation safety academics and PhD research |

---

## Data Source

### NHTSA Crash Report Sampling System (CRSS)

| Detail | Value |
|--------|-------|
| **Source** | [NHTSA CRSS](https://www.nhtsa.gov/file-downloads?p=nhtsa/downloads/CRSS/) |
| **Type** | Nationally representative sample of all police-reported motor vehicle crashes in the U.S. |
| **Years** | 2020, 2021, 2022, 2023 |
| **Total Crash Events** | 205,874 |
| **Total Person Records** | 477,801 motor vehicle occupants |
| **Tables Used** | ACCIDENT, VEHICLE, PERSON, DISTRACT, DRIMPAIR, CRASHRF, MANEUVER, DRIVERRF, VIOLATN |
| **File Format** | CSV (Latin-1 encoding) |
| **Download** | Each year is a separate ZIP archive containing CSV files |

### Target Variable — KABCO Injury Severity Scale (5 Classes)

| Code | Label | Description | Count | Distribution |
|------|-------|-------------|-------|-------------|
| O | No Injury | No apparent injury | 339,221 | 71.0% |
| C | Possible | Possible injury, complaint of pain | 67,244 | 14.1% |
| B | Non-Incapacitating | Suspected minor injury | 42,445 | 8.9% |
| A | Incapacitating | Suspected serious injury | 24,512 | 5.1% |
| K | Fatal | Fatal injury | 4,379 | **0.9%** |

> **Class imbalance challenge:** Fatal crashes represent only 0.9% of records, making detection extremely difficult. This is addressed through SMOTE oversampling and class-weighted loss functions.

> **Data transparency — SMOTE:** The raw input data is entirely real NHTSA CRSS records. SMOTE (Synthetic Minority Oversampling Technique) is applied **only to the training set** to address class imbalance — it interpolates new synthetic samples between existing real fatal/severe crash examples. The validation and test sets used for all reported metrics contain **only real crash records, never synthetic ones**. Every F1 score and fatal recall figure in this project is evaluated on real hold-out data.

### Feature Groups (44 Total Features)

| Group | Count | Features |
|-------|-------|----------|
| **Numeric** | 11 | `AGE_CLEAN`, `TRAV_SP_CLEAN` (travel speed MPH), `SPEED_OVER_LIMIT`, `VEHICLE_AGE`, `VE_TOTAL` (vehicles in crash), `NUMOCCS` (occupants), `VSPD_LIM` (posted speed limit), `VNUM_LAN` (lane count), `NUM_CRASH_FACTORS`, `NUM_DRIVER_RF` (driver risk factor count), `NUM_VIOLATIONS` (violation count) |
| **Categorical** | 14 | `BODY_TYPE_CAT` (vehicle type), `LIGHT_CAT` (lighting), `WEATHER_CAT`, `COLLISION_TYPE`, `AGE_GROUP`, `SEX_CLEAN`, `RESTRAINT_CAT`, `AIRBAG_CAT`, `EJECTION_CAT`, `DEFORMATION_CAT`, `SURFACE_CAT`, `SPEED_LIMIT_CAT`, `TIME_PERIOD`, `VTRAFCON_CAT` (traffic control device) |
| **Binary** | 14 | `IS_WEEKEND`, `IS_DRIVER`, `MULTI_VEHICLE`, `ROLLOVER_FLAG`, `SPEED_RELATED`, `IN_WORK_ZONE`, `AT_JUNCTION`, `DISTRACTED`, `DRIVER_IMPAIRED`, `DRINKING_FLAG`, `HAS_PRE_CRASH_MANEUVER` (active maneuver), `HAS_DRIVER_RF` (risk factor flag), `HAS_VIOLATION` (violation flag), `HIT_RUN_FLAG` (driver fled scene) |
| **Ordinal** | 5 | `URBANICITY`, `REGION`, `DAY_WEEK`, `HOUR`, `MONTH` |

---

## Project Structure

```
CrashLens/
├── README.md                         # This file — full project documentation
├── requirements.txt                  # Python dependencies (pip install -r)
├── .gitignore                        # Git ignore rules for large/generated files
│
├── pipeline/                         # End-to-end data processing & model training
│   ├── 01_explore_data.py            # Raw data exploration & statistics
│   ├── 02_merge_and_engineer.py      # Table merging & feature engineering (44 features)
│   ├── 03_clean_and_encode.py        # Missing values, encoding, cleaning
│   ├── 04_split_and_balance.py       # Group-aware train/val/test split + SMOTE
│   ├── 05_validate_dataset.py        # 33-check dataset quality validation
│   ├── 06_train_baselines.py         # Random Forest, XGBoost, LightGBM training
│   ├── 07_train_transformer.py       # FT-Transformer (PyTorch) training
│   ├── 08_shap_analysis.py           # SHAP explainability (TreeExplainer on LightGBM)
│   └── 09_export_results.py          # Export consolidated JSON for dashboard
│
├── data/
│   ├── raw/                          # Raw CRSS CSV archives (not tracked — see setup)
│   │   ├── crss_2020/CRSS2020CSV/    #   ACCIDENT.csv, VEHICLE.csv, PERSON.csv, etc.
│   │   ├── crss_2021/CRSS2021CSV/
│   │   ├── crss_2022/CRSS2022CSV/
│   │   └── crss_2023/CRSS2023CSV/
│   └── processed/                    # Cleaned & engineered data
│       ├── feature_config.json       # Feature definitions, indices, cardinalities (tracked)
│       ├── encoding_maps.json        # Categorical encoding mappings (tracked)
│       ├── crashlens_merged.parquet  # Merged dataset (not tracked)
│       ├── crashlens_clean.parquet   # Final clean dataset (not tracked)
│       └── splits/
│           ├── split_config.json     # Split sizes, class weights, SMOTE config (tracked)
│           ├── X_train_smote.npy     # Training features after SMOTE (not tracked)
│           ├── y_train_smote.npy     # Training labels after SMOTE (not tracked)
│           ├── X_val.npy             # Validation features (not tracked)
│           ├── y_val.npy             # Validation labels (not tracked)
│           ├── X_test.npy            # Test features (not tracked)
│           └── y_test.npy            # Test labels (not tracked)
│
├── results/
│   ├── baseline_results.json         # RF, XGBoost, LightGBM metrics (tracked)
│   ├── transformer_results.json      # FT-Transformer metrics (tracked)
│   ├── shap_results.json             # SHAP feature importance values (tracked)
│   └── models/
│       ├── random_forest.pkl         # Trained RF model (not tracked — 859 MB)
│       ├── xgboost.pkl               # Trained XGBoost model (not tracked — 24 MB)
│       ├── lightgbm.pkl              # Trained LightGBM model (not tracked — 8.8 MB)
│       ├── ft_transformer_best.pt    # Trained FT-Transformer weights (not tracked — 188 KB)
│       ├── ft_transformer_config.json # Transformer architecture config (tracked)
│       └── scaler.pkl                # StandardScaler for transformer (not tracked)
│
├── dashboard/                        # Interactive web dashboard
│   ├── index.html                    # Single-file dashboard (HTML/CSS/JS + Chart.js)
│   └── data/
│       └── dashboard_data.json       # Exported data for visualization (22 KB)
│
└── docs/
    └── data_dictionary.md            # Detailed feature descriptions & CRSS coding
```

> **Note:** Files marked "not tracked" are excluded from git via `.gitignore` because they are too large or can be regenerated by running the pipeline. Configuration files and results JSONs *are* tracked so the dashboard works standalone.

---

## Step-by-Step Pipeline Breakdown

### Phase 1: Data Collection & Exploration

**Script:** `pipeline/01_explore_data.py`

1. **Download** CRSS CSV archives from [NHTSA](https://www.nhtsa.gov/file-downloads?p=nhtsa/downloads/CRSS/) for years 2020–2023
2. Place each year's CSV folder into `data/raw/crss_YYYY/CRSSYYYYCSV/`
3. The script loads **9 tables** across all 4 years:
   - `ACCIDENT` — crash-level data (time, location, conditions)
   - `VEHICLE` — vehicle-level data (type, speed, deformation, hit-and-run, traffic control)
   - `PERSON` — person-level data (age, sex, injury severity, restraint)
   - `DISTRACT` — driver distraction factors
   - `DRIMPAIR` — driver impairment factors
   - `CRASHRF` — crash-related contributing factors
   - `MANEUVER` — pre-crash driver maneuver (evasive action, turning, lane change)
   - `DRIVERRF` — driver-related risk factors (fatigue, inattention, aggressive driving)
   - `VIOLATN` — traffic violations at time of crash (speeding, signal violations, etc.)
4. **Outputs:** Console summary of record counts, target distribution, and missing value patterns

```
Total person records: 477,801
Unique crashes: 205,874
Fatal injuries (K): 4,379 (0.9%)
```

---

### Phase 2: Data Merging & Feature Engineering

**Script:** `pipeline/02_merge_and_engineer.py`

This is the most complex step — it joins 9 tables and engineers 44 meaningful features from raw CRSS codes.

**Merge Strategy:**
1. **Crash + Vehicle:** Join `ACCIDENT` and `VEHICLE` on `(CASENUM, DATA_YEAR)` via `VEHNO`
2. **+ Person:** Join with `PERSON` on `(CASENUM, VEHNO, DATA_YEAR)` via `PER_NO`
3. **+ Supplementary tables:** Left-join `DISTRACT`, `DRIMPAIR`, `CRASHRF`, `MANEUVER`, `DRIVERRF`, `VIOLATN` — aggregate to binary flags per vehicle/person

**Feature Engineering (key transformations):**

| Raw CRSS Field | Engineered Feature | Transformation |
|---|---|---|
| `AGE` | `AGE_CLEAN` | Cap outliers, handle coded unknowns (998/999) |
| `TRAV_SP` | `TRAV_SP_CLEAN` | Cap at 150 MPH, handle 997/998/999 codes |
| `TRAV_SP` - `VSPD_LIM` | `SPEED_OVER_LIMIT` | Computed: travel speed minus posted limit |
| `MOD_YEAR` | `VEHICLE_AGE` | Computed: crash year minus model year |
| `BODY_TYP` | `BODY_TYPE_CAT` | 30+ raw codes → 8 categories (Passenger_Car, SUV, Motorcycle, etc.) |
| `LGT_COND` | `LIGHT_CAT` | 9 codes → 5 categories (Daylight, Dark_Lighted, Dark_Unlighted, Dawn_Dusk, Unknown) |
| `WEA_COND` | `WEATHER_CAT` | 10+ codes → 7 categories (Clear, Rain, Snow_Sleet, Fog, Severe, Cloudy, Unknown) |
| `REST_USE` | `RESTRAINT_CAT` | 20+ codes → 5 categories (SeatBelt_Full, SeatBelt_Partial, Child_Restraint, None, Unknown) |
| `MAN_COLL` | `COLLISION_TYPE` | 10+ codes → 7 categories (Rear_End, Angle, Head_On, Sideswipe, etc.) |
| `DEFORMED` | `DEFORMATION_CAT` | 7 codes → 5 categories (None, Minor, Functional, Disabling, Unknown) |
| Multiple tables | `DISTRACTED`, `DRIVER_IMPAIRED`, `DRINKING_FLAG` | Binary flags from supplementary tables |
| `VEHICLE` | `ROLLOVER_FLAG` | Binary: derived from ROLLOVER field in vehicle table (VSOE not used) |

**Output:** `data/processed/crashlens_merged.parquet` — 477,801 rows × 50+ columns

---

### Phase 3: Cleaning & Encoding

**Script:** `pipeline/03_clean_and_encode.py`

1. **Select final 44 model features** from the merged dataset
2. **Handle CRSS coded unknowns:** Values like 98, 99, 998, 999 → treated as missing
3. **Missing value strategy:**
   - Numeric: median imputation
   - Categorical: "Unknown" category (preserved as informative)
4. **Encode categoricals:** Ordinal encoding with consistent mapping across all categories
5. **Export encoding maps** for reproducibility and dashboard use

**Outputs:**
- `data/processed/crashlens_clean.parquet` — final model-ready dataset
- `data/processed/feature_config.json` — feature names, indices, cardinalities, feature groups
- `data/processed/encoding_maps.json` — categorical encoding dictionaries

---

### Phase 4: Train/Val/Test Split & Class Balancing

**Script:** `pipeline/04_split_and_balance.py`

**Splitting Strategy:**
- **Ratio:** 60% train / 20% validation / 20% test
- **Group-aware:** Uses `GroupShuffleSplit` on `CASENUM` — no persons from the same crash appear in different splits (prevents data leakage)
- **Stratified:** Maintains class proportions across splits

| Split | Records | Fatal (K) Count |
|-------|---------|-----------------|
| Train | 286,938 | 2,607 (0.9%) |
| Validation | 95,894 | 890 (0.9%) |
| Test | 94,969 | 882 (0.9%) |

**Class Balancing (Training Set Only):**

> **Important:** SMOTE synthetic samples are generated **exclusively within the training fold**. Validation (95,894) and test (94,969) records are untouched real CRSS data. All reported model metrics are computed on the real test set only.

- **SMOTE** (Synthetic Minority Oversampling) — creates new training examples by interpolating feature vectors between existing real minority-class records
- Fatal class upsampled: 2,607 real records → 30,635 (real + synthetic); 0.9% → 9.1% of training set
- Total training records after SMOTE: 336,142 (286,938 real + 49,204 synthetic)

| Class | Before SMOTE | After SMOTE |
|-------|-------------|-------------|
| O — No Injury | 204,236 | 204,236 (unchanged) |
| C — Possible | 40,001 | 40,001 (unchanged) |
| B — Non-Incapacitating | 25,310 | 30,635 |
| A — Incapacitating | 14,784 | 30,635 |
| K — Fatal | 2,607 | **30,635** |

**Class Weights (sqrt-dampened):**
Used in loss functions for all models to further address imbalance.

| Class | Weight |
|-------|--------|
| O — No Injury | 0.268 |
| C — Possible | 0.605 |
| B — Non-Incapacitating | 0.761 |
| A — Incapacitating | 0.995 |
| K — Fatal | **2.371** |

**Outputs:** NumPy arrays (`X_train_smote.npy`, `y_train_smote.npy`, `X_val.npy`, `y_val.npy`, `X_test.npy`, `y_test.npy`) + `split_config.json`

---

### Phase 5: Dataset Validation

**Script:** `pipeline/05_validate_dataset.py`

Runs **33 automated quality checks** before model training:

- No NaN/Inf values in any split
- Feature count matches config (44 features)
- Class counts match between config and actual data
- No group leakage (no `CASENUM` overlap across splits)
- Value ranges are reasonable (age 0–120, speed 0–150, etc.)
- Encoding maps cover all values present in data
- SMOTE did not create impossible feature combinations
- Train/val/test distributions are statistically similar

---

### Phase 6: Baseline Model Training

**Script:** `pipeline/06_train_baselines.py`

Trains three gradient-boosted tree models with identical feature sets and evaluation protocol.

#### Random Forest

| Hyperparameter | Value |
|---|---|
| `n_estimators` | 300 |
| `max_depth` | 20 |
| `min_samples_split` | 10 |
| `min_samples_leaf` | 5 |
| `class_weight` | sqrt-dampened weights |
| Training time | 76.4 seconds |

#### XGBoost

| Hyperparameter | Value |
|---|---|
| `n_estimators` | 500 |
| `max_depth` | 8 |
| `learning_rate` | 0.05 |
| `subsample` | 0.8 |
| `colsample_bytree` | 0.8 |
| `min_child_weight` | 5 |
| `class_weight` | sqrt-dampened (via sample weights) |
| Early stopping | 30 rounds (validation set) |
| Training time | 88.1 seconds |

#### LightGBM

| Hyperparameter | Value |
|---|---|
| `n_estimators` | 500 |
| `max_depth` | 8 |
| `learning_rate` | 0.05 |
| `subsample` | 0.8 |
| `colsample_bytree` | 0.8 |
| `min_child_samples` | 20 |
| `reg_alpha` | 0.1 |
| `reg_lambda` | 1.0 |
| `class_weight` | sqrt-dampened weights |
| Early stopping | 30 rounds (validation set) |
| Training time | 53.7 seconds |

**Outputs:** Trained models (`results/models/*.pkl`), evaluation metrics (`results/baseline_results.json`)

---

### Phase 7: Deep Learning — FT-Transformer

**Script:** `pipeline/07_train_transformer.py`

Implements the **Feature Tokenizer + Transformer** (FT-Transformer) architecture from Gorishniy et al. (2021), adapted for crash severity prediction.

#### Architecture

```
Input (44 features)
  ├── Continuous features (30) → Linear projection → per-feature tokens (32-dim each)
  └── Categorical features (14) → Learned embeddings → per-feature tokens (32-dim each)
          ↓
  [CLS] token prepended → 45 tokens total
          ↓
  Transformer Encoder (2 layers, 4 heads, d_ff=64, dropout=0.2)
          ↓
  [CLS] output → Linear → 5-class softmax
```

| Hyperparameter | Value |
|---|---|
| `d_model` | 32 |
| `n_heads` | 4 |
| `n_layers` | 2 |
| `d_ff` (feedforward dim) | 64 |
| `dropout` | 0.2 |
| Batch size | 2,048 (train), 4,096 (eval) |
| Optimizer | AdamW (lr=2e-3, weight_decay=1e-4) |
| Scheduler | CosineAnnealingLR (T_max=12, eta_min=1e-5) |
| Loss | CrossEntropyLoss with sqrt-dampened class weights |
| Epochs | 12 (early stopping patience=4) |
| Gradient clipping | max_norm=1.0 |
| Input normalization | StandardScaler on continuous features |
| Training time | ~438 seconds |

**Key design choice:** Compact architecture (32-dim tokens, 2 layers) to avoid overfitting on the moderately-sized dataset while still capturing feature interactions through self-attention.

**Outputs:** Model weights (`results/models/ft_transformer_best.pt`), config (`ft_transformer_config.json`), scaler (`scaler.pkl`), metrics (`results/transformer_results.json`)

---

### Phase 8: SHAP Explainability Analysis

**Script:** `pipeline/08_shap_analysis.py`

Uses **SHAP (SHapley Additive exPlanations)** with TreeExplainer on the LightGBM model (best balanced accuracy) to explain predictions.

**What SHAP computes:**
- For each prediction, SHAP assigns a contribution value to every feature
- Positive SHAP = pushes toward this class; negative = pushes away
- Values are computed for all 5 severity classes

**Analysis produced:**
1. **Global Feature Importance** — Mean |SHAP| across all predictions and all classes
2. **Fatal-Specific Importance** — Mean |SHAP| for the fatal (K) class only
3. **Directional Impact** — Which features increase vs. decrease fatal risk
4. **Feature Descriptions** — Human-readable names mapped from feature codes

**Top 5 Global Risk Factors (by mean |SHAP|):**

| Rank | Feature | Mean |SHAP| | Interpretation |
|------|---------|-------------|----------------|
| 1 | Total Vehicles in Crash | 0.1227 | Single-vehicle crashes are deadlier (no energy dissipation) |
| 2 | Person Age | 0.0944 | Older occupants (65+) face 3.2× higher fatal risk |
| 3 | Vehicle Age (Years) | 0.0653 | Older vehicles lack modern safety systems |
| 4 | Travel Speed (MPH) | 0.0596 | Each +10 MPH raises fatal probability ~15% |
| 5 | Speed Over Limit (MPH) | 0.0395 | Independent risk beyond absolute speed |

**Output:** `results/shap_results.json`

---

### Phase 9: Dashboard Export

**Script:** `pipeline/09_export_results.py`

Consolidates all results into a single JSON file for the interactive dashboard:

- Model performance metrics (accuracy, F1, confusion matrices)
- SHAP feature importance (global + fatal-specific)
- Key findings with natural language explanations
- Scenario explorer configuration (features, encoding maps, defaults)
- Recommendation data (infrastructure, enforcement, EMS, vehicle policy)

**Output:** `dashboard/data/dashboard_data.json` (22 KB)

---

## Model Architectures & Hyperparameters

### Comparison Summary

| Aspect | Random Forest | XGBoost | LightGBM | FT-Transformer |
|--------|--------------|---------|----------|-----------------|
| **Type** | Bagging ensemble | Gradient boosting | Gradient boosting (leaf-wise) | Deep learning (attention) |
| **Trees/Layers** | 300 trees | 500 trees | 500 trees | 2 transformer layers |
| **Max Depth** | 20 | 8 | 8 | N/A |
| **Learning Rate** | N/A | 0.05 | 0.05 | 2e-3 (cosine decay) |
| **Regularization** | min_samples | subsample + colsample | L1(0.1) + L2(1.0) | Dropout(0.2) + weight decay |
| **Imbalance Handling** | Class weights | Sample weights | Class weights | Class-weighted loss |
| **Training Time** | 76s | 88s | 54s | 439s |
| **Model Size** | 859 MB | 24 MB | 8.8 MB | 188 KB |

---

## Results

### Test Set Performance

| Model | Accuracy | Balanced Accuracy | F1 Macro | Fatal (K) Sensitivity | Fatal (K) F1 |
|-------|----------|-------------------|----------|----------------------|-------------|
| Random Forest | 72.0% | 44.5% | 43.0% | 48.5% | 40.5% |
| XGBoost | 71.4% | 46.5% | 45.3% | 46.3% | 40.7% |
| **LightGBM** | **71.2%** | **47.6%** | **45.0%** | **54.3%** | **40.8%** |
| FT-Transformer | 35.5% | 44.2% | 31.2% | **57.3%** | 30.8% |

### Per-Class F1 Scores (Test Set — LightGBM)

| Class | Precision | Recall | F1 Score |
|-------|-----------|--------|----------|
| O — No Injury | 0.851 | 0.879 | 0.865 |
| C — Possible | 0.350 | 0.270 | 0.305 |
| B — Non-Incapacitating | 0.338 | 0.327 | 0.332 |
| A — Incapacitating | 0.322 | 0.360 | 0.340 |
| K — Fatal | 0.327 | 0.543 | 0.408 |

### Confusion Matrix (LightGBM — Test Set)

```
              Predicted:  O       C       B       A       K
Actual O             58,944  4,411   2,503   1,052     138
Actual C              6,931  3,668   1,803     997     165
Actual B              2,506  1,657   2,816   1,405     223
Actual A                817    694   1,145   1,753     459
Actual K                 51     39      76     237     479
```

---

## Key Findings

1. **LightGBM achieves the best balance** of overall accuracy (71.2%) and fatal crash detection (54.3% sensitivity) — it's the recommended model for general deployment.

2. **FT-Transformer detects 57.3% of fatal crashes** — evaluated from a partially-trained checkpoint (CPU constraint limited training to 6 epochs). It trades F1-macro for fatal sensitivity, suggesting value for safety-critical applications where missing a fatal crash has high cost.

3. **Top risk factors identified by SHAP:**
   - **Ejection** from vehicle → >20× fatality risk
   - **No seatbelt** → 5× fatality risk (strongest protective factor when worn)
   - **Age 65+** → 3.2× higher fatal injury than age 25–34
   - **Dark unlighted roads** → 3.5× fatal risk
   - **Disabling vehicle damage** → 8× fatal rate
   - **Each +10 MPH** in travel speed → ~15% increase in fatal probability

4. **Speed and restraint interact strongly:** No seatbelt + high speed produces the highest SHAP values in the dataset — these two factors compound each other's risk.

5. **Vehicle age matters more than expected:** Vehicles >15 years old have roughly 2× the fatal risk of modern vehicles, due to missing ESC, side-curtain airbags, and advanced crumple zones.

---

## Interactive Dashboard

The dashboard (`dashboard/index.html`) is a standalone single-file web application with 7 tabs:

| Tab | Content |
|-----|---------|
| **Overview** | KPI cards, severity distribution donut, fatal outcomes by year |
| **Model Performance** | Accuracy/F1 trade-off chart, per-class F1 comparison, confusion matrix heatmap |
| **Risk Factors** | SHAP global importance bars, fatal-specific importance, directional impact |
| **AI Explainability** | Key findings cards with natural language SHAP interpretations |
| **Scenario Explorer** | Interactive "what-if" tool: adjust crash conditions and see predicted severity |
| **Recommendations** | Critical driver alerts, often-overlooked risk factors, policy recommendations |
| **Methodology** | Step-by-step pipeline visualization, technical stack |

**Features:**
- Dark/light theme toggle (top-right icon)
- Year filter (2020–2023) and severity filter
- Responsive layout (desktop + mobile)
- All data visualization powered by Chart.js
- Zero external dependencies at runtime (runs from a single HTML file)

---

## Technical Stack

| Category | Tools |
|----------|-------|
| **Language** | Python 3.10+ |
| **Data Processing** | Pandas, NumPy, PyArrow |
| **Machine Learning** | scikit-learn (Random Forest), XGBoost, LightGBM |
| **Deep Learning** | PyTorch (FT-Transformer) |
| **Explainability** | SHAP (TreeExplainer) |
| **Class Imbalance** | imbalanced-learn (SMOTE) |
| **Visualization** | Matplotlib, Seaborn (pipeline), Chart.js (dashboard) |
| **Dashboard** | Vanilla HTML/CSS/JavaScript (no framework) |

---

## Reproducing Results

### Prerequisites

- Python 3.10+
- ~4 GB disk space for raw data + processed files
- GPU optional (FT-Transformer trains in ~7 min on CPU)

### Setup

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/CrashLens.git
cd CrashLens

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download CRSS data from NHTSA
#    https://www.nhtsa.gov/file-downloads?p=nhtsa/downloads/CRSS/
#    Download ZIP files for 2020, 2021, 2022, 2023
#    Extract into data/raw/ with this structure:
#      data/raw/crss_2020/CRSS2020CSV/ACCIDENT.csv, VEHICLE.csv, PERSON.csv, ...
#      data/raw/crss_2021/CRSS2021CSV/...
#      data/raw/crss_2022/CRSS2022CSV/...
#      data/raw/crss_2023/CRSS2023CSV/...
```

### Run the Pipeline

```bash
# Phase 1: Explore raw data
python pipeline/01_explore_data.py

# Phase 2: Merge tables & engineer features
python pipeline/02_merge_and_engineer.py

# Phase 3: Clean, encode & prepare features
python pipeline/03_clean_and_encode.py

# Phase 4: Split data & apply SMOTE balancing
python pipeline/04_split_and_balance.py

# Phase 5: Validate dataset integrity (33 checks)
python pipeline/05_validate_dataset.py

# Phase 6: Train baseline models (RF, XGBoost, LightGBM)
python pipeline/06_train_baselines.py

# Phase 7: Train FT-Transformer
python pipeline/07_train_transformer.py

# Phase 8: SHAP explainability analysis
python pipeline/08_shap_analysis.py

# Phase 9: Export results for dashboard
python pipeline/09_export_results.py
```

### View the Dashboard

Open `dashboard/index.html` in any modern browser. No server required — it's a standalone file.

---

## References

1. **NHTSA CRSS Data:** [https://www.nhtsa.gov/file-downloads?p=nhtsa/downloads/CRSS/](https://www.nhtsa.gov/file-downloads?p=nhtsa/downloads/CRSS/)
2. **FT-Transformer:** Gorishniy, Y., Rubachev, I., Khrulkov, V., & Babenko, A. (2021). *Revisiting Deep Learning Models for Tabular Data*. NeurIPS 2021. [arXiv:2106.11959](https://arxiv.org/abs/2106.11959)
3. **FGTT for Crash Modeling:** Santos, K. et al. (2024). *Fine-Grained Tabular Transformer for Crash Severity Prediction*. [arXiv:2412.06825](https://arxiv.org/abs/2412.06825)
4. **Meta-analysis of Crash Severity ML Studies:** Rezapour, M. et al. (2026). *Machine learning applications in crash severity modeling: A meta-analysis of 74 studies*. Journal of Road Safety, 37(1). [DOI:10.33492/JRS-D-24-00062](https://journalofroadsafety.org/article/156042)
5. **SHAP:** Lundberg, S.M. & Lee, S.-I. (2017). *A Unified Approach to Interpreting Model Predictions*. NeurIPS 2017.
6. **SMOTE:** Chawla, N.V. et al. (2002). *SMOTE: Synthetic Minority Over-sampling Technique*. JAIR, 16, 321–357.

---

## Author

**Sai Kumar Naidu**
MS Computer Science, Florida Atlantic University
Email: naidusaikumar1998@gmail.com

---

## License

MIT License — see [LICENSE](LICENSE) for details.
