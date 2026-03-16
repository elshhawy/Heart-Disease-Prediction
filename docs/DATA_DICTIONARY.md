# Data Dictionary — Heart Disease Dataset

## Source
**Kaggle:** [Heart Failure Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction)  
**Combined from:** Cleveland, Hungarian, Switzerland, Long Beach VA, Stalog datasets  
**Rows:** 918 patients | **Columns:** 12 (11 features + 1 target)

---

## Features

| Column | Type | Values | Description |
|--------|------|--------|-------------|
| `Age` | int | 28 – 77 | Patient age in years |
| `Sex` | str → int | M=1, F=0 | Biological sex |
| `ChestPainType` | str → dummy | ATA, NAP, ASY, TA | Type of chest pain (see below) |
| `RestingBP` | int | 0 – 200 mm Hg | Resting blood pressure |
| `Cholesterol` | int | 0 – 603 mg/dL | Serum cholesterol. **Note:** 0 = missing value in original data |
| `FastingBS` | int | 0, 1 | Fasting blood sugar > 120 mg/dL (1 = true) |
| `RestingECG` | str → dummy | Normal, ST, LVH | Resting electrocardiogram results (see below) |
| `MaxHR` | int | 60 – 202 bpm | Maximum heart rate achieved during exercise |
| `ExerciseAngina` | str → int | Y=1, N=0 | Exercise-induced angina |
| `Oldpeak` | float | -2.6 – 6.2 | ST depression induced by exercise relative to rest |
| `ST_Slope` | str → dummy | Up, Flat, Down | Slope of the peak exercise ST segment (see below) |

---

## Target

| Column | Type | Values | Description |
|--------|------|--------|-------------|
| `HeartDisease` | int | 0, 1 | **1** = Heart disease present, **0** = Normal |

---

## Categorical Value Details

### ChestPainType
| Value | Full Name | Clinical Meaning |
|-------|-----------|-----------------|
| `ATA` | Atypical Angina | Chest pain not matching classic angina pattern |
| `NAP` | Non-Anginal Pain | Chest pain unrelated to heart |
| `ASY` | Asymptomatic | No chest pain — often highest risk group |
| `TA`  | Typical Angina | Classic angina pattern (pressure, exertion-triggered) |

### RestingECG
| Value | Full Name | Clinical Meaning |
|-------|-----------|-----------------|
| `Normal` | Normal | No significant findings |
| `ST` | ST-T Wave Abnormality | T wave inversions / ST elevation or depression |
| `LVH` | Left Ventricular Hypertrophy | R wave > 11mm (Estes criteria) |

### ST_Slope
| Value | Clinical Meaning |
|-------|-----------------|
| `Up` | Upsloping — generally normal/benign |
| `Flat` | Flat — associated with ischemia |
| `Down` | Downsloping — strongest indicator of disease |

---

## Encoding Applied in Pipeline

```
Sex            M → 1,  F → 0
ExerciseAngina Y → 1,  N → 0

One-Hot (drop_first=True):
  ChestPainType  → ChestPainType_NAP, ChestPainType_ASY, ChestPainType_TA
  RestingECG     → RestingECG_Normal, RestingECG_ST
  ST_Slope       → ST_Slope_Flat,     ST_Slope_Up
```

---

## Known Data Quality Notes

- `Cholesterol = 0` appears in the original dataset and likely represents **missing values** rather than true zero cholesterol. Consider imputing (median) in future iterations.
- `RestingBP = 0` for one patient — likely a data entry error. Can be filtered or imputed.
