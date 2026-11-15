# Mental Health Predictive Analytics

An educational HealthTech data-science project using public CDC NHANES 2017–2018 data to explore factors associated with elevated depressive symptoms in adults and compare baseline machine-learning models.

> **Research prototype only.** This project does not diagnose depression, provide treatment recommendations, or replace professional assessment. It is not validated for clinical use.

## Research question

Among adults aged 18 and older represented in the NHANES 2017–2018 sample, how well can selected demographic, sleep, physical-activity, and body-measurement variables distinguish participants with PHQ-9 scores of 10 or higher from those with scores below 10?

The outcome is **elevated depressive symptoms by a PHQ-9 screening threshold**, not a diagnosis of major depressive disorder.

## Data sources

The download script retrieves public NHANES SAS transport (`.XPT`) files from CDC:

| Component | File | Purpose |
|---|---|---|
| Demographics | `DEMO_J.XPT` | Age, sex, education, income-to-poverty ratio, survey design fields |
| Depression Screener | `DPQ_J.XPT` | Nine PHQ-9 questionnaire items |
| Sleep Disorders | `SLQ_J.XPT` | Self-reported sleep-related indicators |
| Sleep Duration | `SLD_J.XPT` | Weekday/workday sleep duration |
| Body Measures | `BMX_J.XPT` | BMI |
| Physical Activity | `PAQ_J.XPT` | Physical activity indicators |

Official source: [CDC NHANES 2017–2018](https://wwwn.cdc.gov/nchs/nhanes/continuousnhanes/default.aspx?BeginYear=2017)

Component documentation:
- [DEMO_J](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/DEMO_J.htm)
- [DPQ_J](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/DPQ_J.htm)
- [SLQ_J](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/SLQ_J.htm)
- [SLD_J](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/SLD_J.htm)
- [BMX_J](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/BMX_J.htm)
- [PAQ_J](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/PAQ_J.htm)

Files are downloaded on demand. Do not commit raw or processed participant-level data to Git.

## Outcome definition and scoring

The nine DPQ items are `DPQ010` through `DPQ090`. Valid responses are 0–3. NHANES special response codes (commonly 7 = refused and 9 = don't know) are treated as missing, never as zero.

A complete-case PHQ-9 sum is calculated only when all nine items have valid responses:
- PHQ-9 total: 0–27
- `elevated_symptoms = 1` when total >= 10
- `elevated_symptoms = 0` when total < 10

This is a screening threshold for research classification, not a clinical diagnosis. PHQ-9 items and the total score are excluded from model predictors to prevent target leakage.

## Quick start

Python 3.10+ is recommended.

```bash
git clone https://github.com/SafaeHL/mental-health-predictive-analytics.git
cd mental-health-predictive-analytics
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python -m src.download_data
python -m src.build_dataset
python -m src.train_model
streamlit run dashboard/app.py
```

## Project workflow

1. Download six public NHANES component files.
2. Merge records by `SEQN` (participant identifier).
3. Restrict the cohort to adults (age >= 18).
4. Score PHQ-9 only for participants with all nine valid responses.
5. Exclude DPQ items and PHQ-9 total from predictors to prevent target leakage.
6. Fit Logistic Regression and Random Forest with a stratified train/test split.
7. Evaluate accuracy, precision, recall, F1, ROC-AUC, PR-AUC, and confusion matrices.
8. Explore aggregate patterns in the Streamlit dashboard.

## Methodological limitations

- NHANES is a US survey; results are not representative of Canada or Morocco.
- The selected 2017–2018 cycle is historical and cross-sectional.
- Self-reported measures may contain recall and reporting error.
- Complete-case PHQ-9 scoring can introduce selection bias.
- The scikit-learn workflow is an **unweighted exploratory sample-level analysis** and does not produce nationally representative estimates. Population inference requires appropriate NHANES weights, strata, PSU variables, and survey methods.
- Random train/test splitting does not account for complex survey design.
- Associations and feature importance do not establish causation.
- Predicted probabilities must not be interpreted as an individual's clinical risk or diagnosis.

## Repository layout

```text
data/                 # downloaded data; ignored by Git
notebooks/            # analysis notebook(s)
src/                  # download, preprocessing, training code
dashboard/            # Streamlit app
models/               # local model artifacts; ignored by Git
reports/              # generated metrics and figures
tests/                # unit tests
```

## License and disclaimer

Code is provided under the MIT License. NHANES data are public-use data supplied by CDC; consult CDC terms and component documentation. This repository does not redistribute NHANES files. For education and research only—not for diagnosis, treatment, triage, or individual healthcare decisions.
