from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

CYCLE_BASE = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles"
COMPONENTS = {
    "DEMO": "DEMO_J", "DPQ": "DPQ_J", "SLQ": "SLQ_J",
    "SLD": "SLD_J", "BMX": "BMX_J", "PAQ": "PAQ_J",
}
DPQ_ITEMS = [f"DPQ{i:03d}" for i in range(10, 100, 10)]
NUMERIC_FEATURES = ["RIDAGEYR", "INDFMPIR", "SLD012", "BMXBMI"]
CATEGORICAL_FEATURES = ["RIAGENDR", "DMDEDUC2", "SLQ050", "SLQ060", "PAQ605", "PAQ620", "PAQ650", "PAQ665"]
TARGET = "elevated_symptoms"
PARTICIPANT_ID = "SEQN"
