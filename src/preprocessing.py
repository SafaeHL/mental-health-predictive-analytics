import pandas as pd
import pyreadstat
from .config import (RAW_DIR, PROCESSED_DIR, COMPONENTS, DPQ_ITEMS,
                     NUMERIC_FEATURES, CATEGORICAL_FEATURES, PARTICIPANT_ID, TARGET)


def read_xpt(component_key: str) -> pd.DataFrame:
    path = RAW_DIR / (COMPONENTS[component_key] + ".XPT")
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `python -m src.download_data` first.")
    frame, _ = pyreadstat.read_xport(str(path))
    return frame


def merge_components() -> pd.DataFrame:
    keys = list(COMPONENTS)
    merged = read_xpt(keys[0])
    for key in keys[1:]:
        part = read_xpt(key)
        if PARTICIPANT_ID not in part.columns:
            raise ValueError(f"{PARTICIPANT_ID} missing from {key} component")
        merged = merged.merge(part, on=PARTICIPANT_ID, how="left", validate="one_to_one")
    return merged


def score_phq9(df: pd.DataFrame) -> pd.DataFrame:
    missing = [col for col in DPQ_ITEMS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing PHQ-9 columns: {missing}")
    out = df.copy()
    for col in DPQ_ITEMS:
        out[col] = pd.to_numeric(out[col], errors="coerce")
        out.loc[~out[col].isin([0, 1, 2, 3]), col] = float("nan")
    complete = out[DPQ_ITEMS].notna().all(axis=1)
    out["phq9_total"] = out[DPQ_ITEMS].sum(axis=1, min_count=len(DPQ_ITEMS))
    out.loc[~complete, "phq9_total"] = float("nan")
    out[TARGET] = pd.NA
    valid = out["phq9_total"].notna()
    out.loc[valid, TARGET] = (out.loc[valid, "phq9_total"] >= 10).astype(int)
    return out


def build_analytic_dataset() -> pd.DataFrame:
    df = merge_components()
    if "RIDAGEYR" not in df.columns:
        raise ValueError("RIDAGEYR not found in demographics component")
    df = df.loc[df["RIDAGEYR"] >= 18].copy()
    df = score_phq9(df)
    available = [c for c in NUMERIC_FEATURES + CATEGORICAL_FEATURES if c in df.columns]
    absent = sorted(set(NUMERIC_FEATURES + CATEGORICAL_FEATURES) - set(available))
    if absent:
        print(f"NOTE: optional predictors absent from downloaded components: {absent}")
    keep = [PARTICIPANT_ID, "phq9_total", TARGET] + available
    for col in ["WTMEC2YR", "WTINT2YR", "SDMVSTRA", "SDMVPSU"]:
        if col in df.columns and col not in keep:
            keep.append(col)
    analytic = df[keep].copy()
    analytic = analytic.loc[analytic[TARGET].notna()].copy()
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    output = PROCESSED_DIR / "nhanes_2017_2018_mental_health.csv"
    analytic.to_csv(output, index=False)
    print(f"Saved {len(analytic):,} eligible adult records to {output}")
    print(f"Outcome counts:\n{analytic[TARGET].value_counts(dropna=False)}")
    return analytic


if __name__ == "__main__":
    build_analytic_dataset()
