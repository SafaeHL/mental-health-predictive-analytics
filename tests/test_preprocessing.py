import pandas as pd
from src.preprocessing import score_phq9
from src.config import DPQ_ITEMS, TARGET


def test_phq9_complete_valid_responses():
    row = {column: 1 for column in DPQ_ITEMS}
    result = score_phq9(pd.DataFrame([row]))
    assert result.loc[0, "phq9_total"] == 9
    assert int(result.loc[0, TARGET]) == 0


def test_phq9_elevated_threshold():
    row = {column: 1 for column in DPQ_ITEMS}
    row["DPQ010"] = 2
    result = score_phq9(pd.DataFrame([row]))
    assert result.loc[0, "phq9_total"] == 10
    assert int(result.loc[0, TARGET]) == 1


def test_special_codes_are_missing_not_zero():
    row = {column: 0 for column in DPQ_ITEMS}
    row["DPQ090"] = 7
    result = score_phq9(pd.DataFrame([row]))
    assert pd.isna(result.loc[0, "phq9_total"])
    assert pd.isna(result.loc[0, TARGET])
