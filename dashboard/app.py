from pathlib import Path
import json
import sys
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.config import PROCESSED_DIR, REPORTS_DIR, TARGET

st.set_page_config(page_title="Mental Health Analytics", page_icon="🧠", layout="wide")
st.title("Mental Health Predictive Analytics")
st.caption("NHANES 2017–2018 | Educational research prototype | Not a diagnostic tool")
data_path = PROCESSED_DIR / "nhanes_2017_2018_mental_health.csv"
if not data_path.exists():
    st.warning("No prepared dataset found. Run `python -m src.download_data` and `python -m src.build_dataset` from the project root.")
    st.stop()
df = pd.read_csv(data_path)
if df.empty:
    st.error("The prepared dataset is empty.")
    st.stop()
st.info("Historical US survey sample only. This dashboard does not provide individual diagnosis, treatment, or clinical risk assessment.")
c1, c2, c3 = st.columns(3)
c1.metric("Eligible adult records", f"{len(df):,}")
c2.metric("Elevated-symptom records", f"{int((df[TARGET] == 1).sum()):,}")
c3.metric("PHQ-9 complete-case mean", f"{df['phq9_total'].mean():.1f}")
st.subheader("PHQ-9 score distribution")
fig, ax = plt.subplots()
sns.histplot(df["phq9_total"].dropna(), bins=14, discrete=True, ax=ax)
ax.set_xlabel("PHQ-9 total score")
ax.set_ylabel("Number of participants")
st.pyplot(fig)
plt.close(fig)
st.subheader("Elevated depressive symptoms by selected characteristic")
groups = [c for c in ["RIAGENDR", "DMDEDUC2", "SLQ050", "SLQ060"] if c in df.columns]
if groups:
    selected = st.selectbox("Choose a grouping variable", groups)
    summary = df.groupby(selected, dropna=False)[TARGET].agg(["mean", "count"]).reset_index()
    summary["percentage"] = summary["mean"] * 100
    st.dataframe(summary, use_container_width=True)
    fig, ax = plt.subplots()
    sns.barplot(data=summary, x=selected, y="percentage", ax=ax)
    ax.set_ylabel("Percentage with PHQ-9 >= 10")
    ax.set_xlabel(selected)
    st.pyplot(fig)
    plt.close(fig)
else:
    st.caption("No selected grouping variables were available in this prepared file.")
st.subheader("Model evaluation")
metrics_path = REPORTS_DIR / "model_metrics.json"
if metrics_path.exists():
    st.dataframe(pd.DataFrame(json.loads(metrics_path.read_text(encoding="utf-8"))).T,
                 use_container_width=True)
else:
    st.caption("Train models with `python -m src.train_model` to display evaluation metrics.")
st.subheader("Data preview")
st.dataframe(df.drop(columns=["SEQN"], errors="ignore").head(20), use_container_width=True)
st.caption("Survey weights and complex sample design are not incorporated into exploratory scikit-learn model metrics.")
