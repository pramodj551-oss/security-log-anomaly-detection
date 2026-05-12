"""
SOC Dashboard — Security Log Anomaly Detection
================================================
Author  : Pramod Prakash Jadhav
GitHub  : https://github.com/pramodj551-oss
LinkedIn: https://linkedin.com/in/pramod-jadhav-42ba2281

Run: streamlit run soc_dashboard.py
"""

import os
import warnings
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ── Page Config ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SOC Anomaly Dashboard",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .metric-card {
        background: #1e2130; border-radius: 10px;
        padding: 16px 20px; margin: 6px 0;
    }
    .critical { border-left: 4px solid #ff4b4b; }
    .high     { border-left: 4px solid #ffa500; }
    .medium   { border-left: 4px solid #ffd700; }
    .normal   { border-left: 4px solid #00cc88; }
    h1 { color: #00cc88 !important; }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────
FEATURE_COLS = [
    "login_attempts", "unique_ips", "off_hours",
    "failed_auths", "data_volume_mb", "session_duration_min",
]
SEV_COLOR = {
    "CRITICAL": "#ff4b4b",
    "HIGH":     "#ffa500",
    "MEDIUM":   "#ffd700",
    "LOW":      "#64b5f6",
    "NORMAL":   "#00cc88",
}


# ── Data & Model ─────────────────────────────────────────────────────────
@st.cache_data
def load_and_detect(contamination: float, n_estimators: int):
    data_path = "data/sample_logs.csv"
    if not os.path.exists(data_path):
        st.error("❌ data/sample_logs.csv not found. Run `python3 generate_data.py` first.")
        st.stop()

    df = pd.read_csv(data_path)

    # Feature engineering
    df["attempt_ip_ratio"] = df["login_attempts"] / (df["unique_ips"] + 1)
    df["failure_rate"]     = df["failed_auths"]   / (df["login_attempts"] + 1)
    df["risk_score"]       = (
        df["login_attempts"] * 0.30 +
        df["failed_auths"]   * 0.35 +
        df["unique_ips"]     * 0.20 +
        df["off_hours"]      * 0.15
    )

    all_feats = FEATURE_COLS + ["attempt_ip_ratio", "failure_rate", "risk_score"]
    X = df[all_feats].fillna(0)
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model               = IsolationForest(
        n_estimators=n_estimators, contamination=contamination, random_state=42
    )
    df["anomaly_flag"]  = model.fit_predict(X_scaled)
    df["anomaly_score"] = model.decision_function(X_scaled)
    df["is_anomaly"]    = df["anomaly_flag"] == -1

    def severity(row):
        if not row["is_anomaly"]:
            return "NORMAL"
        s = row["anomaly_score"]
        if s <= -0.155: return "CRITICAL"
        if s <= -0.12:  return "HIGH"
        if s <= -0.05:  return "MEDIUM"
        return "LOW"

    df["severity"] = df.apply(severity, axis=1)
    return df


# ── Sidebar ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.shields.io/badge/SOC-Dashboard-00cc88?style=for-the-badge&logo=shield&logoColor=white")
    st.markdown("## ⚙️ Model Parameters")

    contamination = st.slider(
        "Contamination (Expected Anomaly %)",
        min_value=0.01, max_value=0.15, value=0.06, step=0.01,
        help="Expected proportion of anomalies in data"
    )
    n_estimators = st.slider(
        "Number of Trees", min_value=50, max_value=500, value=200, step=50
    )

    severity_filter = st.multiselect(
        "Filter by Severity",
        options=["CRITICAL", "HIGH", "MEDIUM", "LOW", "NORMAL"],
        default=["CRITICAL", "HIGH", "MEDIUM"],
    )

    st.markdown("---")
    st.markdown("**👤 Author:** Pramod Prakash Jadhav")
    st.markdown("**🎓** IIT Patna — Applied AI & ML")
    st.markdown("[GitHub](https://github.com/pramodj551-oss) | [LinkedIn](https://linkedin.com/in/pramod-jadhav-42ba2281)")


# ── Load Data ─────────────────────────────────────────────────────────────
df = load_and_detect(contamination, n_estimators)

# ── Header ────────────────────────────────────────────────────────────────
st.title("🔐 Security Log Anomaly Detection — SOC Dashboard")
st.caption("Unsupervised ML (Isolation Forest) | Real-time access-log analysis | IIT Patna Applied AI & ML")

# ── KPI Row ───────────────────────────────────────────────────────────────
total      = len(df)
anomalies  = df["is_anomaly"].sum()
critical_n = (df["severity"] == "CRITICAL").sum()
high_n     = (df["severity"] == "HIGH").sum()
fp_rate    = f"{(1 - anomalies/total)*100:.1f}%"

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("📋 Total Logs",    f"{total:,}")
col2.metric("🚨 Anomalies",     int(anomalies),  delta=f"{anomalies/total*100:.1f}%")
col3.metric("🔴 Critical",      int(critical_n))
col4.metric("🟠 High",          int(high_n))
col5.metric("✅ Normal Rate",   fp_rate)

st.divider()

# ── Charts Row 1 ──────────────────────────────────────────────────────────
col_a, col_b = st.columns(2)

with col_a:
    st.subheader("📊 Severity Distribution")
    sev_counts = df["severity"].value_counts().reset_index()
    sev_counts.columns = ["Severity", "Count"]
    order = ["CRITICAL", "HIGH", "MEDIUM", "LOW", "NORMAL"]
    sev_counts["Severity"] = pd.Categorical(sev_counts["Severity"], categories=order, ordered=True)
    sev_counts = sev_counts.sort_values("Severity")
    colors = [SEV_COLOR.get(s, "#888") for s in sev_counts["Severity"]]
    fig1 = px.bar(
        sev_counts, x="Severity", y="Count",
        color="Severity", color_discrete_map=SEV_COLOR,
        template="plotly_dark"
    )
    fig1.update_layout(showlegend=False, margin=dict(t=20, b=20))
    st.plotly_chart(fig1, use_container_width=True)

with col_b:
    st.subheader("🎯 Anomaly Score Distribution")
    fig2 = px.histogram(
        df, x="anomaly_score", color="is_anomaly",
        color_discrete_map={True: "#ff4b4b", False: "#00cc88"},
        nbins=40, template="plotly_dark",
        labels={"anomaly_score": "Anomaly Score", "is_anomaly": "Is Anomaly"},
    )
    fig2.update_layout(margin=dict(t=20, b=20))
    st.plotly_chart(fig2, use_container_width=True)

# ── Charts Row 2 ──────────────────────────────────────────────────────────
col_c, col_d = st.columns(2)

with col_c:
    st.subheader("⚡ Risk Score vs Login Attempts")
    fig3 = px.scatter(
        df, x="login_attempts", y="risk_score",
        color="severity", color_discrete_map=SEV_COLOR,
        size="failed_auths", hover_data=["user_id", "anomaly_score"],
        template="plotly_dark",
        labels={"login_attempts": "Login Attempts", "risk_score": "Risk Score"},
    )
    fig3.update_layout(margin=dict(t=20, b=20))
    st.plotly_chart(fig3, use_container_width=True)

with col_d:
    st.subheader("📈 Failed Auth vs Data Volume")
    fig4 = px.scatter(
        df, x="failed_auths", y="data_volume_mb",
        color="severity", color_discrete_map=SEV_COLOR,
        hover_data=["user_id", "off_hours"],
        template="plotly_dark",
        labels={"failed_auths": "Failed Auths", "data_volume_mb": "Data Volume (MB)"},
    )
    fig4.update_layout(margin=dict(t=20, b=20))
    st.plotly_chart(fig4, use_container_width=True)

# ── Anomaly Table ─────────────────────────────────────────────────────────
st.divider()
st.subheader("🚨 Suspicious Events — Detailed View")

filtered = df[df["severity"].isin(severity_filter)].copy()
filtered = filtered.sort_values("anomaly_score")

display_cols = [
    "user_id", "timestamp", "severity", "anomaly_score",
    "login_attempts", "failed_auths", "unique_ips",
    "off_hours", "data_volume_mb", "risk_score"
]
display_cols = [c for c in display_cols if c in filtered.columns]

def color_severity(val):
    colors_map = {
        "CRITICAL": "background-color: #3d0000; color: #ff4b4b; font-weight: bold",
        "HIGH":     "background-color: #3d2000; color: #ffa500; font-weight: bold",
        "MEDIUM":   "background-color: #3d3300; color: #ffd700",
        "LOW":      "background-color: #1a2a3a; color: #64b5f6",
        "NORMAL":   "background-color: #003d1e; color: #00cc88",
    }
    return colors_map.get(val, "")

st.dataframe(
    filtered[display_cols].style.applymap(color_severity, subset=["severity"]).format(
        {"anomaly_score": "{:.3f}", "risk_score": "{:.2f}", "data_volume_mb": "{:.1f}"}
    ),
    use_container_width=True,
    height=400,
)

# ── Download ──────────────────────────────────────────────────────────────
st.download_button(
    label="⬇️ Download Full Results CSV",
    data=filtered.to_csv(index=False).encode("utf-8"),
    file_name="anomaly_results.csv",
    mime="text/csv",
)

# ── Footer ────────────────────────────────────────────────────────────────
st.divider()
st.caption("Built by Pramod Prakash Jadhav · Applied AI & ML Essentials · IIT Patna (Vishlesan i-Hub) · github.com/pramodj551-oss")
