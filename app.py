"""
SOC Dashboard — Security Log Anomaly Detection
================================================
Author  : Pramod Prakash Jadhav
GitHub  : https://github.com/pramodj551-oss
LinkedIn: https://linkedin.com/in/pramod-jadhav-42ba2281
Built as part of Applied AI & ML Essentials — IIT Patna (Vishlesan i-Hub)
"""

import warnings
import random
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from datetime import datetime, timedelta
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

st.set_page_config(page_title="SOC Anomaly Dashboard", page_icon="🔐",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
h1 { color: #00cc88 !important; }
</style>""", unsafe_allow_html=True)

FEATURE_COLS = ["login_attempts","unique_ips","off_hours",
                "failed_auths","data_volume_mb","session_duration_min"]
SEV_COLOR = {"CRITICAL":"#ff4b4b","HIGH":"#ffa500","MEDIUM":"#ffd700",
             "LOW":"#64b5f6","NORMAL":"#00cc88"}

@st.cache_data
def generate_logs():
    np.random.seed(42); random.seed(42)
    user_ids  = [f"USR{str(i).zfill(3)}" for i in range(1, 51)]
    locations = ["Mumbai","Pune","Bangalore"]
    base      = datetime(2024, 1, 1, 6, 0, 0)
    records   = []

    for _ in range(470):
        ts = base + timedelta(minutes=random.randint(0, 43200))
        records.append({
            "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
            "user_id": random.choice(user_ids[:40]),
            "login_attempts": random.randint(1, 2),
            "unique_ips": 1,
            "off_hours": 0 if 6 <= ts.hour <= 20 else 1,
            "failed_auths": random.randint(0, 1),
            "data_volume_mb": round(random.uniform(0.1, 50.0), 2),
            "session_duration_min": round(random.uniform(5, 480), 1),
            "location": random.choice(locations),
            "device_type": random.choice(["laptop","desktop"]),
        })

    for i in range(30):
        ts = base + timedelta(minutes=random.randint(0, 43200))
        p  = i % 3
        if p == 0:
            rec = {"login_attempts": random.randint(30,50),"unique_ips": random.randint(5,10),
                   "off_hours":1,"failed_auths":random.randint(28,48),
                   "data_volume_mb":round(random.uniform(0.1,5.0),2),
                   "session_duration_min":round(random.uniform(1,10),1),
                   "location":"Unknown","device_type":"unknown"}
        elif p == 1:
            rec = {"login_attempts":1,"unique_ips":1,"off_hours":1,"failed_auths":0,
                   "data_volume_mb":round(random.uniform(800,2000),2),
                   "session_duration_min":round(random.uniform(60,300),1),
                   "location":random.choice(locations),"device_type":"laptop"}
        else:
            rec = {"login_attempts":random.randint(10,20),"unique_ips":random.randint(6,15),
                   "off_hours":random.randint(0,1),"failed_auths":random.randint(8,18),
                   "data_volume_mb":round(random.uniform(0.5,10.0),2),
                   "session_duration_min":round(random.uniform(2,30),1),
                   "location":"Unknown","device_type":"unknown"}
        rec["timestamp"] = ts.strftime("%Y-%m-%d %H:%M:%S")
        rec["user_id"]   = random.choice(user_ids[40:])
        records.append(rec)

    return pd.DataFrame(records).sample(frac=1, random_state=42).reset_index(drop=True)

@st.cache_data
def run_detection(contamination, n_estimators):
    df = generate_logs()
    df["attempt_ip_ratio"] = df["login_attempts"] / (df["unique_ips"] + 1)
    df["failure_rate"]     = df["failed_auths"]   / (df["login_attempts"] + 1)
    df["risk_score"]       = (df["login_attempts"]*0.30 + df["failed_auths"]*0.35 +
                               df["unique_ips"]*0.20    + df["off_hours"]*0.15)
    all_feats = FEATURE_COLS + ["attempt_ip_ratio","failure_rate","risk_score"]
    X_scaled  = StandardScaler().fit_transform(df[all_feats].fillna(0))
    model     = IsolationForest(n_estimators=n_estimators,
                                contamination=contamination, random_state=42)
    df["anomaly_flag"]  = model.fit_predict(X_scaled)
    df["anomaly_score"] = model.decision_function(X_scaled)
    df["is_anomaly"]    = df["anomaly_flag"] == -1
    def sev(row):
        if not row["is_anomaly"]: return "NORMAL"
        s = row["anomaly_score"]
        if s <= -0.155: return "CRITICAL"
        if s <= -0.120: return "HIGH"
        if s <= -0.050: return "MEDIUM"
        return "LOW"
    df["severity"] = df.apply(sev, axis=1)
    return df

# Sidebar
with st.sidebar:
    st.markdown("## ⚙️ Model Parameters")
    contamination   = st.slider("Contamination (Anomaly %)", 0.01, 0.15, 0.06, 0.01)
    n_estimators    = st.slider("Number of Trees", 50, 500, 200, 50)
    severity_filter = st.multiselect("Filter Severity",
        ["CRITICAL","HIGH","MEDIUM","LOW","NORMAL"],
        default=["CRITICAL","HIGH","MEDIUM"])
    st.markdown("---")
    st.markdown("**👤 Pramod Prakash Jadhav**")
    st.markdown("🎓 IIT Patna — Applied AI & ML")
    st.markdown("[GitHub](https://github.com/pramodj551-oss) | [LinkedIn](https://linkedin.com/in/pramod-jadhav-42ba2281)")

df = run_detection(contamination, n_estimators)

st.title("🔐 Security Log Anomaly Detection — SOC Dashboard")
st.caption("Unsupervised ML · Isolation Forest · IIT Patna Applied AI & ML · Pramod Prakash Jadhav")

total      = len(df)
anom_n     = df["is_anomaly"].sum()
critical_n = (df["severity"]=="CRITICAL").sum()
high_n     = (df["severity"]=="HIGH").sum()

c1,c2,c3,c4,c5 = st.columns(5)
c1.metric("📋 Total Logs",  f"{total:,}")
c2.metric("🚨 Anomalies",   int(anom_n), delta=f"{anom_n/total*100:.1f}%")
c3.metric("🔴 Critical",    int(critical_n))
c4.metric("🟠 High",        int(high_n))
c5.metric("✅ Normal Rate", f"{(total-anom_n)/total*100:.1f}%")

st.divider()
col_a, col_b = st.columns(2)

with col_a:
    st.subheader("📊 Severity Distribution")
    sev_df = df["severity"].value_counts().reset_index()
    sev_df.columns = ["Severity","Count"]
    fig1 = px.bar(sev_df, x="Severity", y="Count", color="Severity",
                  color_discrete_map=SEV_COLOR, template="plotly_dark")
    fig1.update_layout(showlegend=False, margin=dict(t=10,b=10))
    st.plotly_chart(fig1, use_container_width=True)

with col_b:
    st.subheader("🎯 Anomaly Score Distribution")
    fig2 = px.histogram(df, x="anomaly_score", color="is_anomaly",
                        color_discrete_map={True:"#ff4b4b",False:"#00cc88"},
                        nbins=40, template="plotly_dark")
    fig2.update_layout(margin=dict(t=10,b=10))
    st.plotly_chart(fig2, use_container_width=True)

col_c, col_d = st.columns(2)

with col_c:
    st.subheader("⚡ Risk Score vs Login Attempts")
    fig3 = px.scatter(df, x="login_attempts", y="risk_score", color="severity",
                      color_discrete_map=SEV_COLOR, size="failed_auths",
                      hover_data=["user_id","anomaly_score"], template="plotly_dark")
    fig3.update_layout(margin=dict(t=10,b=10))
    st.plotly_chart(fig3, use_container_width=True)

with col_d:
    st.subheader("📈 Failed Auths vs Data Volume")
    fig4 = px.scatter(df, x="failed_auths", y="data_volume_mb", color="severity",
                      color_discrete_map=SEV_COLOR, hover_data=["user_id","off_hours"],
                      template="plotly_dark")
    fig4.update_layout(margin=dict(t=10,b=10))
    st.plotly_chart(fig4, use_container_width=True)

st.divider()
st.subheader("🚨 Suspicious Events")
filtered  = df[df["severity"].isin(severity_filter)].sort_values("anomaly_score")
show_cols = [c for c in ["user_id","timestamp","severity","anomaly_score",
             "login_attempts","failed_auths","unique_ips","off_hours",
             "data_volume_mb","risk_score"] if c in filtered.columns]

def color_sev(val):
    m = {"CRITICAL":"background-color:#3d0000;color:#ff4b4b;font-weight:bold",
         "HIGH":"background-color:#3d2000;color:#ffa500;font-weight:bold",
         "MEDIUM":"background-color:#3d3300;color:#ffd700",
         "LOW":"background-color:#1a2a3a;color:#64b5f6",
         "NORMAL":"background-color:#003d1e;color:#00cc88"}
    return m.get(val,"")

st.dataframe(
    filtered[show_cols].style.map(color_sev, subset=["severity"])
    .format({"anomaly_score":"{:.3f}","risk_score":"{:.2f}","data_volume_mb":"{:.1f}"}),
    use_container_width=True, height=400)

st.download_button("⬇️ Download Results CSV",
                   filtered.to_csv(index=False).encode("utf-8"),
                   "anomaly_results.csv","text/csv")

st.divider()
st.caption("Built by Pramod Prakash Jadhav · IIT Patna (Vishlesan i-Hub) · github.com/pramodj551-oss")
