"""
╔══════════════════════════════════════════════════════════════════╗
║   Login Anomaly Detection — Streamlit Dashboard                  ║
║   IIT Patna | Applied AI & ML Essentials                         ║
║   GitHub: pramodj551-oss                                         ║
╚══════════════════════════════════════════════════════════════════╝

चालवण्यासाठी:
    pip install streamlit plotly pandas numpy scikit-learn tensorflow
    streamlit run anomaly_dashboard.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
import random

# ══════════════════════════════════════════════════════════════════
# PAGE CONFIGURATION
# ══════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Login Anomaly Detection | IIT Patna",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ══════════════════════════════════════════════════════════════════
# CUSTOM CSS — Dark SOC Theme
# ══════════════════════════════════════════════════════════════════

st.markdown("""
<style>
    /* Dark SOC Theme */
    .stApp {
        background-color: #0a0e1a;
        color: #e0e6f0;
    }

    /* Main Title */
    .main-title {
        font-family: 'Courier New', monospace;
        font-size: 2.2rem;
        font-weight: 700;
        color: #00d4ff;
        text-align: center;
        padding: 1.5rem 0 0.5rem 0;
        letter-spacing: 2px;
        text-shadow: 0 0 20px rgba(0, 212, 255, 0.4);
    }

    .sub-title {
        font-family: 'Courier New', monospace;
        font-size: 0.9rem;
        color: #5a7fa0;
        text-align: center;
        margin-bottom: 2rem;
        letter-spacing: 1px;
    }

    /* Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, #111827 0%, #1a2535 100%);
        border: 1px solid #1e3a5f;
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0, 212, 255, 0.08);
        margin-bottom: 1rem;
    }

    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        font-family: 'Courier New', monospace;
        margin: 0;
    }

    .metric-label {
        font-size: 0.75rem;
        color: #5a7fa0;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-top: 0.3rem;
    }

    .metric-green  { color: #00ff88; text-shadow: 0 0 10px rgba(0,255,136,0.4); }
    .metric-red    { color: #ff4560; text-shadow: 0 0 10px rgba(255,69,96,0.4); }
    .metric-yellow { color: #ffd700; text-shadow: 0 0 10px rgba(255,215,0,0.4); }
    .metric-blue   { color: #00d4ff; text-shadow: 0 0 10px rgba(0,212,255,0.4); }

    /* Section Headers */
    .section-header {
        font-family: 'Courier New', monospace;
        font-size: 1rem;
        color: #00d4ff;
        border-left: 3px solid #00d4ff;
        padding-left: 0.8rem;
        margin: 1.5rem 0 1rem 0;
        letter-spacing: 1px;
    }

    /* Alert Box */
    .alert-high {
        background: rgba(255, 69, 96, 0.1);
        border: 1px solid #ff4560;
        border-radius: 8px;
        padding: 0.8rem 1rem;
        margin: 0.5rem 0;
        font-family: 'Courier New', monospace;
        font-size: 0.85rem;
        color: #ff4560;
    }

    .alert-medium {
        background: rgba(255, 215, 0, 0.1);
        border: 1px solid #ffd700;
        border-radius: 8px;
        padding: 0.8rem 1rem;
        margin: 0.5rem 0;
        font-family: 'Courier New', monospace;
        font-size: 0.85rem;
        color: #ffd700;
    }

    /* Sidebar */
    .css-1d391kg, [data-testid="stSidebar"] {
        background-color: #0d1221 !important;
    }

    /* Streamlit elements override */
    div[data-testid="metric-container"] {
        background: #111827;
        border: 1px solid #1e3a5f;
        border-radius: 10px;
        padding: 1rem;
    }

    /* Plotly chart background */
    .js-plotly-plot {
        border-radius: 12px;
    }

    /* Footer */
    .footer {
        text-align: center;
        font-family: 'Courier New', monospace;
        font-size: 0.75rem;
        color: #2a4060;
        margin-top: 2rem;
        padding: 1rem;
        border-top: 1px solid #1e3a5f;
    }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# DEMO DATA GENERATOR
# (प्रत्यक्षात तुमचा trained model आणि real data वापरा)
# ══════════════════════════════════════════════════════════════════

@st.cache_data
def generate_demo_data():
    """Demo data — तुमचा actual model data येथे replace करा"""
    np.random.seed(42)
    random.seed(42)

    # ── Training History ──────────────────────────
    epochs = 100
    train_loss = []
    val_loss = []
    base = 0.08

    for i in range(epochs):
        decay = np.exp(-i * 0.04)
        noise_t = np.random.normal(0, 0.001)
        noise_v = np.random.normal(0, 0.0015)
        tl = base * decay + 0.005 + noise_t
        vl = base * decay + 0.007 + noise_v
        train_loss.append(max(0.003, tl))
        val_loss.append(max(0.004, vl))

    history_df = pd.DataFrame({
        'epoch': range(1, epochs + 1),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'train_mae': [l * 0.7 + np.random.normal(0, 0.0005) for l in train_loss],
        'val_mae':   [l * 0.75 + np.random.normal(0, 0.0006) for l in val_loss]
    })

    # ── Reconstruction Errors ─────────────────────
    n_normal   = 900
    n_anomaly  = 100

    normal_errors  = np.random.lognormal(mean=-4.5, sigma=0.6, size=n_normal)
    anomaly_errors = np.random.lognormal(mean=-2.8, sigma=0.5, size=n_anomaly)

    labels  = ['Normal'] * n_normal + ['Anomaly'] * n_anomaly
    errors  = np.concatenate([normal_errors, anomaly_errors])
    threshold = np.percentile(normal_errors, 95)

    risk_scores = np.clip((errors / (threshold * 2)) * 100, 0, 100)

    errors_df = pd.DataFrame({
        'mse': errors,
        'label': labels,
        'risk_score': risk_scores
    })

    # ── Live Login Events ─────────────────────────
    now = datetime.now()
    login_times = [now - timedelta(minutes=random.randint(0, 120)) for _ in range(50)]
    login_times.sort(reverse=True)

    live_df = pd.DataFrame({
        'timestamp': login_times,
        'user_id': [f"USR_{random.randint(1000, 9999)}" for _ in range(50)],
        'ip': [f"{random.randint(10,200)}.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(0,255)}"
               for _ in range(50)],
        'mse': np.concatenate([
            np.random.lognormal(-4.5, 0.5, 42),
            np.random.lognormal(-2.8, 0.4, 8)
        ]),
        'is_anomaly': [False] * 42 + [True] * 8
    })
    live_df = live_df.sample(frac=1, random_state=42).reset_index(drop=True)
    live_df['risk_score'] = np.clip((live_df['mse'] / (threshold * 2)) * 100, 0, 100)
    live_df['risk_level'] = live_df['risk_score'].apply(
        lambda x: '🔴 HIGH' if x > 80 else ('🟡 MEDIUM' if x > 50 else '🟢 LOW')
    )

    # ── Hourly Anomaly Trend ──────────────────────
    hours = list(range(24))
    hourly_normal  = [random.randint(80, 300) for _ in hours]
    hourly_anomaly = [random.randint(0, 8) + (15 if h in [2, 3, 14, 15] else 0)
                      for h in hours]

    hourly_df = pd.DataFrame({
        'hour': hours,
        'normal_logins': hourly_normal,
        'anomalies': hourly_anomaly
    })

    return history_df, errors_df, live_df, hourly_df, threshold


# ══════════════════════════════════════════════════════════════════
# PLOTLY CHART STYLE — Dark SOC Theme
# ══════════════════════════════════════════════════════════════════

CHART_LAYOUT = dict(
    paper_bgcolor='rgba(10,14,26,0)',
    plot_bgcolor='rgba(17,24,39,0.6)',
    font=dict(family='Courier New, monospace', color='#8aa0be', size=11),
    xaxis=dict(gridcolor='#1e3a5f', linecolor='#1e3a5f', zerolinecolor='#1e3a5f'),
    yaxis=dict(gridcolor='#1e3a5f', linecolor='#1e3a5f', zerolinecolor='#1e3a5f'),
    legend=dict(bgcolor='rgba(17,24,39,0.8)', bordercolor='#1e3a5f', borderwidth=1),
    margin=dict(l=50, r=20, t=50, b=50)
)


# ══════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("### 🛡️ Control Panel")
    st.markdown("---")

    model_type = st.selectbox(
        "मॉडेल निवडा",
        ["Stacked Autoencoder", "LSTM Autoencoder", "Variational AE (VAE)"]
    )

    threshold_pct = st.slider(
        "Anomaly Threshold Percentile",
        min_value=85, max_value=99, value=95, step=1,
        help="जास्त value = कमी False Positives, पण काही Anomalies miss होतील"
    )

    st.markdown("---")
    st.markdown("**डेटा फिल्टर:**")
    show_high   = st.checkbox("🔴 HIGH Risk",   value=True)
    show_medium = st.checkbox("🟡 MEDIUM Risk", value=True)
    show_low    = st.checkbox("🟢 LOW Risk",    value=False)

    st.markdown("---")
    auto_refresh = st.checkbox("⟳ Auto Refresh (30s)", value=False)
    if auto_refresh:
        st.info("Live mode ON")

    st.markdown("---")
    st.markdown("""
    <div style='font-family: Courier New; font-size: 0.72rem; color: #2a4060;'>
    IIT Patna<br>Applied AI & ML Essentials<br>
    GitHub: pramodj551-oss<br>
    Model v1.0 | 2024
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════════════════════════════════

history_df, errors_df, live_df, hourly_df, threshold = generate_demo_data()


# ══════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════

st.markdown('<div class="main-title">🛡️ LOGIN ANOMALY DETECTION DASHBOARD</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Deep Learning | Autoencoder | IIT Patna Applied AI & ML Essentials</div>', unsafe_allow_html=True)

# Live timestamp
col_ts, col_model = st.columns([3, 1])
with col_ts:
    st.markdown(f"""
    <div style='font-family: Courier New; font-size: 0.8rem; color: #2a4060;'>
    ⏱ Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} &nbsp;|&nbsp; 
    Model: {model_type} &nbsp;|&nbsp; Threshold: {threshold_pct}th percentile
    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# SECTION 1: KPI METRICS
# ══════════════════════════════════════════════════════════════════

st.markdown('<div class="section-header">◈ REAL-TIME METRICS</div>', unsafe_allow_html=True)

total       = len(live_df)
anomalies   = live_df['is_anomaly'].sum()
normal      = total - anomalies
high_risk   = (live_df['risk_score'] > 80).sum()
medium_risk = ((live_df['risk_score'] > 50) & (live_df['risk_score'] <= 80)).sum()
anomaly_rate = round(anomalies / total * 100, 1)

c1, c2, c3, c4, c5 = st.columns(5)

with c1:
    st.markdown(f"""
    <div class="metric-card">
        <p class="metric-value metric-blue">{total}</p>
        <p class="metric-label">Total Logins</p>
    </div>""", unsafe_allow_html=True)

with c2:
    st.markdown(f"""
    <div class="metric-card">
        <p class="metric-value metric-green">{normal}</p>
        <p class="metric-label">Normal</p>
    </div>""", unsafe_allow_html=True)

with c3:
    st.markdown(f"""
    <div class="metric-card">
        <p class="metric-value metric-red">{anomalies}</p>
        <p class="metric-label">Anomalies</p>
    </div>""", unsafe_allow_html=True)

with c4:
    st.markdown(f"""
    <div class="metric-card">
        <p class="metric-value metric-red">{high_risk}</p>
        <p class="metric-label">High Risk</p>
    </div>""", unsafe_allow_html=True)

with c5:
    color = "metric-red" if anomaly_rate > 10 else "metric-yellow"
    st.markdown(f"""
    <div class="metric-card">
        <p class="metric-value {color}">{anomaly_rate}%</p>
        <p class="metric-label">Anomaly Rate</p>
    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# SECTION 2: TRAINING LOSS GRAPHS
# ══════════════════════════════════════════════════════════════════

st.markdown('<div class="section-header">◈ AUTOENCODER TRAINING PROGRESS</div>', unsafe_allow_html=True)

col_loss, col_mae = st.columns(2)

with col_loss:
    # MSE Loss Graph
    fig_loss = go.Figure()

    # Training Loss
    fig_loss.add_trace(go.Scatter(
        x=history_df['epoch'],
        y=history_df['train_loss'],
        name='Training Loss (MSE)',
        line=dict(color='#00d4ff', width=2),
        fill='tozeroy',
        fillcolor='rgba(0,212,255,0.05)',
        hovertemplate='Epoch %{x}<br>Train Loss: %{y:.6f}<extra></extra>'
    ))

    # Validation Loss
    fig_loss.add_trace(go.Scatter(
        x=history_df['epoch'],
        y=history_df['val_loss'],
        name='Validation Loss (MSE)',
        line=dict(color='#ff6b35', width=2, dash='dot'),
        hovertemplate='Epoch %{x}<br>Val Loss: %{y:.6f}<extra></extra>'
    ))

    # Early Stopping marker (best epoch)
    best_epoch = history_df['val_loss'].idxmin() + 1
    best_val   = history_df['val_loss'].min()

    fig_loss.add_vline(
        x=best_epoch, line_color='#00ff88',
        line_dash='dash', line_width=1,
        annotation_text=f"Best: Epoch {best_epoch}",
        annotation_font_color='#00ff88',
        annotation_font_size=10
    )

    fig_loss.update_layout(
        title=dict(text="MSE Training Loss", font=dict(color='#00d4ff', size=14)),
        xaxis_title="Epochs",
        yaxis_title="MSE Loss",
        height=320,
        **CHART_LAYOUT
    )
    st.plotly_chart(fig_loss, use_container_width=True)

with col_mae:
    # MAE Graph
    fig_mae = go.Figure()

    fig_mae.add_trace(go.Scatter(
        x=history_df['epoch'],
        y=history_df['train_mae'],
        name='Training MAE',
        line=dict(color='#a78bfa', width=2),
        fill='tozeroy',
        fillcolor='rgba(167,139,250,0.05)',
        hovertemplate='Epoch %{x}<br>Train MAE: %{y:.6f}<extra></extra>'
    ))

    fig_mae.add_trace(go.Scatter(
        x=history_df['epoch'],
        y=history_df['val_mae'],
        name='Validation MAE',
        line=dict(color='#f472b6', width=2, dash='dot'),
        hovertemplate='Epoch %{x}<br>Val MAE: %{y:.6f}<extra></extra>'
    ))

    # Overfitting zone highlight
    fig_mae.add_vrect(
        x0=best_epoch, x1=100,
        fillcolor='rgba(255,69,96,0.04)',
        layer='below', line_width=0,
        annotation_text="Overfitting Risk",
        annotation_font_color='#ff4560',
        annotation_font_size=9
    )

    fig_mae.update_layout(
        title=dict(text="MAE Training Loss", font=dict(color='#a78bfa', size=14)),
        xaxis_title="Epochs",
        yaxis_title="MAE Loss",
        height=320,
        **CHART_LAYOUT
    )
    st.plotly_chart(fig_mae, use_container_width=True)


# ══════════════════════════════════════════════════════════════════
# SECTION 3: ERROR DISTRIBUTION
# ══════════════════════════════════════════════════════════════════

st.markdown('<div class="section-header">◈ RECONSTRUCTION ERROR DISTRIBUTION</div>', unsafe_allow_html=True)

col_hist, col_scatter = st.columns([3, 2])

with col_hist:
    fig_hist = go.Figure()

    normal_errors  = errors_df[errors_df['label'] == 'Normal']['mse']
    anomaly_errors = errors_df[errors_df['label'] == 'Anomaly']['mse']

    # Normal distribution
    fig_hist.add_trace(go.Histogram(
        x=normal_errors,
        name='Normal Logins',
        nbinsx=50,
        marker_color='rgba(0,255,136,0.7)',
        marker_line=dict(color='rgba(0,255,136,0.3)', width=0.5),
        hovertemplate='MSE: %{x:.5f}<br>Count: %{y}<extra>Normal</extra>'
    ))

    # Anomaly distribution
    fig_hist.add_trace(go.Histogram(
        x=anomaly_errors,
        name='Anomalous Logins',
        nbinsx=30,
        marker_color='rgba(255,69,96,0.7)',
        marker_line=dict(color='rgba(255,69,96,0.3)', width=0.5),
        hovertemplate='MSE: %{x:.5f}<br>Count: %{y}<extra>Anomaly</extra>'
    ))

    # Threshold line
    fig_hist.add_vline(
        x=threshold,
        line_color='#ffd700', line_width=2, line_dash='dash',
        annotation_text=f"Threshold = {threshold:.4f}",
        annotation_font_color='#ffd700',
        annotation_font_size=11,
        annotation_position='top right'
    )

    fig_hist.update_layout(
        title=dict(text="Reconstruction Error (MSE) Distribution", font=dict(color='#00d4ff', size=14)),
        xaxis_title="Reconstruction Error (MSE)",
        yaxis_title="Login Count",
        barmode='overlay',
        height=350,
        **CHART_LAYOUT
    )
    st.plotly_chart(fig_hist, use_container_width=True)

with col_scatter:
    # Risk Score Gauge / Donut
    normal_pct  = round(len(normal_errors) / len(errors_df) * 100, 1)
    anomaly_pct = round(len(anomaly_errors) / len(errors_df) * 100, 1)

    fig_donut = go.Figure(go.Pie(
        labels=['Normal 🟢', 'Anomaly 🔴'],
        values=[len(normal_errors), len(anomaly_errors)],
        hole=0.65,
        marker=dict(
            colors=['#00ff88', '#ff4560'],
            line=dict(color='#0a0e1a', width=3)
        ),
        textfont=dict(family='Courier New', size=12),
        hovertemplate='%{label}<br>Count: %{value}<br>%{percent}<extra></extra>'
    ))

    fig_donut.add_annotation(
        text=f"<b>{anomaly_pct}%</b><br>Anomaly",
        x=0.5, y=0.5,
        font=dict(size=16, color='#ff4560', family='Courier New'),
        showarrow=False
    )

    fig_donut.update_layout(
        title=dict(text="Detection Overview", font=dict(color='#00d4ff', size=14)),
        height=350,
        showlegend=True,
        legend=dict(
            orientation='v', x=0.7, y=0.5,
            font=dict(family='Courier New', size=10)
        ),
        **CHART_LAYOUT
    )
    st.plotly_chart(fig_donut, use_container_width=True)


# ══════════════════════════════════════════════════════════════════
# SECTION 4: RISK SCORE DISTRIBUTION + HOURLY TREND
# ══════════════════════════════════════════════════════════════════

col_risk, col_hourly = st.columns(2)

with col_risk:
    st.markdown('<div class="section-header">◈ RISK SCORE DISTRIBUTION</div>', unsafe_allow_html=True)

    fig_risk = go.Figure()

    # Color based on risk level
    colors = errors_df['risk_score'].apply(
        lambda x: '#ff4560' if x > 80 else ('#ffd700' if x > 50 else '#00ff88')
    )

    fig_risk.add_trace(go.Scatter(
        x=list(range(len(errors_df))),
        y=errors_df['risk_score'],
        mode='markers',
        marker=dict(
            color=errors_df['risk_score'],
            colorscale=[[0, '#00ff88'], [0.5, '#ffd700'], [1, '#ff4560']],
            size=5,
            opacity=0.7,
            colorbar=dict(
                title='Risk',
                tickfont=dict(family='Courier New', color='#5a7fa0'),
                thickness=12
            )
        ),
        hovertemplate='Login #%{x}<br>Risk Score: %{y:.1f}/100<extra></extra>'
    ))

    # Threshold lines
    fig_risk.add_hline(y=80, line_color='#ff4560', line_dash='dash',
                       line_width=1, annotation_text="HIGH (80)",
                       annotation_font_color='#ff4560', annotation_font_size=9)
    fig_risk.add_hline(y=50, line_color='#ffd700', line_dash='dash',
                       line_width=1, annotation_text="MEDIUM (50)",
                       annotation_font_color='#ffd700', annotation_font_size=9)

    fig_risk.update_layout(
        title=dict(text="Per-Login Risk Score", font=dict(color='#00d4ff', size=14)),
        xaxis_title="Login Index",
        yaxis_title="Risk Score (0-100)",
        yaxis=dict(range=[0, 110], **CHART_LAYOUT['yaxis']),
        height=320,
        **CHART_LAYOUT
    )
    st.plotly_chart(fig_risk, use_container_width=True)

with col_hourly:
    st.markdown('<div class="section-header">◈ HOURLY ANOMALY TREND</div>', unsafe_allow_html=True)

    fig_hourly = make_subplots(specs=[[{"secondary_y": True}]])

    fig_hourly.add_trace(go.Bar(
        x=hourly_df['hour'],
        y=hourly_df['normal_logins'],
        name='Normal Logins',
        marker_color='rgba(0,212,255,0.3)',
        marker_line=dict(color='rgba(0,212,255,0.5)', width=0.5),
        hovertemplate='Hour %{x}:00<br>Normal: %{y}<extra></extra>'
    ), secondary_y=False)

    fig_hourly.add_trace(go.Scatter(
        x=hourly_df['hour'],
        y=hourly_df['anomalies'],
        name='Anomalies',
        line=dict(color='#ff4560', width=2.5),
        mode='lines+markers',
        marker=dict(size=7, color='#ff4560',
                    line=dict(color='#0a0e1a', width=1)),
        hovertemplate='Hour %{x}:00<br>Anomalies: %{y}<extra></extra>'
    ), secondary_y=True)

    fig_hourly.update_layout(
        title=dict(text="24-Hour Login Activity", font=dict(color='#00d4ff', size=14)),
        xaxis_title="Hour of Day",
        height=320,
        **CHART_LAYOUT
    )
    fig_hourly.update_yaxes(title_text="Normal Count", secondary_y=False,
                            gridcolor='#1e3a5f', color='#8aa0be')
    fig_hourly.update_yaxes(title_text="Anomaly Count", secondary_y=True,
                            gridcolor='#1e3a5f', color='#ff4560')

    st.plotly_chart(fig_hourly, use_container_width=True)


# ══════════════════════════════════════════════════════════════════
# SECTION 5: LIVE EVENTS TABLE
# ══════════════════════════════════════════════════════════════════

st.markdown('<div class="section-header">◈ LIVE LOGIN EVENTS</div>', unsafe_allow_html=True)

# Recent Alerts
high_risk_logins = live_df[live_df['risk_score'] > 80].head(3)
if not high_risk_logins.empty:
    for _, row in high_risk_logins.iterrows():
        st.markdown(f"""
        <div class="alert-high">
        🚨 HIGH RISK DETECTED | User: {row['user_id']} | IP: {row['ip']} | 
        Risk Score: {row['risk_score']:.1f}/100 | MSE: {row['mse']:.5f} | 
        Action: BLOCK & ALERT SOC
        </div>""", unsafe_allow_html=True)

medium_risk_logins = live_df[(live_df['risk_score'] > 50) & (live_df['risk_score'] <= 80)].head(2)
if not medium_risk_logins.empty:
    for _, row in medium_risk_logins.iterrows():
        st.markdown(f"""
        <div class="alert-medium">
        ⚠️ MEDIUM RISK | User: {row['user_id']} | IP: {row['ip']} | 
        Risk Score: {row['risk_score']:.1f}/100 | Action: REQUIRE MFA
        </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Detailed Table
display_df = live_df[[
    'timestamp', 'user_id', 'ip', 'mse', 'risk_score', 'risk_level'
]].copy()
display_df['timestamp'] = display_df['timestamp'].dt.strftime('%H:%M:%S')
display_df['mse']        = display_df['mse'].round(6)
display_df['risk_score'] = display_df['risk_score'].round(1)
display_df.columns       = ['Time', 'User ID', 'IP Address', 'MSE Error', 'Risk Score', 'Risk Level']

st.dataframe(
    display_df.head(20),
    use_container_width=True,
    hide_index=True,
    column_config={
        'Risk Score': st.column_config.ProgressColumn(
            'Risk Score', min_value=0, max_value=100, format="%.1f"
        ),
        'MSE Error': st.column_config.NumberColumn('MSE Error', format="%.6f")
    }
)


# ══════════════════════════════════════════════════════════════════
# SECTION 6: MODEL SUMMARY STATS
# ══════════════════════════════════════════════════════════════════

st.markdown('<div class="section-header">◈ MODEL PERFORMANCE SUMMARY</div>', unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
with c1:
    final_train = history_df['train_loss'].iloc[-1]
    st.metric("Final Train Loss", f"{final_train:.6f}",
              delta=f"-{(history_df['train_loss'].iloc[0] - final_train):.4f}",
              delta_color="normal")
with c2:
    final_val = history_df['val_loss'].iloc[-1]
    st.metric("Final Val Loss", f"{final_val:.6f}")
with c3:
    st.metric("Best Epoch (EarlyStopping)", f"{best_epoch}")
with c4:
    st.metric("Anomaly Threshold (MSE)", f"{threshold:.5f}",
              help=f"{threshold_pct}th percentile of normal login reconstruction errors")


# ══════════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════════

st.markdown("""
<div class="footer">
🎓 IIT Patna | Applied AI & ML Essentials | Login Anomaly Detection Project<br>
Deep Learning: Stacked Autoencoder + LSTM + VAE | GitHub: pramodj551-oss<br>
Built with Streamlit + Plotly + TensorFlow/Keras
</div>
""", unsafe_allow_html=True)
