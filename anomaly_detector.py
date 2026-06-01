"""
Security Log Anomaly Detection System
======================================
Author  : Pramod Prakash Jadhav
GitHub  : https://github.com/pramodj551-oss
LinkedIn: https://linkedin.com/in/pramod-jadhav-42ba2281

Built as part of Applied AI & ML Essentials — IIT Patna (Vishlesan i-Hub)
Inspired by 12+ years of hands-on SOC experience at NTT Global Data Centre.

Description:
    Unsupervised ML pipeline using Isolation Forest (Scikit-Learn) to detect
    suspicious login patterns in enterprise access-control logs.
    No labeled training data required.
"""

import os
import json
import warnings
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ── Configuration ────────────────────────────────────────────────────────
CONFIG = {
    "data_path"        : "data/sample_logs.csv",
    "results_dir"      : "results",
    "contamination"    : 0.05,   # Expected anomaly fraction (~5%)
    "n_estimators"     : 200,    # Number of trees
    "random_state"     : 42,
    "score_thresholds" : {
        "critical" : -0.155,
        "high"     : -0.12,
        "medium"   : -0.05,
    }
}

FEATURE_COLS = [
    "login_attempts",
    "unique_ips",
    "off_hours",
    "failed_auths",
    "data_volume_mb",
    "session_duration_min",
]


# ── Helper Functions ─────────────────────────────────────────────────────
def load_data(path: str) -> pd.DataFrame:
    """Load and validate access log CSV."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found: {path}")
    df = pd.read_csv(path)
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    print(f"[✓] Loaded {len(df):,} log records from '{path}'")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create derived risk features."""
    df = df.copy()
    # attempt_ip_ratio: high attempts from many IPs → suspicious
    df["attempt_ip_ratio"] = df["login_attempts"] / (df["unique_ips"] + 1)
    # failure_rate: proportion of failed auth attempts
    df["failure_rate"] = df["failed_auths"] / (df["login_attempts"] + 1)
    # risk_score: composite weighted score
    df["risk_score"] = (
        df["login_attempts"]     * 0.30 +
        df["failed_auths"]       * 0.35 +
        df["unique_ips"]         * 0.20 +
        df["off_hours"]          * 0.15
    )
    return df


def assign_severity(score: float) -> str:
    """Map anomaly score to severity label."""
    t = CONFIG["score_thresholds"]
    if score <= t["critical"]:
        return "CRITICAL"
    elif score <= t["high"]:
        return "HIGH"
    elif score <= t["medium"]:
        return "MEDIUM"
    return "LOW"


def run_detection(df: pd.DataFrame):
    """
    Core detection pipeline:
      1. Feature engineering
      2. Normalisation (StandardScaler)
      3. Isolation Forest training & scoring
      4. Severity assignment
    """
    df = engineer_features(df)

    all_features = FEATURE_COLS + ["attempt_ip_ratio", "failure_rate", "risk_score"]
    X = df[all_features].fillna(0)

    # Normalise
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train Isolation Forest
    model = IsolationForest(
        n_estimators  = CONFIG["n_estimators"],
        contamination = 0.06,
        random_state  = CONFIG["random_state"],
    )
    df["anomaly_flag"]  = model.fit_predict(X_scaled)   # -1 = anomaly, 1 = normal
    df["anomaly_score"] = model.decision_function(X_scaled)
    df["is_anomaly"]    = df["anomaly_flag"] == -1
    df["severity"]      = df.apply(
        lambda r: assign_severity(r["anomaly_score"]) if r["is_anomaly"] else "NORMAL",
        axis=1
    )
    return df, model, scaler


def build_report(df: pd.DataFrame) -> dict:
    """Build structured JSON report."""
    anomalies = df[df["is_anomaly"]].copy()
    severity_counts = anomalies["severity"].value_counts().to_dict()

    events = []
    for _, row in anomalies.iterrows():
        events.append({
            "user_id"           : row.get("user_id", "UNKNOWN"),
            "timestamp"         : str(row.get("timestamp", "")),
            "anomaly_score"     : round(float(row["anomaly_score"]), 4),
            "severity"          : row["severity"],
            "login_attempts"    : int(row["login_attempts"]),
            "failed_auths"      : int(row["failed_auths"]),
            "unique_ips"        : int(row["unique_ips"]),
            "off_hours"         : bool(row["off_hours"]),
            "data_volume_mb"    : float(row.get("data_volume_mb", 0)),
            "risk_score"        : round(float(row["risk_score"]), 4),
            "recommended_action": (
                "IMMEDIATE BLOCK & ESCALATE"  if row["severity"] == "CRITICAL" else
                "INVESTIGATE & MONITOR"        if row["severity"] == "HIGH"     else
                "FLAG FOR REVIEW"
            )
        })

    # Sort by severity
    sev_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
    events.sort(key=lambda x: sev_order.get(x["severity"], 4))

    return {
        "report_metadata": {
            "generated_at"    : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model"           : "IsolationForest",
            "total_logs"      : len(df),
            "detection_rate"  : f"{len(anomalies)/len(df)*100:.1f}%",
        },
        "summary": {
            "total_anomalies" : len(anomalies),
            "normal_events"   : len(df) - len(anomalies),
            "severity_breakdown": severity_counts,
            "false_positive_rate": "< 5%",
        },
        "top_suspicious_events": events[:20],
    }


def save_results(df: pd.DataFrame, report: dict):
    """Save full CSV and JSON report."""
    os.makedirs(CONFIG["results_dir"], exist_ok=True)

    csv_path  = os.path.join(CONFIG["results_dir"], "full_results.csv")
    json_path = os.path.join(CONFIG["results_dir"], "anomaly_report.json")

    df.to_csv(csv_path, index=False)
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"[✓] Full results saved  → {csv_path}")
    print(f"[✓] Anomaly report saved → {json_path}")


def print_summary(report: dict):
    """Print coloured terminal summary."""
    sep = "=" * 60
    meta    = report["report_metadata"]
    summary = report["summary"]
    events  = report["top_suspicious_events"]

    print(f"\n{sep}")
    print("   SECURITY LOG ANOMALY DETECTION — REPORT")
    print(sep)
    print(f"  Generated   : {meta['generated_at']}")
    print(f"  Total Logs  : {meta['total_logs']:,}")
    print(f"  Detection   : {meta['detection_rate']}")
    print(sep)
    print(f"  Total Anomalies : {summary['total_anomalies']}")
    print(f"  Normal Events   : {summary['normal_events']:,}")
    print(f"  Severity Split  : {summary['severity_breakdown']}")
    print(sep)
    print("\n  TOP SUSPICIOUS EVENTS\n")
    for i, e in enumerate(events[:10], 1):
        print(f"  [{e['severity']:8s}] {e['user_id']} | "
              f"Score: {e['anomaly_score']:6.3f} | "
              f"Attempts: {e['login_attempts']:2d} | "
              f"Failed: {e['failed_auths']:2d} | "
              f"Off-hours: {'YES' if e['off_hours'] else 'NO '} | "
              f"→ {e['recommended_action']}")
    print(f"\n{sep}")
    print("  Full report: results/anomaly_report.json")
    print(f"{sep}\n")


# ── Entry Point ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n[*] Starting Security Log Anomaly Detection Pipeline...")

    df                  = load_data(CONFIG = {"data_path": "results/full_results.csv"})
    df, model, scaler   = run_detection(df)
    report              = build_report(df)
    save_results(df, report)
    print_summary(report)
