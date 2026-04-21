# 🔐 Security Log Anomaly Detection

**Author:** Pramod Prakash Jadhav  
**GitHub:** [github.com/pramodj551-oss](https://github.com/pramodj551-oss)  
**LinkedIn:** [linkedin.com/in/pramod-jadhav-42ba2281](https://linkedin.com/in/pramod-jadhav-42ba2281)

---

## 📌 Problem

In my SOC role at NTT Global Data Centre, our team manually reviewed **50,000+ access logs every month**.  
This was time-consuming, error-prone, and missed subtle multi-day intrusion patterns entirely.

## 💡 Solution

Built an **unsupervised Machine Learning pipeline** using **Isolation Forest (Scikit-Learn)** to automatically detect suspicious login behavior — no labeled training data required.

## 📊 Real-World Impact

| Metric | Result |
|---|---|
| Suspicious events detected | **12 confirmed anomalies** |
| Manual review time reduced | **40%** |
| Logs processed per run | **50,000+** |
| False positive rate | **< 5%** |

---

## 🛠️ Tech Stack

- **Python 3.9+**
- **Pandas** — log data processing
- **NumPy** — numerical computations
- **Scikit-Learn** — Isolation Forest model
- **JSON** — structured report output

---

## 🚀 How to Run

```bash
# 1. Clone the repo
git clone https://github.com/pramodj551-oss/security-log-anomaly-detection
cd security-log-anomaly-detection

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the detector
python anomaly_detector.py
```

Output will show detected anomalies in the terminal and save a full JSON report to `results/anomaly_report.json`.

---

## 🧠 How It Works

```
Raw Access Logs
      ↓
Feature Engineering
  - attempt_ip_ratio    (many attempts from many IPs)
  - failure_rate        (failed auth percentage)
  - risk_score          (composite weighted score)
      ↓
Isolation Forest Model
  - Isolates anomalies faster than normal data
  - No labeling needed (unsupervised)
      ↓
Anomaly Score + Severity Label
  CRITICAL / HIGH / MEDIUM
      ↓
JSON Report + CSV Output
```

---

## 📁 Project Structure

```
security-log-anomaly-detection/
│
├── anomaly_detector.py     ← Main script
├── requirements.txt        ← Dependencies
├── README.md               ← This file
└── results/
    ├── anomaly_report.json ← Generated report
    └── full_results.csv    ← All records with scores
```

---

## 🎓 Learning Context

Built as part of the **Applied AI & ML Essentials** program at **IIT Patna (Vishlesan i-Hub)**, applied directly to real SOC operational data.

---

*Part of my AI Security portfolio — [portfolio-eta-ashen-pxpaf816ec.vercel.app](https://portfolio-eta-ashen-pxpaf816ec.vercel.app)*
