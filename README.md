# customer_segmentation_RFM

This project applies **RFM (Recency, Frequency, Monetary)** analysis for customer segmentation and extends traditional methods using **3D visualization** and a full **Streamlit web UI**. By plotting RFM dimensions together, it reveals hidden patterns, overlaps, and high-value behaviours not visible in 2D — emphasising interpretability for academic and analytics use.

The application demonstrates how multidimensional visualisation can reveal patterns that are not always visible in pairwise (2D) plots.

---

## 📌 Objectives

- Calculate RFM metrics for every customer
- Segment customers using **KMeans clustering** with Elbow Method
- Train a **Random Forest classifier** to predict segments on new data
- Visualise customer segments using:
  - 2D scatter plots (Recency × Monetary, Frequency × Monetary, etc.)
  - **Interactive 3D scatter plot (Recency × Frequency × Monetary)**
  - Cluster heatmaps & segment bar charts
  - Confusion matrix & classification report
- Derive actionable business insights from multidimensional clustering

---

## 📊 Key Features

- Clean RFM feature engineering from raw transaction data
- Automatic segment naming (Champions, Loyal, At Risk, Lost)
- Segment-based colour coding across all charts
- **Dark / Light theme toggle** in the UI
- 6-tab Streamlit dashboard:
  | Tab | Description |
  |-----|-------------|
  | 📊 Dashboard | KPIs, RFM table, confusion matrix, segment distribution |
  | 📈 Visualisations | Elbow, 2D scatters, heatmap, interactive 3D (Plotly) |
  | 🔮 Predict Customer | Enter R / F / M values → instant segment prediction with confidence scores |
  | ⌨️ Manual Entry | Type raw transaction records → auto-compute RFM & predict |
  | 📁 Predict from CSV | Upload a new transaction file → batch predictions + download |
  | 💡 Business Insights | Per-segment descriptions & recommended actions |
- Auto-trains on `synthetic_online_retail.csv` at startup — no upload needed
- Retrain on any new CSV / Excel with one click
- Currency in Indian Rupees (₹)

---

## 🧩 Why 3D RFM Visualisation?

Traditional 2D plots show only partial relationships.
The 3D plot:

- Reveals overlapping segments hidden in 2D
- Highlights extreme high-value or dormant customers
- Improves intuition about customer lifecycle positioning
- Helps stakeholders visually grasp segmentation logic

---

## 🛠️ Tech Stack

- Python 3.10+
- Pandas
- NumPy
- scikit-learn (KMeans, RandomForest)
- Matplotlib & Seaborn
- Plotly (interactive 3D)
- Streamlit (web UI)
- openpyxl (Excel support)

---

## 📁 Repository Structure

```
customer_segmentation_rfm/
│
├── app.py                  # Streamlit web UI (6 tabs, dark/light theme)
├── rfm_engine.py           # Headless backend (load → cluster → train → predict)
├── requirements.txt        # Python dependencies
│
├── .streamlit/
│   └── config.toml         # Streamlit theme & server config
│
├── synthetic_online_retail.csv   # Default training dataset (20 000 rows)
├── OnlineRetail.csv              # Alternative real-world dataset
│
├── Dockerfile              # Container deployment
├── Procfile                # Heroku / Railway deployment
├── DEPLOY.md               # Detailed deployment guide
├── README.md
└── .gitignore
```

---

## 📋 Required CSV Columns

| Column        | Description              |
|--------------|--------------------------|
| `InvoiceNo`  | Invoice / transaction ID |
| `InvoiceDate`| Date of purchase         |
| `Quantity`   | Number of items          |
| `UnitPrice`  | Price per item (₹)       |
| `CustomerID` | Customer identifier      |

---

## 🚀 How to Run

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/customer_segmentation_rfm.git
   cd customer_segmentation_rfm
   ```

2. **Create a virtual environment (optional but recommended):**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate        # Windows
   # source .venv/bin/activate   # macOS / Linux
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch the app:**
   ```bash
   streamlit run app.py
   ```

5. **Open in browser:**
   ```
   http://localhost:8501
   ```

The model auto-trains on `synthetic_online_retail.csv` — the full dashboard loads instantly.

---

## ☁️ Deployment

### Option 1 — Streamlit Community Cloud (Recommended, Free)

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click **New app** → select your repo → set **Main file** to `app.py`
4. Click **Deploy** — done!

### Option 2 — Railway / Render (Docker)

See [DEPLOY.md](DEPLOY.md) for detailed instructions.

> **Note:** Vercel does not natively support Streamlit's WebSocket architecture.

---

## .gitignore

```gitignore
# ========================
# Python
# ========================
__pycache__/
*.py[cod]
*.pyo
*.pyd

# ========================
# Virtual Environments
# ========================
venv/
env/
.venv/

# ========================
# Environment Variables
# ========================
.env
.env.*

# ========================
# OS / Editor Files
# ========================
.DS_Store
Thumbs.db
.vscode/
.idea/

# ========================
# Outputs / Generated Files
# ========================
outputs/
figures/
plots/

# ========================
# Logs
# ========================
*.log
```
