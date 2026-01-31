# customer_segmentation_RFM
This project applies RFM (Recency, Frequency, Monetary) analysis for customer segmentation and extends traditional methods using 3D visualization. By plotting RFM dimensions together, it reveals hidden patterns, overlaps, and high-value behaviors not visible in 2D, emphasizing interpretability for academic and analytics use.
The notebook demonstrates how multidimensional visualization can reveal patterns that are not always visible in pairwise (2D) plots.
---

## 📌 Objectives
- Calculate RFM metrics for customers
- Segment customers based on RFM scores
- Visualize customer segments using:
  - 2D scatter plots
  - **3D scatter plot (Recency × Frequency × Monetary)**
- Derive business insights from multidimensional clustering
---

## 📊 Key Features
- Clean RFM feature engineering
- Segment-based color coding
- 3D visualization using `matplotlib` and `Axes3D`
- Clear analytical reasoning embedded in markdown
- Insight summary comparing 2D vs 3D views
---

## 🧩 Why 3D RFM Visualization?
Traditional 2D plots show only partial relationships.  
The 3D plot:
- Reveals overlapping segments hidden in 2D
- Highlights extreme high-value or dormant customers
- Improves intuition about customer lifecycle positioning
- Helps stakeholders visually grasp segmentation logic
---

## 🛠️ Tech Stack
- Python 3.8+
- Pandas
- NumPy
- Matplotlib
- Jupyter Notebook
---

📁 Recommended Repository Structure
rfm-customer-segmentation/
│
├── notebooks/
│   └── rfm_analysis_3d.ipynb
│
├── data/
│   └── (optional dataset files)
│
├── outputs/
│   └── figures/
│
├── README.md
├── requirements.txt
├── .gitignore
└── LICENSE

## 🚀 How to Run
1. Clone the repository:
   git clone https://github.com/your-username/rfm-customer-segmentation.git
   cd rfm-customer-segmentation
2.Create a virtual environment (optional but recommended):
python -m venv venv
source venv/bin/activate
3.Install dependencies:
pip install -r requirements.txt
4.Launch Jupyter:
jupyter notebook
5.Open:
notebooks/rfm_analysis_3d.ipynb

##.gitignore
# ========================
# Python
# ========================
__pycache__/
*.py[cod]
*.pyo
*.pyd

# ========================
# Jupyter Notebook
# ========================
.ipynb_checkpoints/

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
