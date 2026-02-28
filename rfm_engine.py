# ============================================================================
#  RFM Customer Intelligence Engine — Core Backend
#  ------------------------------------------------
#  Pure logic module (no CLI, no plt.show). Imported by the Streamlit app.
# ============================================================================

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
matplotlib.use("Agg")           # headless — figures returned, never shown
plt.style.use("default")

# ========================  COLOUR PALETTE  ================================

SEGMENT_COLORS = {
    "VIP Customers":     "#FFD700",
    "Loyal Customers":   "#2196F3",
    "Regular Customers": "#4CAF50",
    "At-Risk Customers": "#F44336",
    "New Customers":     "#9C27B0",
    "Other":             "#9E9E9E",
}

SEGMENT_ICONS = {
    "VIP Customers":     "👑",
    "Loyal Customers":   "💎",
    "Regular Customers": "🛒",
    "At-Risk Customers": "⚠️",
    "New Customers":     "🌱",
    "Other":             "❓",
}

# ========================  DATA HELPERS  ==================================


def load_dataset(source) -> pd.DataFrame:
    """Load from a file path (str) or an uploaded file-like object."""
    if isinstance(source, str):
        ext = os.path.splitext(source)[1].lower()
        if ext == ".csv":
            return pd.read_csv(source, encoding="latin1", on_bad_lines="skip")
        elif ext in (".xlsx", ".xls"):
            return pd.read_excel(source, engine="openpyxl")
        raise ValueError(f"Unsupported file type '{ext}'")
    else:
        # Streamlit UploadedFile
        name = getattr(source, "name", "file.csv")
        if name.endswith(".csv"):
            return pd.read_csv(source, encoding="latin1", on_bad_lines="skip")
        elif name.endswith((".xlsx", ".xls")):
            return pd.read_excel(source, engine="openpyxl")
        raise ValueError("Upload a .csv, .xlsx, or .xls file.")


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Validate, clean, and compute TotalPrice."""
    df.columns = df.columns.str.strip()

    required = ["InvoiceNo", "InvoiceDate", "Quantity", "UnitPrice", "CustomerID"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.dropna(subset=["CustomerID"]).copy()
    df["Quantity"]    = pd.to_numeric(df["Quantity"], errors="coerce").fillna(0)
    df["UnitPrice"]   = pd.to_numeric(df["UnitPrice"], errors="coerce").fillna(0)
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    df = df.dropna(subset=["InvoiceDate"])

    df["InvoiceNo"] = df["InvoiceNo"].astype(str)
    df = df[~df["InvoiceNo"].str.startswith("C")]
    df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]
    df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]
    return df


def compute_rfm(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Recency, Frequency, Monetary per customer."""
    ref = df["InvoiceDate"].max() + pd.Timedelta(days=1)
    rfm = df.groupby("CustomerID").agg(
        Recency=("InvoiceDate", lambda x: (ref - x.max()).days),
        Frequency=("InvoiceNo", "nunique"),
        Monetary=("TotalPrice", "sum"),
    )
    if rfm.shape[0] >= 10:
        rfm["Monetary"] = rfm["Monetary"].clip(upper=rfm["Monetary"].quantile(0.95))
    return rfm


# ========================  SEGMENT NAMING  ================================


def assign_segment_names(cluster_summary: pd.DataFrame) -> dict:
    """Dynamically name clusters based on centroid characteristics."""
    n = len(cluster_summary)
    if n <= 1:
        return {cluster_summary.index[0]: "Regular Customers"}

    cs = cluster_summary.copy()
    cs["score"] = (
        -cs["Recency"].rank() + cs["Frequency"].rank() + cs["Monetary"].rank()
    )
    ordered = cs.sort_values("score", ascending=False).index.tolist()

    labels = [
        "VIP Customers", "Loyal Customers", "Regular Customers",
        "At-Risk Customers", "New Customers",
    ]
    return {cid: labels[i] if i < len(labels) else f"Segment {cid}"
            for i, cid in enumerate(ordered)}


# ========================  CHART BUILDERS  ================================
# Each function *returns* a matplotlib Figure or a Plotly Figure object.


def fig_elbow(rfm_scaled: np.ndarray):
    max_k = min(10, rfm_scaled.shape[0])
    K = list(range(1, max_k + 1))
    ssd = []
    for k in K:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(rfm_scaled)
        ssd.append(km.inertia_)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(K, ssd, "o-", color="#2196F3", linewidth=2, markersize=8)
    ax.set_xlabel("Number of Clusters (k)")
    ax.set_ylabel("Inertia (SSD)")
    ax.set_title("Elbow Method — Optimal Cluster Selection", fontweight="bold")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def fig_scatter_2d(rfm: pd.DataFrame, x: str, y: str, title: str = ""):
    fig, ax = plt.subplots(figsize=(8, 5))
    for seg in rfm["Segment"].unique():
        m = rfm["Segment"] == seg
        ax.scatter(rfm.loc[m, x], rfm.loc[m, y],
                   label=seg, alpha=0.6, s=30,
                   color=SEGMENT_COLORS.get(seg, "#9E9E9E"),
                   edgecolors="white", linewidths=0.3)
    ax.set_xlabel(x); ax.set_ylabel(y)
    ax.set_title(title or f"{x} vs {y}", fontweight="bold")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.2)
    fig.tight_layout()
    return fig


def fig_heatmap(cluster_summary: pd.DataFrame, seg_map: dict):
    ds = cluster_summary.copy()
    ds.index = ds.index.map(lambda x: seg_map.get(x, f"Cluster {x}"))
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.heatmap(ds, annot=True, fmt=".1f", cmap="YlOrRd",
                linewidths=1, linecolor="white",
                cbar_kws={"label": "Mean Value"}, ax=ax)
    ax.set_title("Cluster Characteristics Heatmap", fontweight="bold")
    ax.set_ylabel("Customer Segment")
    fig.tight_layout()
    return fig


def fig_segment_bar(rfm: pd.DataFrame):
    counts = rfm["Segment"].value_counts()
    colors = [SEGMENT_COLORS.get(s, "#9E9E9E") for s in counts.index]
    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.bar(counts.index, counts.values, color=colors,
                  edgecolor="white", linewidth=1.5)
    for b, v in zip(bars, counts.values):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 5,
                str(v), ha="center", va="bottom", fontweight="bold")
    ax.set_xlabel("Customer Segment"); ax.set_ylabel("Count")
    ax.set_title("Customer Segment Distribution", fontweight="bold")
    plt.xticks(rotation=15); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def fig_confusion_matrix(cm, labels):
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels,
                linewidths=0.5, ax=ax)
    ax.set_title("Confusion Matrix", fontweight="bold")
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    fig.tight_layout()
    return fig


def fig_3d(rfm: pd.DataFrame, title: str = "3D RFM Customer Segments"):
    df_plot = rfm.reset_index()
    cmap = {s: SEGMENT_COLORS.get(s, "#9E9E9E") for s in df_plot["Segment"].unique()}
    fig = px.scatter_3d(
        df_plot, x="Recency", y="Frequency", z="Monetary",
        color="Segment", color_discrete_map=cmap,
        hover_name="CustomerID", title=title,
        labels={"Recency": "Recency (Days)",
                "Frequency": "Frequency (Orders)",
                "Monetary": "Monetary (₹)"},
        opacity=0.7,
    )
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=40),
                      legend=dict(bgcolor="rgba(255,255,255,0.8)"))
    return fig


def fig_3d_combined(rfm_train: pd.DataFrame, pred_df: pd.DataFrame, seg_map: dict):
    """3D scatter with training data + new predictions overlaid."""
    ex = rfm_train.reset_index().copy()
    ex["Source"] = "Training Data"

    pr = pred_df.copy()
    pr["CustomerID"] = [f"NEW_{i+1}" for i in range(len(pr))]
    pr["Segment"] = pr["Predicted_Segment"]
    pr["Source"] = "New Prediction"

    combined = pd.concat([
        ex[["CustomerID", "Recency", "Frequency", "Monetary", "Segment", "Source"]],
        pr[["CustomerID", "Recency", "Frequency", "Monetary", "Segment", "Source"]],
    ], ignore_index=True)

    cmap = {s: SEGMENT_COLORS.get(s, "#9E9E9E") for s in combined["Segment"].unique()}
    fig = px.scatter_3d(
        combined, x="Recency", y="Frequency", z="Monetary",
        color="Segment", symbol="Source", color_discrete_map=cmap,
        hover_name="CustomerID",
        title="3D RFM — Training + New Predictions",
        opacity=0.7,
    )
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=40))
    return fig


# ========================  BUSINESS INSIGHTS  =============================

INSIGHTS = {
    "VIP Customers": {
        "desc": "Most recent & frequent buyers with highest spend",
        "actions": [
            "Loyalty rewards & early product access",
            "Personalised premium offers",
            "Dedicated account management",
        ],
    },
    "Loyal Customers": {
        "desc": "Consistent high-value buyers",
        "actions": [
            "Cross-selling & bundle discounts",
            "Referral program incentives",
            "Exclusive member benefits",
        ],
    },
    "Regular Customers": {
        "desc": "Occasional buyers with moderate spend",
        "actions": [
            "Targeted discount coupons",
            "Product recommendation emails",
            "Engagement campaigns",
        ],
    },
    "At-Risk Customers": {
        "desc": "Long time since last purchase",
        "actions": [
            "Win-back campaigns with urgency",
            "Time-limited discounts",
            "Feedback surveys to understand churn",
        ],
    },
    "New Customers": {
        "desc": "Recently acquired with few transactions",
        "actions": [
            "Welcome series & onboarding",
            "First-purchase discounts",
            "Product education content",
        ],
    },
}


# ========================  CORE ENGINE CLASS  =============================


class RFMEngine:
    """Stateful engine: load → cluster → train → predict."""

    def __init__(self):
        self.df: pd.DataFrame | None = None
        self.rfm: pd.DataFrame | None = None
        self.rfm_scaled: np.ndarray | None = None
        self.scaler = StandardScaler()
        self.kmeans: KMeans | None = None
        self.classifier: RandomForestClassifier | None = None
        self.optimal_k: int = 4
        self.segment_map: dict = {}
        self.cluster_summary = pd.DataFrame()
        self.accuracy: float | None = None
        self.report_text: str = ""
        self.cm: np.ndarray | None = None
        self.cm_labels: list = []
        self.silhouette: float | None = None
        self.is_trained: bool = False

    # --- pipeline steps ------------------------------------------------

    def load(self, source) -> dict:
        """Load & clean. Returns info dict."""
        raw = load_dataset(source)
        self.df = clean_data(raw)
        self.rfm = compute_rfm(self.df)
        return {
            "raw_rows": len(raw),
            "clean_rows": len(self.df),
            "customers": len(self.rfm),
            "date_min": self.df["InvoiceDate"].min().date(),
            "date_max": self.df["InvoiceDate"].max().date(),
        }

    def cluster(self, k: int = 4) -> dict:
        """Scale + KMeans. Returns info dict."""
        self.optimal_k = min(k, self.rfm.shape[0])
        self.rfm_scaled = self.scaler.fit_transform(
            self.rfm[["Recency", "Frequency", "Monetary"]]
        )
        self.kmeans = KMeans(n_clusters=self.optimal_k, random_state=42, n_init=10)
        self.rfm["Cluster"] = self.kmeans.fit_predict(self.rfm_scaled)

        self.silhouette = (
            silhouette_score(self.rfm_scaled, self.rfm["Cluster"])
            if self.optimal_k >= 2 else None
        )

        self.cluster_summary = (
            self.rfm.groupby("Cluster")[["Recency", "Frequency", "Monetary"]]
            .mean().round(2)
        )
        self.segment_map = assign_segment_names(self.cluster_summary)
        self.rfm["Segment"] = self.rfm["Cluster"].map(self.segment_map).fillna("Other")

        return {
            "k": self.optimal_k,
            "silhouette": self.silhouette,
            "distribution": self.rfm["Segment"].value_counts().to_dict(),
        }

    def train(self) -> dict:
        """Train RandomForest on clustered data. Returns metrics."""
        X, y = self.rfm_scaled, self.rfm["Cluster"].values
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y,
        )
        self.classifier = RandomForestClassifier(
            n_estimators=100, random_state=42, n_jobs=-1,
        )
        self.classifier.fit(X_tr, y_tr)
        y_pred = self.classifier.predict(X_te)

        self.accuracy = accuracy_score(y_te, y_pred)
        self.cm_labels = [
            self.segment_map.get(c, f"Cluster {c}")
            for c in sorted(np.unique(y))
        ]
        self.report_text = classification_report(
            y_te, y_pred, target_names=self.cm_labels,
        )
        self.cm = confusion_matrix(y_te, y_pred)
        self.is_trained = True

        return {
            "accuracy": self.accuracy,
            "train_size": len(X_tr),
            "test_size": len(X_te),
        }

    # --- predictions ---------------------------------------------------

    def predict_single(self, recency, frequency, monetary):
        """Returns (cluster_id, segment_name, proba_dict)."""
        arr = np.array([[recency, frequency, monetary]])
        scaled = self.scaler.transform(arr)
        cid = self.classifier.predict(scaled)[0]
        seg = self.segment_map.get(cid, "Unknown")
        proba = {}
        if hasattr(self.classifier, "predict_proba"):
            p = self.classifier.predict_proba(scaled)[0]
            for i, v in enumerate(p):
                cls = self.classifier.classes_[i]
                proba[self.segment_map.get(cls, f"Cluster {cls}")] = round(v * 100, 1)
        return cid, seg, proba

    def predict_csv(self, source):
        """Predict on new transaction file → returns new rfm DF."""
        raw = load_dataset(source)
        df2 = clean_data(raw)
        rfm2 = compute_rfm(df2)
        scaled = self.scaler.transform(rfm2[["Recency", "Frequency", "Monetary"]])
        rfm2["Cluster"] = self.classifier.predict(scaled)
        rfm2["Segment"] = rfm2["Cluster"].map(self.segment_map).fillna("Other")
        return rfm2
