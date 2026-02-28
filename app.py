# ============================================================================
#  RFM Customer Intelligence Engine — Streamlit UI
#  Run:  streamlit run app.py
# ============================================================================

import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rfm_engine import (
    RFMEngine,
    SEGMENT_COLORS,
    SEGMENT_ICONS,
    INSIGHTS,
    fig_elbow,
    fig_scatter_2d,
    fig_heatmap,
    fig_segment_bar,
    fig_confusion_matrix,
    fig_3d,
    fig_3d_combined,
)

# ========================  CONSTANTS  =====================================

DEFAULT_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "synthetic_online_retail.csv")

# ========================  PAGE CONFIG  ===================================

st.set_page_config(
    page_title="RFM Customer Intelligence Engine",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ========================  THEME STATE  ===================================

if "theme" not in st.session_state:
    st.session_state.theme = "dark"  # default theme

is_dark = st.session_state.theme == "dark"

# ========================  CUSTOM CSS  ====================================

# -- Palette --------------------------------------------------------------
if is_dark:
    BG_MAIN       = "#0e1117"
    BG_SECONDARY  = "#1a1d23"
    BG_CARD       = "#1e2128"
    TEXT_PRIMARY  = "#e6e6e6"
    TEXT_SECONDARY= "#b0b0b0"
    ACCENT        = "#FFD700"
    BORDER        = "#2d3139"
    HEADER_BG     = "linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%)"
    PRED_BG       = "linear-gradient(135deg, #1b3a2d, #1e4d35)"
    PRED_BORDER   = "#4CAF50"
    PRED_TEXT     = "#e0e0e0"
    CODE_BG       = "#161b22"
    TAB_BG        = "#1a1d23"
    TAB_ACTIVE    = "#FFD700"
    METRIC_BORDER = "#FFD700"
else:
    BG_MAIN       = "#ffffff"
    BG_SECONDARY  = "#f8f9fa"
    BG_CARD       = "#ffffff"
    TEXT_PRIMARY  = "#1a1a2e"
    TEXT_SECONDARY= "#555555"
    ACCENT        = "#0f3460"
    BORDER        = "#e0e0e0"
    HEADER_BG     = "linear-gradient(135deg, #0f3460 0%, #1a5276 50%, #2980b9 100%)"
    PRED_BG       = "linear-gradient(135deg, #e8f5e9, #c8e6c9)"
    PRED_BORDER   = "#4CAF50"
    PRED_TEXT     = "#333333"
    CODE_BG       = "#f5f5f5"
    TAB_BG        = "#f0f0f0"
    TAB_ACTIVE    = "#0f3460"
    METRIC_BORDER = "#2196F3"

st.markdown(f"""
<style>
    /* ---------- Global overrides ---------- */
    .stApp {{
        background-color: {BG_MAIN};
        color: {TEXT_PRIMARY};
    }}

    /* ---------- ALL text elements ---------- */
    .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6 {{
        color: {TEXT_PRIMARY} !important;
    }}
    .stApp p, .stApp span, .stApp li, .stApp label, .stApp div {{
        color: {TEXT_PRIMARY};
    }}
    .stApp .stMarkdown, .stApp .stMarkdown p,
    .stApp .stMarkdown span, .stApp .stMarkdown li {{
        color: {TEXT_PRIMARY} !important;
    }}
    .stApp .stCaption, .stApp .stCaption p {{
        color: {TEXT_SECONDARY} !important;
    }}

    /* ---------- Sidebar ---------- */
    section[data-testid="stSidebar"] {{
        background-color: {BG_SECONDARY};
    }}
    section[data-testid="stSidebar"] * {{
        color: {TEXT_PRIMARY} !important;
    }}

    /* ---------- Input widgets ---------- */
    .stApp input, .stApp textarea {{
        color: {TEXT_PRIMARY} !important;
        background-color: {BG_CARD} !important;
        border-color: {BORDER} !important;
    }}
    .stApp .stSelectbox label,
    .stApp .stNumberInput label,
    .stApp .stTextInput label,
    .stApp .stDateInput label,
    .stApp .stSlider label,
    .stApp .stFileUploader label,
    .stApp .stRadio label {{
        color: {TEXT_PRIMARY} !important;
    }}
    .stApp .stSelectbox [data-baseweb="select"] {{
        color: {TEXT_PRIMARY} !important;
        background-color: {BG_CARD} !important;
    }}

    /* ---------- Buttons ---------- */
    .stApp button[kind="primary"] {{
        color: #fff !important;
    }}
    .stApp button:not([kind="primary"]) {{
        color: {TEXT_PRIMARY} !important;
        border-color: {BORDER} !important;
    }}

    /* ---------- File uploader / Browse button ---------- */
    .stApp [data-testid="stFileUploader"] button {{
        color: {TEXT_PRIMARY} !important;
        background-color: {BG_CARD} !important;
        border: 1px solid {BORDER} !important;
    }}
    .stApp [data-testid="stFileUploader"] section {{
        background-color: {BG_CARD} !important;
        border-color: {BORDER} !important;
    }}
    .stApp [data-testid="stFileUploader"] section > div {{
        color: {TEXT_SECONDARY} !important;
    }}
    .stApp [data-testid="stFileUploaderDropzone"] {{
        background-color: {BG_CARD} !important;
        border-color: {BORDER} !important;
    }}
    .stApp [data-testid="stFileUploaderDropzone"] span,
    .stApp [data-testid="stFileUploaderDropzone"] small,
    .stApp [data-testid="stFileUploaderDropzone"] p,
    .stApp [data-testid="stFileUploaderDropzone"] div {{
        color: {TEXT_SECONDARY} !important;
    }}
    .stApp [data-testid="stFileUploaderDropzone"] button {{
        color: {ACCENT} !important;
        border-color: {ACCENT} !important;
        background-color: transparent !important;
    }}

    /* ---------- Header banner ---------- */
    .main-header {{
        background: {HEADER_BG};
        padding: 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        text-align: center;
    }}
    .main-header h1 {{ color: #FFD700 !important; margin: 0; font-size: 2.2rem; }}
    .main-header p  {{ color: #e0e0e0 !important; margin: 0.3rem 0 0; font-size: 1rem; }}

    /* ---------- Metric cards ---------- */
    [data-testid="stMetric"] {{
        background: {BG_CARD};
        border-left: 4px solid {METRIC_BORDER};
        padding: 0.8rem 1rem;
        border-radius: 8px;
        border: 1px solid {BORDER};
    }}
    [data-testid="stMetricValue"] {{ color: {ACCENT} !important; }}
    [data-testid="stMetricLabel"] {{ color: {TEXT_SECONDARY} !important; }}

    /* ---------- Prediction result ---------- */
    .prediction-result {{
        background: {PRED_BG};
        padding: 1.5rem;
        border-radius: 12px;
        border: 2px solid {PRED_BORDER};
        text-align: center;
    }}
    .prediction-result h2 {{ color: {ACCENT} !important; }}
    .prediction-result p  {{ color: {PRED_TEXT} !important; }}

    /* ---------- Tabs ---------- */
    .stTabs [data-baseweb="tab-list"] {{ gap: 8px; }}
    .stTabs [data-baseweb="tab"] {{
        padding: 10px 20px;
        border-radius: 8px 8px 0 0;
        background: {TAB_BG};
        color: {TEXT_PRIMARY} !important;
    }}
    .stTabs [aria-selected="true"] {{
        border-bottom: 3px solid {TAB_ACTIVE} !important;
        font-weight: 700;
    }}

    /* ---------- Dataframes ---------- */
    .stDataFrame {{ border: 1px solid {BORDER}; border-radius: 8px; }}

    /* ---------- Expanders ---------- */
    .streamlit-expanderHeader, [data-testid="stExpanderToggleIcon"] {{
        background: {BG_CARD};
        border-radius: 8px;
        color: {TEXT_PRIMARY} !important;
    }}
    .stApp details summary span {{
        color: {TEXT_PRIMARY} !important;
    }}

    /* ---------- Code blocks ---------- */
    code, .stCode {{
        background: {CODE_BG} !important;
        color: {TEXT_PRIMARY} !important;
    }}

    /* ---------- Alerts / info / success / warning boxes ---------- */
    .stApp .stAlert p, .stApp .stAlert span {{
        color: {TEXT_PRIMARY} !important;
    }}

    /* ---------- Progress bars ---------- */
    .stProgress > div > div > div {{
        background: {ACCENT};
    }}
</style>
""", unsafe_allow_html=True)

# ========================  SESSION STATE  =================================

if "engine" not in st.session_state:
    st.session_state.engine = RFMEngine()
if "predictions" not in st.session_state:
    st.session_state.predictions = []
if "trained_file" not in st.session_state:
    st.session_state.trained_file = None

engine: RFMEngine = st.session_state.engine

# ========================  AUTO-TRAIN ON STARTUP  =========================

if not engine.is_trained and os.path.isfile(DEFAULT_CSV):
    engine.load(DEFAULT_CSV)
    engine.cluster(4)
    engine.train()
    st.session_state.trained_file = "synthetic_online_retail.csv"

# ========================  HEADER  ========================================

st.markdown("""
<div class="main-header">
    <h1>🧠 RFM Customer Intelligence Engine</h1>
    <p>Segment · Train · Predict · Visualise — All in one place</p>
</div>
""", unsafe_allow_html=True)

# ========================  SIDEBAR  =======================================

with st.sidebar:
    # --- Theme toggle ---
    theme_col1, theme_col2 = st.columns([3, 1])
    with theme_col1:
        st.title("Control Panel")
    with theme_col2:
        theme_icon = "🌙" if is_dark else "☀️"
        if st.button(theme_icon, help="Toggle dark / light theme",
                     use_container_width=True, key="theme_toggle"):
            st.session_state.theme = "light" if is_dark else "dark"
            st.rerun()

    st.caption(f"Theme: {'Dark' if is_dark else 'Light'}")
    st.divider()

    # --- Upload ---
    st.subheader("📂 Training Data")
    upload = st.file_uploader(
        "Upload CSV / Excel",
        type=["csv", "xlsx", "xls"],
        help="Columns required: InvoiceNo, InvoiceDate, Quantity, UnitPrice, CustomerID",
    )

    n_clusters = st.slider("Number of clusters (k)", 2, 10, 4,
                           help="Use the Elbow chart to pick the best k")

    train_btn = st.button("🚀 Train Model", use_container_width=True, type="primary")

    st.divider()

    # Status
    if engine.is_trained:
        st.success(f"✅ Model Accuracy: **{engine.accuracy * 100:.1f}%**")
        st.caption(f"Dataset: {st.session_state.trained_file}")
        st.caption(f"Clusters: {engine.optimal_k} · Customers: {len(engine.rfm):,}")
    else:
        st.info("Upload data and click **Train Model** to start")

    st.divider()
    st.caption("Built with Streamlit · RFM Engine v2.0")

# ========================  TRAINING FLOW  =================================

if train_btn:
    if upload is not None:
        src, name = upload, upload.name
    elif os.path.isfile(DEFAULT_CSV):
        src, name = DEFAULT_CSV, "synthetic_online_retail.csv"
    else:
        st.error("No data available. Upload a CSV.")
        st.stop()

    with st.spinner(f"Training on {name}..."):
        try:
            engine.load(src)
            engine.cluster(n_clusters)
            engine.train()
            st.session_state.trained_file = name
            st.session_state.predictions = []
        except Exception as e:
            st.error(f"Training error: {e}")
            st.stop()
    st.rerun()

# ========================  MAIN CONTENT  ==================================

if not engine.is_trained:
    st.info("No model trained. Upload data in the sidebar and click **Train Model**.")
    st.stop()

# --------- TABS ----------------------------------------------------------

tabs = st.tabs([
    "📊 Dashboard",
    "📈 Visualisations",
    "🔮 Predict Customer",
    "⌨️ Manual Entry",
    "📁 Predict from CSV",
    "💡 Business Insights",
])

# ========================  TAB 1: DASHBOARD  ==============================

with tabs[0]:
    st.header("Training Dashboard")

    # KPI row
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Model Accuracy", f"{engine.accuracy * 100:.1f}%")
    c2.metric("Silhouette Score", f"{engine.silhouette:.3f}" if engine.silhouette else "N/A")
    c3.metric("Clusters", engine.optimal_k)
    c4.metric("Customers", f"{len(engine.rfm):,}")

    st.divider()

    # RFM table preview
    col_left, col_right = st.columns([3, 2])
    with col_left:
        st.subheader("RFM Table Preview")
        st.dataframe(
            engine.rfm.reset_index().head(50),
            use_container_width=True,
            height=400,
        )
    with col_right:
        st.subheader("Cluster Summary")
        summary_display = engine.cluster_summary.copy()
        summary_display.index = summary_display.index.map(
            lambda x: f"{SEGMENT_ICONS.get(engine.segment_map.get(x,''), '')} {engine.segment_map.get(x, f'Cluster {x}')}"
        )
        st.dataframe(summary_display, use_container_width=True)

        st.subheader("Segment Distribution")
        dist = engine.rfm["Segment"].value_counts()
        for seg, cnt in dist.items():
            icon = SEGMENT_ICONS.get(seg, "")
            color = SEGMENT_COLORS.get(seg, "#9E9E9E")
            pct = cnt / len(engine.rfm) * 100
            st.markdown(
                f"**{icon} {seg}** — {cnt:,} customers ({pct:.1f}%)"
            )
            st.progress(pct / 100)

    st.divider()

    # Classification report + confusion matrix
    st.subheader("Classification Report")
    r_col, cm_col = st.columns([1, 1])
    with r_col:
        st.code(engine.report_text, language="text")
    with cm_col:
        st.pyplot(fig_confusion_matrix(engine.cm, engine.cm_labels))

# ========================  TAB 2: VISUALISATIONS  =========================

with tabs[1]:
    st.header("Interactive Visualisations")

    viz_source = st.radio(
        "Data source",
        ["Training Data", "Latest CSV Prediction"],
        horizontal=True,
        help="Switch between training-set charts and prediction-set charts",
    )
    rfm_viz = engine.rfm

    if viz_source == "Latest CSV Prediction" and "pred_rfm" in st.session_state:
        rfm_viz = st.session_state.pred_rfm
    elif viz_source == "Latest CSV Prediction":
        st.warning("No CSV prediction data yet. Showing training data.")

    # Elbow
    with st.expander("🔎 Elbow Method", expanded=False):
        st.pyplot(fig_elbow(engine.rfm_scaled))

    # 2D Scatters
    st.subheader("2D Scatter Plots")
    s1, s2 = st.columns(2)
    with s1:
        st.pyplot(fig_scatter_2d(rfm_viz, "Recency", "Monetary", "Recency vs Monetary"))
    with s2:
        st.pyplot(fig_scatter_2d(rfm_viz, "Frequency", "Monetary", "Frequency vs Monetary"))

    s3, s4 = st.columns(2)
    with s3:
        st.pyplot(fig_scatter_2d(rfm_viz, "Recency", "Frequency", "Recency vs Frequency"))
    with s4:
        st.pyplot(fig_segment_bar(rfm_viz))

    # Heatmap
    st.subheader("Cluster Heatmap")
    if viz_source == "Latest CSV Prediction" and "pred_rfm" in st.session_state:
        pred_summary = rfm_viz.groupby("Cluster")[["Recency", "Frequency", "Monetary"]].mean().round(2)
        st.pyplot(fig_heatmap(pred_summary, engine.segment_map))
    else:
        st.pyplot(fig_heatmap(engine.cluster_summary, engine.segment_map))

    # 3D
    st.subheader("Interactive 3D Scatter")
    st.plotly_chart(fig_3d(rfm_viz), use_container_width=True)

# ========================  TAB 3: PREDICT SINGLE  =========================

with tabs[2]:
    st.header("🔮 Predict Customer Segment")
    st.markdown("Enter RFM values to instantly predict which segment a customer belongs to.")

    pcol1, pcol2, pcol3 = st.columns(3)
    with pcol1:
        p_recency = st.number_input(
            "Recency (days since last purchase)", min_value=0, value=30, step=1,
        )
    with pcol2:
        p_frequency = st.number_input(
            "Frequency (number of orders)", min_value=1, value=5, step=1,
        )
    with pcol3:
        p_monetary = st.number_input(
            "Monetary (total spend ₹)", min_value=0.0, value=500.0, step=10.0,
        )

    if st.button("⚡ Predict Segment", type="primary", use_container_width=True):
        cid, seg, proba = engine.predict_single(p_recency, p_frequency, p_monetary)
        icon = SEGMENT_ICONS.get(seg, "")
        color = SEGMENT_COLORS.get(seg, "#9E9E9E")

        st.markdown(f"""
        <div class="prediction-result">
            <h2 style="margin:0">{icon} {seg}</h2>
            <p style="margin:0.5rem 0 0; font-size:1.1rem; color:#333">
                Recency {p_recency}d · Frequency {p_frequency} · Monetary ₹{p_monetary:,.0f}
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Confidence bars
        if proba:
            st.subheader("Confidence Scores")
            for seg_name, pct in sorted(proba.items(), key=lambda x: -x[1]):
                seg_icon = SEGMENT_ICONS.get(seg_name, "")
                st.markdown(f"**{seg_icon} {seg_name}** — {pct:.1f}%")
                st.progress(pct / 100)

        # Save prediction
        st.session_state.predictions.append({
            "Recency": p_recency,
            "Frequency": p_frequency,
            "Monetary": p_monetary,
            "Predicted_Segment": seg,
        })

    # Prediction history
    if st.session_state.predictions:
        st.divider()
        st.subheader("Prediction History")
        hist_df = pd.DataFrame(st.session_state.predictions)
        st.dataframe(hist_df, use_container_width=True)

        if len(hist_df) >= 1:
            st.subheader("Predictions on 3D Map")
            st.plotly_chart(
                fig_3d_combined(engine.rfm, hist_df, engine.segment_map),
                use_container_width=True,
            )

        if st.button("🗑️ Clear History"):
            st.session_state.predictions = []
            st.rerun()

# ========================  TAB 4: MANUAL ENTRY  ==========================

with tabs[3]:
    st.header("⌨️ Predict from Manual Transaction Records")
    st.markdown(
        "Enter raw transaction data (at least **2 records**) and the model "
        "will compute RFM scores and predict customer segments automatically."
    )

    # --- number of records ------------------------------------------------
    n_records = st.number_input(
        "How many transaction records?",
        min_value=2, max_value=50, value=2, step=1,
        key="manual_n_records",
    )

    st.divider()
    st.subheader("Enter Transaction Details")

    manual_rows = []
    for i in range(int(n_records)):
        with st.expander(f"Record {i + 1}", expanded=(i < 2)):
            mc1, mc2 = st.columns(2)
            with mc1:
                inv_no = st.text_input(
                    "Invoice No", value=f"INV{5000 + i}",
                    key=f"man_inv_{i}",
                )
                inv_date = st.date_input(
                    "Invoice Date",
                    value=pd.Timestamp.today().date(),
                    key=f"man_date_{i}",
                )
                cust_id = st.number_input(
                    "Customer ID", min_value=1, value=90000 + i,
                    step=1, key=f"man_cid_{i}",
                )
            with mc2:
                qty = st.number_input(
                    "Quantity", min_value=1, value=1,
                    step=1, key=f"man_qty_{i}",
                )
                price = st.number_input(
                    "Unit Price (₹)", min_value=0.01, value=100.0,
                    step=1.0, key=f"man_price_{i}",
                )
                desc = st.text_input(
                    "Description (optional)", value="Product",
                    key=f"man_desc_{i}",
                )

            manual_rows.append({
                "InvoiceNo": str(inv_no),
                "InvoiceDate": pd.Timestamp(inv_date),
                "CustomerID": int(cust_id),
                "Quantity": int(qty),
                "UnitPrice": float(price),
                "Description": desc,
                "StockCode": f"STK{i}",
                "Country": "India",
            })

    st.divider()

    # --- preview ----------------------------------------------------------
    manual_df_preview = pd.DataFrame(manual_rows)
    st.subheader("Preview of Entered Records")
    st.dataframe(manual_df_preview, use_container_width=True)

    # --- predict ----------------------------------------------------------
    if st.button("🚀 Compute RFM & Predict", type="primary",
                 use_container_width=True, key="manual_predict"):
        manual_df = pd.DataFrame(manual_rows)

        # Validate: need at least 2 records
        if len(manual_df) < 2:
            st.error("Please enter at least 2 records.")
        else:
            try:
                from rfm_engine import clean_data, compute_rfm

                cleaned = clean_data(manual_df)
                if cleaned.empty:
                    st.error("All records were filtered out during cleaning. "
                             "Check that Quantity > 0, UnitPrice > 0, and "
                             "InvoiceNo does not start with 'C'.")
                    st.stop()

                rfm_manual = compute_rfm(cleaned)

                # Predict using the trained scaler + classifier
                scaled = engine.scaler.transform(
                    rfm_manual[["Recency", "Frequency", "Monetary"]]
                )
                rfm_manual["Cluster"] = engine.classifier.predict(scaled)
                rfm_manual["Segment"] = (
                    rfm_manual["Cluster"].map(engine.segment_map)
                                         .fillna("Other")
                )

                st.success(
                    f"Predicted segments for **{len(rfm_manual)}** "
                    f"customer(s) from {len(cleaned)} transaction(s)!"
                )

                # Results table
                st.subheader("RFM Scores & Predicted Segments")
                st.dataframe(
                    rfm_manual.reset_index(), use_container_width=True
                )

                # Segment badges
                for _, row in rfm_manual.iterrows():
                    seg = row["Segment"]
                    icon = SEGMENT_ICONS.get(seg, "")
                    st.markdown(
                        f"**Customer {int(row.name)}** — "
                        f"R={row['Recency']:.0f}d  F={row['Frequency']:.0f}  "
                        f"M=₹{row['Monetary']:,.0f} → "
                        f"{icon} **{seg}**"
                    )

                # Confidence scores per customer
                if hasattr(engine.classifier, "predict_proba"):
                    st.subheader("Confidence Scores")
                    probas = engine.classifier.predict_proba(scaled)
                    for idx, (cust_idx, row) in enumerate(
                        rfm_manual.iterrows()
                    ):
                        with st.expander(
                            f"Customer {int(cust_idx)} — "
                            f"{SEGMENT_ICONS.get(row['Segment'], '')} "
                            f"{row['Segment']}"
                        ):
                            p = probas[idx]
                            for ci, v in enumerate(p):
                                cls = engine.classifier.classes_[ci]
                                sn = engine.segment_map.get(
                                    cls, f"Cluster {cls}"
                                )
                                si = SEGMENT_ICONS.get(sn, "")
                                st.markdown(
                                    f"**{si} {sn}** — {v * 100:.1f}%"
                                )
                                st.progress(min(v, 1.0))

                # 3D scatter
                if len(rfm_manual) >= 1:
                    st.subheader("3D Scatter — Manual Entries")
                    st.plotly_chart(
                        fig_3d(rfm_manual,
                               "3D RFM — Manual Transaction Predictions"),
                        use_container_width=True,
                    )

                # Download
                csv_out = rfm_manual.reset_index().to_csv(index=False)
                st.download_button(
                    "⬇️ Download Results CSV", csv_out,
                    file_name="rfm_manual_predictions.csv",
                    mime="text/csv",
                    key="manual_download",
                )

            except Exception as e:
                st.error(f"Prediction error: {e}")

# ========================  TAB 5: PREDICT CSV  ============================

with tabs[4]:
    st.header("📁 Predict Segments from New CSV")
    st.markdown("Upload a new transaction file to predict customer segments using the trained model.")

    pred_upload = st.file_uploader(
        "Upload new transaction data",
        type=["csv", "xlsx", "xls"],
        key="pred_upload",
    )

    if pred_upload and st.button("🔮 Run Predictions on File", type="primary"):
        with st.spinner("Processing new data..."):
            try:
                pred_rfm = engine.predict_csv(pred_upload)
                st.session_state.pred_rfm = pred_rfm
            except Exception as e:
                st.error(f"Error: {e}")
                st.stop()

        st.success(f"Predicted segments for **{len(pred_rfm):,}** customers!")

        # Results
        st.subheader("Prediction Results")
        st.dataframe(pred_rfm.reset_index(), use_container_width=True, height=400)

        # Distribution
        dc1, dc2 = st.columns(2)
        with dc1:
            st.subheader("Segment Distribution")
            for seg, cnt in pred_rfm["Segment"].value_counts().items():
                icon = SEGMENT_ICONS.get(seg, "")
                st.markdown(f"**{icon} {seg}** — {cnt:,}")
        with dc2:
            st.pyplot(fig_segment_bar(pred_rfm))

        # Charts
        st.subheader("Visualisations (New Data)")
        v1, v2 = st.columns(2)
        with v1:
            st.pyplot(fig_scatter_2d(pred_rfm, "Recency", "Monetary", "Recency vs Monetary (New)"))
        with v2:
            st.pyplot(fig_scatter_2d(pred_rfm, "Frequency", "Monetary", "Frequency vs Monetary (New)"))

        pred_summary = pred_rfm.groupby("Cluster")[["Recency", "Frequency", "Monetary"]].mean().round(2)
        st.pyplot(fig_heatmap(pred_summary, engine.segment_map))

        st.subheader("3D Scatter (New Data)")
        st.plotly_chart(fig_3d(pred_rfm, "3D RFM — New Data Predictions"), use_container_width=True)

        # Download
        csv_out = pred_rfm.reset_index().to_csv(index=False)
        st.download_button(
            "⬇️ Download Predictions CSV",
            csv_out,
            file_name="rfm_predictions.csv",
            mime="text/csv",
        )

# ========================  TAB 6: BUSINESS INSIGHTS  ======================

with tabs[5]:
    st.header("💡 Actionable Business Insights")

    for cluster_id, seg_name in sorted(engine.segment_map.items()):
        icon = SEGMENT_ICONS.get(seg_name, "❓")
        color = SEGMENT_COLORS.get(seg_name, "#9E9E9E")
        info = INSIGHTS.get(seg_name, {"desc": "Custom segment", "actions": []})

        # Segment stats from cluster summary
        if cluster_id in engine.cluster_summary.index:
            row = engine.cluster_summary.loc[cluster_id]
            stats_str = (f"Avg Recency: **{row['Recency']:.0f}** days · "
                         f"Avg Frequency: **{row['Frequency']:.0f}** orders · "
                         f"Avg Monetary: **₹{row['Monetary']:,.0f}**")
        else:
            stats_str = ""

        count = (engine.rfm["Segment"] == seg_name).sum()
        pct = count / len(engine.rfm) * 100

        with st.expander(f"{icon}  {seg_name}  —  {count:,} customers ({pct:.1f}%)", expanded=True):
            st.markdown(f"**{info['desc']}**")
            st.caption(stats_str)
            st.markdown("**Recommended Actions:**")
            for action in info["actions"]:
                st.markdown(f"- ✅ {action}")
