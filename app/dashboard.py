from pathlib import Path

import json

import pandas as pd

import plotly.express as px

import plotly.graph_objects as go

import streamlit as st

# =========================================================
# DESIGN TOKENS
# =========================================================

COLORWAY = ["#3267D6", "#13A8B8", "#E9C46A", "#E76F51", "#2A9D8F", "#8E6CD8"]
FONT_FAMILY = "Inter, 'Segoe UI', Arial, sans-serif"

# Keep every Plotly chart readable on the light dashboard theme.
px.defaults.template = "plotly_white"
px.defaults.color_discrete_sequence = COLORWAY


def style_chart(fig):
    """Apply the same polished light theme to every Plotly chart."""
    fig.update_layout(
        template="plotly_white",
        colorway=COLORWAY,
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font=dict(family=FONT_FAMILY, color="#263A56", size=12),
        title=dict(font=dict(color="#16345C", size=17, family=FONT_FAMILY), x=0.015, xanchor="left"),
        legend=dict(
            font=dict(color="#263A56", size=12),
            bgcolor="rgba(255,255,255,0)",
            title_font=dict(color="#6B7A90", size=11),
        ),
        hoverlabel=dict(
            bgcolor="#16345C",
            font=dict(color="#FFFFFF", family=FONT_FAMILY, size=12),
            bordercolor="#16345C",
        ),
        bargap=0.32,
        margin=dict(l=22, r=22, t=60, b=22),
    )
    fig.update_xaxes(
        showgrid=True, gridcolor="#EEF2F8", zerolinecolor="#DCE4EF",
        linecolor="#DCE4EF", tickfont=dict(color="#52647D"),
        title_font=dict(color="#6B7A90", size=12),
    )
    fig.update_yaxes(
        showgrid=True, gridcolor="#EEF2F8", zerolinecolor="#DCE4EF",
        linecolor="#DCE4EF", tickfont=dict(color="#52647D"),
        title_font=dict(color="#6B7A90", size=12),
    )
    fig.update_coloraxes(colorbar=dict(outlinewidth=0, thickness=14, tickfont=dict(color="#52647D")))
    return fig


# =========================================================
# PATHS
# =========================================================

ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = ROOT / "data" / "processed" / "training_snapshots.csv"

PREDICTIONS_PATH = ROOT / "reports" / "test_predictions.csv"

METRICS_PATH = ROOT / "reports" / "model_metrics.json"

SHAP_PATH = ROOT / "reports" / "shap_customer_explanations.csv"

SHAP_GLOBAL_PATH = ROOT / "reports" / "shap_global_importance.csv"


# =========================================================
# PAGE CONFIGURATION AND STYLE
# =========================================================

st.set_page_config(
    page_title="CustomerIQ",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root {
    --navy: #16345C;
    --navy-deep: #0F2745;
    --blue: #3267D6;
    --blue-soft: #EAF1FF;
    --cyan: #13A8B8;
    --ink: #203047;
    --muted: #6B7A90;
    --line: #E4EAF2;
    --surface: #FFFFFF;
    --page: #F4F7FB;
    --radius: 16px;
    --shadow-sm: 0 2px 8px rgba(27, 49, 82, 0.05);
    --shadow-md: 0 8px 24px rgba(27, 49, 82, 0.08);
}

/* ---------- Base ---------- */
html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"],
button, input, textarea, select {
    font-family: 'Inter', 'Segoe UI', Arial, sans-serif !important;
}
html, body, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
    color-scheme: light !important;
}
.stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
    background: var(--page) !important;
    color: var(--ink) !important;
}
.block-container {
    padding-top: 1.15rem;
    padding-bottom: 2rem;
    max-width: 1450px;
}
#MainMenu, footer { visibility: hidden; }
[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stToolbar"] { background: transparent !important; }

h1, h2, h3, p, label, [data-testid="stMarkdownContainer"] { color: var(--ink); }
h1, h2, h3 { letter-spacing: -0.4px; }
h1 { font-weight: 800 !important; }
h2, h3 { font-weight: 700 !important; }

/* Section headings with a small accent bar */
[data-testid="stMain"] h3 {
    color: var(--navy) !important;
    font-size: 1.2rem !important;
    padding-left: 0.75rem;
    border-left: 4px solid var(--blue);
    line-height: 1.25;
    margin: 0.4rem 0 0.9rem 0;
}

hr {
    border: none !important;
    height: 1px;
    background: linear-gradient(90deg, transparent, #D9E2EF 12%, #D9E2EF 88%, transparent) !important;
    margin: 1.4rem 0 !important;
}

/* ---------- Sidebar ---------- */
section[data-testid="stSidebar"],
section[data-testid="stSidebar"] > div {
    background: #FFFFFF !important;
    border-right: 1px solid #E5EAF2;
    box-shadow: 3px 0 18px rgba(28, 47, 77, 0.04);
}
section[data-testid="stSidebar"] > div { padding-top: 1.25rem; }
section[data-testid="stSidebar"] * { color: #263A56; }
section[data-testid="stSidebar"] h1 {
    color: var(--navy) !important;
    font-size: 1.55rem !important;
    letter-spacing: -0.8px;
}
section[data-testid="stSidebar"] h3 {
    font-size: 0.78rem !important;
    text-transform: uppercase;
    letter-spacing: 1.3px;
    color: #7A8AA0 !important;
    font-weight: 700 !important;
    border-left: none !important;
    padding-left: 0 !important;
    margin-bottom: 0.4rem !important;
}
section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {
    color: #7A8AA0 !important;
    letter-spacing: 1.4px;
    font-size: 0.70rem;
    font-weight: 750;
}
section[data-testid="stSidebar"] hr { background: #E5EAF2 !important; height: 1px; }

/* Navigation radio rendered as nav pills */
section[data-testid="stSidebar"] [data-testid="stRadio"] > label { display: none; }
section[data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] { gap: 0.2rem; }
section[data-testid="stSidebar"] [data-testid="stRadio"] label[data-baseweb="radio"] {
    padding: 0.6rem 0.8rem;
    border-radius: 10px;
    font-weight: 600;
    width: 100%;
    transition: background 0.15s ease, transform 0.15s ease;
    border-left: 3px solid transparent;
}
section[data-testid="stSidebar"] [data-testid="stRadio"] label[data-baseweb="radio"] > div:first-child {
    display: none;
}
section[data-testid="stSidebar"] [data-testid="stRadio"] label[data-baseweb="radio"]:hover {
    background: #F0F5FF;
    transform: translateX(2px);
}
section[data-testid="stSidebar"] [data-testid="stRadio"] label[data-baseweb="radio"]:has(input:checked) {
    background: var(--blue-soft);
    border-left: 3px solid var(--blue);
}
section[data-testid="stSidebar"] [data-testid="stRadio"] label[data-baseweb="radio"]:has(input:checked) * {
    color: #244C91 !important;
    font-weight: 700;
}
section[data-testid="stSidebar"] [data-testid="stSlider"] { padding-top: 0.25rem; }

/* ---------- KPI cards ---------- */
[data-testid="stMetric"] {
    background: #FFFFFF !important;
    border: 1px solid var(--line);
    border-top: 3px solid var(--blue);
    padding: 1.1rem 1.2rem;
    border-radius: var(--radius);
    box-shadow: var(--shadow-sm);
    min-height: 112px;
    transition: transform 0.18s ease, box-shadow 0.18s ease;
}
[data-testid="stMetric"]:hover {
    transform: translateY(-3px);
    box-shadow: var(--shadow-md);
}
[data-testid="stHorizontalBlock"] > div:nth-child(2) [data-testid="stMetric"] { border-top-color: var(--cyan); }
[data-testid="stHorizontalBlock"] > div:nth-child(3) [data-testid="stMetric"] { border-top-color: #E9A23B; }
[data-testid="stHorizontalBlock"] > div:nth-child(4) [data-testid="stMetric"] { border-top-color: #2A9D8F; }
[data-testid="stMetricLabel"], [data-testid="stMetricLabel"] * {
    color: var(--muted) !important;
    font-size: 0.82rem !important;
    font-weight: 650 !important;
    text-transform: uppercase;
    letter-spacing: 0.6px;
}
[data-testid="stMetricValue"], [data-testid="stMetricValue"] * {
    color: var(--navy) !important;
    font-weight: 800 !important;
    letter-spacing: -0.6px;
}

/* ---------- Charts and tables ---------- */
div[data-testid="stPlotlyChart"] {
    background: #FFFFFF !important;
    border: 1px solid var(--line);
    border-radius: 14px;
    padding: 0.25rem;
    box-shadow: var(--shadow-sm);
    transition: box-shadow 0.18s ease;
}
div[data-testid="stPlotlyChart"]:hover { box-shadow: var(--shadow-md); }
[data-testid="stDataFrame"], [data-testid="stTable"] {
    background: #FFFFFF !important;
    border: 1px solid var(--line);
    border-radius: 14px;
    overflow: hidden;
    box-shadow: var(--shadow-sm);
}
[data-testid="stDataFrame"] [role="grid"] { border-radius: 14px; }
[data-testid="stJson"] {
    background: #FFFFFF !important;
    border: 1px solid var(--line);
    border-radius: 14px;
    padding: 0.9rem 1rem;
    box-shadow: var(--shadow-sm);
}

/* ---------- Inputs ---------- */
div[data-baseweb="input"] > div,
div[data-baseweb="select"] > div,
div[data-baseweb="textarea"] > div {
    border-radius: 10px;
    border-color: #D5DFED;
    background: #FFFFFF;
    transition: border-color 0.15s ease, box-shadow 0.15s ease;
}
div[data-baseweb="input"] > div:focus-within,
div[data-baseweb="select"] > div:focus-within {
    border-color: var(--blue) !important;
    box-shadow: 0 0 0 3px rgba(50, 103, 214, 0.15);
}
[data-testid="stWidgetLabel"] p {
    font-weight: 600;
    color: #33465F;
    font-size: 0.88rem;
}
/* Multiselect tags */
span[data-baseweb="tag"] {
    background: var(--blue-soft) !important;
    border-radius: 8px !important;
}
span[data-baseweb="tag"] * { color: #244C91 !important; font-weight: 600; }
/* Slider */
[data-testid="stSlider"] [role="slider"] {
    background: var(--blue) !important;
    box-shadow: 0 0 0 4px rgba(50, 103, 214, 0.18);
}
[data-testid="stSlider"] [data-testid="stThumbValue"],
[data-testid="stSlider"] [data-testid="stSliderThumbValue"] {
    color: var(--blue) !important;
    font-weight: 700;
}

/* ---------- Buttons ---------- */
button[kind="primary"], .stDownloadButton button {
    border-radius: 10px !important;
    font-weight: 650 !important;
    padding: 0.55rem 1.1rem !important;
    transition: all 0.15s ease-in-out;
}
.stDownloadButton button {
    background: #FFFFFF !important;
    color: #244C91 !important;
    border: 1px solid #CAD8EE !important;
    box-shadow: var(--shadow-sm);
}
.stDownloadButton button:hover {
    background: var(--blue) !important;
    color: #FFFFFF !important;
    border-color: var(--blue) !important;
    transform: translateY(-1px);
    box-shadow: 0 6px 16px rgba(50, 103, 214, 0.25);
}
.stDownloadButton button:hover * { color: #FFFFFF !important; }

/* ---------- Compact, minimal header ---------- */
.customeriq-hero {
    background: #FFFFFF;
    border: 1px solid #E3EAF4;
    border-left: 5px solid #3267D6;
    border-radius: 14px;
    padding: 1.15rem 1.5rem 1.2rem;
    margin: 0 0 1.15rem 0;
    box-shadow: 0 3px 12px rgba(27, 49, 82, 0.045);
    position: relative;
}
.customeriq-hero .eyebrow {
    color: #3267D6;
    font-size: 0.68rem;
    letter-spacing: 1.5px;
    font-weight: 800;
    text-transform: uppercase;
    margin-bottom: 0.35rem;
}
.customeriq-hero h1 {
    color: #16345C !important;
    font-size: 2rem;
    line-height: 1.12;
    margin: 0;
    letter-spacing: -0.8px;
}
.customeriq-hero p {
    color: #6B7A90 !important;
    margin: 0.45rem 0 0 0;
    font-size: 0.9rem;
}

/* ---------- Alerts, captions ---------- */
div[data-testid="stAlert"] { border-radius: 12px; border: 1px solid var(--line); }
[data-testid="stMain"] [data-testid="stCaptionContainer"] p {
    color: var(--muted) !important;
    font-size: 0.82rem;
}
code { border-radius: 6px; }

@media (max-width: 800px) {
    .customeriq-hero { padding: 1rem; }
    .customeriq-hero h1 { font-size: 1.65rem; }
}
</style>
""", unsafe_allow_html=True)


# =========================================================
# LOAD FILES
# =========================================================

@st.cache_data
def load_csv(path, modified_time):
    return pd.read_csv(path)


@st.cache_data
def load_json(path, modified_time):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


required_files = [
    DATA_PATH,
    PREDICTIONS_PATH,
    METRICS_PATH,
]

missing_files = [
    str(path)
    for path in required_files
    if not path.exists()
]

if missing_files:
    st.error("Required project files are missing.")
    st.code("\n".join(missing_files))
    st.stop()

snapshots = load_csv(
    str(DATA_PATH), DATA_PATH.stat().st_mtime
)

predictions = load_csv(
    str(PREDICTIONS_PATH),
    PREDICTIONS_PATH.stat().st_mtime,
)

metrics = load_json(
    str(METRICS_PATH), METRICS_PATH.stat().st_mtime
)


# =========================================================
# PREPARE CUSTOMER DATA
# =========================================================

for frame in [snapshots, predictions]:
    frame["CustomerID"] = frame["CustomerID"].astype(str)
    frame["SnapshotDate"] = pd.to_datetime(
        frame["SnapshotDate"]
    ).dt.strftime("%Y-%m-%d")

test_features = snapshots[
    snapshots["Split"] == "test"
].drop(
    columns=["Churn", "Split"],
    errors="ignore",
).copy()

customers = predictions.merge(
    test_features,
    on=["CustomerID", "SnapshotDate"],
    how="left",
    validate="one_to_one",
)

if customers.empty or customers["ChurnRisk"].isna().any():
    st.error(
        "Could not match predictions with customer features. "
        "Run the model-training script again."
    )
    st.stop()

customers["ActualLabel"] = customers["Churn"].astype(int)

customers["RiskPercent"] = customers["ChurnRisk"] * 100


# =========================================================
# LOAD SHAP EXPLANATIONS
# =========================================================

shap_customers = pd.DataFrame()

shap_global = pd.DataFrame()

if SHAP_PATH.exists():
    shap_customers = load_csv(
        str(SHAP_PATH), SHAP_PATH.stat().st_mtime
    )

    shap_customers["CustomerID"] = (
        shap_customers["CustomerID"].astype(str)
    )

    shap_customers["SnapshotDate"] = pd.to_datetime(
        shap_customers["SnapshotDate"]
    ).dt.strftime("%Y-%m-%d")

if SHAP_GLOBAL_PATH.exists():
    shap_global = load_csv(
        str(SHAP_GLOBAL_PATH),
        SHAP_GLOBAL_PATH.stat().st_mtime,
    )


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:
    st.title("📊 CustomerIQ")
    st.caption("CUSTOMER INTELLIGENCE")
    st.divider()

    page = st.radio(
        "Navigation",
        [
            "Overview",
            "Customer Explorer",
            "Model Performance",
            "Feature Insights",
        ],
    )

    st.divider()

    st.subheader("Evaluation Dataset")
    st.write(f"**Snapshot:** {metrics['test_cutoff']}")
    st.write(f"**Customers:** {len(customers):,}")

    st.divider()

    st.subheader("Risk Thresholds")

    high_threshold = st.slider(
        "High-risk threshold (%)",
        min_value=50,
        max_value=95,
        value=70,
        step=5,
    )

    medium_threshold = st.slider(
        "Medium-risk threshold (%)",
        min_value=10,
        max_value=80,
        value=40,
        step=5,
    )

    if medium_threshold >= high_threshold:
        st.warning(
            "The medium-risk threshold must be lower "
            "than the high-risk threshold."
        )
        st.stop()


# =========================================================
# DYNAMIC RISK CATEGORIES
# =========================================================

high_value = high_threshold / 100

medium_value = medium_threshold / 100


def assign_risk(score):
    if score >= high_value:
        return "High"
    elif score >= medium_value:
        return "Medium"
    return "Low"


customers["RiskCategory"] = customers["ChurnRisk"].apply(
    assign_risk
)


# =========================================================
# HEADER
# =========================================================

st.markdown("""
<div class="customeriq-hero">
    <div class="eyebrow">Customer Intelligence Platform</div>
    <h1>CustomerIQ</h1>
    <p>Customer Churn &amp; Retention Intelligence&nbsp; · &nbsp;Retail Analytics&nbsp; · &nbsp;Explainable AI</p>
</div>
""", unsafe_allow_html=True)


# =========================================================
# PAGE 1: OVERVIEW
# =========================================================

if page == "Overview":

    total_customers = len(customers)

    observed_churn = customers["ActualLabel"].mean() * 100

    high_risk_count = (
        customers["RiskCategory"] == "High"
    ).sum()

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Customers Analysed",
        f"{total_customers:,}",
    )

    col2.metric(
        "Observed Churn",
        f"{observed_churn:.1f}%",
    )

    col3.metric(
        "High-Risk Customers",
        f"{high_risk_count:,}",
    )

    col4.metric(
        "Test ROC-AUC",
        f"{metrics['test_roc_auc']:.3f}",
    )

    st.divider()

    left, right = st.columns(2)

    with left:
        st.subheader("Observed Churn")

        churn_data = (
            customers["ActualLabel"]
            .map({0: "Retained", 1: "Churned"})
            .value_counts()
            .rename_axis("Status")
            .reset_index(name="Customers")
        )

        fig = px.pie(
            churn_data,
            names="Status",
            values="Customers",
            hole=0.62,
            color="Status",
            color_discrete_map={
                "Churned": "#E76F51",
                "Retained": "#2A9D8F",
            },
        )

        fig.update_traces(
            textinfo="percent+label",
            textposition="inside",
            textfont=dict(color="#FFFFFF", size=13),
            marker=dict(line=dict(color="#FFFFFF", width=3)),
        )

        style_chart(fig)
        fig.update_layout(
            legend=dict(orientation="h", y=-0.08, x=0.5, xanchor="center"),
        )
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("Estimated Risk Segments")

        risk_data = (
            customers["RiskCategory"]
            .value_counts()
            .reindex(
                ["High", "Medium", "Low"],
                fill_value=0,
            )
            .rename_axis("Risk")
            .reset_index(name="Customers")
        )

        fig = px.bar(
            risk_data,
            x="Risk",
            y="Customers",
            color="Risk",
            text="Customers",
            color_discrete_map={
                "High": "#E76F51",
                "Medium": "#E9C46A",
                "Low": "#2A9D8F",
            },
        )

        fig.update_layout(showlegend=False)

        fig.update_traces(
            textposition="outside",
            textfont=dict(color="#263A56", size=13),
            marker_line_width=0,
        )

        style_chart(fig)
        fig.update_xaxes(showgrid=False)
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    st.subheader("Highest-Risk Customers")

    top_customers = customers.sort_values(
        "ChurnRisk", ascending=False
    ).head(10)

    st.dataframe(
        top_customers[
            [
                "CustomerID",
                "RiskPercent",
                "RiskCategory",
                "TotalSpend",
                "TotalOrders",
                "RecencyDays",
            ]
        ].round(2),
        use_container_width=True,
        hide_index=True,
    )

    st.download_button(
        "Download High-Risk Customers",
        data=top_customers.to_csv(
            index=False
        ).encode("utf-8"),
        file_name="high_risk_customers.csv",
        mime="text/csv",
    )


# =========================================================
# PAGE 2: CUSTOMER EXPLORER
# =========================================================

elif page == "Customer Explorer":

    st.subheader("Customer Explorer")

    col1, col2 = st.columns(2)

    with col1:
        search_id = st.text_input(
            "Search Customer ID",
            placeholder="Enter customer ID",
        )

    with col2:
        minimum_risk = st.slider(
            "Minimum estimated risk (%)",
            min_value=0,
            max_value=100,
            value=0,
            step=5,
        )

    available_categories = st.multiselect(
        "Risk categories",
        ["High", "Medium", "Low"],
        default=["High", "Medium", "Low"],
    )

    view = customers[
        (customers["RiskPercent"] >= minimum_risk)
        & customers["RiskCategory"].isin(
            available_categories
        )
    ].copy()

    if search_id.strip():
        view = view[
            view["CustomerID"].str.contains(
                search_id.strip(),
                regex=False,
                case=False,
            )
        ]

    st.write(f"**Matching customers: {len(view):,}**")

    display_columns = [
        "CustomerID",
        "RiskPercent",
        "RiskCategory",
        "TotalSpend",
        "TotalOrders",
        "TotalItems",
        "UniqueProducts",
        "RecencyDays",
        "CustomerTenureDays",
        "AverageOrderValue",
    ]

    st.dataframe(
        view[display_columns].round(2),
        use_container_width=True,
        hide_index=True,
    )

    st.download_button(
        "Download Filtered Customers",
        data=view.to_csv(
            index=False
        ).encode("utf-8"),
        file_name="filtered_customers.csv",
        mime="text/csv",
    )

    if not view.empty:

        st.divider()

        st.subheader("Customer Profile")

        selected_id = st.selectbox(
            "Select a customer",
            view["CustomerID"].tolist(),
        )

        customer = view[
            view["CustomerID"] == selected_id
        ].iloc[0]

        p1, p2, p3, p4 = st.columns(4)

        p1.metric(
            "Estimated Risk",
            f"{customer['RiskPercent']:.1f}%",
        )

        p2.metric(
            "Total Spend",
            f"{customer['TotalSpend']:,.2f}",
        )

        p3.metric(
            "Orders",
            f"{int(customer['TotalOrders']):,}",
        )

        p4.metric(
            "Days Since Purchase",
            f"{int(customer['RecencyDays'])}",
        )

        behaviour_features = [
            "TotalSpend",
            "TotalOrders",
            "TotalItems",
            "UniqueProducts",
            "RecencyDays",
            "CustomerTenureDays",
            "AverageOrderValue",
        ]

        profile_data = pd.DataFrame({
            "Feature": behaviour_features,
            "Value": [
                customer[feature]
                for feature in behaviour_features
            ],
        })

        fig = px.bar(
            profile_data,
            x="Value",
            y="Feature",
            orientation="h",
            title="Customer Behaviour",
        )

        style_chart(fig)
        fig.update_yaxes(showgrid=False)
        st.plotly_chart(
            fig, use_container_width=True
        )

        st.divider()

        st.subheader("Individual SHAP Explanation")

        match = pd.DataFrame()

        if not shap_customers.empty:
            match = shap_customers[
                (
                    shap_customers["CustomerID"]
                    == str(selected_id)
                )
                & (
                    shap_customers["SnapshotDate"]
                    == str(customer["SnapshotDate"])
                )
            ]

        if not match.empty:

            contribution_row = match.iloc[0]

            feature_columns = [
                col for col in shap_customers.columns
                if col not in ["CustomerID", "SnapshotDate"]
            ]

            explanation = pd.DataFrame({
                "Feature": feature_columns,
                "Contribution": [
                    float(contribution_row[feature])
                    for feature in feature_columns
                ],
            })

            explanation = explanation.sort_values(
                "Contribution"
            )

            explanation["Direction"] = explanation[
                "Contribution"
            ].apply(
                lambda value: (
                    "Toward churn"
                    if value > 0
                    else "Away from churn"
                )
            )

            fig = px.bar(
                explanation,
                x="Contribution",
                y="Feature",
                orientation="h",
                color="Direction",
                color_discrete_map={
                    "Toward churn": "#E76F51",
                    "Away from churn": "#2A9D8F",
                },
                title="Feature Contributions to Model Output",
            )

            style_chart(fig)
            fig.update_yaxes(showgrid=False)
            fig.update_layout(
                legend=dict(orientation="h", y=-0.12, x=0.5, xanchor="center"),
            )
            st.plotly_chart(
                fig, use_container_width=True
            )

        else:
            st.info(
                "No saved individual explanation is available "
                "for this customer. Run src/explain_model.py "
                "to generate explanations."
            )

        st.caption(
            "SHAP contributions explain model behaviour. "
            "They are not causal effects."
        )


# =========================================================
# PAGE 3: MODEL PERFORMANCE
# =========================================================

elif page == "Model Performance":

    st.subheader("Model Performance")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Selected Model",
        metrics["selected_model"],
    )

    c2.metric(
        "ROC-AUC",
        f"{metrics['test_roc_auc']:.4f}",
    )

    c3.metric(
        "Average Precision",
        f"{metrics['test_average_precision']:.4f}",
    )

    c4.metric(
        "Accuracy",
        f"{metrics['test_accuracy']:.1%}",
    )

    actual = customers["ActualLabel"]

    predicted = customers["PredictedChurn"]

    matrix = pd.crosstab(
        actual.map({0: "Retained", 1: "Churned"}),
        predicted.map({
            0: "Predicted Retained",
            1: "Predicted Churned",
        }),
    )

    matrix = matrix.reindex(
        index=["Retained", "Churned"],
        columns=[
            "Predicted Retained",
            "Predicted Churned",
        ],
        fill_value=0,
    )

    fig = px.imshow(
        matrix,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="Blues",
        title="Confusion Matrix",
        labels={
            "x": "Prediction",
            "y": "Actual Outcome",
            "color": "Customers",
        },
    )

    fig.update_traces(textfont=dict(size=18), xgap=4, ygap=4)

    style_chart(fig)
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Evaluation Details")

    st.json(metrics)

    st.download_button(
        "Download Test Predictions",
        data=customers.to_csv(
            index=False
        ).encode("utf-8"),
        file_name="test_predictions.csv",
        mime="text/csv",
    )


# =========================================================
# PAGE 4: FEATURE INSIGHTS
# =========================================================

elif page == "Feature Insights":

    st.subheader("Global Feature Importance")

    if not shap_global.empty:

        importance_column = "MeanAbsoluteSHAP"

        if importance_column in shap_global.columns:

            importance = shap_global.sort_values(
                importance_column,
                ascending=True,
            )

            fig = px.bar(
                importance,
                x=importance_column,
                y="Feature",
                orientation="h",
                title="Mean Absolute SHAP Contribution",
            )

            style_chart(fig)
            fig.update_yaxes(showgrid=False)
            st.plotly_chart(
                fig, use_container_width=True
            )

            st.dataframe(
                shap_global,
                use_container_width=True,
                hide_index=True,
            )

            st.download_button(
                "Download Feature Importance",
                data=shap_global.to_csv(
                    index=False
                ).encode("utf-8"),
                file_name="feature_importance.csv",
                mime="text/csv",
            )

        else:
            st.warning(
                "The SHAP importance file has an unexpected schema."
            )

    else:
        st.info(
            "No global SHAP report found. Run "
            "python src/explain_model.py first."
        )

    st.divider()

    st.subheader("Feature Relationships")

    numeric_columns = customers.select_dtypes(
        include="number"
    ).columns.tolist()

    excluded_columns = [
        "Churn",
        "ActualLabel",
        "PredictedChurn",
        "ChurnRisk",
        "RiskPercent",
    ]

    numeric_columns = [
        column for column in numeric_columns
        if column not in excluded_columns
    ]

    if len(numeric_columns) >= 2:

        x_feature = st.selectbox(
            "Horizontal axis",
            numeric_columns,
            index=0,
        )

        y_feature = st.selectbox(
            "Vertical axis",
            numeric_columns,
            index=min(1, len(numeric_columns) - 1),
        )

        fig = px.scatter(
            customers,
            x=x_feature,
            y=y_feature,
            color="ActualLabel",
            color_continuous_scale=["#2A9D8F", "#E76F51"],
            opacity=0.65,
            hover_data=["CustomerID", "RiskPercent"],
            title=f"{x_feature} vs {y_feature}",
        )

        fig.update_traces(marker=dict(size=8, line=dict(width=0.5, color="#FFFFFF")))

        style_chart(fig)
        st.plotly_chart(
            fig, use_container_width=True
        )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "CustomerIQ | Data-driven customer analytics"
)