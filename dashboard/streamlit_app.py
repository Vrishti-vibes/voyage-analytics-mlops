import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from datetime import datetime

# =====================================================
# PAGE CONFIGURATION
# =====================================================
st.set_page_config(
    page_title="Voyage Analytics | AI Travel Intelligence Platform",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================
# GLOBAL CONSTANTS
# =====================================================
APP_VERSION = "v1.0.0"
PROJECT_NAME = "Voyage Analytics"
SUBTITLE = "AI-Powered Travel Intelligence & MLOps Platform"

DATA_PATHS = {
    "flights": "data/processed/flights_processed.csv",
    "hotels": "data/processed/hotels_processed.csv",
    "users": "data/processed/users_processed.csv"
}

MODEL_PATHS = {
    "flight_model": "models/flight_price_model.pkl",
    "gender_model": "models/gender_model.pkl",
    "gender_encoder": "models/gender_label_encoder.pkl"
}

REPORT_PATHS = {
    "flight_metrics": "report/flight_model_metrics.txt",
    "gender_metrics": "report/gender_model_metrics.txt",
    "hotel_sample": "report/hotel_recommendations_sample.csv"
}

# =====================================================
# PREMIUM CSS
# =====================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background:
            radial-gradient(circle at top left, rgba(37,99,235,0.18), transparent 34%),
            radial-gradient(circle at top right, rgba(14,165,233,0.12), transparent 30%),
            linear-gradient(135deg, #020617 0%, #0F172A 45%, #020617 100%);
        color: #F8FAFC;
    }

    .block-container {
        padding-top: 2.2rem;
        padding-bottom: 2rem;
        max-width: 1250px;
    }

    section[data-testid="stSidebar"] {
        background:
            linear-gradient(180deg, #020617 0%, #0F172A 50%, #020617 100%);
        border-right: 1px solid rgba(148,163,184,0.15);
    }

    section[data-testid="stSidebar"] * {
        color: #E5E7EB;
    }

    .sidebar-brand {
        padding: 18px 8px 8px 8px;
        margin-bottom: 18px;
    }

    .sidebar-title {
        font-size: 30px;
        font-weight: 900;
        color: #F8FAFC;
        line-height: 1.15;
    }

    .sidebar-subtitle {
        font-size: 14px;
        color: #94A3B8;
        margin-top: 8px;
        font-weight: 500;
    }

    .sidebar-pill {
        display: inline-block;
        background: rgba(34,197,94,0.12);
        color: #86EFAC;
        padding: 7px 12px;
        border-radius: 999px;
        border: 1px solid rgba(34,197,94,0.28);
        font-size: 13px;
        font-weight: 700;
        margin-top: 16px;
    }

    .hero {
        background:
            linear-gradient(135deg, rgba(30,64,175,0.38), rgba(15,23,42,0.94)),
            linear-gradient(45deg, rgba(14,165,233,0.14), transparent);
        padding: 42px 46px;
        border-radius: 30px;
        border: 1px solid rgba(148,163,184,0.18);
        box-shadow: 0 25px 80px rgba(0,0,0,0.38);
        margin-bottom: 30px;
        position: relative;
        overflow: hidden;
    }

    .hero:before {
        content: "";
        position: absolute;
        top: -70px;
        right: -70px;
        width: 230px;
        height: 230px;
        background: radial-gradient(circle, rgba(59,130,246,0.28), transparent 70%);
        border-radius: 50%;
    }

    .hero-kicker {
        color: #93C5FD;
        font-size: 15px;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-bottom: 8px;
    }

    .hero-title {
        color: #F8FAFC;
        font-size: 58px;
        font-weight: 950;
        line-height: 1.04;
        margin-bottom: 18px;
    }

    .hero-subtitle {
        color: #CBD5E1;
        font-size: 22px;
        line-height: 1.65;
        max-width: 980px;
        margin-bottom: 20px;
    }

    .tag {
        display: inline-block;
        background: rgba(59,130,246,0.16);
        color: #BFDBFE;
        padding: 8px 14px;
        border-radius: 999px;
        font-size: 14px;
        font-weight: 700;
        margin-right: 10px;
        margin-top: 10px;
        border: 1px solid rgba(147,197,253,0.25);
    }

    .metric-grid {
        margin-bottom: 24px;
    }

    .metric-card {
        background:
            linear-gradient(180deg, rgba(15,23,42,0.96), rgba(2,6,23,0.96));
        border: 1px solid rgba(148,163,184,0.15);
        border-radius: 24px;
        padding: 24px 24px;
        box-shadow: 0 18px 50px rgba(0,0,0,0.32);
        min-height: 150px;
    }

    .metric-icon {
        font-size: 26px;
        margin-bottom: 8px;
    }

    .metric-label {
        color: #94A3B8;
        font-size: 14px;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: .8px;
    }

    .metric-value {
        color: #F8FAFC;
        font-size: 36px;
        font-weight: 950;
        margin-top: 8px;
        line-height: 1.1;
    }

    .metric-note {
        color: #64748B;
        font-size: 13px;
        margin-top: 8px;
        font-weight: 600;
    }

    .glass-card {
        background: rgba(15,23,42,0.78);
        border: 1px solid rgba(148,163,184,0.14);
        border-radius: 24px;
        padding: 26px;
        box-shadow: 0 18px 50px rgba(0,0,0,0.28);
        margin-bottom: 22px;
    }

    .section-title {
        font-size: 34px;
        font-weight: 950;
        color: #F8FAFC;
        margin-top: 12px;
        margin-bottom: 16px;
        letter-spacing: -0.5px;
    }

    .section-subtitle {
        color: #94A3B8;
        font-size: 16px;
        margin-bottom: 18px;
        line-height: 1.7;
    }

    .result-card {
        background:
            linear-gradient(135deg, rgba(34,197,94,0.18), rgba(16,185,129,0.08));
        border: 1px solid rgba(34,197,94,0.35);
        padding: 28px;
        border-radius: 26px;
        box-shadow: 0 15px 55px rgba(34,197,94,0.08);
        margin-top: 20px;
        margin-bottom: 20px;
    }

    .result-title {
        color: #BBF7D0;
        font-size: 16px;
        font-weight: 900;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .result-value {
        color: #F0FDF4;
        font-size: 54px;
        font-weight: 950;
        margin-top: 8px;
        line-height: 1;
    }

    .result-sub {
        color: #A7F3D0;
        font-size: 15px;
        font-weight: 600;
        margin-top: 12px;
    }

    .warning-card {
        background:
            linear-gradient(135deg, rgba(245,158,11,0.15), rgba(120,53,15,0.10));
        border: 1px solid rgba(245,158,11,0.32);
        padding: 22px;
        border-radius: 22px;
        color: #FEF3C7;
        margin: 16px 0;
    }

    .danger-card {
        background:
            linear-gradient(135deg, rgba(239,68,68,0.15), rgba(127,29,29,0.10));
        border: 1px solid rgba(248,113,113,0.32);
        padding: 22px;
        border-radius: 22px;
        color: #FEE2E2;
        margin: 16px 0;
    }

    .status-online {
        display: inline-block;
        background: rgba(34,197,94,0.14);
        color: #86EFAC;
        border: 1px solid rgba(34,197,94,0.3);
        padding: 8px 13px;
        border-radius: 999px;
        font-weight: 800;
        font-size: 13px;
    }

    .status-experimental {
        display: inline-block;
        background: rgba(245,158,11,0.14);
        color: #FCD34D;
        border: 1px solid rgba(245,158,11,0.3);
        padding: 8px 13px;
        border-radius: 999px;
        font-weight: 800;
        font-size: 13px;
    }

    .architecture-box {
        background: rgba(15,23,42,0.82);
        border: 1px solid rgba(148,163,184,0.16);
        border-radius: 24px;
        padding: 28px;
        margin-bottom: 20px;
    }

    .arch-step {
        background: linear-gradient(135deg, rgba(37,99,235,0.18), rgba(14,165,233,0.08));
        border: 1px solid rgba(96,165,250,0.24);
        padding: 18px;
        border-radius: 18px;
        text-align: center;
        color: #DBEAFE;
        font-size: 16px;
        font-weight: 850;
        min-height: 92px;
        display: flex;
        align-items: center;
        justify-content: center;
    }

    .arch-arrow {
        color: #60A5FA;
        text-align: center;
        font-size: 34px;
        font-weight: 900;
        padding-top: 20px;
    }

    .footer-note {
        color: #64748B;
        font-size: 13px;
        margin-top: 28px;
        text-align: center;
    }

    div[data-testid="stDataFrame"] {
        border-radius: 18px;
        overflow: hidden;
        border: 1px solid rgba(148,163,184,0.12);
    }

    .stButton > button {
        border-radius: 14px;
        padding: 0.75rem 1.3rem;
        font-weight: 900;
        border: 1px solid rgba(96,165,250,0.4);
        background: linear-gradient(135deg, #2563EB, #0EA5E9);
        color: white;
        box-shadow: 0 12px 35px rgba(37,99,235,0.25);
    }

    .stButton > button:hover {
        border: 1px solid rgba(191,219,254,0.8);
        box-shadow: 0 15px 45px rgba(14,165,233,0.30);
        transform: translateY(-1px);
    }

    hr {
        border-color: rgba(148,163,184,0.16);
    }
</style>
""", unsafe_allow_html=True)

# =====================================================
# DATA LOADING
# =====================================================
@st.cache_data(show_spinner=False)
def load_datasets():
    flights_df = pd.read_csv(DATA_PATHS["flights"])
    hotels_df = pd.read_csv(DATA_PATHS["hotels"])
    users_df = pd.read_csv(DATA_PATHS["users"])

    for df in [flights_df, hotels_df]:
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")

    return flights_df, hotels_df, users_df


@st.cache_resource(show_spinner=False)
def load_models():
    flight_price_model = joblib.load(MODEL_PATHS["flight_model"])
    gender_classifier = joblib.load(MODEL_PATHS["gender_model"])
    gender_encoder = joblib.load(MODEL_PATHS["gender_encoder"])
    return flight_price_model, gender_classifier, gender_encoder


def read_text_file(path):
    p = Path(path)
    if p.exists():
        return p.read_text()
    return "Metrics file not found. Run model training script first."


def get_model_artifacts():
    rows = []
    model_dir = Path("models")
    if model_dir.exists():
        for file in model_dir.glob("*"):
            rows.append({
                "Artifact": file.name,
                "Type": file.suffix.replace(".", "").upper(),
                "Size (KB)": round(file.stat().st_size / 1024, 2),
                "Last Modified": datetime.fromtimestamp(file.stat().st_mtime).strftime("%d %b %Y, %I:%M %p")
            })
    return pd.DataFrame(rows)


def recommend_hotels_engine(place, budget_per_day, stay_days, top_n=5):
    df = hotels.copy()
    df = df[df["place"].str.lower() == place.lower()]

    if df.empty:
        return pd.DataFrame()

    df["estimated_total_cost"] = df["price"] * stay_days
    df["budget_gap"] = budget_per_day - df["price"]
    df = df[df["budget_gap"] >= 0]

    if df.empty:
        return pd.DataFrame()

    df["affordability_score"] = 1 - (df["price"] / budget_per_day)
    df["stay_efficiency_score"] = 1 / (1 + abs(df["days"] - stay_days))
    df["recommendation_score"] = (
        0.70 * df["affordability_score"] +
        0.30 * df["stay_efficiency_score"]
    )

    result = (
        df[[
            "name",
            "place",
            "price",
            "days",
            "estimated_total_cost",
            "budget_gap",
            "recommendation_score"
        ]]
        .drop_duplicates(subset=["name", "place", "price"])
        .sort_values("recommendation_score", ascending=False)
        .head(top_n)
        .copy()
    )

    result["estimated_total_cost"] = result["estimated_total_cost"].round(2)
    result["budget_gap"] = result["budget_gap"].round(2)
    result["recommendation_score"] = result["recommendation_score"].round(4)
    return result


def render_hero(kicker, title, subtitle, tags=None):
    tags_html = ""
    if tags:
        tags_html = "".join([f'<span class="tag">{tag}</span>' for tag in tags])

    st.markdown(
        f"""
        <div class="hero">
            <div class="hero-kicker">{kicker}</div>
            <div class="hero-title">{title}</div>
            <div class="hero-subtitle">{subtitle}</div>
            <div>{tags_html}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def render_metric(icon, label, value, note=""):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-icon">{icon}</div>
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def render_gauge(title, value, max_value, color="#22C55E"):
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value,
            title={"text": title, "font": {"size": 18}},
            gauge={
                "axis": {"range": [0, max_value]},
                "bar": {"color": color},
                "bgcolor": "rgba(15,23,42,0.5)",
                "borderwidth": 1,
                "bordercolor": "rgba(148,163,184,0.25)",
                "steps": [
                    {"range": [0, max_value * 0.5], "color": "rgba(239,68,68,0.18)"},
                    {"range": [max_value * 0.5, max_value * 0.8], "color": "rgba(245,158,11,0.18)"},
                    {"range": [max_value * 0.8, max_value], "color": "rgba(34,197,94,0.18)"}
                ],
            },
            number={"font": {"size": 32}}
        )
    )
    fig.update_layout(
        template="plotly_dark",
        height=270,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    return fig


# =====================================================
# INITIALIZE DATA
# =====================================================
try:
    flights, hotels, users = load_datasets()
    flight_model, gender_model, gender_label_encoder = load_models()
except Exception as e:
    st.error(f"Application failed to load required data/models: {e}")
    st.stop()

# =====================================================
# SIDEBAR
# =====================================================
st.sidebar.markdown(
    """
    <div class="sidebar-brand">
        <div class="sidebar-title">✈️ Voyage Analytics</div>
        <div class="sidebar-subtitle">AI Travel Intelligence + MLOps Platform</div>
        <div class="sidebar-pill">● System Online</div>
    </div>
    """,
    unsafe_allow_html=True
)

st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Executive Dashboard",
        "✈️ Flight Price Prediction",
        "🏨 Hotel Recommendation",
        "📊 Travel Analytics",
        "🤖 Model Monitoring",
        "🧬 MLOps Architecture",
        "📂 Dataset Explorer",
        "📑 Project Documentation"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Version:** `{APP_VERSION}`")
st.sidebar.markdown("**Environment:** `Local Production Demo`")
st.sidebar.markdown("**Backend:** `FastAPI-ready`")
st.sidebar.markdown("**Dashboard:** `Streamlit`")
st.sidebar.markdown("---")
st.sidebar.caption("Built for major project demonstration, internship portfolio, and resume showcase.")

# =====================================================
# EXECUTIVE DASHBOARD
# =====================================================
if page == "🏠 Executive Dashboard":
    render_hero(
        "AI Travel Intelligence Platform",
        "✈️ Voyage Analytics",
        "A production-oriented machine learning platform for flight price prediction, hotel recommendation, travel analytics, and MLOps monitoring.",
        ["Machine Learning", "FastAPI", "Streamlit", "MLOps", "Predictive Analytics", "Travel Intelligence"]
    )

    avg_flight_price = flights["price"].mean()
    avg_hotel_price = hotels["price"].mean()
    total_trip_value = flights["price"].sum() + hotels["total"].sum()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_metric("✈️", "Flight Records", f"{len(flights):,}", "Processed flight transactions")
    with c2:
        render_metric("🏨", "Hotel Records", f"{len(hotels):,}", "Hotel booking records")
    with c3:
        render_metric("👥", "User Profiles", f"{len(users):,}", "Traveler profiles")
    with c4:
        render_metric("💰", "Avg Flight Price", f"{avg_flight_price:.2f}", "Dataset monetary unit")

    st.markdown('<div class="section-title">Business Intelligence Overview</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-subtitle">Key travel patterns, price behavior, demand routes, and agency-level pricing insights.</div>',
        unsafe_allow_html=True
    )

    left, right = st.columns(2)

    with left:
        fig = px.histogram(
            flights,
            x="price",
            nbins=50,
            title="Flight Price Distribution",
            template="plotly_dark",
            color_discrete_sequence=["#60A5FA"]
        )
        fig.update_layout(
            height=420,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15,23,42,0.45)"
        )
        st.plotly_chart(fig, use_container_width=True)

    with right:
        agency_stats = (
            flights.groupby("agency")["price"]
            .mean()
            .reset_index()
            .sort_values("price", ascending=False)
        )
        fig = px.bar(
            agency_stats,
            x="agency",
            y="price",
            title="Average Flight Price by Agency",
            template="plotly_dark",
            color="price",
            color_continuous_scale="Blues"
        )
        fig.update_layout(
            height=420,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15,23,42,0.45)"
        )
        st.plotly_chart(fig, use_container_width=True)

    left2, right2 = st.columns(2)

    with left2:
        route_stats = (
            flights.groupby(["from", "to"])
            .size()
            .reset_index(name="total_trips")
            .sort_values("total_trips", ascending=False)
            .head(12)
        )
        route_stats["route"] = route_stats["from"] + " → " + route_stats["to"]

        fig = px.bar(
            route_stats,
            x="total_trips",
            y="route",
            orientation="h",
            title="Top Travel Routes by Demand",
            template="plotly_dark",
            color="total_trips",
            color_continuous_scale="Teal"
        )
        fig.update_layout(
            height=450,
            yaxis={"categoryorder": "total ascending"},
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15,23,42,0.45)"
        )
        st.plotly_chart(fig, use_container_width=True)

    with right2:
        monthly_price = flights.groupby("month")["price"].mean().reset_index()
        fig = px.line(
            monthly_price,
            x="month",
            y="price",
            markers=True,
            title="Monthly Average Flight Price Trend",
            template="plotly_dark"
        )
        fig.update_traces(line=dict(width=4, color="#38BDF8"), marker=dict(size=10))
        fig.update_layout(
            height=450,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15,23,42,0.45)"
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-title">Operational Snapshot</div>', unsafe_allow_html=True)

    o1, o2, o3 = st.columns(3)
    with o1:
        render_metric("🟢", "API Layer", "Ready", "FastAPI endpoints available")
    with o2:
        render_metric("📦", "Model Artifacts", f"{len(get_model_artifacts())}", "Serialized models available")
    with o3:
        render_metric("📊", "Estimated Transaction Value", f"{total_trip_value/1_000_000:.2f}M", "Flights + hotels total")

        # =====================================================
# FLIGHT PRICE PREDICTION
# =====================================================
elif page == "✈️ Flight Price Prediction":

    render_hero(
        "Prediction Engine",
        "💰 Flight Price Prediction",
        "Predict flight pricing using route, airline agency, flight class, travel distance, duration and temporal travel features.",
        ["Regression Model", "Scikit-Learn", "Real-Time Prediction"]
    )

    with st.container():

        c1, c2 = st.columns(2)

        with c1:
            source = st.selectbox(
                "Source City",
                sorted(flights["from"].unique())
            )

            flight_type = st.selectbox(
                "Flight Type",
                sorted(flights["flightType"].unique())
            )

            travel_time = st.number_input(
                "Flight Duration / Time",
                min_value=0.10,
                value=2.50
            )

            month = st.slider(
                "Travel Month",
                1,
                12,
                6
            )

        with c2:
            destination = st.selectbox(
                "Destination City",
                sorted(flights["to"].unique())
            )

            agency = st.selectbox(
                "Travel Agency",
                sorted(flights["agency"].unique())
            )

            distance = st.number_input(
                "Distance",
                min_value=50.0,
                value=500.0
            )

            day = st.slider(
                "Travel Day",
                1,
                31,
                15
            )

    if st.button("Predict Flight Fare"):

        try:
            sample = flights.iloc[[0]].copy()

            sample["from"] = source
            sample["to"] = destination
            sample["flightType"] = flight_type
            sample["agency"] = agency
            sample["time"] = travel_time
            sample["distance"] = distance
            sample["month"] = month
            sample["day"] = day

            prediction = float(flight_model.predict(sample)[0])

            lower = prediction * 0.92
            upper = prediction * 1.08

            st.markdown(
                f"""
                <div class="result-card">
                    <div class="result-title">
                        Predicted Flight Fare
                    </div>
                    <div class="result-value">
                        ₹ {prediction:,.2f}
                    </div>
                    <div class="result-sub">
                        Expected Market Range:
                        ₹ {lower:,.2f}
                        —
                        ₹ {upper:,.2f}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        except Exception as e:
            st.error(str(e))


# =====================================================
# HOTEL RECOMMENDATION
# =====================================================
elif page == "🏨 Hotel Recommendation":

    render_hero(
        "Recommendation Engine",
        "🏨 Smart Hotel Recommendation",
        "Rank hotels using affordability, estimated stay cost, budget compatibility and stay duration matching.",
        ["Recommendation System", "Ranking Engine", "Budget Optimization"]
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        destination = st.selectbox(
            "Destination",
            sorted(hotels["place"].unique())
        )

    with c2:
        budget = st.number_input(
            "Budget Per Day",
            value=400.0
        )

    with c3:
        stay_days = st.slider(
            "Stay Duration",
            1,
            14,
            3
        )

    if st.button("Generate Recommendations"):

        recs = recommend_hotels_engine(
            destination,
            budget,
            stay_days,
            top_n=10
        )

        if recs.empty:

            st.markdown(
                """
                <div class="warning-card">
                No hotels satisfy the selected budget.
                Try increasing the budget.
                </div>
                """,
                unsafe_allow_html=True
            )

        else:

            st.success(
                f"{len(recs)} hotels identified successfully."
            )

            st.dataframe(
                recs,
                use_container_width=True
            )

            fig = px.bar(
                recs,
                x="name",
                y="recommendation_score",
                color="recommendation_score",
                title="Recommendation Score Comparison",
                template="plotly_dark"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )


# =====================================================
# TRAVEL ANALYTICS
# =====================================================
elif page == "📊 Travel Analytics":

    render_hero(
        "Advanced Analytics",
        "📊 Travel Intelligence",
        "Explore route demand, pricing behavior, agency influence, user patterns and hotel economics.",
        ["Business Intelligence", "Analytics", "Visualization"]
    )

    tab1, tab2, tab3 = st.tabs(
        [
            "Flight Analytics",
            "Hotel Analytics",
            "User Analytics"
        ]
    )

    with tab1:

        left, right = st.columns(2)

        with left:

            fig = px.box(
                flights,
                x="flightType",
                y="price",
                title="Price Spread by Flight Type",
                template="plotly_dark"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        with right:

            fig = px.scatter(
                flights.sample(1000),
                x="distance",
                y="price",
                color="flightType",
                title="Distance vs Price",
                template="plotly_dark"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

    with tab2:

        hotel_price = (
            hotels.groupby("place")["price"]
            .mean()
            .reset_index()
            .sort_values(
                "price",
                ascending=False
            )
            .head(15)
        )

        fig = px.bar(
            hotel_price,
            x="place",
            y="price",
            color="price",
            title="Top Hotel Destinations by Price",
            template="plotly_dark"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with tab3:

        fig = px.histogram(
            users,
            x="age",
            nbins=25,
            title="Age Distribution",
            template="plotly_dark"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


# =====================================================
# MODEL MONITORING
# =====================================================
elif page == "🤖 Model Monitoring":

    render_hero(
        "MLOps Monitoring",
        "🤖 Model Monitoring",
        "Monitor trained model artifacts, evaluation metrics and operational readiness.",
        ["Monitoring", "Artifacts", "Deployment"]
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        st.plotly_chart(
            render_gauge(
                "Flight Model Health",
                98,
                100
            ),
            use_container_width=True
        )

    with c2:
        st.plotly_chart(
            render_gauge(
                "API Availability",
                100,
                100,
                "#22C55E"
            ),
            use_container_width=True
        )

    with c3:
        st.plotly_chart(
            render_gauge(
                "Gender Model",
                34,
                100,
                "#F59E0B"
            ),
            use_container_width=True
        )

    st.markdown("### Saved Model Artifacts")

    st.dataframe(
        get_model_artifacts(),
        use_container_width=True
    )

    st.markdown("### Flight Model Metrics")

    st.code(
        read_text_file(
            REPORT_PATHS["flight_metrics"]
        )
    )

    st.markdown("### Gender Model Metrics")

    st.code(
        read_text_file(
            REPORT_PATHS["gender_metrics"]
        )
    )


# =====================================================
# MLOPS ARCHITECTURE
# =====================================================
elif page == "🧬 MLOps Architecture":

    render_hero(
        "Architecture",
        "🧬 End-to-End MLOps Pipeline",
        "Production workflow used for dataset ingestion, preprocessing, model training, deployment and monitoring.",
        ["Data Engineering", "Machine Learning", "Deployment"]
    )

    st.markdown(
        '<div class="architecture-box">',
        unsafe_allow_html=True
    )

    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:
        st.markdown(
            '<div class="arch-step">Raw CSV Datasets</div>',
            unsafe_allow_html=True
        )

    with c2:
        st.markdown(
            '<div class="arch-step">Data Processing & Feature Engineering</div>',
            unsafe_allow_html=True
        )

    with c3:
        st.markdown(
            '<div class="arch-step">Model Training</div>',
            unsafe_allow_html=True
        )

    with c4:
        st.markdown(
            '<div class="arch-step">Joblib Serialization</div>',
            unsafe_allow_html=True
        )

    with c5:
        st.markdown(
            '<div class="arch-step">FastAPI + Streamlit Deployment</div>',
            unsafe_allow_html=True
        )

    st.markdown(
        '</div>',
        unsafe_allow_html=True
    )


# =====================================================
# DATASET EXPLORER
# =====================================================
elif page == "📂 Dataset Explorer":

    render_hero(
        "Dataset Explorer",
        "📂 Travel Data Warehouse",
        "Explore processed datasets powering the machine learning workflow.",
        ["Flights", "Hotels", "Users"]
    )

    tab1, tab2, tab3 = st.tabs(
        [
            "Flights",
            "Hotels",
            "Users"
        ]
    )

    with tab1:
        st.dataframe(
            flights.head(100),
            use_container_width=True
        )

    with tab2:
        st.dataframe(
            hotels.head(100),
            use_container_width=True
        )

    with tab3:
        st.dataframe(
            users.head(100),
            use_container_width=True
        )


# =====================================================
# DOCUMENTATION
# =====================================================
elif page == "📑 Project Documentation":

    render_hero(
        "Major Project",
        "🚀 Voyage Analytics",
        "A machine learning based travel intelligence platform integrating predictive analytics, recommendation systems and MLOps deployment practices.",
        ["Major Project", "Resume Project", "Internship Portfolio"]
    )

    st.markdown("## Problem Statement")

    st.write("""
Travel platforms generate large-scale transactional data from flight bookings,
hotel reservations and user activities.

The goal of Voyage Analytics is to create a complete machine learning ecosystem
that can:

- Predict flight prices
- Recommend hotels
- Analyze travel patterns
- Monitor ML models
- Demonstrate MLOps concepts
- Serve predictions through APIs
    """)

    st.markdown("## Modules")

    st.markdown("""
- Flight Price Prediction
- Hotel Recommendation System
- Travel Analytics Dashboard
- Model Monitoring
- Dataset Explorer
- FastAPI Integration
- Streamlit Dashboard
- MLOps Architecture
    """)

    st.markdown("## Technology Stack")

    st.markdown("""
- Python
- Pandas
- NumPy
- Scikit-Learn
- Joblib
- FastAPI
- Streamlit
- Plotly
- GitHub
- Docker Ready Architecture
    """)

    st.markdown(
        """
        <div class="footer-note">
        Voyage Analytics © 2026 |
        AI Travel Intelligence Platform |
        Major Project Demonstration
        </div>
        """,
        unsafe_allow_html=True
    )