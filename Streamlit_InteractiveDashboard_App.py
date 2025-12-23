import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# -------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------
st.set_page_config(
    page_title="Medical Cost Explorer",
    layout="wide"
)

st.title("🏥 Medical Cost Explorer – Who Costs Us Money and Why?")
st.caption("A business-friendly dashboard to understand spending, customer risk, pricing fairness, and prevention opportunities.")

# -------------------------------------------------------------------
# LOAD DATA
# -------------------------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("medical_insurance_cleaned.csv")

df = load_data()

# region reconstruction (if encoded)
region_cols = [c for c in df.columns if c.startswith("region_")]
df["region_original"] = df[region_cols].idxmax(axis=1).str.replace("region_", "")

# age grouping
df["age_group"] = pd.cut(
    df["age"],
    bins=[18, 35, 50, 65, 100],
    labels=["18–35", "36–50", "51–65", "65+"]
)

# -------------------------------------------------------------------
# SIDEBAR FILTERS
# -------------------------------------------------------------------
st.sidebar.header("🎛 Filters")

# Age
age_range = st.sidebar.slider("Age Range", 18, 100, (30, 65))
df_filtered = df[(df.age >= age_range[0]) & (df.age <= age_range[1])]

# Smoker
smoker_mult = st.sidebar.multiselect(
    "Smoking Behaviour",
    df_filtered["smoker"].unique(),
    df_filtered["smoker"].unique()
)
df_filtered = df_filtered[df_filtered["smoker"].isin(smoker_mult)]

# Region
region_mult = st.sidebar.multiselect(
    "Region",
    df_filtered["region_original"].unique(),
    df_filtered["region_original"].unique()
)
df_filtered = df_filtered[df_filtered["region_original"].isin(region_mult)]

# BMI
bmi_range = st.sidebar.slider(
    "BMI Range",
    float(df_filtered["bmi"].min()),
    float(df_filtered["bmi"].max()),
    (18.0, 35.0)
)
df_filtered = df_filtered[(df_filtered.bmi >= bmi_range[0]) & (df_filtered.bmi <= bmi_range[1])]

# Chronic Conditions
chronic_range = st.sidebar.slider(
    "Chronic Condition Count",
    int(df_filtered["chronic_count"].min()),
    int(df_filtered["chronic_count"].max()),
    (0, 3)
)
df_filtered = df_filtered[
    (df_filtered.chronic_count >= chronic_range[0]) &
    (df_filtered.chronic_count <= chronic_range[1])
]

# Sidebar summary
st.sidebar.markdown("### 📌 Current Selection")
st.sidebar.write(f"Members in view: **{len(df_filtered):,}**")

# -------------------------------------------------------------------
# GLOBAL KPIs
# -------------------------------------------------------------------
avg_cost = df_filtered["annual_medical_cost"].mean()
risk80 = df_filtered["annual_medical_cost"].quantile(0.80)
pct_top20 = (df_filtered["annual_medical_cost"] > risk80).mean() * 100
variability = df_filtered["annual_medical_cost"].std() / avg_cost

k1, k2, k3 = st.columns(3)
k1.metric("💰 Avg Yearly Spend", f"${avg_cost:,.0f}")
k2.metric("🔥 % Driving Highest Spend", f"{pct_top20:.1f}%")
k3.metric("⚖️ Cost Stability (std/mean)", f"{variability:.2f}")

st.markdown("---")

# -------------------------------------------------------------------
# TABS
# -------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Overview",
    "🚬 Behaviour & Lifestyle",
    "👵 Age & Health",
    "🗺 Geographic Fairness",
    "🔮 Predictive Tool"
])

# -------------------------------------------------------------------
# TAB 1 — OVERVIEW
# -------------------------------------------------------------------
with tab1:
    st.header("📊 Overview – Where Is Our Money Going?")

    pareto_sorted = df_filtered["annual_medical_cost"].sort_values(ascending=False).reset_index(drop=True)
    cum_pct = pareto_sorted.cumsum() / pareto_sorted.sum()
    x_vals = np.arange(1, len(cum_pct)+1) / len(cum_pct)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_vals, y=cum_pct, mode="lines"))
    fig.add_hline(y=0.80, line_dash="dot", annotation_text="80% of Spend")
    fig.add_vline(x=0.20, line_dash="dot", annotation_text="Top 20% Customers")
    fig.update_layout(
        title="A Few Customers Drive Most Spending (Pareto)",
        xaxis_title="% of Customers",
        yaxis_title="% of Total Cost"
    )
    st.plotly_chart(fig, use_container_width=True)


# -------------------------------------------------------------------
# TAB 2 — BEHAVIOUR & LIFESTYLE
# -------------------------------------------------------------------
with tab2:
    st.header("🚬 Behaviour – Do Smokers Cost Us More?")
    fig = px.violin(
        df_filtered,
        x="smoker",
        y="annual_medical_cost",
        color="smoker",
        box=True,
        points=False,
        title="Cost Differences by Smoking Habit"
    )
    st.plotly_chart(fig, use_container_width=True)



# -------------------------------------------------------------------
# TAB 3 — AGE & HEALTH
# -------------------------------------------------------------------
with tab3:
    st.header("👵 Age – How Much More Do Older People Cost?")
    age_stats = df_filtered.groupby("age")["annual_medical_cost"].agg(["mean", "std"]).reset_index()
    age_stats["upper"] = age_stats["mean"] + age_stats["std"]
    age_stats["lower"] = age_stats["mean"] - age_stats["std"]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=age_stats["age"], y=age_stats["upper"], line=dict(width=0)))
    fig.add_trace(go.Scatter(x=age_stats["age"], y=age_stats["lower"], fill="tonexty", name="Volatility Band"))
    fig.add_trace(go.Scatter(x=age_stats["age"], y=age_stats["mean"], name="Average Cost", line_color="black"))
    fig.update_layout(
        title="Costs Increase With Age",
        xaxis_title="Age",
        yaxis_title="Annual Cost"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.write("---")
    st.subheader("💉 Chronic Conditions – Who Needs Support?")
    heat = df_filtered.groupby(["age_group", "chronic_count"])["annual_medical_cost"].mean().reset_index()
    fig = px.density_heatmap(
        heat,
        x="age_group",
        y="chronic_count",
        z="annual_medical_cost",
        color_continuous_scale="Reds",
        title="Chronic Conditions Raise Costs — Especially at Older Ages"
    )
    st.plotly_chart(fig, use_container_width=True)

# -------------------------------------------------------------------
# TAB 4 — GEOGRAPHY
# -------------------------------------------------------------------
with tab4:
    st.header("🗺 Region – Do Costs Really Differ by Location?")
    region_stats = df_filtered.groupby("region_original")["annual_medical_cost"].agg(["mean", "std"]).reset_index()

    fig = px.scatter(
        region_stats,
        x="std",
        y="mean",
        text="region_original",
        size="mean",
        size_max=50,
        title="Average Cost vs Variation by Region"
    )
    fig.update_traces(textposition="top center")
    fig.update_layout(
        xaxis_title="Cost Variability (Std Dev)",
        yaxis_title="Average Cost"
    )
    st.plotly_chart(fig, use_container_width=True)


# -------------------------------------------------------------------
# TAB 5 — PREDICTIVE TOOL (REAL COEFFICIENTS — OPTION A)
# -------------------------------------------------------------------
with tab5:
    st.header("🔮 Predict a Customer’s Annual Medical Cost")
    st.caption("Adjust customer characteristics to simulate expected spending using your real regression model.")

    # ========================
    # User Inputs
    # ========================
    colA, colB, colC = st.columns(3)
    age = colA.slider("Age", 18, 100, 45)
    bmi = colB.slider("BMI", 12, 50, 27)
    chronic = colC.slider("Chronic Conditions", 0, 6, 1)

    colD, colE, colF = st.columns(3)
    visits = colD.slider("Visits Last Year", 0, 20, 5)
    hosp = colE.slider("Hospitalizations (3 yrs)", 0, 5, 0)
    days = colF.slider("Days Hospitalized (3 yrs)", 0, 40, 2)

    colG, colH = st.columns(2)
    smoker_input = colG.selectbox("Smoking Status", ["Never", "Former", "Current"])
    risk_score = colH.slider("Risk Score (1=Low, 10=High)", 1, 10, 4)

    # ========================
    # Dummy variables
    # ========================
    smoker_current = 1 if smoker_input == "Current" else 0
    smoker_former = 1 if smoker_input == "Former" else 0

    # ========================
    # REAL MODEL COEFFICIENTS
    # ========================
    b0 = 6.885904
    coef = {
        "risk_score": 0.757449,
        "smoker_Current": 0.134983,
        "smoker_Former": 0.106674,
        "hospitalizations": 0.102814,
        "chronic": 0.095844,
        "days_hosp": 0.055368,
        "visits": 0.006889,
        "bmi": 0.004990,
        "age": -0.001841
    }

    # ========================
    # Log prediction
    # ========================
    pred_log = (
        b0
        + coef["risk_score"] * risk_score
        + coef["smoker_Current"] * smoker_current
        + coef["smoker_Former"] * smoker_former
        + coef["hospitalizations"] * hosp
        + coef["chronic"] * chronic
        + coef["days_hosp"] * days
        + coef["visits"] * visits
        + coef["bmi"] * bmi
        + coef["age"] * age
    )

    pred_cost = np.expm1(pred_log)

    st.metric("💵 Predicted Yearly Cost", f"${pred_cost:,.0f}")

    # ========================
    # Risk Classification
    # ========================
    if pred_cost >= df["annual_medical_cost"].quantile(0.80):
        st.error("Risk Level: HIGH-COST MEMBER")
    elif pred_cost >= df["annual_medical_cost"].quantile(0.50):
        st.warning("Risk Level: MEDIUM-COST MEMBER")
    else:
        st.success("Risk Level: LOW-COST MEMBER")

    # ========================
    # Driver Bar Chart
    # ========================
    driver_df = pd.DataFrame({
        "Driver": [
            "Risk Score",
            "Smoking",
            "Hospitalizations",
            "Chronic Conditions",
            "Days Hospitalized",
            "BMI",
            "Visits",
            "Age"
        ],
        "Impact": [
            coef["risk_score"] * risk_score,
            coef["smoker_Current"] * smoker_current + coef["smoker_Former"] * smoker_former,
            coef["hospitalizations"] * hosp,
            coef["chronic"] * chronic,
            coef["days_hosp"] * days,
            coef["bmi"] * bmi,
            coef["visits"] * visits,
            coef["age"] * age
        ]
    })

    fig_bar = px.bar(
        driver_df.sort_values("Impact", ascending=False),
        x="Impact", y="Driver",
        orientation="h",
        title="Which Factors Drive This Prediction?"
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    st.info("""
    How to use this:
    - Adjust sliders to simulate customer profiles
    - Compare lifestyle vs health impact
    - Use to support pricing and prevention strategies
    """)
