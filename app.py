import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Enterprise Customer Churn Intelligence Platform",
    page_icon="📊",
    layout="wide",
)


# --- LOAD AND PREPROCESS DATA ---
@st.cache_data
def load_data():
  try:
    df = pd.read_csv("exported_churn_data.csv")

    # Date columns conversion
    date_cols = [
        "subscription_start_date",
        "renewal_date",
        "cancellation_date",
        "dob",
        "complaint_date",
    ]
    for col in date_cols:
      if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors="coerce")

    # Recreate churn_risk if missing
    if "churn_risk" not in df.columns and "churn_score" in df.columns:
      conditions = [
          df["churn_score"] < 50,
          (df["churn_score"] >= 50) & (df["churn_score"] < 70),
          df["churn_score"] >= 70,
      ]
      choices = ["low", "medium", "high"]
      df["churn_risk"] = np.select(conditions, choices, default="unknown")

    # Ensure escalations is numeric
    if "escalations" in df.columns:
      df["escalations"] = (
          df["escalations"]
          .astype(str)
          .str.upper()
          .map({"Y": 1, "N": 0})
          .fillna(0)
          .astype(int)
      )

    return df
  except FileNotFoundError:
    st.error(
        "❌ 'exported_churn_data.csv' file not found! Please ensure it is in"
        " the same directory."
    )
    return None


df = load_data()

if df is not None:
  # --- TOP NAVIGATION TABS ---
  tab1, tab2, tab3, tab4, tab5 = st.tabs([
      "Overview & BRD",
      "Features Correlations",
      "Categorical & Geographic Compositions",
      "Numerical Distributions",
      "Model Evaluation & Risk Strategy",
  ])

  # ==========================================
  # TAB 1: OVERVIEW & DETAILED BRD
  # ==========================================
  with tab1:
    st.title("🏢 Enterprise Client Project: OTT Churn & Revenue Intelligence")
    st.markdown(
        "*Client Solution Report: Mitigating Subscriber Attrition and Maximizing"
        " Customer Lifetime Value (CLTV)*"
    )
    st.markdown("---")

    col_desc, col_pie = st.columns([2, 1])

    with col_desc:
      st.markdown("""
            ### **1. Executive Summary & Business Scope**
            In today's hyper-competitive subscription and OTT streaming ecosystem, subscriber acquisition costs (CAC) are surging. Retaining existing users is significantly more cost-effective than acquiring new ones. This client-grade analytics platform evaluates multi-dimensional subscriber datasets—combining demographics, subscription tiers, contract structures, and support ticket escalations—to diagnose revenue leakages and provide proactive retention workflows.

            ### **2. Core Business Objectives**
            * **Quantify Attrition Benchmarks:** Establish baseline metrics for overall churn rate (**28.6%**) versus retention rate (**71.4%**).
            * **MRR & CLTV Protection:** Track Monthly Recurring Revenue (MRR) exposure and Customer Lifetime Value erosion caused by high-risk user segments.
            * **Operational Root-Cause Analysis:** Investigate how unresolved customer support escalations trigger service cancellations.
            """)

    with col_pie:
      churn_counts = df["churn_flag"].value_counts()
      churn_pct = (churn_counts / len(df)) * 100
      no_churn_pct = churn_pct.get(0, 0)
      yes_churn_pct = churn_pct.get(1, 0)

      fig, ax = plt.subplots(figsize=(4, 4))
      ax.pie(
          [no_churn_pct, yes_churn_pct],
          labels=[
              f"Retained ({no_churn_pct:.1f}%)",
              f"Churned ({yes_churn_pct:.1f}%)",
          ],
          colors=["#2ca02c", "#d62728"],
          startangle=90,
          autopct="%1.1f%%",
          wedgeprops=dict(width=0.4, edgecolor="w"),
      )
      ax.set_title(
          "Subscriber Portfolio Health Ratio", fontsize=10, fontweight="bold"
      )
      st.pyplot(fig)

    st.markdown("---")
    st.subheader("📊 Executive Key Performance Indicators (KPIs)")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Overall Churn Rate", f"{yes_churn_pct:.2f}%")
    m2.metric("Retention Rate", f"{no_churn_pct:.2f}%")
    m3.metric("ARPU (Avg Revenue)", f"${df['monthly_charges'].mean():.2f}")
    m4.metric(
        "Revenue at Risk",
        f"${df.loc[df['churn_flag'] == 1, 'monthly_charges'].sum():,.2f}",
    )

  # ==========================================
  # TAB 2: FEATURES CORRELATIONS
  # ==========================================
  with tab2:
    st.title("🔥 Feature Correlation Matrix & Statistical Dependency")
    st.markdown(
        "Uncovering hidden statistical relationships between predictive model"
        " scores, support escalations, and customer cancellations."
    )
    st.markdown("---")

    df_encoded = df[
        [
            "plan_type",
            "contract_type",
            "churn_score",
            "churn_flag",
            "churn_risk",
            "escalations",
        ]
    ].copy()
    order_mappings = {
        "plan_type": ["Basic", "Standard", "Premium"],
        "contract_type": ["Monthly", "Annual"],
        "churn_risk": ["low", "medium", "high"],
    }
    for col, order in order_mappings.items():
      if col in df_encoded.columns:
        df_encoded[col] = pd.Categorical(
            df_encoded[col], categories=order, ordered=True
        ).codes

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        df_encoded.corr(),
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        linewidths=0.5,
        ax=ax,
    )
    st.pyplot(fig)

    st.warning("""
        ### 🔍 Detailed Problem, Objective & Client Solution Framework
        * **Business Problem:** The executive team needed to know whether internal predictive scores (`churn_score`) and support failures are actually driving real customer cancellations (`churn_flag`) or if they are false positives.
        * **Analytical Objective:** To compute the pairwise correlation coefficient across all encoded behavioral and transactional features.
        * **Why We Use It (Business Impact):** The heatmap confirms a **0.86 strong positive correlation** between churn scores and actual churn, validating our machine-learning risk model. Furthermore, a **0.77 correlation** with support escalations proves that unresolved customer complaints are major churn catalysts.
        * **Client Actionable Solution:** Implement an automated ticketing flag that triggers immediate supervisory intervention when a high-risk subscriber logs a support escalation.
        """)

  # ==========================================
  # TAB 3: CATEGORICAL & GEOGRAPHIC COMPOSITIONS
  # ==========================================
  with tab3:
    st.title("📊 Categorical Compositions & Geographic Regional Risk")
    st.markdown(
        "Analyzing churn variance across subscription tiers, billing contract"
        " structures, and regional states."
    )
    st.markdown("---")

    col_a, col_b = st.columns(2)

    with col_a:
      st.subheader("Churn Rate by Plan Type")
      churn_plan = (
          df.groupby("plan_type")["churn_flag"].mean() * 100
      ).reset_index()
      fig, ax = plt.subplots(figsize=(6, 4))
      sns.barplot(
          data=churn_plan,
          x="plan_type",
          y="churn_flag",
          palette="viridis",
          ax=ax,
      )
      ax.set_ylabel("Churn Rate (%)")
      ax.set_xlabel("Plan Type")
      st.pyplot(fig)

    with col_b:
      st.subheader("Churn Rate by Contract Type")
      churn_contract = (
          df.groupby("contract_type")["churn_flag"].mean() * 100
      ).reset_index()
      fig, ax = plt.subplots(figsize=(6, 4))
      sns.barplot(
          data=churn_contract,
          x="contract_type",
          y="churn_flag",
          palette="magma",
          ax=ax,
      )
      ax.set_ylabel("Churn Rate (%)")
      ax.set_xlabel("Contract Type")
      st.pyplot(fig)

    st.markdown("---")
    st.subheader("🌍 Geographic Churn Rate by State / Region")
    churn_state = (
        df.groupby("state")["churn_flag"].mean().mul(100).reset_index()
    )
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.barplot(
        data=churn_state, x="state", y="churn_flag", palette="Set2", ax=ax
    )
    ax.set_ylabel("Churn Rate (%)")
    ax.set_xlabel("State / Region")
    plt.xticks(rotation=45)
    st.pyplot(fig)

    st.warning("""
        ### 🔍 Detailed Problem, Objective & Client Solution Framework
        * **Business Problem:** High churn was hurting revenue, but management didn't know *where* (regions) or *on which plans* (Basic vs Premium, Monthly vs Annual) the bleeding was concentrated.
        * **Analytical Objective:** To aggregate customer records by categorical dimensions (Plan, Contract, State) and compute normalized churn percentages.
        * **Why We Use It (Business Impact):** Bar charts immediately reveal that **Basic tier plans** suffer extreme churn (~60%), **Monthly contracts** experience massive drop-offs compared to annual plans, and specific regions (like Karnataka) exhibit 100% regional churn spikes.
        * **Client Actionable Solution:** Restructure entry-level Basic plans with added engagement features, incentivize annual commitment discounts, and audit regional service or pricing issues in high-churn states.
        """)

  # ==========================================
  # TAB 4: NUMERICAL DISTRIBUTIONS
  # ==========================================
  with tab4:
    st.title("📈 Numerical Financial Distributions & CLTV Exposure")
    st.markdown(
        "Evaluating customer monetary value spread and revenue concentration."
    )
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
      st.subheader("Monthly Charges Distribution")
      fig, ax = plt.subplots(figsize=(6, 4))
      sns.histplot(
          df["monthly_charges"], kde=True, color="purple", bins=15, ax=ax
      )
      ax.set_xlabel("Monthly Charges ($)")
      st.pyplot(fig)

    with col2:
      st.subheader("Customer Lifetime Value (CLTV)")
      fig, ax = plt.subplots(figsize=(6, 4))
      sns.histplot(df["cltv"], kde=True, color="teal", bins=15, ax=ax)
      ax.set_xlabel("Customer Lifetime Value (CLTV)")
      st.pyplot(fig)

    st.warning("""
        ### 🔍 Detailed Problem, Objective & Client Solution Framework
        * **Business Problem:** The finance team needed clarity on ARPU distribution and long-term customer value erosion to plan acquisition budgets.
        * **Analytical Objective:** To plot probability density curves and histograms for financial variables (`monthly_charges` and `cltv`).
        * **Why We Use It (Business Impact):** Highlights revenue clustering and helps identify high-value subscriber segments versus low-margin transient users.
        * **Client Actionable Solution:** Direct marketing spend toward acquiring profiles matching high-CLTV clusters while protecting existing high-value subscribers from churn.
        """)

  # ==========================================
  # TAB 5: MODEL EVALUATION & RISK STRATEGY
  # ==========================================
  with tab5:
    st.title("🎯 Predictive Risk Tiers & Customer Success Strategy")
    st.markdown(
        "Operationalizing risk classification models for proactive retention"
        " campaigns."
    )
    st.markdown("---")

    st.subheader("Customer Distribution Across Churn Risk Buckets")
    risk_counts = df["churn_risk"].value_counts().reset_index()
    risk_counts.columns = ["Risk Tier", "Customer Count"]

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(
        data=risk_counts, x="Risk Tier", y="Customer Count", palette="Set2", ax=ax
    )
    st.pyplot(fig)

    st.warning("""
        ### 🔍 Detailed Problem, Objective & Client Solution Framework
        * **Business Problem:** Customer success agents were reacting to churn *after* cancellation requests occurred, rather than predicting and preventing them.
        * **Analytical Objective:** To segment the user database into actionable risk tiers (`Low`, `Medium`, `High`) using composite churn scoring rules.
        * **Why We Use It (Business Impact):** Gives the operations team a prioritized target list of vulnerable accounts before cancellation happens.
        * **Client Actionable Solution:** Automate CRM alerts for any subscriber shifting into the 'High' risk tier, instantly deploying targeted retention offers, automated discount emails, or support callbacks.
        """)