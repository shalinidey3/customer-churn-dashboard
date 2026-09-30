import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Customer Churn Analysis",
    page_icon="📊",
    layout="wide"
)

# -----------------------------
# Load Customer Churn Dataset
# -----------------------------
DATA_URL = "https://raw.githubusercontent.com/shalinidey3/Customer-Churn-Analysis/main/customer_churn_dataset-training-master.csv"

train = pd.read_csv(DATA_URL)

# Remove missing values
train_clean = train.dropna().copy()

# -----------------------------
# Dashboard Title
# -----------------------------
st.title("📊 Customer Churn Analysis Dashboard")

st.write(
    "Interactive dashboard for exploring customer churn patterns "
    "using Python, Pandas, and Matplotlib."
)

st.divider()

# -----------------------------
# Key Metrics
# -----------------------------
churn_rate = train_clean["Churn"].mean() * 100
non_churn_rate = 100 - churn_rate

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Total Customers",
        f"{len(train_clean):,}"
    )

with col2:
    st.metric(
        "Churn Rate",
        f"{churn_rate:.2f}%"
    )

with col3:
    st.metric(
        "Non-Churn Rate",
        f"{non_churn_rate:.2f}%"
    )

st.divider()

# -----------------------------
# Subscription Type Analysis
# -----------------------------
st.subheader("Churn Rate by Subscription Type")

subscription_churn = (
    train_clean.groupby("Subscription Type")["Churn"]
    .mean() * 100
)

fig, ax = plt.subplots(figsize=(8, 4))

subscription_churn.sort_values().plot(
    kind="bar",
    ax=ax
)

ax.set_xlabel("Subscription Type")
ax.set_ylabel("Churn Rate (%)")
ax.set_ylim(0, 100)
ax.set_title("Churn Rate by Subscription Type")

plt.xticks(rotation=0)

st.pyplot(fig)

st.dataframe(
    subscription_churn.round(2).reset_index(),
    use_container_width=True
)

# -----------------------------
# Contract Length Analysis
# -----------------------------
st.subheader("Churn Rate by Contract Length")

contract_churn = (
    train_clean.groupby("Contract Length")["Churn"]
    .mean() * 100
)

fig, ax = plt.subplots(figsize=(8, 4))

contract_churn.sort_values().plot(
    kind="bar",
    ax=ax
)

ax.set_xlabel("Contract Length")
ax.set_ylabel("Churn Rate (%)")
ax.set_ylim(0, 100)
ax.set_title("Churn Rate by Contract Length")

plt.xticks(rotation=0)

st.pyplot(fig)

st.dataframe(
    contract_churn.round(2).reset_index(),
    use_container_width=True
)

# -----------------------------
# Support Calls Analysis
# -----------------------------
st.subheader("Churn Rate by Support Calls")

support_churn = (
    train_clean.groupby("Support Calls")["Churn"]
    .mean() * 100
)

fig, ax = plt.subplots(figsize=(9, 4))

support_churn.plot(
    kind="line",
    marker="o",
    ax=ax
)

ax.set_xlabel("Support Calls")
ax.set_ylabel("Churn Rate (%)")
ax.set_ylim(0, 110)
ax.set_title("Churn Rate by Number of Support Calls")
ax.grid(True)

st.pyplot(fig)

# -----------------------------
# Project Observations
# -----------------------------
st.subheader("Key Observations")

st.write(
    f"""
- Overall churn rate in the cleaned training dataset: **{churn_rate:.2f}%**
- Monthly contracts show **100% churn** in this dataset.
- Customers with 5 support calls show **94.71% churn**.
- These findings describe associations observed in the dataset and
  should not be interpreted as causal relationships.
"""
)

st.divider()

st.caption(
    "Dataset: Customer Churn Dataset | "
    "Analysis using Python, Pandas and Matplotlib"
)
