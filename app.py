import streamlit as st
import pandas as pd
import os
import csv
import plotly.express as px
from datetime import datetime

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="AI Expense Manager Pro",
    page_icon="💸",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =====================================================
# PREMIUM CSS
# =====================================================
st.markdown("""
<style>
.block-container{
    padding-top:1.1rem;
    padding-bottom:2rem;
    padding-left:2rem;
    padding-right:2rem;
}
h1,h2,h3{
    letter-spacing:-0.4px;
}
.stTabs [data-baseweb="tab-list"]{
    gap:10px;
}
.stTabs [data-baseweb="tab"]{
    height:52px;
    border-radius:12px;
    padding-left:18px;
    padding-right:18px;
    font-weight:600;
}
.stButton>button{
    width:100%;
    height:46px;
    border-radius:12px;
    font-weight:700;
    margin-top:8px;
}
.stTextInput input,
.stNumberInput input{
    border-radius:12px !important;
}
div[data-testid="metric-container"]{
    border:1px solid rgba(255,255,255,0.08);
    padding:18px;
    border-radius:16px;
}
</style>
""", unsafe_allow_html=True)

# =====================================================
# FILES
# =====================================================
records_file = "records.csv"
budget_file = "budget.txt"
training_file = "expenses.csv"

# create files if missing
if not os.path.exists(records_file):
    pd.DataFrame(
        columns=["Date", "Expense", "Amount", "Category"]
    ).to_csv(records_file, index=False)

if not os.path.exists(budget_file):
    with open(budget_file, "w") as f:
        f.write("20000")

# =====================================================
# LOAD TRAINING DATA
# =====================================================
train = pd.read_csv(training_file, on_bad_lines="skip")

X = train["Description"].astype(str)
y = train["Category"].astype(str)

vectorizer = TfidfVectorizer(
    analyzer="char_wb",
    ngram_range=(2, 5)
)

X_vec = vectorizer.fit_transform(X)

model = LogisticRegression(max_iter=3000)
model.fit(X_vec, y)

# =====================================================
# LOAD USER DATA
# =====================================================
records = pd.read_csv(records_file)

with open(budget_file, "r") as f:
    budget = int(f.read())

# =====================================================
# HEADER
# =====================================================
st.title("💸 AI Expense Manager Pro")
st.caption("Smart finance dashboard powered by Artificial Intelligence")

# =====================================================
# TABS
# =====================================================
tab1, tab2, tab3, tab4 = st.tabs(
    ["➕ Add Expense", "📋 Records", "📊 Analytics", "🎯 Budget & AI"]
)

# =====================================================
# TAB 1 - ADD EXPENSE
# =====================================================
with tab1:

    st.subheader("Add New Expense")

    c1, c2 = st.columns([2, 1])

    with c1:
        expense = st.text_input(
            "Expense Name",
            placeholder="Example: pizza hut / uber / airtel recharge"
        )

    with c2:
        amount = st.number_input(
            "Amount ₹",
            min_value=0,
            step=50
        )

    categories = sorted(list(train["Category"].unique()))
    predicted_category = categories[0]
    confidence_level = ""

    if expense.strip():

        clean_expense = expense.strip()

        vec = vectorizer.transform([clean_expense])

        predicted_category = model.predict(vec)[0]

        probs = model.predict_proba(vec)[0]
        sorted_probs = sorted(probs, reverse=True)

        top1 = sorted_probs[0]
        top2 = sorted_probs[1] if len(sorted_probs) > 1 else 0

        margin = top1 - top2

        if margin > 0.20:
            confidence_level = "Very High"
        elif margin > 0.10:
            confidence_level = "High"
        elif margin > 0.05:
            confidence_level = "Medium"
        else:
            confidence_level = "Low"

        st.info(
            f"Suggested Category: {predicted_category} | Confidence: {confidence_level}"
        )

    selected_category = st.selectbox(
        "Choose / Confirm Category",
        categories,
        index=categories.index(predicted_category)
        if predicted_category in categories else 0
    )

    if st.button("Add Expense"):

        clean_expense = expense.strip()

        if clean_expense and amount > 0:

            new_row = pd.DataFrame(
                [[
                    datetime.now().strftime("%Y-%m-%d"),
                    clean_expense,
                    amount,
                    selected_category
                ]],
                columns=["Date", "Expense", "Amount", "Category"]
            )

            records = pd.concat([records, new_row], ignore_index=True)
            records.to_csv(records_file, index=False)

            # self learning
            learn_row = pd.DataFrame(
                [[clean_expense, selected_category]],
                columns=["Description", "Category"]
            )

            learn_row.to_csv(
                training_file,
                mode="a",
                header=False,
                index=False,
                quoting=csv.QUOTE_ALL
            )

            st.success(
                f"Added • {clean_expense} • ₹{amount} • {selected_category}"
            )
            st.rerun()

# =====================================================
# EMPTY CHECK
# =====================================================
if len(records) == 0:

    with tab2:
        st.info("No records yet.")

    with tab3:
        st.info("No analytics yet.")

    with tab4:
        st.info("Add expenses first.")

    st.stop()

# =====================================================
# CALCULATIONS
# =====================================================
records["Amount"] = pd.to_numeric(
    records["Amount"],
    errors="coerce"
).fillna(0)

total = int(records["Amount"].sum())

category_summary = records.groupby("Category")["Amount"].sum()

top_category = category_summary.idxmax()
top_amount = int(category_summary.max())

budget_left = max(budget - total, 0)

records["Date"] = pd.to_datetime(
    records["Date"],
    errors="coerce"
)

daily = records.groupby("Date")["Amount"].sum()

# =====================================================
# TAB 2 - RECORDS
# =====================================================
with tab2:

    st.subheader("Saved Expenses")

    display_records = records.copy()

    display_records["Date"] = display_records["Date"].dt.strftime("%d %b %Y")

    display_records.insert(
        0,
        "No.",
        range(1, len(display_records) + 1)
    )

    st.dataframe(
        display_records,
        use_container_width=True,
        hide_index=True,
        height=360
    )

    st.markdown("### Delete Record")

    d1, d2 = st.columns([3, 1])

    with d1:

        options = []

        for i, row in records.iterrows():
            options.append(
                f"{i+1} | {row['Expense']} | ₹{int(row['Amount'])} | {row['Category']}"
            )

        selected = st.selectbox(
            "Choose record to delete",
            options
        )

    with d2:

        st.write("")
        st.write("")

        if st.button("Delete"):

            idx = int(selected.split("|")[0].strip()) - 1

            records = records.drop(idx).reset_index(drop=True)
            records.to_csv(records_file, index=False)

            st.success("Record deleted.")
            st.rerun()

# =====================================================
# TAB 3 - ANALYTICS
# =====================================================
with tab3:

    st.subheader("Expense Analytics")

    left, right = st.columns(2)

    with left:

        fig = px.pie(
            values=category_summary.values,
            names=category_summary.index,
            title="Spending by Category"
        )

        st.plotly_chart(fig, use_container_width=True)

    with right:

        bar = px.bar(
            x=category_summary.index,
            y=category_summary.values,
            labels={"x": "Category", "y": "Amount"},
            title="Category Comparison"
        )

        st.plotly_chart(bar, use_container_width=True)

    line = px.line(
        x=daily.index,
        y=daily.values,
        labels={"x": "Date", "y": "Amount"},
        title="Daily Spending Trend"
    )

    st.plotly_chart(line, use_container_width=True)

# =====================================================
# TAB 4 - BUDGET & AI
# =====================================================
with tab4:

    st.subheader("Budget & Smart Insights")

    m1, m2, m3 = st.columns(3)

    with m1:
        st.metric("💰 Total Spent", f"₹{total}")

    with m2:
        st.metric("🎯 Budget Left", f"₹{budget_left}")

    with m3:
        st.metric("🏆 Top Category", top_category)

    new_budget = st.number_input(
        "Monthly Budget ₹",
        min_value=0,
        value=budget,
        step=500
    )

    if st.button("Save Budget"):

        with open(budget_file, "w") as f:
            f.write(str(int(new_budget)))

        st.success("Budget updated.")
        st.rerun()

    if total > budget:
        st.error("Budget exceeded.")
    elif total > budget * 0.8:
        st.warning("You have used over 80% of your budget.")
    else:
        st.success("You are within budget.")

    st.markdown("### AI Suggestions")

    if top_category == "Food & Dining":
        st.info("You spend most on food. Reduce eating out to save more.")

    elif top_category == "Transport":
        st.info("Transport spending is highest. Try metro or ride pooling.")

    elif top_category == "Shopping & Retail":
        st.info("Shopping is highest. Delay impulse purchases.")

    elif top_category == "Entertainment":
        st.info("Entertainment leads spending. Review subscriptions.")

    else:
        st.info(f"Top category is {top_category}. Keep tracking it.")
