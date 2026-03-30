import streamlit as st
import pandas as pd
import plotly.express as px

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import re


# PAGE CONFIG


st.set_page_config(page_title="AI Data Analyst", layout="wide")

st.title("AI Data Analyst — Shopping Trends Dashboard")


# LOAD DATA


df = pd.read_csv("shopping_trends.csv")


# ML TRAINING DATA


training_data = [

    ("which location spends most", "location"),
    ("top spending locations", "location"),
    ("location revenue", "location"),
    ("sales by location", "location"),

    ("most popular product", "product"),
    ("top products", "product"),
    ("top selling items", "product"),
    ("best selling products", "product"),

    ("category revenue", "category"),
    ("best selling category", "category"),
    ("top categories", "category"),

    ("spending by gender", "gender"),
    ("which gender spends more", "gender"),
    ("male vs female spending", "gender"),

    ("sales by season", "season"),
    ("season revenue", "season"),

    ("age distribution", "age"),
    ("customer age analysis", "age")
]

questions = [q for q, i in training_data]
labels = [i for q, i in training_data]

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(questions)

model = LogisticRegression()
model.fit(X, labels)


# SIDEBAR FILTERS


st.sidebar.header("Filters")

gender_filter = st.sidebar.multiselect(
    "Gender",
    options=df["Gender"].unique(),
    default=df["Gender"].unique()
)

category_filter = st.sidebar.multiselect(
    "Category",
    options=df["Category"].unique(),
    default=df["Category"].unique()
)

season_filter = st.sidebar.multiselect(
    "Season",
    options=df["Season"].unique(),
    default=df["Season"].unique()
)

filtered_df = df[
    (df["Gender"].isin(gender_filter)) &
    (df["Category"].isin(category_filter)) &
    (df["Season"].isin(season_filter))
]


# DATA PREVIEW


with st.expander("Dataset Preview"):
    st.dataframe(filtered_df.head())


# KEY METRICS


st.subheader("Key Metrics")

c1, c2, c3, c4 = st.columns(4)

total_revenue = filtered_df["Purchase Amount (USD)"].sum()
avg_purchase = filtered_df["Purchase Amount (USD)"].mean()
top_product = filtered_df["Item Purchased"].value_counts().idxmax()
top_location = filtered_df.groupby(
    "Location")["Purchase Amount (USD)"].sum().idxmax()

c1.metric("Total Revenue", f"${total_revenue:,.0f}")
c2.metric("Average Purchase", f"${avg_purchase:.2f}")
c3.metric("Top Product", top_product)
c4.metric("Top Location", top_location)

st.divider()


# DASHBOARD


st.subheader("Insights Dashboard")

col1, col2 = st.columns(2)

with col1:

    gender_spending = filtered_df.groupby(
        "Gender")["Purchase Amount (USD)"].sum().reset_index()

    fig = px.pie(
        gender_spending,
        names="Gender",
        values="Purchase Amount (USD)",
        title="Spending by Gender"
    )

    st.plotly_chart(fig, use_container_width=True, key="gender_chart")

with col2:

    category_spending = filtered_df.groupby(
        "Category")["Purchase Amount (USD)"].sum().reset_index()

    fig = px.bar(
        category_spending,
        x="Purchase Amount (USD)",
        y="Category",
        orientation="h",
        title="Revenue by Category"
    )

    st.plotly_chart(fig, use_container_width=True, key="category_chart")

col3, col4 = st.columns(2)

with col3:

    season_spending = filtered_df.groupby(
        "Season")["Purchase Amount (USD)"].sum().reset_index()

    fig = px.line(
        season_spending,
        x="Season",
        y="Purchase Amount (USD)",
        markers=True,
        title="Seasonal Sales"
    )

    st.plotly_chart(fig, use_container_width=True, key="season_chart")

with col4:

    location_spending = (
        filtered_df.groupby("Location")["Purchase Amount (USD)"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
    )

    fig = px.bar(
        location_spending,
        x="Purchase Amount (USD)",
        y="Location",
        orientation="h",
        title="Top Locations"
    )

    st.plotly_chart(fig, use_container_width=True, key="location_chart")

st.divider()


# SMART QUERY PARSER



def parse_query_filters(query):

    query = query.lower()

    gender = None
    category = None
    season = None

    if "male" in query or "men" in query:
        gender = "Male"

    if "female" in query or "women" in query:
        gender = "Female"

    for cat in df["Category"].unique():
        if cat.lower() in query:
            category = cat

    for s in df["Season"].unique():
        if s.lower() in query:
            season = s

    return gender, category, season


def detect_top_n(query):

    numbers = re.findall(r'\d+', query)

    if numbers:
        return int(numbers[0])

    if "top" in query:
        return 5

    return 10


# AI BOT



st.subheader("AI Data Analyst Bot")

user_question = st.text_input(
    "Ask the data anything",
    placeholder="Examples: male footwear in winter, top 5 products, female clothing sales"
)

if user_question:

    intent = model.predict(vectorizer.transform([user_question]))[0]

    st.write("Detected analysis:", intent)

    gender_q, category_q, season_q = parse_query_filters(user_question)

    query_df = filtered_df.copy()

    if gender_q:
        query_df = query_df[query_df["Gender"] == gender_q]

    if category_q:
        query_df = query_df[query_df["Category"] == category_q]

    if season_q:
        query_df = query_df[query_df["Season"] == season_q]

    top_n = detect_top_n(user_question)

 
    # LOCATION
   

    if intent == "location":

        result = (
            query_df.groupby("Location")["Purchase Amount (USD)"]
            .sum()
            .sort_values(ascending=False)
            .head(top_n)
            .reset_index()
        )

        fig = px.bar(
            result,
            x="Purchase Amount (USD)",
            y="Location",
            orientation="h",
            title="Top Locations"
        )

        st.plotly_chart(fig, use_container_width=True, key="bot_location")

   
    # PRODUCT
    

    elif intent == "product":

        result = query_df["Item Purchased"].value_counts().reset_index()
        result.columns = ["Product", "Count"]

        fig = px.bar(
            result.head(top_n),
            x="Count",
            y="Product",
            orientation="h",
            title="Top Products"
        )

        st.plotly_chart(fig, use_container_width=True, key="bot_product")

    
    # CATEGORY
    

    elif intent == "category":

        result = query_df.groupby("Category")[
            "Purchase Amount (USD)"].sum().reset_index()

        fig = px.bar(
            result,
            x="Purchase Amount (USD)",
            y="Category",
            orientation="h",
            title="Category Revenue"
        )

        st.plotly_chart(fig, use_container_width=True, key="bot_category")

  
    # SEASON
    

    elif intent == "season":

        result = query_df.groupby(
            "Season")["Purchase Amount (USD)"].sum().reset_index()

        fig = px.line(
            result,
            x="Season",
            y="Purchase Amount (USD)",
            markers=True,
            title="Season Sales"
        )

        st.plotly_chart(fig, use_container_width=True, key="bot_season")

   
    # GENDER
    

    elif intent == "gender":

        result = query_df.groupby(
            "Gender")["Purchase Amount (USD)"].sum().reset_index()

        fig = px.pie(
            result,
            names="Gender",
            values="Purchase Amount (USD)",
            title="Spending by Gender"
        )

        st.plotly_chart(fig, use_container_width=True, key="bot_gender")

   
    # AGE
    

    elif intent == "age":

        fig = px.histogram(
            query_df,
            x="Age",
            title="Age Distribution"
        )

        st.plotly_chart(fig, use_container_width=True, key="bot_age")

    else:

        st.warning("Sorry, I couldn't understand the question.")
