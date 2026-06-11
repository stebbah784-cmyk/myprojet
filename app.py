import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import plotly.express as px
import pickle
import re
from collections import Counter

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="UCD - IA S6 Sentiment Dashboard",
    page_icon="🎓",
    layout="wide"
)

# =========================
# HEADER
# =========================
st.title("🎓 Université Chouaib Doukkali - Faculté des Sciences El Jadida")
st.subheader("Filière : Informatique Appliquée (S6)")
st.markdown("🦠 ML for sentiment analysis: case of Covid 19")

st.markdown("---")

# =========================
# STYLE
# =========================
st.markdown("""
<style>
.stApp {
    background-color: #fff7ed;
}
</style>
""", unsafe_allow_html=True)

# =========================
# LOAD MODEL
# =========================
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# =========================
# CLEAN TEXT
# =========================
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#", "", text)
    text = re.sub(r"[^\w\sÀ-ÿ]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# =========================
# PREDICT
# =========================
def predict_sentiment(text):
    cleaned = clean_text(text)

    if cleaned.strip() == "":
        return "Neutral"

    vector = vectorizer.transform([cleaned])
    prediction = model.predict(vector)[0]

    return prediction

# =========================
# LOAD DATA
# =========================
@st.cache_data
def load_data():

    df = pd.read_csv("Corona_NLP_train.csv", encoding="latin-1")

    mapping = {
        "Extremely Positive": "Positive",
        "Positive": "Positive",
        "Neutral": "Neutral",
        "Negative": "Negative",
        "Extremely Negative": "Negative"
    }

    df["Sentiment"] = df["Sentiment"].map(mapping)

    df["Clean_Tweet"] = df["OriginalTweet"].apply(clean_text)
    df["AI_Sentiment"] = df["Clean_Tweet"].apply(predict_sentiment)

    return df

df = load_data()

# =========================
# FILTERS
# =========================
st.sidebar.header("🔎 Filters")

sentiment_filter = st.sidebar.selectbox(
    "Sentiment",
    ["All", "Positive", "Neutral", "Negative"]
)

search = st.sidebar.text_input("🔍 Search Tweet")

df_filtered = df.copy()

if sentiment_filter != "All":
    df_filtered = df_filtered[
        df_filtered["AI_Sentiment"].str.lower() == sentiment_filter.lower()
    ]

if search:
    df_filtered = df_filtered[
        df_filtered["Clean_Tweet"].str.contains(search.lower(), na=False)
    ]

# =========================
# KPI
# =========================
sent_counts = df_filtered["AI_Sentiment"].value_counts()

c1, c2, c3 = st.columns(3)

c1.metric("📊 Tweets", len(df_filtered))
c2.metric("🟢 Positive", sent_counts.get("Positive", 0))
c3.metric("🔴 Negative", sent_counts.get("Negative", 0))

st.markdown("---")

# =========================
# TABS
# =========================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Data",
    "☁️ Word Analysis",
    "📈 Visualisation",
    "🔮 Live Test",
    "📊 Evaluation"
])

# =========================
# TAB 1
# =========================
with tab1:
    st.subheader("Dataset Preview")
    st.dataframe(df_filtered[["OriginalTweet", "Clean_Tweet", "AI_Sentiment"]].head(20))

# =========================
# TAB 2
# =========================
with tab2:

    option = st.selectbox(
        "Select sentiment:",
        ["All", "Positive", "Negative", "Neutral"]
    )

    if option != "All":
        data_words = df_filtered[
            df_filtered["AI_Sentiment"].str.lower() == option.lower()
        ]
    else:
        data_words = df_filtered

    text = " ".join(data_words["Clean_Tweet"].dropna().astype(str))

    if text.strip() != "":

        wc = WordCloud(width=1200, height=500, background_color="white").generate(text)

        fig, ax = plt.subplots()
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        st.pyplot(fig)

        words = text.split()
        common = Counter(words).most_common(15)

        words_df = pd.DataFrame(common, columns=["Word", "Count"])

        fig2 = px.bar(words_df, x="Word", y="Count")
        st.plotly_chart(fig2, use_container_width=True)

# =========================
# TAB 3
# =========================
with tab3:

    fig1 = px.pie(df_filtered, names="AI_Sentiment")
    st.plotly_chart(fig1, use_container_width=True)

    fig2 = px.histogram(df_filtered, x="AI_Sentiment", color="AI_Sentiment")
    st.plotly_chart(fig2, use_container_width=True)

# =========================
# TAB 4
# =========================
with tab4:

    user_text = st.text_area("Enter text")

    if st.button("Analyze"):

        if user_text.strip() == "":
            st.warning("Please enter text")
        else:
            pred = predict_sentiment(user_text)

            if "positive" in pred.lower():
                st.success("😊 POSITIVE")
            elif "negative" in pred.lower():
                st.error("😡 NEGATIVE")
            else:
                st.info("😐 NEUTRAL")

# =========================
# TAB 5 - EVALUATION
# =========================
with tab5:

    st.subheader("📊 Model Evaluation Dashboard")

    y_true = df["Sentiment"]
    y_pred = df["AI_Sentiment"]

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    metrics = pd.DataFrame({
        "Metric": ["Accuracy", "Precision", "Recall", "F1 Score"],
        "Score": [acc, prec, rec, f1]
    })

    st.dataframe(metrics)

    # BAR
    fig1 = px.bar(metrics, x="Metric", y="Score", text="Score", color="Metric")
    fig1.update_layout(yaxis=dict(range=[0, 1]))
    st.plotly_chart(fig1, use_container_width=True)

    # PIE
    fig2 = px.pie(metrics, names="Metric", values="Score")
    st.plotly_chart(fig2, use_container_width=True)

    # RADAR
    fig3 = px.line_polar(metrics, r="Score", theta="Metric", line_close=True)
    st.plotly_chart(fig3, use_container_width=True)

    # CONFUSION MATRIX
    st.subheader("📉 Confusion Matrix")

    labels = ["Positive", "Neutral", "Negative"]

    cm = confusion_matrix(y_true, y_pred, labels=labels)

    fig4, ax = plt.subplots()
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, cmap="Blues", values_format="d")

    st.pyplot(fig4)

# =========================
# FOOTER
# =========================
st.markdown("---")

st.markdown("""
🎓 UCD - FS El Jadida | IA S6 | Data Science  
🦠 Covid-19 Sentiment Analysis Project
""")
