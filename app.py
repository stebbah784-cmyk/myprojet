
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import plotly.express as px
import re
import string
from textblob import TextBlob
from collections import Counter

# =========================
# CONFIG
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
st.subheader("📘 Filière : Intelligence Artificielle & Big Data (S6)")
st.markdown("🦠 COVID-19 Sentiment Analysis Dashboard (Projet Data Science)")

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

    def clean_text(text):
        text = str(text).lower()
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'#', '', text)
        text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
        return text

    df["Clean_Tweet"] = df["OriginalTweet"].apply(clean_text)

    def sentiment_ai(text):
        score = TextBlob(text).sentiment.polarity
        if score > 0:
            return "Positive"
        elif score < 0:
            return "Negative"
        return "Neutral"

    df["AI_Sentiment"] = df["Clean_Tweet"].apply(sentiment_ai)

    return df

df = load_data()

# =========================
# SIDEBAR
# =========================
st.sidebar.header("🔎 Filters")

sentiment_filter = st.sidebar.selectbox(
    "Sentiment",
    ["All", "Positive", "Neutral", "Negative"]
)

search = st.sidebar.text_input("Search Tweet")

df_filtered = df.copy()

if sentiment_filter != "All":
    df_filtered = df_filtered[df_filtered["AI_Sentiment"] == sentiment_filter]

if search:
    df_filtered = df_filtered[df_filtered["Clean_Tweet"].str.contains(search.lower(), na=False)]

# =========================
# KPI
# =========================
c1, c2, c3 = st.columns(3)

sent_counts = df_filtered["AI_Sentiment"].value_counts()

c1.metric("Tweets", len(df_filtered))
c2.metric("Positive", sent_counts.get("Positive", 0))
c3.metric("Negative", sent_counts.get("Negative", 0))

st.markdown("---")

# =========================
# TABS
# =========================
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Data",
    "☁️ Word Analysis",
    "📈 Visualisation",
    "🔮 Live Test"
])

# =========================
# TAB 1
# =========================
with tab1:
    st.subheader("📁 Dataset Preview")
    st.dataframe(df_filtered[["OriginalTweet", "Clean_Tweet", "AI_Sentiment"]].head(20))

# =========================
# TAB 2 - WORDCLOUD
# =========================
with tab2:
    st.subheader("☁️ WordCloud Analysis")

    option = st.selectbox("Select sentiment:", ["All", "Positive", "Negative", "Neutral"])

    if option != "All":
        data_words = df_filtered[df_filtered["AI_Sentiment"] == option]
    else:
        data_words = df_filtered

    text = " ".join(data_words["Clean_Tweet"].astype(str))

    wc = WordCloud(width=1000, height=400, background_color="white").generate(text)

    fig, ax = plt.subplots()
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig)

    st.markdown("### 🔥 Top Words")

    words = text.split()
    common = Counter(words).most_common(15)

    words_df = pd.DataFrame(common, columns=["Word", "Count"])

    fig2 = px.bar(words_df, x="Word", y="Count", title="Most Frequent Words")
    st.plotly_chart(fig2, use_container_width=True)

# =========================
# TAB 3
# =========================
with tab3:
    st.subheader("📊 Sentiment Analytics")

    fig1 = px.pie(df_filtered, names="AI_Sentiment", title="Sentiment Distribution")
    st.plotly_chart(fig1, use_container_width=True)

    fig2 = px.histogram(df_filtered, x="AI_Sentiment", color="AI_Sentiment")
    st.plotly_chart(fig2, use_container_width=True)

    df_filtered["length"] = df_filtered["Clean_Tweet"].apply(len)

    fig3 = px.box(df_filtered, x="AI_Sentiment", y="length", color="AI_Sentiment")
    st.plotly_chart(fig3, use_container_width=True)

# =========================
# TAB 4
# =========================
with tab4:
    st.subheader("🔮 Live Sentiment Test (AI)")

    user_text = st.text_area("Enter text:")

    if st.button("Analyze"):

        if user_text.strip() == "":
            st.warning("Please enter text")
        else:
            score = TextBlob(user_text).sentiment.polarity

            if score > 0:
                st.success("Positive 😊")
            elif score < 0:
                st.error("Negative 😡")
            else:
                st.info("Neutral 😐")

            st.metric("Score", round(score, 2))

# =========================
# FOOTER
# =========================
st.markdown("---")
st.markdown("🎓 UCD - FS El Jadida | IA S6 | Data Science Project | COVID-19 Sentiment Analysis")
