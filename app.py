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
    page_title="COVID-19 Sentiment Dashboard",
    page_icon="🦠",
    layout="wide"
)

# =========================
# STYLE
# =========================
st.markdown("""
<style>
.stApp {
    background-color: #fff8f0;
}
</style>
""", unsafe_allow_html=True)

st.title("🦠 COVID-19 Sentiment Analysis Dashboard")
st.write("Dashboard professionnel d’analyse des tweets")

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

search = st.sidebar.text_input("Search tweet")

df_filtered = df.copy()

if sentiment_filter != "All":
    df_filtered = df_filtered[df_filtered["AI_Sentiment"] == sentiment_filter]

if search:
    df_filtered = df_filtered[df_filtered["Clean_Tweet"].str.contains(search.lower(), na=False)]

# =========================
# KPI
# =========================
col1, col2, col3, col4 = st.columns(4)

sent_counts = df_filtered["AI_Sentiment"].value_counts()

col1.metric("Tweets", len(df_filtered))
col2.metric("Positive", sent_counts.get("Positive", 0))
col3.metric("Negative", sent_counts.get("Negative", 0))
col4.metric("Neutral", sent_counts.get("Neutral", 0))

st.markdown("---")

# =========================
# TABS
# =========================
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Data",
    "☁️ WordCloud",
    "📈 Analytics",
    "🔮 Live Test"
])

# =========================
# TAB 1
# =========================
with tab1:
    st.subheader("Dataset Preview")
    st.dataframe(df_filtered[["OriginalTweet", "Clean_Tweet", "AI_Sentiment"]].head(20))

# =========================
# TAB 2 WORDCLOUD
# =========================
with tab2:
    st.subheader("WordCloud")

    text = " ".join(df_filtered["Clean_Tweet"].astype(str))

    wc = WordCloud(width=1200, height=500, background_color="white").generate(text)

    fig, ax = plt.subplots()
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig)

    # Top words
    st.subheader("🔥 Top Words")

    words = " ".join(df_filtered["Clean_Tweet"]).split()
    common = Counter(words).most_common(15)

    words_df = pd.DataFrame(common, columns=["Word", "Count"])

    fig2 = px.bar(words_df, x="Word", y="Count", title="Top 15 words")
    st.plotly_chart(fig2, use_container_width=True)

# =========================
# TAB 3 ANALYTICS
# =========================
with tab3:
    st.subheader("📊 Sentiment Distribution")

    fig1 = px.histogram(
        df_filtered,
        x="AI_Sentiment",
        color="AI_Sentiment"
    )
    st.plotly_chart(fig1, use_container_width=True)

    fig2 = px.pie(
        df_filtered,
        names="AI_Sentiment"
    )
    st.plotly_chart(fig2, use_container_width=True)

    # Tweet length
    df_filtered["length"] = df_filtered["Clean_Tweet"].apply(len)

    st.subheader("📏 Tweet Length Analysis")

    fig3 = px.box(
        df_filtered,
        x="AI_Sentiment",
        y="length",
        color="AI_Sentiment"
    )
    st.plotly_chart(fig3, use_container_width=True)

    fig4 = px.histogram(
        df_filtered,
        x="length",
        nbins=40
    )
    st.plotly_chart(fig4, use_container_width=True)

    # Correlation
    st.subheader("🔥 Correlation")

    corr = df_filtered[["length"]].corr()

    fig5, ax = plt.subplots()
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig5)

# =========================
# TAB 4 LIVE TEST
# =========================
with tab4:
    st.subheader("🔮 Test Sentiment IA")

    text_input = st.text_area("Enter tweet:")

    if st.button("Analyze"):

        if text_input.strip() == "":
            st.warning("Enter text")
        else:
            score = TextBlob(text_input).sentiment.polarity

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
st.markdown("🚀 Professional Data Science Project | COVID-19 Sentiment Analysis")
