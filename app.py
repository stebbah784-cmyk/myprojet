
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

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="COVID-19 Sentiment Dashboard",
    page_icon="🦠",
    layout="wide"
)

# =========================
# UI STYLE
# =========================
st.markdown(
    """
    <style>
    .stApp {
        background-color: blue;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# =========================
# TITLE
# =========================
st.title("🦠 Application d'Analyse des Sentiments - COVID-19")
st.write("Dashboard professionnel pour analyse des tweets et sentiments")

# =========================
# LOAD + CLEAN DATA
# =========================
@st.cache_data
def load_data():
    df = pd.read_csv("Corona_NLP_train.csv", encoding="latin-1")

    # mapping sentiments
    mapping = {
        "Extremely Positive": "Positive",
        "Positive": "Positive",
        "Neutral": "Neutral",
        "Negative": "Negative",
        "Extremely Negative": "Negative"
    }
    df["Sentiment"] = df["Sentiment"].map(mapping)

    # clean text
    def clean_text(text):
        text = str(text).lower()
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'#', '', text)
        text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
        return text

    df["Clean_Tweet"] = df["OriginalTweet"].apply(clean_text)

    # sentiment via TextBlob
    def sentiment_ai(text):
        score = TextBlob(text).sentiment.polarity
        if score > 0:
            return "Positive"
        elif score < 0:
            return "Negative"
        else:
            return "Neutral"

    df["AI_Sentiment"] = df["Clean_Tweet"].apply(sentiment_ai)

    return df

df = load_data()

st.markdown("---")

# =========================
# SIDEBAR FILTERS
# =========================
st.sidebar.header("🔎 Filtres")

sentiment_filter = st.sidebar.selectbox(
    "Filtrer par sentiment",
    ["All", "Positive", "Neutral", "Negative"]
)

search = st.sidebar.text_input("🔍 Rechercher un tweet")

df_filtered = df.copy()

if sentiment_filter != "All":
    df_filtered = df_filtered[df_filtered["AI_Sentiment"] == sentiment_filter]

if search:
    df_filtered = df_filtered[
        df_filtered["Clean_Tweet"].str.contains(search.lower(), na=False)
    ]

# =========================
# KPI
# =========================
c1, c2, c3 = st.columns(3)

sent_counts = df_filtered["AI_Sentiment"].value_counts()

c1.metric("📊 Tweets", df_filtered.shape[0])
c2.metric("🧠 Positifs", sent_counts.get("Positive", 0))
c3.metric("😡 Négatifs", sent_counts.get("Negative", 0))

st.markdown("---")

# =========================
# TABS
# =========================
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Analyse",
    "☁️ WordCloud",
    "📈 Graphiques",
    "🔮 Test IA"
])

# =========================
# TAB 1 - DATA
# =========================
with tab1:
    st.subheader("Aperçu des données")
    st.dataframe(df_filtered[["OriginalTweet", "Clean_Tweet", "AI_Sentiment"]].head(20))

# =========================
# TAB 2 - WORDCLOUD
# =========================
with tab2:
    st.subheader("Nuage de mots")

    text = " ".join(df_filtered["Clean_Tweet"].astype(str))

    wc = WordCloud(
        width=1200,
        height=500,
        background_color="white",
        colormap="viridis"
    ).generate(text)

    fig, ax = plt.subplots()
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig)

# =========================
# TAB 3 - GRAPHS
# =========================
with tab3:
    st.subheader("Distribution des sentiments")

    fig1 = px.histogram(
        df_filtered,
        x="AI_Sentiment",
        color="AI_Sentiment",
        title="Sentiments Distribution"
    )
    st.plotly_chart(fig1, use_container_width=True)

    fig2 = px.pie(
        df_filtered,
        names="AI_Sentiment",
        title="Répartition des sentiments"
    )
    st.plotly_chart(fig2, use_container_width=True)

# =========================
# TAB 4 - LIVE TEST
# =========================
with tab4:
    st.subheader("🔮 Test de sentiment en temps réel")

    user_text = st.text_area("Écris un tweet ici:")

    if st.button("Analyser"):

        if user_text.strip() == "":
            st.warning("Veuillez entrer un texte")
        else:
            score = TextBlob(user_text).sentiment.polarity

            if score > 0:
                st.success("😊 POSITIF")
            elif score < 0:
                st.error("😡 NEGATIF")
            else:
                st.info("😐 NEUTRE")

            st.metric("Score", round(score, 2))

# =========================
# FOOTER
# =========================
st.markdown("---")
st.markdown("🚀 Projet Data Science | COVID-19 Sentiment Analysis | Streamlit")
