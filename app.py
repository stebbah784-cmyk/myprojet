import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import plotly.express as px
import seaborn as sns
import pickle
import re

# =========================
# PAGE CONFIG
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
    background-color: #FF9800;
}
</style>
""", unsafe_allow_html=True)

# =========================
# TITLE
# =========================
st.title("🦠 Application d'Analyse des Sentiments - COVID-19")
st.write("Dashboard interactif avec Machine Learning")
# hhhouda  
import os

st.write("Files found:")
st.write(os.listdir())

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

    text = re.sub(r"[^a-zA-Z\s]", "", text)

    return text

# =========================
# PREDICTION FUNCTION
# =========================
def predict_sentiment(text):

    cleaned = clean_text(text)

    vector = vectorizer.transform([cleaned])

    prediction = model.predict(vector)[0]

    return prediction

# =========================
# LOAD DATA
# =========================
@st.cache_data
def load_data():

    df = pd.read_csv("Corona_NLP_test.csv", encoding="latin-1")

    df["Clean_Tweet"] = df["OriginalTweet"].apply(clean_text)

    df["AI_Sentiment"] = df["Clean_Tweet"].apply(predict_sentiment)

    return df

df = load_data()

# =========================
# SIDEBAR
# =========================
st.sidebar.header("🔎 Filtres")

sentiment_filter = st.sidebar.selectbox(
    "Filtrer par sentiment",
    ["All", "Positive", "Neutral", "Negative"]
)

search = st.sidebar.text_input("🔍 Rechercher un tweet")

# =========================
# FILTER LOGIC
# =========================
df_filtered = df.copy()

if sentiment_filter != "All":
    df_filtered = df_filtered[
        df_filtered["AI_Sentiment"] == sentiment_filter
    ]

if search:

    search = search.lower()

    df_filtered = df_filtered[
        df_filtered["Clean_Tweet"].str.contains(search, na=False, regex=False)
    ]

# =========================
# KPI
# =========================
sent_counts = df_filtered["AI_Sentiment"].value_counts()

c1, c2, c3 = st.columns(3)

c1.metric("📊 Tweets", df_filtered.shape[0])

c2.metric(
    "🟢 Positifs",
    sent_counts.get("Positive", 0)
)

c3.metric(
    "🔴 Négatifs",
    sent_counts.get("Negative", 0)
)

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
# TAB 1
# =========================
with tab1:

    st.subheader("Aperçu des données")

    st.dataframe(
        df_filtered[
            ["OriginalTweet", "Clean_Tweet", "AI_Sentiment"]
        ].head(20),
        use_container_width=True
    )

# =========================
# TAB 2
# =========================
with tab2:

    st.subheader("Nuage de mots")

    text = " ".join(
        df_filtered["Clean_Tweet"].astype(str)
    )

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
# TAB 3
# =========================
with tab3:

    st.subheader("Distribution des sentiments")

    fig1 = px.histogram(
        df_filtered,
        x="AI_Sentiment",
        color="AI_Sentiment",
        title="Distribution des sentiments"
    )

    st.plotly_chart(fig1, use_container_width=True)

    fig2 = px.pie(
        df_filtered,
        names="AI_Sentiment",
        title="Répartition des sentiments"
    )

    st.plotly_chart(fig2, use_container_width=True)

# =========================
# TAB 4
# =========================
with tab4:

    st.subheader("🔮 Analyse IA en temps réel")

    user_text = st.text_area(
        "✍️ Écris un tweet en anglais"
    )

    if st.button("Analyser"):

        if user_text.strip() == "":

            st.warning("Veuillez entrer un texte")

        else:

            prediction = predict_sentiment(user_text)

            if prediction == "Positive":

                st.success("😊 POSITIF")

            elif prediction == "Negative":

                st.error("😡 NEGATIF")

            else:

                st.info("😐 NEUTRE")

# =========================
# FOOTER
# =========================
st.markdown("---")

st.markdown(
    "🚀 Projet Data Science | COVID-19 Sentiment Analysis | Streamlit + Machine Learning"
)
