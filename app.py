import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import plotly.express as px
import pickle
import re
import string
from collections import Counter

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
st.markdown("Réalisée par : Wakil chaimae et Stebba houda ")


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

    text = re.sub(r"[%s]" % re.escape(string.punctuation), "", text)

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

    df = pd.read_csv("Corona_NLP_train.csv", encoding="latin-1")

    mapping = {
        "Extremely Positive": "Positive",
        "Positive": "Positive",
        "Neutral": "Neutral",
        "Negative": "Negative",
        "Extremely Negative": "Negative"
    }

    if "Sentiment" in df.columns:
        df["Sentiment"] = df["Sentiment"].map(mapping)

    df["Clean_Tweet"] = df["OriginalTweet"].apply(clean_text)

    df["AI_Sentiment"] = df["Clean_Tweet"].apply(
        predict_sentiment
    )

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

search = st.sidebar.text_input(
    "🔍 Search Tweet"
)

# =========================
# FILTERS
# =========================
df_filtered = df.copy()

if sentiment_filter != "All":

    df_filtered = df_filtered[
        df_filtered["AI_Sentiment"] == sentiment_filter
    ]

if search:

    df_filtered = df_filtered[
        df_filtered["Clean_Tweet"].str.contains(
            search.lower(),
            na=False,
            regex=False
        )
    ]

# =========================
# KPI
# =========================
sent_counts = df_filtered["AI_Sentiment"].value_counts()

c1, c2, c3 = st.columns(3)

c1.metric(
    "📊 Tweets",
    len(df_filtered)
)

c2.metric(
    "🟢 Positive",
    sent_counts.get("Positive", 0)
)

c3.metric(
    "🔴 Negative",
    sent_counts.get("Negative", 0)
)

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

    st.dataframe(
        df_filtered[
            [
                "OriginalTweet",
                "Clean_Tweet",
                "AI_Sentiment"
            ]
        ].head(20),
        use_container_width=True
    )

# =========================
# TAB 2
# =========================
with tab2:

    st.subheader("☁️ WordCloud Analysis")

    option = st.selectbox(
        "Select sentiment:",
        ["All", "Positive", "Negative", "Neutral"]
    )

    if option != "All":

        data_words = df_filtered[
            df_filtered["AI_Sentiment"] == option
        ]

    else:

        data_words = df_filtered

    text = " ".join(
        data_words["Clean_Tweet"].astype(str)
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

    # =====================
    # TOP WORDS
    # =====================
    st.markdown("### 🔥 Top Words")

    words = text.split()

    common = Counter(words).most_common(15)

    words_df = pd.DataFrame(
        common,
        columns=["Word", "Count"]
    )

    fig2 = px.bar(
        words_df,
        x="Word",
        y="Count",
        title="Most Frequent Words"
    )

    st.plotly_chart(
        fig2,
        use_container_width=True
    )

# =========================
# TAB 3
# =========================
with tab3:

    st.subheader("📊 Sentiment Analytics")

    fig1 = px.pie(
        df_filtered,
        names="AI_Sentiment",
        title="Sentiment Distribution"
    )

    st.plotly_chart(
        fig1,
        use_container_width=True
    )

    fig2 = px.histogram(
        df_filtered,
        x="AI_Sentiment",
        color="AI_Sentiment",
        title="Sentiment Histogram"
    )

    st.plotly_chart(
        fig2,
        use_container_width=True
    )

    df_filtered["length"] = df_filtered[
        "Clean_Tweet"
    ].apply(len)

    fig3 = px.box(
        df_filtered,
        x="AI_Sentiment",
        y="length",
        color="AI_Sentiment",
        title="Tweet Length Analysis"
    )

    st.plotly_chart(
        fig3,
        use_container_width=True
    )

# =========================
# TAB 4
# =========================
with tab4:

    st.subheader(
        "🔮 Live Sentiment Test (AI)"
    )

    user_text = st.text_area(
        "✍️ Enter text:"
    )

    if st.button("Analyze"):

        if user_text.strip() == "":

            st.warning(
                "Please enter text"
            )

        else:

            prediction = predict_sentiment(
                user_text
            )

            if prediction == "Positive":

                st.success(
                    "😊 POSITIVE"
                )

            elif prediction == "Negative":

                st.error(
                    "😡 NEGATIVE"
                )

            else:

                st.info(
                    "😐 NEUTRAL"
                )

# =========================
# FOOTER
# =========================
st.markdown("---")

st.markdown("""
🎓 UCD - FS El Jadida | IA S6 | Data Science  
🦠 Sujet : ML for sentiment analysis: case of Covid 19
""")
