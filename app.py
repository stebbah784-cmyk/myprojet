import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import re
import string

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc
)

from sklearn.preprocessing import label_binarize

# =========================
# PAGE STYLE
# =========================
st.set_page_config(
    page_title="UCD Sentiment AI",
    page_icon="🎓",
    layout="wide"
)

st.markdown("""
    <style>
    .main {
        background-color: #0f172a;
        color: white;
    }
    h1 {
        color: #38bdf8;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🎓 Sentiment Analysis Dashboard (UCD Project)")

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

    return df

df = load_data()

# =========================
# TRAIN MODEL
# =========================
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["Clean_Tweet"])
y = df["Sentiment"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# =========================
# SIDEBAR FILTER
# =========================
st.sidebar.title("⚙️ Filters")

sentiment_filter = st.sidebar.selectbox(
    "Choose Sentiment",
    ["All", "Positive", "Neutral", "Negative"]
)

df_view = df.copy()

if sentiment_filter != "All":
    df_view = df_view[df_view["Sentiment"] == sentiment_filter]

# =========================
# TABS
# =========================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Data",
    "📈 Visualization",
    "💬 Live Prediction",
    "📉 Metrics",
    "📡 ROC Curve"
])

# =========================
# TAB 1
# =========================
with tab1:
    st.subheader("Dataset Preview")
    st.dataframe(df_view[["OriginalTweet", "Sentiment"]].head(20))

# =========================
# TAB 2
# =========================
with tab2:
    st.subheader("Sentiment Distribution")

    fig = px.pie(df_view, names="Sentiment", hole=0.4)
    st.plotly_chart(fig, use_container_width=True)

# =========================
# TAB 3
# =========================
with tab3:
    st.subheader("Try Your Text")

    text = st.text_area("Write something...")

    if st.button("Predict"):
        cleaned = clean_text(text)
        vector = vectorizer.transform([cleaned])
        pred = model.predict(vector)[0]

        st.success(f"Prediction: {pred}")

# =========================
# TAB 4
# =========================
with tab4:
    st.subheader("Model Performance")

    acc = accuracy_score(y_test, y_pred)
    st.metric("Accuracy", f"{acc:.2f}")

    st.text(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay(cm, display_labels=model.classes_).plot(ax=ax)
    st.pyplot(fig)

# =========================
# TAB 5
# =========================
with tab5:
    st.subheader("ROC Curve")

    classes = model.classes_
    y_bin = label_binarize(y_test, classes=classes)
    y_score = model.predict_proba(X_test)

    plt.figure()

    for i in range(len(classes)):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_score[:, i])
        plt.plot(fpr, tpr, label=classes[i])

    plt.plot([0, 1], [0, 1], "--")
    plt.legend()
    plt.title("ROC Curve")

    st.pyplot(plt)
