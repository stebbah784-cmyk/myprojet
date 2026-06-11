import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
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
    ConfusionMatrixDisplay,
    roc_curve,
    auc
)

from sklearn.preprocessing import label_binarize

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="UCD - IA S6 Sentiment Dashboard",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 UCD Sentiment Analysis Dashboard")
st.markdown("🦠 ML Project - Covid-19 Sentiment Analysis")

st.markdown("---")

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
    return text

# =========================
# PREDICT
# =========================
def predict_sentiment(text):
    cleaned = clean_text(text)

    if cleaned.strip() == "":
        return "Neutral"

    vector = vectorizer.transform([cleaned])
    return model.predict(vector)[0]

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
# SIDEBAR
# =========================
st.sidebar.header("Filters")

sentiment_filter = st.sidebar.selectbox(
    "Sentiment",
    ["All", "Positive", "Neutral", "Negative"]
)

df_filtered = df.copy()

if sentiment_filter != "All":
    df_filtered = df_filtered[
        df_filtered["AI_Sentiment"].str.lower() == sentiment_filter.lower()
    ]

# =========================
# TABS
# =========================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Data",
    "Visualization",
    "Live Test",
    "Evaluation",
    "Advanced ML"
])

# =========================
# TAB 1
# =========================
with tab1:
    st.dataframe(df_filtered[["OriginalTweet", "Sentiment", "AI_Sentiment"]].head(20))

# =========================
# TAB 2
# =========================
with tab2:
    fig = px.pie(df_filtered, names="AI_Sentiment")
    st.plotly_chart(fig, use_container_width=True)

# =========================
# TAB 3
# =========================
with tab3:
    text = st.text_area("Enter text")

    if st.button("Predict"):
        pred = predict_sentiment(text)
        st.write("Prediction:", pred)

# =========================
# TAB 4 - METRICS + CONFUSION MATRIX
# =========================
with tab4:

    st.subheader("Model Metrics")

    y_true = df["Sentiment"]
    y_pred = df["AI_Sentiment"]

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    metrics = pd.DataFrame({
        "Metric": ["Accuracy", "Precision", "Recall", "F1"],
        "Score": [acc, prec, rec, f1]
    })

    st.dataframe(metrics)

    fig = px.bar(metrics, x="Metric", y="Score", color="Metric")
    st.plotly_chart(fig, use_container_width=True)

    # Confusion Matrix
    st.subheader("Confusion Matrix")

    labels = model.classes_   # ✅ FIX IMPORTANT

    cm = confusion_matrix(y_true, y_pred, labels=labels)

    fig2, ax = plt.subplots()
    ConfusionMatrixDisplay(cm, display_labels=labels).plot(ax=ax, cmap="Blues")

    st.pyplot(fig2)

# =========================
# TAB 5 - ROC FIXED
# =========================
with tab5:

    st.subheader("ROC Curve + AUC (FIXED)")

    # ✅ IMPORTANT FIX: use model.classes_
    classes = model.classes_

    y_true_bin = label_binarize(df["Sentiment"], classes=classes)

    y_score = model.predict_proba(vectorizer.transform(df["Clean_Tweet"]))

    fig = plt.figure()

    for i in range(len(classes)):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_score[:, i])
        roc_auc = auc(fpr, tpr)

        plt.plot(fpr, tpr, label=f"{classes[i]} (AUC={roc_auc:.2f})")

    plt.plot([0, 1], [0, 1], "k--")
    plt.legend()
    plt.title("ROC Curve (Corrected)")

    st.pyplot(fig)

    # =========================
    # MISCLASSIFIED
    # =========================
    st.subheader("Misclassified Tweets")

    df_errors = df[df["Sentiment"] != df["AI_Sentiment"]]

    st.write("Errors:", len(df_errors))

    st.dataframe(df_errors[["OriginalTweet", "Sentiment", "AI_Sentiment"]].head(20))
