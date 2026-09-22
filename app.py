import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import re
import string

from wordcloud import WordCloud

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Université Chouaib Doukkali - Faculté des Sciences El Jadida",
    page_icon="🎓",
    layout="wide"
)

# =========================
# TITLE
# =========================
st.title("ML for Sentiment Analysis : Case of Covid-19")

st.subheader("Filière : Informatique Appliquée (S6)")

st.markdown("""
<style>

.main {
    background-color:#0f172a;
    color:white;
}

h1 {
    color:#38bdf8;
    text-align:center;
}

.stTabs [data-baseweb="tab"] {
    font-size:16px;
}

</style>
""", unsafe_allow_html=True)

st.markdown("### Réalisée par : Wakil Chaimae , Stebba Houda")

# =========================
# CLEAN TEXT
# =========================
def clean_text(text):

    text = str(text).lower()

    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#", "", text)

    text = re.sub(
        r"[%s]" % re.escape(string.punctuation),
        "",
        text
    )

    text = re.sub(r"[^a-zA-Z\s]", "", text)

    return text


# =========================
# LOAD DATA
# =========================
@st.cache_data
def load_data():

    df = pd.read_csv(
        "Corona_NLP_test.csv",
        encoding="latin-1"
    )

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
# VECTORIZE
# =========================
vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(df["Clean_Tweet"])

y = df["Sentiment"]

# =========================
# SPLIT DATA
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# =========================
# TRAIN MODEL
# =========================
model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# =========================
# SIDEBAR
# =========================
st.sidebar.title("⚙️ Filter")

sent_filter = st.sidebar.selectbox(
    "Choose Sentiment",
    ["All", "Positive", "Neutral", "Negative"]
)

df_view = df.copy()

if sent_filter != "All":

    df_view = df_view[
        df_view["Sentiment"] == sent_filter
    ]

# =========================
# TABS
# =========================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Data",
    "📈 Visualization",
    "💬 Live Test",
    "📉 Metrics",
    "📡 ROC"
])

# =========================
# TAB 1 - DATASET
# =========================
with tab1:

    st.subheader("Dataset")

    st.dataframe(
        df_view[
            ["OriginalTweet", "Sentiment"]
        ].head(20)
    )

    st.markdown("""
### 📌 Définition

Ce dataset contient des tweets liés au COVID-19.

Les émotions sont classées en :

- Positive
- Neutral
- Negative
""")

# =========================
# TAB 2 - VISUALIZATION
# =========================
with tab2:

    # =====================
    # PIE CHART
    # =====================
    st.subheader("Sentiment Distribution")

    fig = px.pie(
        df_view,
        names="Sentiment",
        hole=0.4
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.markdown("""
### 📌 Définition

Ce graphique montre la répartition des sentiments dans le dataset.
""")

    # =====================
    # WORD CLOUD
    # =====================
    st.subheader("Most Repeated Words by Sentiment")

    selected_sentiment = st.selectbox(
        "Choose Sentiment for WordCloud",
        ["Positive", "Neutral", "Negative"]
    )

    text_data = " ".join(
        df[
            df["Sentiment"] == selected_sentiment
        ]["Clean_Tweet"]
    )

    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color="black",
        colormap="cool"
    ).generate(text_data)

    fig_wc, ax = plt.subplots(figsize=(10, 5))

    ax.imshow(
        wordcloud,
        interpolation="bilinear"
    )

    ax.axis("off")

    st.pyplot(fig_wc)

    st.markdown(f"""
📌 Cette image montre les mots les plus fréquents dans les tweets **{selected_sentiment}**.

Les mots les plus grands sont les plus répétés.
""")

# =========================
# TAB 3 - LIVE TEST
# =========================
with tab3:

    st.subheader("Test du modèle")

    text = st.text_area(
        "Entrez un texte"
    )

    if st.button("Predict"):

        cleaned = clean_text(text)

        vector = vectorizer.transform(
            [cleaned]
        )

        pred = model.predict(vector)[0]

        st.success(
            f"Résultat : {pred}"
        )

    st.markdown("""
### 📌 Définition

Le modèle prédit le sentiment d’un texte
en utilisant TF-IDF + Logistic Regression.
""")

# =========================
# TAB 4 - METRICS
# =========================
with tab4:

    st.subheader("Performance du modèle")

    acc = accuracy_score(
        y_test,
        y_pred
    )

    st.metric(
        "Accuracy",
        f"{acc:.2f}"
    )

    # =====================
    # CLASSIFICATION REPORT
    # =====================
    report = classification_report(
        y_test,
        y_pred,
        output_dict=True
    )

    df_metrics = pd.DataFrame({

        "Classe": [
            "Negative",
            "Neutral",
            "Positive"
        ],

        "Precision": [
            report["Negative"]["precision"],
            report["Neutral"]["precision"],
            report["Positive"]["precision"]
        ],

        "Recall": [
            report["Negative"]["recall"],
            report["Neutral"]["recall"],
            report["Positive"]["recall"]
        ],

        "F1-score": [
            report["Negative"]["f1-score"],
            report["Neutral"]["f1-score"],
            report["Positive"]["f1-score"]
        ]
    })

    st.subheader("Table des scores")

    st.dataframe(df_metrics)

    # =====================
    # BAR CHART
    # =====================
    df_melt = df_metrics.melt(
        id_vars="Classe",
        value_vars=[
            "Precision",
            "Recall",
            "F1-score"
        ],
        var_name="Métrique",
        value_name="Score"
    )

    fig_bar = px.bar(
        df_melt,
        x="Classe",
        y="Score",
        color="Métrique",
        barmode="group"
    )

    st.plotly_chart(
        fig_bar,
        use_container_width=True
    )

    st.markdown("""
### 📌 Définitions

- Precision : qualité des prédictions positives
- Recall : capacité à détecter tous les vrais cas
- F1-score : équilibre entre précision et rappel
- Accuracy : taux global de bonnes prédictions
""")

    # =====================
    # CONFUSION MATRIX
    # =====================
    st.subheader("Matrice de confusion")

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=model.classes_
    )

    fig_cm, ax = plt.subplots()

    ConfusionMatrixDisplay(
        cm,
        display_labels=model.classes_
    ).plot(
        ax=ax,
        cmap="Blues"
    )

    st.pyplot(fig_cm)

    st.markdown("""
### 📌 Définition

La matrice de confusion montre les bonnes
et mauvaises prédictions du modèle.
""")

# =========================
# TAB 5 - ROC CURVE
# =========================
with tab5:

    st.subheader("Courbe ROC")

    classes = model.classes_

    y_bin = label_binarize(
        y_test,
        classes=classes
    )

    y_score = model.predict_proba(
        X_test
    )

    plt.figure()

    for i in range(len(classes)):

        fpr, tpr, _ = roc_curve(
            y_bin[:, i],
            y_score[:, i]
        )

        plt.plot(
            fpr,
            tpr,
            label=classes[i]
        )

    plt.plot([0, 1], [0, 1], "--")

    plt.legend()

    plt.title("ROC Curve")

    st.pyplot(plt)

    st.markdown("""
### 📌 Définition

La courbe ROC mesure la capacité du modèle
à distinguer les classes.

Plus elle est proche du coin supérieur gauche,
meilleur est le modèle.
""")
