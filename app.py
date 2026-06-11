import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import string
import pickle

from wordcloud import WordCloud

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# =========================
# LOAD DATA
# =========================
df = pd.read_csv("Corona_NLP_test.csv", encoding="latin-1")

print(df.head())

print(df.info())

print(df.isnull().sum())

# =========================
# SENTIMENT MAPPING
# =========================
mapping = {
    "Extremely Positive": "Positive",
    "Positive": "Positive",
    "Neutral": "Neutral",
    "Negative": "Negative",
    "Extremely Negative": "Negative"
}

df["Sentiment"] = df["Sentiment"].map(mapping)

# =========================
# VISUALIZATION
# =========================
sns.countplot(x='Sentiment', data=df)

plt.title("Sentiment Distribution")

plt.xticks(rotation=45)

plt.show()

# =========================
# WORD CLOUD
# =========================
text = " ".join(df['OriginalTweet'].astype(str))

wordcloud = WordCloud(
    width=1000,
    height=500,
    background_color='white'
).generate(text)

plt.figure(figsize=(12,6))

plt.imshow(wordcloud)

plt.axis("off")

plt.title("Word Cloud")

plt.show()

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
# APPLY CLEANING
# =========================
df['Clean_Tweet'] = df['OriginalTweet'].apply(clean_text)

# =========================
# FEATURES
# =========================
vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(df['Clean_Tweet'])

y = df['Sentiment']

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
# MODEL
# =========================
model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

# =========================
# PREDICTION
# =========================
y_pred = model.predict(X_test)

# =========================
# EVALUATION
# =========================
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)

print(classification_report(y_test, y_pred))

# =========================
# SAVE MODEL
# =========================
pickle.dump(model, open("model.pkl", "wb"))

pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("✅ Model & Vectorizer saved!")
