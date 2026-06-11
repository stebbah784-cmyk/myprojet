import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import pickle

from wordcloud import WordCloud

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# =========================
# LOAD DATASET
# =========================
df = pd.read_csv("Corona_NLP_train.csv")

print(df.head())
print(df.info())

# =========================
# CHECK MISSING VALUES
# =========================
print(df.isnull().sum())

# =========================
# MERGE 5 CLASSES INTO 3
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
plt.figure(figsize=(8, 5))
sns.countplot(x="Sentiment", data=df)
plt.title("Sentiment Distribution")
plt.show()

# =========================
# WORD CLOUD
# =========================
text = " ".join(df["OriginalTweet"].astype(str))

wordcloud = WordCloud(
    width=1000,
    height=500,
    background_color="white"
).generate(text)

plt.figure(figsize=(12, 6))
plt.imshow(wordcloud)
plt.axis("off")
plt.show()

# =========================
# TEXT CLEANING
# =========================
def clean_text(text):

    text = str(text).lower()

    text = re.sub(r"http\S+", "", text)

    text = re.sub(r"@\w+", "", text)

    text = re.sub(r"#", "", text)

    text = re.sub(r"[^a-zA-Z\s]", "", text)

    text = re.sub(r"\s+", " ", text).strip()

    return text

df["Clean_Tweet"] = df["OriginalTweet"].apply(clean_text)

# =========================
# FEATURES & TARGET
# =========================
X_text = df["Clean_Tweet"]

y = df["Sentiment"]

# =========================
# TF-IDF
# =========================
vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=10000
)

X = vectorizer.fit_transform(X_text)

# =========================
# TRAIN TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =========================
# MODEL
# =========================
model = LogisticRegression(
    max_iter=1000
)

model.fit(X_train, y_train)

# =========================
# EVALUATION
# =========================
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy =", accuracy)

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

print("\nClasses learned by model:")
print(model.classes_)

# =========================
# QUICK TEST
# =========================
samples = [
    "very nice",
    "excellent work",
    "i love this",
    "terrible",
    "very bad",
    "awful"
]

print("\nQuick Test:\n")

for s in samples:
    x = vectorizer.transform([clean_text(s)])
    pred = model.predict(x)[0]
    print(f"{s} ---> {pred}")

# =========================
# SAVE MODEL
# =========================
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("\n✅ model.pkl saved")
print("✅ vectorizer.pkl saved")
