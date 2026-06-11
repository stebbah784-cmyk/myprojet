import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import pickle

from wordcloud import WordCloud

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report

# =========================
# LOAD DATA
# =========================
df = pd.read_csv("Corona_NLP_train.csv")

# =========================
# MERGE TO 3 CLASSES
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
# CLEANING
# =========================
def clean_text(text):

    text = str(text).lower()

    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#", "", text)

    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    text = re.sub(r"\s+", " ", text).strip()

    return text

df["Clean_Tweet"] = df["OriginalTweet"].apply(clean_text)

# =========================
# FEATURES
# =========================
X_text = df["Clean_Tweet"]
y = df["Sentiment"]

vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=15000,
    ngram_range=(1, 2)
)

X = vectorizer.fit_transform(X_text)

# =========================
# SPLIT
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
model = LinearSVC()

model.fit(X_train, y_train)

# =========================
# EVALUATION
# =========================
y_pred = model.predict(X_test)

print("Accuracy =", accuracy_score(y_test, y_pred))

print(classification_report(y_test, y_pred))

print("Classes:")
print(model.classes_)

# =========================
# TESTS
# =========================
tests = [
    "very nice",
    "excellent",
    "amazing",
    "i love it",
    "good job",
    "bad",
    "terrible",
    "awful",
    "i hate it",
    "dislike",
    "normal day",
    "nothing special"
]

print("\n===== TESTS =====")

for t in tests:
    pred = model.predict(
        vectorizer.transform([clean_text(t)])
    )[0]

    print(f"{t} --> {pred}")

# =========================
# SAVE
# =========================
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("✅ model.pkl saved")
print("✅ vectorizer.pkl saved")
