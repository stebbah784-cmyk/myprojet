import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re

from wordcloud import WordCloud

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv("Corona_NLP_train.csv")
df.head()
df.info()
df.isnull().sum()

sns.countplot(x='Sentiment', data=df)
plt.xticks(rotation=45)
plt.show()

text = " ".join(df['OriginalTweet'])

wordcloud = WordCloud(width=800,height=400).generate(text)

plt.imshow(wordcloud)
plt.axis("off")
plt.show()

def clean_text(text):

    text = text.lower()

    text = re.sub(r"http\S+", "", text)

    text = re.sub(r"@\w+", "", text)

    text = re.sub(r"#", "", text)

    text = re.sub(r"[^a-zA-Z\s]", "", text)

    return text

df['Clean_Tweet'] = df['OriginalTweet'].apply(clean_text)

vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(df['Clean_Tweet'])

y = df['Sentiment']

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = LogisticRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy_score(y_test, y_pred)

print(classification_report(y_test, y_pred))

import pickle

pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))
print("✅ Model & Vectorizer saved!")


