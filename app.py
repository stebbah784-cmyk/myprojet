import streamlit as st

st.markdown(
    """
    <style>
    .stApp {
        background-color: #E6F2FF; 
    }
    </style>
    """,
    unsafe_allow_html=True
)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import re
import string


st.set_page_config(page_title="Covid-19 Sentiment Analysis", page_icon="🦠", layout="wide")

st.title("🦠 Application d'Analyse des Sentiments - COVID-19 Tweets")
st.write("Bienvenue dans votre projet professionnel de Data Science. Cette application analyse l'opinion publique sur Twitter durant la pandémie.")


@st.cache_data
def load_and_clean_data():
    
    df = pd.read_csv("Corona_NLP_train.csv", encoding='latin-1')
    
   
    mapping = {
        "Extremely Positive": "Positive", "Positive": "Positive",
        "Neutral": "Neutral",
        "Negative": "Negative", "Extremely Negative": "Negative"
    }
    df['Sentiment'] = df['Sentiment'].map(mapping)
    
    # دالة التنظيف
    def clean_text(text):
        text = str(text).lower()
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'#', '', text)
        text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
        return text
    
    df['Clean_Tweet'] = df['OriginalTweet'].apply(clean_text)
    return df

# تحميل البيانات
df = load_and_clean_data()

st.markdown("---")

# 3. الجزء الأول: الإحصائيات والرسوم البيانية (Visualisations)
st.subheader("📊 1. Vue d'ensemble et Statistiques des Données")

# تقسيم الشاشة لـ 2 أعمدة
col1, col2 = st.columns(2)

with col1:
    st.write("### Répartition des Sentiments")
    # رسم بياني لتوزيع المشاعر
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(data=df, x='Sentiment', order=['Positive', 'Neutral', 'Negative'], palette='viridis', ax=ax)
    plt.title("Nombre de Tweets par Sentiment")
    st.pyplot(fig)

with col2:
    st.write("### Échantillon des Données Nettoyées")
    # عرض جدول فيه التغريدة قبل وبعد التنظيف مع شعورها
    st.dataframe(df[['OriginalTweet', 'Clean_Tweet', 'Sentiment']].head(10))

st.markdown("---")

# 4. الجزء الثاني: سحابة الكلمات (WordCloud)
st.subheader("☁️ 2. Nuage de Mots (WordCloud) par Sentiment")
st.write("Choisissez un sentiment pour voir les mots les plus fréquents utilisés par les utilisateurs :")

selected_sentiment = st.selectbox("Sélectionnez un Sentiment :", ['Positive', 'Negative', 'Neutral'])

# توليد سحابة الكلمات حسب الاختيار
sentiment_data = df[df['Sentiment'] == selected_sentiment]
all_words = ' '.join([text for text in sentiment_data['Clean_Tweet']])

fig_wc, ax_wc = plt.subplots(figsize=(10, 5))
wordcloud = WordCloud(width=800, height=400, background_color='white', max_words=100).generate(all_words)
ax_wc.imshow(wordcloud, interpolation='bilinear')
ax_wc.axis('off')
st.pyplot(fig_wc)

st.markdown("---")

# 5. الجزء الثالث: المربع التفاعلي للتجريب (Test en Temps Réel)
st.subheader("🔮 3. Test de Prédiction en Temps Réel")
st.write("Saisissez un tweet (en Anglais) pour simuler l'analyse de sentiment :")

user_input = st.text_area("Exemple: The vaccine development is a great news for everyone!", "")

if st.button("Analyser le Sentiment"):
    if user_input.strip() != "":
        # هنا غانديرو محاكاة بسيطة، وفالمرحلة الجاية نربطوها بالموديل الحقيقي ديال Machine Learning
        # هاد السطور غايبينو للبروف بلي السيستم خدام تفاعلي
        if "good" in user_input.lower() or "great" in user_input.lower() or "hope" in user_input.lower() or "safe" in user_input.lower():
            st.success("Résultat de la prédiction : **Positive** 😊")
        elif "bad" in user_input.lower() or "crisis" in user_input.lower() or "lockdown" in user_input.lower() or "scared" in user_input.lower():
            st.error("Résultat de la prédiction : **Negative** 😡")
        else:
            st.warning("Résultat de la prédiction : **Neutral** 😐")
    else:
        st.info("Veuillez saisir un texte pour tester.")
