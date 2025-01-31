import streamlit as st
import requests
from collections import Counter
import string
import nltk
from nltk.corpus import stopwords
from collections import Counter
import pickle

nltk.download("stopwords")
STOPWORDS = set(stopwords.words("english"))

MODEL_PATH = "/app/models/best_news_classifier.pkl"
VECTORIZER_PATH = "/app/models/tfidf_vectorizer.pkl"

FLASK_API_URL = "http://flask:5000/news"

st.title("📰 News Analysis Dashboard")

def clean_text(text):
    """Remove punctuation and stopwords from text."""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    words = text.split()
    words = [word for word in words if word.isalpha() and word not in STOPWORDS]
    return words

try:
    response = requests.get(FLASK_API_URL)
    
    if response.status_code == 200:
        news = response.json()

        st.subheader("📃 Latest News Articles")
        for article in news:
            st.write(f"### {article['title']}")
            st.write(f"_{article['description']}_")
            st.write(f"📅 Published: {article['publishedAt']} | 🏢 Source: {article['source']}")
            st.write("---")

        all_words = " ".join([article["title"] for article in news])
        words = clean_text(all_words)
        word_freq = Counter(words)
        top_words = dict(word_freq.most_common(10))

        st.subheader("🔍 Most Common Words in News Headlines")
        st.bar_chart(top_words)

    else:
        st.error(f"❌ Failed to fetch news! API returned status code: {response.status_code}")

except requests.exceptions.RequestException as e:
    st.error(f"🚨 API Error: {str(e)}")

with open(MODEL_PATH, "rb") as model_file:
    model = pickle.load(model_file)

with open(VECTORIZER_PATH, "rb") as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

st.title("📰 News Analysis & Prediction Dashboard")

st.subheader("🔮 Predict News Source in Real-Time")

user_input = st.text_input("Type a news headline:")

if user_input:
    X_input = vectorizer.transform([user_input])
    prediction = model.predict(X_input)[0]
    st.success(f"🏢 Predicted Source Category: **{prediction}**")
else:
    st.warning("⚠️ Start typing a headline to see predictions in real-time.")