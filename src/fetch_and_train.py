import requests
import psycopg2
import os
import re
import string
import pandas as pd
import numpy as np
import pickle
import nltk
from dotenv import load_dotenv
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import cosine_similarity

nltk.download("stopwords")
nltk.download("punkt")
nltk.download("punkt_tab")


load_dotenv()

API_KEY = os.getenv("NEWS_API_KEY")
NEWS_API_URL = f"https://newsapi.org/v2/top-headlines?country=us&category=technology&apiKey={API_KEY}"

DB_USER = os.getenv("POSTGRES_USER")
DB_PASSWORD = os.getenv("POSTGRES_PASSWORD")
DB_HOST = os.getenv("POSTGRES_HOST")
DB_NAME = os.getenv("POSTGRES_NAME")
DB_PORT = os.getenv("POSTGRES_PORT")

stop_words = set(stopwords.words("english"))

def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)

response = requests.get(NEWS_API_URL)
if response.status_code == 200:
    data = response.json()
    if "articles" in data:
        articles = data["articles"]
    else:
        print("No articles found in API response!")
        exit(1)
else:
    print(f"Failed to fetch news! API Response Code: {response.status_code}")
    print(f"API Response Text: {response.text}")
    exit(1)

conn = psycopg2.connect(
    dbname=DB_NAME,
    user=DB_USER,
    password=DB_PASSWORD,
    host=DB_HOST,
    port=DB_PORT
)
cursor = conn.cursor()

for article in articles:
    cursor.execute(
        """
        INSERT INTO news (title, description, publishedAt, source)
        VALUES (%s, %s, %s, %s)
        ON CONFLICT (title) DO NOTHING;
        """,
        (article["title"], article["description"], article["publishedAt"], article["source"]["name"])
    )

conn.commit()
cursor.close()
conn.close()
print("News data inserted into PostgreSQL!")


conn = psycopg2.connect(
    dbname=DB_NAME,
    user=DB_USER,
    password=DB_PASSWORD,
    host=DB_HOST,
    port=DB_PORT
)
df = pd.read_sql("SELECT * FROM news", conn)
conn.close()

df = df.dropna(subset=["title"])
df["source"] = df["source"].apply(lambda x: x if isinstance(x, str) else "Unknown")

CATEGORY_RULES = {
    "Gaming": ["IGN", "GameSpot", "Polygon", "Rock Paper Shotgun", "Push Square", "Eurogamer", "Gematsu", "Blizzard"],
    "Tech": ["The Verge", "9to5google", "MacRumors", "Ars Technica", "UploadVR", "Gizmodo"],
    "Finance": ["Bloomberg"],
    "Science": ["Space.com"],
}

def categorize_source(source_name):
    if source_name is None or source_name == "Unknown":
        return None

    for category, keywords in CATEGORY_RULES.items():
        if any(re.search(keyword, source_name, re.IGNORECASE) for keyword in keywords):
            return category

    return None


df["source_category"] = df["source"].apply(categorize_source)
unknown_sources = df[df["source_category"].isna()]["source"].unique()

if len(unknown_sources) > 0:
    print(f"Found {len(unknown_sources)} new sources. Categorizing dynamically...")

    vectorizer = TfidfVectorizer(stop_words="english")
    known_sources = list(CATEGORY_RULES.keys())
    all_sources = known_sources + list(unknown_sources)

    X = vectorizer.fit_transform(all_sources)

    num_clusters = min(len(unknown_sources), 5)
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    labels = kmeans.fit_predict(X[len(known_sources):])

    known_vectors = X[:len(known_sources)]
    unknown_vectors = X[len(known_sources):]

    for i, source in enumerate(unknown_sources):
        similarities = cosine_similarity(unknown_vectors[i], known_vectors)
        best_match = known_sources[np.argmax(similarities)]
        print(f"🔄 Assigning {source} → {best_match}")
        df.loc[df["source"] == source, "source_category"] = best_match

df["source_category"] = df["source_category"].fillna("Other")

vectorizer = TfidfVectorizer(stop_words="english", max_features=5000, ngram_range=(1,2))
X = vectorizer.fit_transform(df["title"])
y = df["source_category"].astype(str)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

models = {
    "Naive Bayes": {"model": MultinomialNB(), "params": {"alpha": [0.1, 0.5, 1.0]}},
    "Logistic Regression": {"model": LogisticRegression(max_iter=2000), "params": {"C": [0.1, 1, 10]}},
    "Random Forest": {"model": RandomForestClassifier(), "params": {"n_estimators": [50, 100, 200], "max_depth": [5, 10, 20]}},
    "SVM": {"model": SVC(), "params": {"C": [0.1, 1, 10], "kernel": ["linear", "rbf"]}},
}

best_model = None
best_score = 0
best_model_name = ""

for model_name, model_info in models.items():
    print(f"Training {model_name}...")
    
    try:
        grid_search = GridSearchCV(model_info["model"], model_info["params"], cv=3, scoring="accuracy")
        grid_search.fit(X_train, y_train)

        best_model_for_this = grid_search.best_estimator_
        accuracy = accuracy_score(y_test, best_model_for_this.predict(X_test))

        print(f"{model_name} Best Accuracy: {accuracy:.2f}")

        if accuracy > best_score:
            best_score = accuracy
            best_model = best_model_for_this
            best_model_name = model_name

    except Exception as e:
        print(f"Error training {model_name}: {e}")

if best_model:
    with open("/app/models/best_news_classifier.pkl", "wb") as model_file:
        pickle.dump(best_model, model_file)

    with open("/app/models/tfidf_vectorizer.pkl", "wb") as vectorizer_file:
        pickle.dump(vectorizer, vectorizer_file)

    print(f"Best Model: {best_model_name} with Accuracy: {best_score:.2f}")
    print("Model trained and saved using pickle!")

else:
    print("No model was successfully trained!")
