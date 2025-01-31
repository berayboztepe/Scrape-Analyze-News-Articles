from flask import Flask, jsonify, render_template
import psycopg2
import os
from dotenv import load_dotenv
import pickle

app = Flask(__name__)
load_dotenv()

DB_USER = os.getenv("POSTGRES_USER")
DB_PASSWORD = os.getenv("POSTGRES_PASSWORD")
DB_HOST = os.getenv("POSTGRES_HOST")
DB_NAME = os.getenv("POSTGRES_NAME")
DB_PORT = os.getenv("POSTGRES_PORT")

MODEL_DIR = "/app/models"

os.makedirs(MODEL_DIR, exist_ok=True)

best_model_path = os.path.join(MODEL_DIR, "best_news_classifier.pkl")
vectorizer_path = os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")

with open(best_model_path, "rb") as model_file:
    model = pickle.load(model_file)

with open(vectorizer_path, "rb") as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

def get_db_connection():
    """Establish database connection."""
    conn = psycopg2.connect(
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT
    )
    return conn

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/news", methods=["GET"])
def get_news():
    """Fetch all news articles from PostgreSQL."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute("SELECT id, title, description, publishedAt, source FROM news")
        news = cursor.fetchall()

        news_list = [
            {"id": row[0], "title": row[1], "description": row[2], "publishedAt": row[3], "source": row[4]}
            for row in news
        ]

        cursor.close()
        conn.close()

        return jsonify(news_list), 200, {"Content-Type": "application/json; charset=utf-8"}

    except Exception as e:
        return jsonify({"error": f"Failed to fetch news: {str(e)}"}), 500

@app.route("/predict/<headline>")
def predict(headline):
    """Predict news category for a given headline."""
    X_input = vectorizer.transform([headline])
    prediction = model.predict(X_input)[0]
    return jsonify({"prediction": prediction})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
