# 📰 Scrape & Analyze News Articles

## 📌 Project Overview
This project automates the process of **scraping, analyzing, and classifying news articles** using **Flask, Streamlit, PostgreSQL, and Machine Learning models** inside Docker containers. The system:

✅ **Fetches news** from an external API (NewsAPI)

✅ **Saves data** into PostgreSQL

✅ **Cleans & preprocesses** the text

✅ **Classifies news sources** using ML models

✅ **Finds the best-performing model** dynamically

✅ **Saves the trained model** for real-time predictions in Streamlit

✅ **Runs inside Docker for full automation**

**It does that everytime you run the docker. So, it always stays up-to-date**

---

## ⚙️ Technologies Used
- **Python** (NLTK, Pandas, Scikit-learn)
- **Flask** (REST API backend)
- **Streamlit** (Dashboard UI)
- **PostgreSQL** (Database for news storage)
- **Docker** (Containerized development)
- **NewsAPI** (External news source) (NewsAPI has a 1,000 requests/day limit on the free plan)

---

## 🏗️ Project Structure
```
Scrape_Analyse_News_Articles/
│── data/                     # Raw & processed data storage
│── database/                 # SQL scripts for DB setup
│── frontend/                 # Streamlit app
│── models/                   # Trained ML models stored here
│── notebooks/                # Jupyter notebooks for EDA (get to know with the data)
│── src/                      # Main source code
│   ├── static/               # CSS & frontend assets
│   ├── templates/            # HTML templates
│   ├── app.py                # Flask API
│   ├── fetch_and_train.py    # Fetch, train & classify news
│── .env                      # Environment variables
│── docker-compose.yml        # Multi-container Docker setup
│── Dockerfile-flask          # Flask app Dockerfile
│── Dockerfile-streamlit      # Streamlit app Dockerfile
│── Dockerfile-fetcher        # Data fetcher Dockerfile
│── requirements.txt          # Python dependencies
│── README.md                 # Documentation (You are here)
```

---

## 🚀 How It Works
### **1️⃣ Fetching News & Storing in PostgreSQL**
- `fetch_and_train.py` pulls articles from NewsAPI
- Articles are **cleaned, preprocessed, and inserted into PostgreSQL**

### **2️⃣ Automatic News Classification**
- If a **new news source appears**, it will be **categorized dynamically**
- Uses **TF-IDF vectorization** & **KMeans clustering** to determine its category
- It does not work well, but worth to try (can be improved)

Some Examples of Classification:
Found 11 new sources. Categorizing dynamically...

🔄 Assigning Forbes → Gaming

🔄 Assigning Nintendo Life → Gaming

🔄 Assigning Dexerto → Gaming

🔄 Assigning Kotaku → Gaming

🔄 Assigning 9to5Mac → Gaming

🔄 Assigning CNET → Gaming      

🔄 Assigning SamMobile → Gaming      

🔄 Assigning Motley Fool → Gaming    

🔄 Assigning TechRadar → Gaming

🔄 Assigning Videocardz.com → Gaming   

🔄 Assigning Jalopnik → Gaming

### **3️⃣ Training & Selecting the Best ML Model**
- Compares **Naive Bayes, Logistic Regression, Random Forest & SVM**
- Uses **GridSearchCV** to find optimal hyperparameters
- Saves the **best-performing model** as `best_news_classifier.pkl`

### **4️⃣ Flask API**
- Serves data from PostgreSQL
- Exposes endpoints for fetching **JSON-formatted news**

### **5️⃣ Streamlit Dashboard**
- Loads the trained ML model
- Accepts user input (news headlines)
- Predicts the **news source category in real-time**

---

## 🐳 Running the Project with Docker
### **1️⃣ Setup Environment Variables**
Create a `.env` file in the root directory:
```ini
NEWS_API_KEY=your_api_key_here
POSTGRES_USER=myuser
POSTGRES_PASSWORD=mypassword
POSTGRES_NAME=newsdb
POSTGRES_HOST=db
POSTGRES_PORT=5432
```

### **2️⃣ Build & Start Containers**
```bash
docker-compose up --build
```
This will start:
- **Flask API** (localhost:5000)
- **PostgreSQL DB**
- **Fetcher (news & ML training)**
- **Streamlit UI** (localhost:8501)

### **3️⃣ Check Running Containers**
```bash
docker ps
```

### **4️⃣ Stop & Remove Containers**
```bash
docker-compose down
```

### **5️⃣ Clean Up**
```bash
docker image prune -f
```

---

## 🔥 API Endpoints
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Returns the home page |
| `/news` | GET | Fetches all stored news articles |
| `/predict` | POST | Predicts news category based on title |

---

## 🎨 Streamlit Dashboard
- Go to **http://localhost:8501**
- Enter a **news headline**
- Get **real-time predictions** based on trained ML models

---

## Testing

- You can check both Flask API or Streamlit to check recent popular news headlines and the source
- For Flask API test, try: **localhost:5000/predict/your-own-headline**
- Some random generated news headlines for testing:

🔹 Tech News Headlines

1️⃣ Apple Unveils AI-Powered iPhone 16 at Global Event

2️⃣ Google’s Quantum Computer Solves Problem in 3 Seconds

3️⃣ Tesla Launches Fully Autonomous Cybertruck in 2025

4️⃣ Meta Introduces AI Avatars for Virtual Meetings

5️⃣ Elon Musk’s Starlink Reaches 10 Million Users Worldwide

🎮 Gaming News Headlines

6️⃣ GTA 6 Trailer Drops, Release Date Set for 2026

7️⃣ Sony Announces PlayStation 6 With 8K Ray Tracing

8️⃣ Elden Ring DLC “Shadow of Erdtree” Gets a New Trailer

9️⃣ Nintendo Switch 2 Rumors Point to Late 2024 Launch

🔟 Minecraft Surpasses 500 Million Copies Sold Worldwide

💰 Finance & Economy News Headlines

1️⃣1️⃣ Bitcoin Hits Record High of $150,000 After ETF Approval

1️⃣2️⃣ Stock Market Surges as Inflation Fears Ease

1️⃣3️⃣ Federal Reserve Cuts Interest Rates for the First Time in 5 Years

1️⃣4️⃣ Amazon Reports $1 Trillion in Revenue for 2024

1️⃣5️⃣ Gold Prices Soar as Global Economic Uncertainty Grows

🔬 Science & Space News Headlines

1️⃣6️⃣ NASA’s Artemis Mission Successfully Lands Astronauts on the Moon

1️⃣7️⃣ Scientists Discover New Exoplanet That Could Support Life

1️⃣8️⃣ Breakthrough in Nuclear Fusion Brings Clean Energy Closer

1️⃣9️⃣ AI-Powered Robots to Assist in Future Mars Missions

2️⃣0️⃣ James Webb Telescope Detects Signs of Water on Distant Planet

The results are not good enough for having lack of labeled data. Since the latest news were all related to either gaming or tech, testing works better in these categories than others.

## 🛠️ Future Improvements
🔹 Expand ML models with **deep learning (LSTMs, BERT)**  
🔹 Add **more news sources & categories** dynamically  
🔹 Improve **data visualization & analytics** in Streamlit  
🔹 Deploy the app **to a cloud platform**  

---

## 👨‍💻 Contributors
- **Emre Beray Boztepe**  

---

## 📝 License
This project is **open-source** and free to use under the MIT License.

