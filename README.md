# 🎵 Song Popularity Prediction

Predict whether a song will become **popular on Spotify** using audio features and genre.  
This project was developed as a **final project** for a Data Science & Machine Learning Bootcamp.

---

## 📂 Project Structure
```
.
├── main.ipynb               # Full EDA, feature engineering, model training & tuning
├── dataset.csv              # Cleaned dataset (Spotify songs)
├── streamlit_app.py         # Interactive web app
├── Requirements.txt         # Python dependencies
├── models/                  # Saved trained models (.pkl)
└── README.md                # Project documentation
```

---

## 🎯 Goal
Predict the probability that a given song will be **popular (1)** or **not popular (0)** based on its audio characteristics.

- The target variable `popularity` originally ranged from **0–100**.
- It was **binarized**:
  - **1 (popular)** if `popularity ≥ 75`
  - **0 (not popular)** otherwise.

---

## 🗂️ Dataset
- Source: **Spotify tracks dataset** (30k+ songs).
- Final features used for modeling:
  - `danceability`, `energy`, `loudness`, `speechiness`, `acousticness`,
    `instrumentalness`, `liveness`, `valence`, `tempo`,
    `track_genre_encoded`, `valence_energy`, `energy_danceability`.
- New engineered features:
  - `valence_energy` (valence × energy)
  - `energy_danceability` (energy × danceability)

---

## ⚙️ Project Steps

### 1. Data Preparation & EDA
- Cleaned and explored the dataset.
- Dealt with outliers and missing values.
- Created interaction features to capture mood and dance potential.

### 2. Target Transformation
- Converted `popularity` from a continuous 0–100 scale to binary classification as described above.

### 3. Modeling
- Baseline: **Random Forest Classifier** (Train Acc ~0.81, Test Acc ~0.81).
- Hyperparameter tuning with:
  - `RandomizedSearchCV` (broad search)
  - `GridSearchCV` (fine tuning)
- Final models trained:
  - **RandomForest** (best performer)
  - **XGBoost**
  - **CatBoost**
  - **Ensemble Voting Classifier**

### 4. Evaluation
- Metrics used: Accuracy, Balanced Accuracy, Weighted F1, ROC AUC.
- Random Forest achieved:
  - **Train Accuracy**: 0.99
  - **Test Accuracy**: 0.98
  - **Macro Avg Precision**: 0.76
  - **Macro Avg Recall**: 0.69
  - **Macro Avg F1-Score**: 0.72

### 5. Interactive Web App (Streamlit)
- Built an easy-to-use interface where anyone can:
  - Enter audio features and genre (or adjust sliders).
  - Choose a model and decision threshold.
  - Instantly get the probability of a song being popular.
- Tabs for:
  - **Predict** – make live predictions.
  - **Models** – compare metrics and bar charts.
  - **Explain** – view feature importances.

---

## 🔍 Key Insights
- **Genre** (`track_genre_encoded`) is the most powerful predictor of popularity.
- Features like **instrumentalness**, **acousticness**, **energy**, and **danceability** also have strong influence.
- Only ~2.5% of songs in the dataset are truly popular, making prediction of class 1 inherently challenging.

---

## ▶️ Running the Project

### 1. Install dependencies
```bash
pip install -r Requirements.txt
```

### 2. Train or load models
- The repository contains `main.ipynb` to re-run data cleaning and training.
- Or place your trained `.pkl` models inside the `models/` folder.

### 3. Launch the Streamlit app
```bash
streamlit run streamlit_app.py
```
The app will open in your browser at `http://localhost:8501`.

---

## 📊 Deliverables
- **Exploratory Data Analysis** with visualizations and insights.
- **Well-documented Python code** (`main.ipynb`).
- **Interactive Streamlit app** (`streamlit_app.py`).
- **Requirements.txt** for environment reproduction.
- Optional: SQL scripts and GitHub commit history.

---

## 🚀 Next Steps
- Connect to the Spotify API to fetch track features from a song link.
- Experiment with deep learning (e.g., CNNs on spectrograms).
- Expand the dataset to new music releases for continued accuracy.

---

> **Author:** Vinicius Jodar Soares Costa  
> *Developed as a final Data Science & Machine Learning Bootcamp project.*
