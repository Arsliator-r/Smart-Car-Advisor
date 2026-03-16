# 🚗 Smart Car Advisor
### A Hybrid AI System for Used Car Valuation & Condition Assessment in Pakistan

![Python](https://img.shields.io/badge/Python-3.9+-blue?style=flat-square&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red?style=flat-square&logo=streamlit)
![Gemini](https://img.shields.io/badge/Gemini-2.5_Flash-orange?style=flat-square&logo=google)
![License](https://img.shields.io/badge/License-Academic-green?style=flat-square)

---

## 📌 Overview

The Pakistani used car market suffers from **high volatility** and **severe information asymmetry** — buyers struggle to determine fair value because traditional tools rely solely on structured data (Year, Make, Model) and completely ignore vehicle condition.

**Smart Car Advisor** solves this by combining:
- 🤖 **Machine Learning** (Random Forest Regressor) for baseline price prediction
- 📝 **NLP** (TF-IDF + Regex rule engine) for condition analysis from seller descriptions
- 💬 **Gemini 2.5 Flash** for AI-powered upgrade recommendations and mechanic chat

---

## 🎯 Key Results

| Metric | Score |
|---|---|
| R² Score (Test Set) | **97.18%** |
| Mean Absolute Percentage Error (MAPE) | **3.2%** |
| Dataset Size | **3,500+ listings** |
| Target Market | Pakistan (Toyota, Honda, Suzuki, Kia, Hyundai) |

---

## ✨ Features

### 💰 Price Estimator
- Select car model from 3,500+ real PakWheels listings
- Input mileage, engine capacity, year, transmission, fuel type
- Paste the seller's ad description — the system reads it automatically
- Get a **condition score (0–10)** and **final price in PKR** with penalty breakdown

### 🧠 Hybrid Logic Controller
Detects critical issues in seller descriptions and applies deterministic market penalties:

| Condition Detected | Penalty |
|---|---|
| Missing / Duplicate File | -35% |
| Major Structural Accident | -25% |
| Duplicate Book / Card | -12% |
| Repainted / Showered | -10% |
| Minor Touchups | -2% |
| Bumper-to-Bumper Genuine | +3% |

### 💡 AI Value Analyst (Gemini)
- Analyzes your specific car + condition
- Suggests **3 high-ROI upgrades** to maximize resale value
- Scored by "Lift Score" (1–10) based on Pakistani market demand

### 🤖 AI Mechanic Chat
- Context-aware conversation powered by Gemini 2.5 Flash
- Remembers last 6 messages for continuity
- Answers in **Pakistani market terminology** (Lakhs, Genuine, On-money, etc.)
- Quick action buttons for common queries

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.9+ |
| ML Framework | Scikit-Learn (Random Forest, TF-IDF) |
| Web App | Streamlit |
| AI / LLM | Google Gemini 2.5 Flash-Lite |
| Data Collection | Selenium WebDriver + BeautifulSoup |
| Data Processing | Pandas, NumPy |
| Model Persistence | Joblib (.pkl) |

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/Smart-Car-Advisor.git
cd Smart-Car-Advisor
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Set Up Your API Key
Create a `.env` file in the root directory:
```
GEMINI_API_KEY=your_gemini_api_key_here
```
Get your free API key at [Google AI Studio](https://aistudio.google.com)

### 4. Train Models
Model files (.pkl) are not included due to size. 
Run notebooks 1 → 2 → 3 → 4 in order to regenerate them.

### 5. Run the App
```bash
streamlit run app.py
```

---

## 📁 Project Structure

```
Smart-Car-Advisor/
│
├── app.py                          ← Main Streamlit application
│
├── models/
│   ├── price_model_optimized.pkl   ← Trained Random Forest model
│   ├── inspector_model.pkl         ← Inspection scoring model
│   └── tfidf_vectorizer.pkl        ← TF-IDF vectorizer
> ⚠️ Model files (.pkl) are not included due to size. Run notebooks 1 → 2 → 3 → 4 in order to regenerate them.
│
├── notebooks/
│   ├── 1_train_inspector.ipynb
│   ├── 2_create_master.ipynb
│   ├── 3_train_price_model.ipynb
│   └── 4_perfect_price_model.ipynb
│
├── scraping/
│   ├── pakwheels_GOLD_selenium.ipynb
│   └── pakwheels_SILVER_selenium.ipynb
│
├── data/
│   ├── gold_data_CLEANED.csv
│   └── pakwheels_silver_data.csv
│
├── assets/
│   └── consistency_check.png
│
├── .env                            ← NOT uploaded (add your own)
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 📊 System Architecture

```
User Input (Structured + Text Description)
        │
        ▼
┌─────────────────────────┐
│   NLP Condition Tagger  │  ← TF-IDF + Regex Rule Engine
│   (smart_condition_tagger)│
└────────────┬────────────┘
             │ condition tags
             ▼
┌─────────────────────────┐
│  Random Forest Regressor│  ← Predicts base price
│   (price_model.pkl)     │
└────────────┬────────────┘
             │ base price
             ▼
┌─────────────────────────┐
│  Hybrid Logic Controller│  ← Applies market penalties/premiums
│  (Rule-Based Guardrail) │
└────────────┬────────────┘
             │ final price
             ▼
┌─────────────────────────┐
│  Gemini AI Recommender  │  ← Upgrade suggestions + Chat
└─────────────────────────┘
             │
             ▼
      Valuation Report (PKR)
```

---


## 🔒 Security Note

This project uses a `.env` file for API key management. The `.env` file is excluded from version control via `.gitignore`. Never hardcode API keys directly in source files.

---

## 👥 Authors

- **Muhammad Arsalan** — University of Haripur, Pakistan
- **Muhammad Annus** — University of Haripur, Pakistan

Supervised by **Dr. Shabih ul Hassan**, Department of Information Technology

---

## 📚 Context

This project was developed as a Final Year Project (FYP) for the degree of **Bachelor of Science in Artificial Intelligence (BSAI)** at the University of Haripur, Pakistan.

---

## 🔮 Future Work

- Image-based condition analysis using Computer Vision
- Real-time price updates from live listings
- Support for additional manufacturers (Changan, MG, Proton)
- Mobile application deployment
