# 🌾 Crop Recommendation System Using Machine Learning

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green.svg)](https://fastapi.tiangolo.com/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.2-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

# 🌱 Crop Recommendation System

**An AI-powered crop recommendation platform built with Machine Learning and FastAPI that predicts the most suitable crop based on soil nutrients and environmental conditions.**

This project uses machine learning algorithms to recommend the best crop based on:

- Nitrogen (N)
- Phosphorus (P)
- Potassium (K)
- Temperature
- Humidity
- pH value
- Rainfall

The system is deployed using **FastAPI** with a modern responsive frontend.

---

# 📑 Table of Contents

- Overview
- Problem Statement
- Objectives
- Features
- Dataset
- Technologies Used
- Algorithms Evaluated
- Model Performance
- Installation
- Usage
- API Endpoints
- Project Structure
- FastAPI Documentation
- Deployment
- Future Enhancements
- Contributors
- License

---

# 🌱 Overview

Agriculture is one of the most important sectors in developing countries. Farmers often struggle to choose the most suitable crop due to changing environmental conditions and lack of scientific guidance.

This project solves that problem using Machine Learning. The application predicts the most suitable crop using soil nutrients and climate conditions.

The system uses a **Random Forest Classifier** trained on agricultural data and achieves **99.32% accuracy**.

---

# ❓ Problem Statement

Farmers face difficulties in selecting suitable crops because of:

- Unpredictable weather
- Lack of soil analysis
- Traditional farming methods
- Limited technical guidance

This project aims to provide a smart crop recommendation system using machine learning techniques.

---

# 🎯 Objectives

- Build an intelligent crop recommendation system
- Analyze soil and environmental conditions
- Compare multiple machine learning algorithms
- Deploy a real-time prediction web application
- Help farmers improve productivity

---

# ✨ Features

- 🌾 Predicts 22 crop categories
- ⚡ High-speed FastAPI backend
- 🤖 Machine Learning powered prediction
- 📈 99.32% Random Forest accuracy
- 📊 Data visualization and analysis
- 📱 Responsive frontend design
- 🔒 Request validation using Pydantic
- 📘 Automatic API documentation
- ☁ Cloud deployment ready
- 🧠 REST API architecture

---

# 🚀 FastAPI Advantages

- Extremely fast backend
- Async support
- Automatic Swagger documentation
- Built-in request validation
- Production-ready architecture
- Easy Docker deployment

---

# 📊 Dataset

Dataset Source:

https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset

## Features

| Column | Description |
|---|---|
| N | Nitrogen |
| P | Phosphorus |
| K | Potassium |
| temperature | Temperature |
| humidity | Humidity |
| ph | Soil pH |
| rainfall | Rainfall |
| label | Crop Name |

---

# 🛠 Technologies Used

| Category | Technologies |
|---|---|
| Backend | FastAPI |
| Machine Learning | Scikit-learn |
| Frontend | HTML5, Tailwind CSS, JavaScript |
| Data Processing | NumPy, Pandas |
| Visualization | Matplotlib, Seaborn |
| Deployment | Docker, Uvicorn |
| Serialization | Joblib |

---

# 🤖 Algorithms Evaluated

- Logistic Regression
- Gaussian Naive Bayes
- Support Vector Machine
- KNN
- Decision Tree
- Random Forest
- Bagging
- AdaBoost
- Gradient Boosting
- Extra Trees

---

# 📈 Model Performance

| Algorithm | Accuracy |
|---|---|
| Random Forest | 99.32% |
| Extra Trees | 99.09% |
| Bagging | 98.86% |
| Gradient Boosting | 98.63% |
| Decision Tree | 97.95% |

---

# 🏆 Best Model

The Random Forest Classifier achieved the best accuracy and was selected for deployment.

---

# ⚙ Installation & Setup

## 1️⃣ Clone Repository

```bash
git clone https://github.com/falcon1614/Crop-Recomendation.git

cd Crop-Recomendation
```

---

## 2️⃣ Create Virtual Environment

### Linux/macOS

```bash
python3 -m venv venv

source venv/bin/activate
```

### Windows

```bash
python -m venv venv

venv\Scripts\activate
```

---

## 3️⃣ Install Requirements

```bash
pip install -r requirements.txt
```

---

## 4️⃣ Run Application

```bash
uvicorn app.main:app --reload
```

---

## 5️⃣ Open Browser

```text
http://127.0.0.1:8000
```

---

# 🚀 Usage

1. Open the web application
2. Enter soil and climate values
3. Click Predict
4. Get recommended crop instantly

---

# 🔌 API Endpoint

## Predict Crop

### Endpoint

```http
POST /predict
```

---

## Request Example

```json
{
  "N": 90,
  "P": 42,
  "K": 43,
  "temperature": 20.87,
  "humidity": 82.00,
  "ph": 6.5,
  "rainfall": 202.93
}
```

---

## Response Example

```json
{
  "recommended_crop": "Rice"
}
```

---

# 📘 Interactive API Documentation

## Swagger UI

```text
http://127.0.0.1:8000/docs
```

---

## ReDoc

```text
http://127.0.0.1:8000/redoc
```

---

# 🧠 Backend Architecture

```text
Frontend UI
    ↓
FastAPI REST API
    ↓
Preprocessing Pipeline
    ↓
Random Forest Model
    ↓
Prediction Response
```

---

# 📂 Project Structure

```bash
Crop-Recommendation/
│
├── app/
│   ├── main.py
│   │
│   ├── routes/
│   │   └── prediction.py
│   │
│   ├── services/
│   │   └── prediction_service.py
│   │
│   ├── models/
│   │   └── request_model.py
│   │
│   ├── templates/
│   │   └── index.html
│   │
│   └── static/
│       ├── css/
│       ├── js/
│       └── images/
│
├── artifacts/
│   ├── model.pkl
│   ├── standscaler.pkl
│   └── minmaxscaler.pkl
│
├── dataset/
│
├── notebooks/
│
├── reports/
│
├── requirements.txt
├── Dockerfile
├── README.md
└── .gitignore
```

---

# 📦 Requirements

```txt
fastapi
uvicorn
jinja2
python-multipart
numpy
pandas
scikit-learn
joblib
matplotlib
seaborn
```

---

# 🐳 Docker Support

## Dockerfile

```dockerfile
FROM python:3.11

WORKDIR /app

COPY . .

RUN pip install -r requirements.txt

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

# ☁ Deployment

Deployable on:

- Render
- Railway
- AWS
- Azure
- Google Cloud
- Docker

---

# 🔮 Future Enhancements

- Real-time weather API
- Fertilizer recommendation
- Soil analysis system
- Multi-language support
- Mobile application
- GPS integration
- Deep learning models
- Microservices architecture

---

# 👨‍💻 Contributors

| Name | Role |
|---|---|
| Jayant | Machine Learning & Backend Development |

---

# 🙏 Acknowledgements

- FastAPI Documentation
- Scikit-learn Documentation
- Kaggle Dataset Contributors

---

# 📚 References

1. https://fastapi.tiangolo.com/

2. https://scikit-learn.org/

3. https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset

---

# 📜 License

MIT License © 2026 Jayant

---

# ⭐ Support

If you like this project:

- Give it a star ⭐
- Fork the repository
- Contribute improvements

---
