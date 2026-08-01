# 🧠 Smart ML Studio

An interactive **Machine Learning Studio** built with **Python, Streamlit, and Scikit-learn** that enables users to upload datasets, preprocess data, train and compare regression and classification models, tune hyperparameters, visualize performance, and generate predictions through an intuitive no-code interface.

---

## ✨ Features

- 📂 Upload CSV datasets
- ⚙️ Select input and target columns
- 🔍 Automatic Regression/Classification detection
- 🤖 Train individual machine learning models
- 📊 Compare multiple machine learning algorithms
- 🎛️ Customize model hyperparameters
- 📈 Interactive visualizations and performance metrics
- 🔮 Generate predictions on new data
- 💾 Download trained models and prediction results

---

## 🤖 Supported Models

### Regression Models
- Linear Regression
- Multiple Linear Regression
- Polynomial Regression

### Classification Models
- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)
- Decision Tree Classifier
- Random Forest Classifier

---

## 🛠️ Tech Stack

| Category | Technologies |
|----------|--------------|
| **Programming Language** | Python |
| **Frontend** | Streamlit |
| **Machine Learning** | Scikit-learn |
| **Data Processing** | Pandas, NumPy |
| **Visualization** | Plotly, Matplotlib |
| **Model Persistence** | Joblib |

---

## 📂 Project Structure

```text
Smart_ML_Studio/
│
├── app.py
├── requirements.txt
├── README.md
│
├── pages/
│   ├── Home.py
│   ├── Upload_Data.py
│   ├── Model_Training.py
│   ├── Prediction.py
│   ├── Visualization.py
│   ├── Comparison.py
│   └── About.py
│
└── utils/
    ├── preprocessing.py
    ├── regression.py
    ├── classification.py
    ├── metrics.py
    ├── visualization.py
    └── helper.py
```

---

## 🚀 Installation

Clone the repository

```bash
git clone https://github.com/panchalshivani04/Smart_ML_Studio.git
```

Navigate to the project directory

```bash
cd Smart_ML_Studio
```

Install the required dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
streamlit run app.py
```

If the above command doesn't work, use:

```bash
python -m streamlit run app.py
```

---

## 📌 Application Workflow

```text
Upload Dataset
       │
       ▼
Select Input & Target Columns
       │
       ▼
Choose Regression or Classification
       │
       ▼
Select Single Model
        or
Compare All Models
       │
       ▼
Train Model(s)
       │
       ▼
Evaluate Performance
       │
       ▼
Generate Predictions
       │
       ▼
Visualize Results
```

---

## 🚀 Future Enhancements

- Cross Validation
- XGBoost & LightGBM Integration
- SHAP Explainability
- AutoML Model Recommendation
- Neural Network Support
- Cloud Deployment
- User Authentication
- Model Versioning

---

## 👩‍💻 Developer

**Shivani Panchal**

Bachelor of Engineering (Computer Science & Technology)

GitHub: https://github.com/panchalshivani04

LinkedIn: https://linkedin.com/in/shivani-panchal-cst

---

⭐ If you found this project useful, consider giving it a **Star** on GitHub!