# 🧠 Smart ML Studio

> An interactive no-code Machine Learning platform built with **Streamlit** that enables users to upload datasets, train machine learning models, compare algorithms, visualize performance, and make predictions—all through an intuitive web interface.

![Python](https://img.shields.io/badge/Python-3.12+-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-Latest-red?logo=streamlit)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange?logo=scikitlearn)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 🚀 Overview

Smart ML Studio is designed for students, beginners, and data enthusiasts who want to experiment with machine learning without writing code.

Simply upload a CSV dataset, choose your input and target columns, select an algorithm, and the application will train the model, evaluate its performance, generate predictions, and display interactive visualizations.

---

# ✨ Features

## 📂 Dataset Handling

- Upload CSV datasets
- Preview uploaded data
- Display dataset shape
- View column data types
- Detect missing values
- Detect duplicate records
- Basic dataset statistics

---

## ⚙️ Data Configuration

- Select input features
- Select target column
- Automatic Regression / Classification detection
- Manual override option

---

## 🧹 Data Preprocessing

- Missing value handling
- Label Encoding
- One-Hot Encoding
- Feature Scaling
- Train-Test Split
- Random State selection

---

## 🤖 Supported Machine Learning Models

### Regression

- Linear Regression
- Multiple Linear Regression
- Polynomial Regression

### Classification

- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)
- Decision Tree
- Random Forest

---

## ⚡ Hyperparameter Tuning

Customize model parameters including:

### Polynomial Regression

- Degree

### KNN

- Number of Neighbors

### SVM

- Kernel
- C Value

### Decision Tree

- Maximum Depth
- Criterion

### Random Forest

- Number of Trees
- Maximum Depth

Or simply use default parameters.

---

## 📊 Model Comparison

Compare multiple models simultaneously.

Regression Comparison includes:

- R² Score
- MAE
- MSE
- RMSE

Classification Comparison includes:

- Accuracy
- Precision
- Recall
- F1 Score

Automatically highlights the best-performing model.

---

## 📈 Interactive Visualizations

### General

- Histograms
- Boxplots
- Correlation Heatmap
- Class Distribution

### Regression

- Actual vs Predicted
- Residual Plot

### Classification

- Confusion Matrix
- ROC Curve
- Precision-Recall Curve
- Feature Importance

---

## 🔮 Prediction

Generate predictions using trained models by entering new feature values directly through the interface.

---

## 💾 Export

Download

- Trained Model
- Prediction Results
- Evaluation Report

---

# 🛠 Tech Stack

| Category | Technology |
|-----------|------------|
| Frontend | Streamlit |
| Programming Language | Python |
| Machine Learning | Scikit-Learn |
| Data Processing | Pandas, NumPy |
| Visualization | Plotly, Matplotlib |
| Model Persistence | Joblib |

---

# 📁 Project Structure

```text
Smart_ML_Studio/
│
├── app.py
├── requirements.txt
├── README.md
│
├── datasets/
├── models/
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

# ⚙️ Installation

Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/Smart_ML_Studio.git
```

Go to the project directory

```bash
cd Smart_ML_Studio
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
streamlit run app.py
```

or

```bash
python -m streamlit run app.py
```

---

# 🧪 Workflow

```text
Upload Dataset
       │
       ▼
Select Features
       │
       ▼
Choose Regression / Classification
       │
       ▼
Choose Single Model
or
Compare All Models
       │
       ▼
Train Model
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

# 🎯 Future Improvements

- Cross Validation
- XGBoost
- LightGBM
- CatBoost
- PCA
- Feature Selection
- SHAP Explainability
- AutoML Recommendations
- Neural Network Support
- Cloud Deployment
- User Authentication
- Model Versioning

---

# 🎓 Learning Outcomes

This project demonstrates practical implementation of

- Data Preprocessing
- Feature Engineering
- Regression
- Classification
- Hyperparameter Tuning
- Model Evaluation
- Data Visualization
- Streamlit Development
- Machine Learning Pipelines

---

# 🤝 Contributing

Contributions, issues, and feature requests are welcome.

Feel free to fork the repository and submit a Pull Request.

---

# 📄 License

This project is licensed under the MIT License.

---

# 👩‍💻 Developer

**Shivani Panchal**

Computer Science & Engineering Student

GitHub: https://github.com/panchalshivani04

LinkedIn: https://linkedin.com/in/YOUR_PROFILE

---

⭐ If you found this project useful, consider giving it a star!