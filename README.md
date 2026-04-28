# 🧬 Breast Cancer Detection & Classification with Medical AI

This project is an end-to-end data science study aimed at diagnosing breast cancer using machine learning techniques. Beyond simple prediction accuracy, the project specifically focuses on the **Recall (Sensitivity)** metric, which is vital in medical diagnostics.

---

## 🎯 Project Objectives
In medical diagnostic processes, a "False Negative" (diagnosing a patient as healthy when they are not) is the most critical error. This project aims to minimize this risk by using feature engineering and advanced classification algorithms to ensure no potential cases are overlooked.

---

## 📊 Data Analysis & Feature Engineering
* **Exploratory Data Analysis (EDA):** Relationships between 30 features were analyzed using a **Heatmap (Correlation Matrix)**.
* **Dimensionality Reduction:** Features with extremely high correlation (**Multicollinearity**), such as `mean radius`, `mean perimeter`, `worst radius`, etc., were removed to ensure the model remains stable and generalizes well.
* **Data Preprocessing:** `StandardScaler` was utilized to bring all numerical features to the same scale.

---

## 🛠️ Technical Architecture
The code structure is built on a production-standard **Scikit-learn Pipeline** and **ColumnTransformer** architecture:

1.  **Preprocessing:** Automatic scaling of numerical data.
2.  **Modelling:** Simultaneous evaluation of multiple algorithms:
    * Logistic Regression
    * Random Forest Classifier
    * XGBoost Classifier

---

## 🏆 Model Performance & Leaderboard
Models were evaluated using Accuracy, Precision, Recall, and F1-Score. The model with the highest **Recall** score was selected as the "Champion Model" due to its medical reliability.

| Model | Accuracy | Precision | Recall | F1 Score | AUC |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **XGBoost** | 0.97 | 0.96 | 0.98 | 0.97 | 0.99 |
| **Random Forest** | 0.96 | 0.95 | 0.97 | 0.96 | 0.99 |
| **Logistic Regression** | 0.95 | 0.94 | 0.96 | 0.95 | 0.99 |

*(Note: Values may vary based on test set results.)*

---

## 🚀 Installation & Execution

1. Clone the repository:
   ```bash
   git clone [https://github.com/your-username/breast-cancer-ai.git](https://github.com/your-username/breast-cancer-ai.git)
