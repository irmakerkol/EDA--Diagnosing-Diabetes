# 🧪 EDA: Diagnosing Diabetes

![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)
![Pandas](https://img.shields.io/badge/Library-pandas-lightgrey)
![NumPy](https://img.shields.io/badge/Library-numpy-lightgrey)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Kaggle Dataset](https://img.shields.io/badge/Data-Kaggle-blue)

This project explores how various medical diagnostic features correlate with diabetes outcomes in female patients. Using the **Pima Indians Diabetes Dataset**, I apply core **EDA (Exploratory Data Analysis)** techniques to inspect, clean, and validate the dataset in preparation for future modeling.

Dataset source: [National Institute of Diabetes and Digestive and Kidney Diseases](https://www.kaggle.com/uciml/pima-indians-diabetes-database)

---

## 📄 Dataset Overview

Each record represents a female patient. The dataset includes the following features:

| Column                  | Description                                                                 |
|------------------------|-----------------------------------------------------------------------------|
| `Pregnancies`          | Number of times the patient has been pregnant (integer)                     |
| `Glucose`              | Plasma glucose concentration (mg/dL) after 2 hours in OGTT                  |
| `BloodPressure`        | Diastolic blood pressure (mm Hg)                                            |
| `SkinThickness`        | Triceps skinfold thickness (mm)                                             |
| `Insulin`              | 2-hour serum insulin (mu U/ml)                                              |
| `BMI`                  | Body mass index (weight in kg / height in m²)                               |
| `DiabetesPedigreeFunction` | A function scoring likelihood of diabetes based on family history     |
| `Age`                  | Age in years (integer)                                                      |
| `Outcome`              | Class label (1: diabetic, 0: non-diabetic)                                  |

---

## ✅ Objectives

- Load and inspect the structure and content of the dataset.
- Identify and handle missing or suspicious values.
- Validate data types and fix any inconsistencies.
- Prepare the dataset for machine learning pipelines.

---

## 🔍 Key EDA Steps

### 1. Load and Inspect the Dataset
```python
import pandas as pd
import numpy as np

diabetes_data = pd.read_csv("diabetes.csv")
print(diabetes_data.head())
```

- ✅ 9 columns, 768 rows
- ✅ Expected data types: mostly `int64` and `float64`

---

### 2. Check for Missing Values
```python
print(diabetes_data.isnull().sum())
```
- Initially, **no missing values** detected.

---

### 3. Dig Deeper with `.describe()`
```python
print(diabetes_data.describe())
```
- Discovered **zeros in Glucose, BloodPressure, BMI, SkinThickness, and Insulin** — biologically impossible values indicating **missing data**.

---

### 4. Replace Zeros with NaNs
```python
columns_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
diabetes_data[columns_with_zeros] = diabetes_data[columns_with_zeros].replace(0, np.NaN)
```

---

### 5. Recheck Missing Values
```python
print(diabetes_data.isnull().sum())
```

---

### 6. Investigate Patterns in Missing Data
```python
print(diabetes_data[diabetes_data.isnull().any(axis=1)])
```

---

### 7. Validate Data Types
```python
diabetes_data.info()
```

---

### 8. Fix Data Type of `Outcome`
```python
# Replace typo (e.g., 'O') with 0 and convert to int
diabetes_data['Outcome'] = diabetes_data['Outcome'].apply(lambda row: 0 if row == 'O' else row)
diabetes_data['Outcome'] = diabetes_data['Outcome'].astype(int)
```

---

## 📊 Observations

- Found **outliers**: e.g., pregnancies as high as 17.
- Confirmed structural integrity after cleaning.
- Prepared clean, validated dataset for modeling.

---

## 📈 Next Steps

- Impute missing values using mean/median strategies.
- Visualize feature relationships using Seaborn and Matplotlib.
- Train classification models (e.g. logistic regression, decision trees).
- Evaluate performance using accuracy, ROC AUC, and confusion matrix.

---

## 🛠️ Tools Used

- Python 3  
- Pandas  
- NumPy  
- Jupyter Notebook  

---

## 📁 Repository Structure

```
📆 EDA-Diabetes
├── diabetes.csv
├── eda_diabetes_notebook.ipynb
├── README.md
```

---

## 📬 Contact

**Irmak Erkol**  
📧 irmakerkol00@gmail.com  
💼 [LinkedIn](https://linkedin.com/in/irmakerkol)

---

