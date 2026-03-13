# JCPenney Customer Review Analysis

A data analysis and machine learning project that explores customer review behaviour on JCPenney's e-commerce platform, using exploratory visualisations and a Decision Tree classifier to predict review sentiment.

---

## Project Overview

This notebook analyses customer reviews to answer key business questions around sentiment, product performance, regional engagement, and pricing strategy. A machine learning model is then built to predict whether a customer will leave a positive or negative review based on product price, customer age, and discount percentage.

---

## Files Required

Place the following data files in the same directory as the notebook before running:

| File | Type | Description |
|------|------|-------------|
| `products.csv` | CSV | Product details — ID, name, price, average score |
| `reviews.csv` | CSV | Customer reviews — username, product ID, score |
| `users.csv` | CSV | User demographics — username, date of birth, state |
| `jcpenney_products.json` | JSON (newline-delimited) | Product catalogue — category, brand, list price, sale price |
| `jcpenney_reviewers.json` | JSON (newline-delimited) | Reviewer profiles — username, date of birth |

---

## Libraries Used

```python
import pandas as pd          # Data loading, cleaning, merging
import numpy as np           # Numerical operations
import matplotlib.pyplot as plt  # All chart visualisations
import seaborn as sns        # Visual theme and heatmap
import json                  # Reading newline-delimited JSON files
from sklearn.tree import DecisionTreeClassifier      # ML model
from sklearn.model_selection import train_test_split # Train/test split
from sklearn.metrics import classification_report, confusion_matrix  # Evaluation
```

Install all dependencies with:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

---

## Notebook Structure

### 1. Data Loading
- Loads all 5 data files (3 CSVs + 2 JSON) into separate dataframes
- Prints shape of each to confirm successful loading

### 2. Data Cleaning
Each dataset is cleaned with the following steps:
- Column name standardisation (lowercase + strip whitespace)
- Duplicate row removal
- Numeric type conversion using `pd.to_numeric(errors="coerce")`
- Null value removal on key columns
- Date of birth parsing and age calculation (as of 2024-01-01)

### 3. Data Merging
A master dataframe `df` is built using left joins in the following order:
1. Reviews → Products CSV (adds name, price, average score)
2. + Users CSV (adds state, age)
3. + Products JSON (adds category, brand, sale price)

### 4. Exploratory Analysis — 7 Visualisations

| # | Chart | Key Finding |
|---|-------|-------------|
| 1 | Distribution of Review Scores (bar) | Scores 0 and 1 dominate — overwhelmingly negative sentiment |
| 2 | Top 10 Brands by Average Rating (horizontal bar) | Clear quality differences across brands |
| 3 | Top 10 Categories by Review Count (bar) | A few categories drive the majority of activity |
| 4 | Product Price vs Average Score (scatter) | No clear relationship between price and rating |
| 5 | Top 15 States by Review Count (bar) | Review activity concentrated in a small number of states |
| 6 | Average Score by Age Group (bar) | Satisfaction levels are consistent across all age groups |
| 7 | Distribution of Discount Percentages (histogram) | Mean discount ~40%, indicating an aggressive pricing strategy |

### 5. Machine Learning Model

**Model:** Decision Tree Classifier (`max_depth=3`, `random_state=42`)

**Features (X):**
- `price` — product price
- `age` — customer age
- `discount_pct` — discount percentage applied to the product

**Target (y):**
- `review_label` — Positive (score ≥ 3) or Negative (score < 3)

**Train / Test Split:** 80% training, 20% testing (`random_state=42`)

**Evaluation outputs:**
- Model accuracy score
- Classification report (Precision, Recall, F1-Score)
- Confusion matrix heatmap
- Feature importance bar chart
- Actual vs Predicted label comparison chart

---

## Key Results

- **Discount percentage** was the most important feature for predicting sentiment
- **Price** was the second most influential factor
- **Customer age** had the least impact on review outcome
- The model demonstrates that perceived value (discount depth) is a stronger driver of positive reviews than product price alone

---

## How to Run

1. Clone or download this repository
2. Place all 5 data files in the same folder as the notebook
3. Open the notebook in Jupyter:

```bash
jupyter notebook 3604945_BD2.ipynb
```

4. Run all cells from top to bottom (`Kernel > Restart & Run All`)

---

## Author

Student ID: 3604945
Data analysis of JCPenny retail
