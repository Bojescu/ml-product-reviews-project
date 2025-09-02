# Sentiment Analysis ML Project (Complete Pipeline)

This repository demonstrates an end-to-end machine learning workflow for **sentiment analysis** of product reviews in Python with scikit-learn.  
It covers all typical phases—from raw data to a ready-to-use saved model.

---

## Project Structure
data/
└── product_reviews_full.csv # dataset

notebooks/
└── exploratory_analysis.ipynb # EDA and preprocessing

src/
├── train_model.py # train and save the model pipeline
└── test_model.py # load the saved model and run predictions

README.md


---

## What We Did in This Module

1) **Project Setup**
- Created a fresh GitHub repository  
- Defined a clean project folder structure  
- Uploaded the raw dataset

2) **Data Exploration**
- Loaded and inspected a large set of product reviews  
- Used `matplotlib` and `seaborn` for visualizations  
- Explored sentiment distribution and text characteristics

3) **Data Cleaning & Preprocessing**
- Removed rows with missing values  
- Standardized sentiment labels to *positive / negative / neutral*  
- Parsed and validated numeric fields (e.g., price if present)  
- Converted free text to a numeric feature (e.g., review length)

4) **Feature Engineering**
- Selected meaningful inputs: `review_title`, `review_text`, `review_length`  
- Removed irrelevant columns  
- Explored correlations (e.g., between price and sentiment)

5) **Model Training & Evaluation**
- Compared multiple models: Logistic Regression, Naive Bayes, Decision Tree, Random Forest, SVM  
- Used `ColumnTransformer` + `Pipeline` for unified preprocessing  
- Evaluated with precision, recall, F1-score, and confusion matrix

6) **Final Model Training**
- Trained the final model on the full dataset  
- Saved the fitted pipeline with `joblib` to `sentiment_model.pkl`

7) **Inference & Usage**
- Loaded the saved pipeline for prediction  
- Built a simple interactive/testing interface for new reviews  
- Enabled quick, real-time tests from the console

---

## How to Use

### 🏋️ Train the Model
```bash
cd src
python train_model.py
