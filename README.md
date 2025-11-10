# 🏪 Superstore Sales Analytics & Forecasting Project

## 📘 Overview  
This project analyzes the **Superstore dataset** to uncover key business insights, predict profit, segment customers, and forecast future sales trends.  
It demonstrates **end-to-end data science skills** — from data preprocessing and feature engineering to machine learning, clustering, and time series forecasting.

---

## 🎯 Objectives  
- Understand sales and profit patterns across regions, categories, and customer segments.  
- Build a regression model to **predict profit** using key business factors.  
- Classify orders based on profitability levels.  
- Segment customers using **K-Means clustering** to identify groups with similar purchase behaviors.  
- Forecast future sales using **SARIMAX** and **Prophet** models.  
- Visualize findings and model performance for business decision-making.

---

## 🧰 Libraries Used  

### 📊 Data Processing  
- pandas  
- numpy  

### 📈 Visualization  
- matplotlib  
- seaborn  

### 🤖 Machine Learning & Forecasting  
- scikit-learn  
- xgboost  
- prophet  
- statsmodels  

### 🧩 Clustering & Dimensionality Reduction  
- sklearn.cluster (KMeans)  
- sklearn.decomposition (PCA)  

### ⚙️ Utilities  
- datetime  
- pickle  
- warnings  

---

## 🧪 Project Structure  

```
Superstore-Analytics/
│
├── data/
│   ├── superstore_raw.csv
│   ├── regression_results.csv
│   ├── classification_results.csv
│   ├── cluster_summary.csv
│   ├── sarimax_forecast.csv
│   └── prophet_forecast.csv
│
├── notebooks/
│   ├── 01_Data_Preprocessing.ipynb
│   ├── 02_Regression_Profit_Prediction.ipynb
│   ├── 03_Classification_Profitability.ipynb
│   ├── 04_Clustering_Customer_Segmentation.ipynb
│   └── 05_TimeSeries_SalesForecast.ipynb
│
├── visuals/
│   ├── regression/
│   ├── classification/
│   ├── clustering/
│   └── timeseries/
│
├── models/
│   ├── xgboost_profit_model.pkl
│   ├── kmeans_model.pkl
│   └── prophet_model.json
│
├── requirements.txt
└── README.md
```

---

## 📊 Key Visuals  
- **Regression:** Actual vs Predicted Profit, Residual Distribution  
- **Classification:** Confusion Matrix, Feature Importance  
- **Clustering:** Elbow Curve, Silhouette Score, Cluster Summary Bar Chart  
- **Time Series:** Prophet Forecast Plot, Components Plot, SARIMAX Forecast  

---

## 🧠 Insights & Results  

- **XGBoost Regression Model:**  
  - R² Score: ~0.83  
  - MAE: ≈ 19.9  
  - RMSE: ≈ 89  

- **Customer Segmentation:**  
  - 3 Optimal clusters based on spending and profitability.  
  - Clear distinction between high-value and discount-sensitive customers.  

- **Forecasting:**  
  - Prophet captured strong **monthly seasonality** and upward trends.  
  - SARIMAX validated the seasonal structure with similar predictions.  

---

## 🚀 Skills Demonstrated  

- Data Cleaning & Feature Engineering  
- Exploratory Data Analysis (EDA)  
- Regression & Classification Modeling  
- Clustering & Dimensionality Reduction (PCA)  
- Time Series Forecasting  
- Data Visualization (Python)  
- Model Evaluation & Interpretation  

---

## 💡 Future Work  
- Automate the entire pipeline using Python scripts.  
- Deploy a simple Power BI dashboard for interactive insights.  
- Extend forecasting to sub-categories and regional trends.  

---

## 🧑‍💻 Author  
**Ricky Samson**  
Aspiring Data Scientist | Machine Learning Enthusiast  
📧 Add your email (optional)  
🌐 [GitHub Profile Link]

---
