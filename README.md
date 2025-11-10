#  Superstore Sales Analytics & Forecasting Project

##  Overview  
This project analyzes the **Superstore dataset** to uncover key business insights, predict profit, segment customers, and forecast future sales trends.  
It demonstrates **end-to-end data science skills** — from data preprocessing and feature engineering to machine learning, clustering, and time series forecasting.

---

##  Objectives  
- Understand sales and profit patterns across regions, categories, and customer segments.  
- Built a regression model to **predict profit** using key business factors.  
- Classified orders based on profitability levels.  
- Segmented customers using **K-Means clustering** to identify groups with similar purchase behaviors.  
- Forecasted future sales using **SARIMAX** and **Prophet** models.  
- Visualized findings and model performance for business decision-making.

---

##  Libraries Used  

###  Data Processing  
- pandas  
- numpy
- sklearn.preprocessing(StandardScaler) 

###  Visualization  
- matplotlib  
- seaborn  

###  Machine Learning & Forecasting  
- scikit-learn  
- xgboost  
- prophet  
- statsmodels  

###  Clustering & Dimensionality Reduction  
- sklearn.cluster (KMeans)  
- sklearn.decomposition (PCA)  

###  Utilities  
- datetime  
- joblib    

---

##  Project Structure  

```
##  Project Structure

Superstore-Analytics/
│
├── data/
│   ├── classification_results.joblib
│   ├── cleaned_superstore_data.xlsx
│   ├── cluster_summary.joblib
│   ├── prophet_forecast.joblib
│   ├── raw_superstore_data.xlsx
│   ├── regression_results.joblib
│   └── sarimax_forecast.joblib
│
├── notebooks/
│   ├── Classification.ipynb
│   ├── Regression.ipynb
│   ├── Clustering.ipynb
│   └── Time_Series_Analysis.ipynb
│
├── models/
│   ├── KMeans.joblib
│   ├── prophet_model.joblib
│   ├── sarimax_model.joblib
│   ├── XGBClassifier.joblib
│   └── XGBRegressor.joblib
│
├── visuals/
│   ├── classification/
│   │   ├── confusion_matrix.png
│   │   └── feature_importance_plot.png
│   │
│   ├── clustering/
│   │   ├── elbow_plot.png
│   │   ├── PCA_Clusters.png
│   │   └── silhouette_plot.png
│   │
│   ├── regression/
│   │   ├── actual_vs_predicted_plot.png
│   │   ├── error_distribution_plot.png
│   │   ├── feature_importance_plot.png
│   │   └── residual_plot.png
│   │
│   └── time_series/
│       ├── prophet_components.png
│       ├── prophet_forecast.png
│       ├── prophet_plot.png
│       └── sarimax_forecast_plot.png
│
└── README.md
```

---

## 📊 Results & Visualizations

###  Regression – Actual vs Predicted
This plot compares the predicted profit values from the XGBoost Regression model against the actual profits, showing how well the model fits the data.

!(visuals/regression/actual_vs_predicted_plot.png)

---

###  Classification – Confusion Matrix
The confusion matrix shows how accurately the model classified profitable vs non-profitable orders.

![Confusion Matrix](visuals/classification/confusion_matrix.png)

---

###  Clustering – PCA Cluster Visualization
This 2D PCA projection illustrates the separation of customer segments identified through K-Means clustering.

![PCA Clusters](visuals/clustering/PCA_Clusters.png)

---

###  Time Series – Prophet Forecast
The Prophet forecast plot highlights future sales predictions based on historical Superstore data.

![Prophet Forecast](visuals/time_series/prophet_forecast.png)


---

##  Insights & Results  

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

##  Skills Demonstrated  

- Data Cleaning & Feature Engineering  
- Exploratory Data Analysis (EDA)  
- Regression & Classification Modeling  
- Clustering & Dimensionality Reduction (PCA)  
- Time Series Forecasting  
- Data Visualization (Python)  
- Model Evaluation & Interpretation  


---

##  Author  
**Ricky Samson**  
Aspiring Data Scientist | Machine Learning Enthusiast   
[LinkedIn: www.linkedin.com/in/ricky-samson-aa6569331]
---
