# **Exploration of Machine Learning Techniques to Predict Crude Oil Price**

## **Presentation and Demo Video (00h:20m:38s):**

## **Problem Statement and purpose:**
Crude oil a.k.a “Industrial Blood” is the largest traded commodity in the world accounting to ~10% of the world traded commodities (1) . Hence, any fluctuations in crude oil price impacts the national and global economy (2). 

Therefore, it is of great importance to investors, businesses, economists and political to be bale to predict the crude-oil price data to build resilience in the economic and energy domains in the national and global scale. 

## **Problem-Solving Frame work:**
We chose machine learning as our problem solving frame work. The typical workflow for ML framework consisting Extract-Transform-Load (ETL) was utilized to define the problem, map the system as time series forecasting ML task, define suitable error metrics, choose suitable ML methods and time the parameters with appropriate values to evaluate the model and the forecasting product before deploying it. 

_Data Extraction  —> Data Transformation —>  Data Loading/Modeling_
![Workflow](https://user-images.githubusercontent.com/115378526/234194266-46ca5dea-3bcf-4df9-9d3d-58bbb0858919.jpg)

Through out the work-flow and various processes in the workflow we followed the CRISP-DM (Cross Industry Standardized Process for Data Mining) framework which emphasizes on iteratively improving the results of each of the processes and the sub-steps in the processes.

## **Background:**

Historically statistically based models like seasonal decompositions and ARIMA family of models along with econometric approaches have been utilized to forecast the crude oil prices and recently ML techniques have been used to capture the non-linear and more complex components of the historical price data (3, 4, 5, 6) .

## **Objective:**

In this work we propose to utilize ML techniques
1. ARIMA family of models from stats libraries
2. SOTA packages FB/Meta Prophet, LinkedIN SilverKite
3. SOTA library ScaleCast for ensemble modeling.
to forecast the crude oil price.

## **Project Description:**

### **Data-Set:**

1. Github Link: https://github.com/datasets/oil-prices/blob/master/README.md. 
2. Description: This is a publicly available data-set provided by EIA (Energy Information Administration) consisting of  price of
    * Europe Brent Spot Price FOB (Dollars per Barrel) - From 20 May 1987 till today excluding weekends and holidays.
    * Cushing, OK WTI Spot Price FOB (Dollars per Barrel) - From 01 February 1986 till today excluding weekends and holidays. 
    * There are no other external regressor(s) or variables.
    * Both the crude oil dataset have been loaded and analyzed till 27 Feb 2023 although they are updated daily. 
3. Domain Knowledge:
    * Brent : A blended crude stream produced in the North Sea region which serves as a reference or "marker" for pricing a number of other crude           streams source.
    * West Texas Intermediate (WTI - Cushing): A crude stream produced in Texas and southern Oklahoma which serves as a reference or "marker" for           pricing a number of other crude streams and which is traded in the domestic spot market at Cushing, Oklahoma. 

### **Methodology and Analysis:**

1. Exploratory Data Analysis: 
   The data was explored using Matplotlib  for visualizations of the time series data and find patterns/insights from the data and Pandas was used to    compute data with aggregation to derive insights on any seasonal trends.
2. Training and Forecast Data: 
   The data from the Jan 1, 2021 till Feb 27, 2023 was used for forecasting by the ML model and to compare the models performance with the actual        data and the validation error. The rest of the historical data was utilized for training and building the ML models for both the crude oil types.
3. Hypothesis:
    * Error-metric: 
       MSE and RMSE was chosen as the error-metric to evaluate the benchmark model (baseline model) and compare it with other ML models to                  evaluate the predictive ability. Models with lower error metrics than the baseline model were chosen for building the forecasting-product. 
    * Wilcoxon Hypothesis Testing: 
       This test was performed on the best models of the two crude oil types to determine if the forecasted data series is significantly                    different from the actual price data series of the crude oils. The hypothesis is defined as follows: 
               H0 : Significant difference between the datasets. So, model cannot be accepted.
	            H1 : Not a significant difference between the dataset. Hence, we reject the null hypothesis and accept the model we built.  
 4. Cross-validation: 
    Slide forward cross-validation was utilized with two business year (506 days) data used to forecast one business month (22 days) of data             splitting the entire training data into ~400 Cross-validation window. The animation below shows the CV strategy that we adopted for both the         crude oil-types where the forecasted values closely follows the actual values. (Note : the forecasting shown in this CV is through the Auto-ARIMA     model).
    A. EU Brent:
    ![ezgif-5-fe739595a2](https://user-images.githubusercontent.com/115378526/234197003-2fb27681-48b4-4b3a-96e8-ca3510c399b5.gif)
    
    B. Cushing:    
    ![ezgif-5-fb49452308](https://user-images.githubusercontent.com/115378526/234197948-bc6e920b-69a0-4c0a-81d0-65675db95c24.gif)

    
 5. ML Methods:
    Baseline Model with training and test data with the mean of the price of the respective data calculated for both he crude oil types to compute       the error metrics. 

    ARIMA Family of Models including seasonal decomposition, Auto-ARIMA and SARIMA were evaluated iteratively to forecast the crude oil price on the     test data.

    Facebook Prophet was utilized to build their different model 1. With default vales 2. With covid period as a one time event 3. Customized yearly     seasonality with Covid event. 

    LinkedIN Silverkite was used to build two models with one model utilizing Ridge regression and the other utilizing Gradient Boosting in the fit       algorithm. 

    Scale Cast Library was used to build 8 different individual models including 1. Naïve Bayes, 2. Gradient Boosted trees, 3. Light Gradient              Boosting    XGBoost (Extreme Gradient Boosting), 5. Prophet, 6. Holt-Winters Exponential Smoothing, 7. LinkedIN Silverkite, and 8. ARIMA model        and all these models were used within the Cat-boost stacked regressor to build an Ensemble model. The ensembling was carried out in different        ways 1. With all the regressors (catboost_all_reg) 2. Without the Auto-regressive component i.e Without the ARIMA family models                      (catboost_signal_only) 3. With only    the ARIMA family of models (catboost_lag_only). 
