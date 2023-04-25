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
 
### **Results:**
1. Exploratory Data Analysis:
    * Fig 1: A1. European Brent crude oil price data time-series plot A. Histogram, B. Distribution of Daily Percentage Price change and C.                 Distribution of Daily percentage price change (within 5%) of European Brent crude oil price data:
    ![EuBrentEDA](https://user-images.githubusercontent.com/115378526/234199156-f95bae69-0f42-494a-996e-2273f648ea54.jpg)
    ![EuBrentHistograms](https://user-images.githubusercontent.com/115378526/234199211-6d6f9e3e-6fb3-4a98-9140-ca9eac3f579b.jpg)

    * Fig 2: A1. U.S. Cushing crude oil price data time-series plot A. Histogram, B. Distribution of Daily Percentage Price change and C.                   Distribution of Daily percentage price change (within 5%) of United States Cushing  crude oil price data: 
    ![CushingEDA](https://user-images.githubusercontent.com/115378526/234199292-296efdbe-73ee-4c9c-8550-88233ad31f95.jpg)
    ![CushingHistograms](https://user-images.githubusercontent.com/115378526/234199333-1e5b2573-828a-46fc-9845-874c1ad87d1f.jpg)
2. ML Model Analysis:
    * Baseline model :
    Table 1: Results of baseline model for EU Brent and U.S. Cushing crude oil price data    
    ![BaselineResults](https://user-images.githubusercontent.com/115378526/234200372-c6032c61-e91a-4981-b739-5f36a32f5275.jpg)
    * ARIMA Family of models:
    The seasonal decomposition model was not used to build a forecasting product as its validation/test error was higher than than the test-error of     baseline (note: no cross validation in seasonal decomposition). Auto-ARIMA and SARIMA was used to build forecasting product and the forecasting       plot of Auto-ARIMA (the best performing model in ARIMA family) is shown below in Fig 3. 
    Table 2: Results of Auto-ARIMA and SARIMA model on EU Brent and US Cushing crude oil price data
    ![ArimaResults](https://user-images.githubusercontent.com/115378526/234200752-b9fc6f80-0bf0-4bc8-a5fc-fc663f2a601a.jpg)
    Fig 3: Auto-ARIMA forecast plot on A.EU Brent and B. U.S Cushing Crude oil price data
    ![ArimaResults](https://user-images.githubusercontent.com/115378526/234201105-f08afbf5-043e-4860-8722-560dadc87fdb.jpg)
    * Facebook/Meta Prophet:
    The model built with customized yearly seasonality and COVID as a one time event minimized the poor to the maximum and results are shown below in 	  table 1 consisting of the error metrics of the analysis ad forecasting product and Fig 4 depicting the visualization of the forecasting product    	 of the best model of Meta/Facebook Prophet. 

    Table 3: Results of Facebook Prophet model with COVID event and customized yearly seasonality on EU Brent and U.S. Cushing crude oil price data
    Fig 4: Facebook/Meta Prophet (with customized yearly seasonality and COVID as one time event) forecast plot on A. EU Brent and B.  U.S. Cushing 	crude oil price data
    ![ProphetResultsPlots](https://user-images.githubusercontent.com/115378526/234201828-e92b91c7-c847-444c-8354-2a5d8a721c81.jpg)
    
    * LinkedIN Silverkite: 
      The model built with gradient boosting regressor performed the best of the two models built as mentioned in the methods section. 

      Table 4: Results of LinkedIn Silverkite model built with gradient boosting regressor fit on EU Brent and U.S. Cushing Crude Oil price data
      ![SilverKiteResults](https://user-images.githubusercontent.com/115378526/234202547-7ecc16e8-a33c-4282-8233-40ca7e1584da.jpg)
      Fig 4: LinkedIn Silverkite with gradient boosting fit method forecasting product on A. EU Brent and B. US Cushing crude oil price data
      ![SilverKiteResultsPlots](https://user-images.githubusercontent.com/115378526/234202758-47d0b1f1-8a83-45fb-bc39-e9001a0b4d8b.jpg)
      
    * Scalecast Library Ensemble model:
      Ensemble models that were built based on the eight individual models mentioned in the methods section (results can be accessed in the notebook       and in the presentation), the model with al regressors and without the lagged (ARIMA) component performed the best.

      Table 5: Results of Scalecast Ensemble model using cat-boost stacking regressor built with various individual models on EU Brent and U.S.             Cushing Crude Oil price data  
      ![StackedResults](https://user-images.githubusercontent.com/115378526/234203522-52a38832-465d-4f52-8cd4-bf04fd8bf43b.jpg)
      ![StackedResults_Cushing](https://user-images.githubusercontent.com/115378526/234203710-ab71d3e3-bc3d-4892-9379-3d538f51b799.jpg)
      Fig 5: Scale Cast Ensemble model using cat-boost stacking regressor using combinations of models forecasting product on A. EU Brent B. U.S.  	 Cushing crude oil price data.
      ![EuBrentStackedPlot](https://user-images.githubusercontent.com/115378526/234204164-2d154597-1ea2-4ef5-af33-27fc5112254f.jpg)
      ![CushingStackedPlot](https://user-images.githubusercontent.com/115378526/234204201-4b6503c2-0ea9-4d1a-8c0d-5d2041043fd4.jpg)
      
    * Wilcoxon Hypothesis Testing:

     The cat boost stacking regressor built with all individual models (catboost_all_reg, RMSE: 6.9) for EU Brent data passed the hypothesis              indicating that there was no significant statistical difference between the forecasted values of this model and the actual price (NOTE: This is      the second best model for the EU Brent data). All other models for EU Brent data and all the models for U.S. Cushing data failed this hypothesis      testing. 
     
  ## **Conclusion:**
  Through the model building and evaluation process for predictive capacity of the crude oil price, we can conclude that we have been able to bring     down the forecast RMSE nearly **~5 times down** compared to the baseline model and **~7 times down** from the lowest performing ML model (auto-       ARIMA) through stacking weak individual models into a stronger ensemble ML model. 

  Since the forecast **RMSE ~3-7** (both crude oil types) for the ensemble model is **comparable with that** of previous research on crude oil price   prediction using **advanced deep learning techniques** with more complex and higher dimensional data (7, 8). 
  
  ## **Future Work:**
  1. Hence, we can for future work we can fine tune the models that we have built and/or utilize better stacking method. 
  2. We would also explore staking ML and DL models and/or increase the complexity of the data to be more representative of the crude-oil price            market.
  3. Explore more advanced techniques like Deep learning and transformers that have been built specific for the oil and gas domain to build hybrid        models to improve the accuracy of predictions 

