# **Exploration of Machine Learning Techniques to Predict Crude Oil Price**
[![Dataset](https://img.shields.io/badge/Dataset-GitHub-brightblue)](https://github.com/datasets/oil-prices) [![Language](https://img.shields.io/badge/Lang-Python-green)](https://www.python.org/) [![ML--Packages](https://img.shields.io/badge/ML--Packages-Statsmodel%2CMeta--Prophet%2C%20LinkedIn--SIlverkite-yellowgreen)](https://www.statsmodels.org/stable/index.html&https://facebook.github.io/prophet/docs/quick_start.html#python-api&https://linkedin.github.io/greykite/docs/0.3.0/html/pages/greykite/overview.html) [![EnsembleML--Library](https://img.shields.io/badge/EnsembleML--Library-Scalecast-blue)](https://scalecast-examples.readthedocs.io/en/latest/misc/introduction/Introduction2.html) [![Capstone Project](https://img.shields.io/badge/CapstoneProject-Dartmouth-brightgreen)](https://em-executive.berkeley.edu/professional-certificate-machine-learning-artificial-intelligence?utm_source=bing&utm_medium=c&utm_term=berkeley%20machine%20learning%20certification&utm_location=86946&utm_campaign=B-365D_US_BG_SE_BH-PCMLAI_Brand&utm_content=Berkeley_MLAI&msclkid=84b4646fc6111be42dcd007fcfb213cf) <a href="https://www.statsmodels.org/stable/index.html">
  <img src="https://www.statsmodels.org/stable/_images/statsmodels-logo-v2-horizontal.svg"
            alt="statsmodel" width="70" height="65"></a>
	
## **Technical Presentation Video (00h:20m:38s):**
https://user-images.githubusercontent.com/115378526/235265185-81044dcf-454a-4052-9386-e8fdc196cf5e.mp4

## **Problem Statement and purpose:**
Crude oil a.k.a “Industrial Blood” is the largest traded commodity in the world accounting to ~10% of the world traded commodities (1) . Hence, any fluctuations in crude oil price impacts the national and global economy (2). 

Therefore, it is of great importance to investors, businesses, economists and political to be bale to predict the crude-oil price data to build resilience in the economic and energy domains in the national and global scale. 

## **Problem-Solving Frame work:**
We chose machine learning as our problem solving frame work. The typical workflow for ML framework consisting Extract-Transform-Load (ETL) was utilized to define the problem, map the system as time series forecasting ML task, define suitable error metrics, choose suitable ML methods and time the parameters with appropriate values to evaluate the model and the forecasting product before deploying it. 

_Data Extraction  —> Data Transformation —>  Data Loading/Modeling_
![Workflow](https://user-images.githubusercontent.com/115378526/234194266-46ca5dea-3bcf-4df9-9d3d-58bbb0858919.jpg)

Through out the work-flow and various processes in the workflow we followed the CRISP-DM (Cross Industry Standardized Process for Data Mining) framework which emphasizes on iteratively improving the results of each of the processes and the sub-steps in the processes.

## **Background:**

Historically statistically based models like seasonal decompositions and ARIMA family of models along with econometric approaches have been utilized to forecast the crude oil prices and recently ML techniques have been used to capture the non-linear and more complex components of the historical price data (3, 4, 5) .

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
    The results of baseline model analysis are presented in Table 1.   
    
    * ARIMA Family of models:
    The seasonal decomposition model was not used to build a forecasting product as its validation/test error was higher than than the test-error of     baseline (note: no cross validation in seasonal decomposition). Auto-ARIMA and SARIMA was used to build forecasting product and the forecasting       plot of Auto-ARIMA (the best performing model in ARIMA family) is shown below in Fig. 3. Since the performance of Auto-ARIMA and SARIMA were comparable the results of the models have been tabulated in Table. 1. 
    
    * Facebook/Meta Prophet:
    The model built with customized yearly seasonality and COVID as a one time event minimized the poor to the maximum and results are shown below in 	  Table. 1 consisting of the error metrics of the analysis. The forecasting product (Fig. 3) depicting the visualization of the forecasted values    	 of the best model of Meta/Facebook Prophet also containes the band for probablity density of the evaluated forecasted values by the model. 
    
    * LinkedIN Silverkite: 
      The model built with gradient boosting regressor performed the best of the two models built as mentioned in the methods section. As observed in Table. 1, the LinkedIn Silverkite model had the best forecasting accuracy of all the individual models built with their respective libraries. The forecasting product of LinnkedIn silverkite built in this analysis is presented in Fig.3. 

      Table. 1: Error-metric (MSE and RMSE) Results of 1. Baseline 2. ARMA family 3. Facebook/Meta Prophet and 4. LinkedIn Silverkite model built with their respective libraries gradient EU Brent and U.S. Cushing Crude Oil price data
     ![Table1a](https://user-images.githubusercontent.com/115378526/235257869-820e3b54-29ef-4c5e-8a02-353c6662ecf3.png)

      Fig. 3:  Forecasting product of best performing model of 1. Auto-ARIMA 2. Facebook/Meta Prophet 3. LinkedIn SIlverkite on A. EU Brent and B. U.S. Cushing crude oil price data
      ![Fig3](https://user-images.githubusercontent.com/115378526/235257154-9ee2af69-c547-414c-99ed-64c86b21d4ee.png)
      
    * Scalecast Library Ensemble model:
      The results of the individual models, as mentioned in the methods section, built with cross-validation and with no validation data set i.e. only train and test/forecast set can be found in Table. 2 for both European Brent and U.S cushing crude oil price. The forecasting products built with these ML models using Scalecast library are presented in Fig. 4. The product of both cross-validate model and model without cross-validation are also presented within the respective plots of the individual models. 
      Ensemble models built with combinations of these eight models are presented in Table. 3 for both types of crude oil. The forecasting product of all the ensemble models built have also been presented in Fig. 5.

      Table. 2: Results of Error metric (MSE and RMSE) of the individual models (with and without cross-validation) built with Scalecast library for European Brent and U.S. Cushing crude oil price data. 
      ![Table2](https://user-images.githubusercontent.com/115378526/235259147-858f2c9e-cb0c-4f22-9d97-e38f957cc501.png)
	
      Fig. 4: Forecasting products of the individual models (1. Naive, 2. GBT (Gradient Boosted Trees), 3. LightGBM, 4. XGBoost (Extreme Gradient Boosting) 5. Facebook/Meta prophet, 6. Holt Winter's Exponential Smoothing, 7. LinkedIn Silverkit and 8. ARIMA built within scalecast) built with and without cross-validation using Scalecast library for A. EU Brent and B. U.S. CUshing crude oil price data. 
      ![Fig4](https://user-images.githubusercontent.com/115378526/235260221-eef92895-5c63-45c6-ab3f-f724f223c56d.png)

      Table. 3: Results of the error-metric of ensemble ML model built using catboost regressors with various combinations of individual models (catboost_all_reg: all regressors included, catboost_signals_only: ARIMA not included, catboost_lags_only: Only ARIMA included) built using scalecast library for EU Brent and U.S. Cushing crude oil price data
      ![Table3](https://user-images.githubusercontent.com/115378526/235261572-817177bc-d0b2-44da-b204-2d2e23d220df.png)

      
      
      
      Fig 5: Forecasting product of each ensemble ML model built using catboost regressor with various combinations of individual  models (catboost_all_reg: all regressors included, catboost_signals_only: ARIMA not included, catboost_lags_only: Only ARIMA included) built using scalecast library for A. EU Brent and B. U.S. Cushing crude oil price data
      ![Fig5](https://user-images.githubusercontent.com/115378526/235262077-5b201c53-c782-448f-a66e-0dc63d302ea7.png)
     
  ## **Conclusion:**
  Through the model building and evaluation process for predictive capacity of the crude oil price, we can conclude that we have been able to bring     down the forecast RMSE nearly **~5 times down** compared to the baseline model and **~7 times down** from the lowest performing ML model (auto-       ARIMA) through stacking weak individual models into a stronger ensemble ML model. 

  Since the forecast **RMSE ~3-7** (both crude oil types) for the ensemble model is **comparable with that** of previous research on crude oil price   prediction using **advanced deep learning techniques** with more complex and higher dimensional data (6, 7), there is scope for further exploration of more ML techniques to improve the predictive capacity for the given crude oil price data.
  
  ## **Future Work:**
  1. Hence, for future work we can fine tune the models that we have built and/or utilize better stacking method using libraries that include methods like N-BEATS etc. 
  2. We would also explore staking ML and DL models and/or increase the complexity of the data to be more representative of the crude-oil price            market.
  3. Explore more advanced techniques like Deep learning and transformers that have been built specific for the oil and gas domain to build hybrid        models to improve the accuracy of predictions 
  
  ## **References:**
  1. https://www.forex.com/ie/news-and-analysis/top-ten-most-traded-global-commodities/#:~:text=Brent%20Crude%20oil%20is%20the,expensive%20than%20WTI%20crude%20oil.
  2. Abd Elaziz, M.; Ewees, A.A.; Alameer, Z. 2020. Improving adaptive neuro-fuzzy inference system based on a modified salp swarm algorithm using genetic algorithm to forecast crude oil price. Natural Resource Research, 29, 2671–2686.
  3. Wu, X.P.; Li, Z.M. 2013. Risk measures for WTI spot market based on GARCH model. J. Hefei Univ. Technol. 9, 1127–1131.
  4. Cheng, F.Z.; Li, T.; Wei, Y.M.; Fan, T.J. The VEC-NAR model for short-term forecasting of oil prices. 2019. Energy Economy. 78, 656–667.
  5. Cheng, F.Z.; Fan, T.J.; Fan, D.D.; Li, S.L. 2018.  The prediction of oil price turning points with log-periodic power law and multi- population genetic algorithm. Energy Econ. 72, 341–355.
  6. Wang, D.; Fang, T. Forecasting Crude Oil Prices with a WT-FNN Model. 2022. Energies. 15, 1955.
  7. Bai, Y., Li, X., & Jia, S. 2021. Crude oil price forecasting incorporating news text. International Journal of Forecasting. 38(1), 367-383.
  
  ## ** Contributors:**
  1. Madhuram Ravichandran (https://github.com/R-Madhuram)
  2. Dr. Pol Cusco (https://github.com/nanakiksc)
  3. Viswajith Karapoonding Nott (https://github.com/viswajithkn)
  

