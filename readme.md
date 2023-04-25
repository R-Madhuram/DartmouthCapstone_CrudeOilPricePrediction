# **Exploration of Machine Learning Techniques to Predict Crude Oil Price**

## **Presentation and Demo Video (00h:20m:38s):**

## **Problem Statement and purpose:**
Crude oil a.k.a “Industrial Blood” is the largest traded commodity in the world accounting to ~10% of the world traded commodities (1) . Hence, any fluctuations in crude oil price impacts the national and global economy (2). 

Therefore, it is of great importance to investors, businesses, economists and political to be bale to predict the crude-oil price data to build resilience in the economic and energy domains in the national and global scale. 

## **Problem-Solving Frame work:**
We chose machine learning as our problem solving frame work. The typical workflow for ML framework consisting Extract-Transform-Load (ETL) was utilized to define the problem, map the system as time series forecasting ML task, define suitable error metrics, choose suitable ML methods and time the parameters with appropriate values to evaluate the model and the forecasting product before deploying it. 

_Data Extraction  —> Data Transformation —>  Data Loading/Modeling_

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

1. <u>Github Link</u>: https://github.com/datasets/oil-prices/blob/master/README.md. 
2. <u>Description</u>: This is a publicly available data-set provided by EIA (Energy Information Administration) consisting of  price of 
