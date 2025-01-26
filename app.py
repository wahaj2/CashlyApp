#!/usr/bin/env python
# coding: utf-8

# In[76]:


#Importing liberaries 
import pandas as pd
import numpy as np 
from sklearn.preprocessing import MinMaxScaler 
import matplotlib.pyplot as plt 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM 
import math 
import base64
import io
import plotly.express as px
import plotly.graph_objs as go
import urllib
from sklearn.model_selection import train_test_split
import xgboost as xgb
import statsmodels.api as sm
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers import Flatten
from prophet import Prophet
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt # library for plots 
from matplotlib import pyplot #plot for loss of model
from statsmodels import api as sm
from statsmodels.tsa.seasonal import seasonal_decompose 
import matplotlib.pyplot as plt
import gurobipy as gp
from gurobipy import GRB
import ipywidgets as widgets
from IPython.display import display
get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
import dash_table
# MSE and MAE
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import skew, kurtosis, norm ,gaussian_kde








#Loading data
#importing the data table
tables = pd.read_html("https://www.feanalytics.com/Table.aspx?SavedResultsID=e33df1a2-7b9a-ee11-b204-002248818b97&xl=1&UserID=B2E540A8-D474-4C56-9088-1F155612F7FE&xlRefreshData=1")
#Saving the data into Dateframe
data = tables[0]
#Taking the transpose of table
data = data.T
#Spliting the dates from string format 
data[1][1][26:36]
#Looping into column and saving the dates in list 
dates = []
for i in range (1,71):
    date = data[1][i][26:36]
    dates.append(date)
#Droping the first and second columns
data = data.drop(columns=[0])
data = data.drop(columns=[1])
#Naming the columns 
data.columns = data.iloc[0]
data = data.iloc[1:].reset_index(drop=True)
#Now droping the Null values rows 
data.dropna(inplace = True)
#Adding the dates 
data['Dates'] = dates
#Converting the date into pandas datetime format
data['Dates'] = pd.to_datetime(data.Dates)
#Setting the Dates as index in our table
data.set_index('Dates', inplace= True)
#Changing the column data into float values
for i in range(len(data.columns)):
    data[data.columns[i]]=data[data.columns[i]].astype(float)













#Risk Matrics dataframe

asset_names = ['Alternatives', 'Alternatives', 'Cash and Term Deposits', 'Cash and Term Deposits', 'Fixed Income', 'Commodities', 'International Securities', 'International Securities', 'Australian Securities', 'Australian Securities', 'Australian Securities', 'Australian Securities', 'International Securities', 'International Securities', 'International Securities', 'International Securities', 'International Securities', 'Infrastructure', 'International Securities', 'Private Equity', 'Fixed Income', 'Fixed Income', 'Fixed Income', 'Fixed Income', 'Fixed Income', 'Fixed Income', 'Fixed Income', 'Alternatives', 'Property', 'Property', 'Property']

historical_data = data

# Clean column names for processing
historical_data_cleaned = historical_data.rename(columns=lambda x: x.strip().split(":")[-1].strip())

# Step 1: Select only numeric columns
numeric_columns = historical_data_cleaned.select_dtypes(include='number')

# Step 2: Arithmetic Mean and Standard Deviation (Volatility)
mean_monthly_returns = numeric_columns.mean()
std_monthly_returns = numeric_columns.std()
annualized_return = mean_monthly_returns * 12
annualized_volatility = std_monthly_returns * (12 ** 0.5)

# Step 3: Geometric Mean Return
geometric_mean_return = (np.prod(1 + numeric_columns) ** (1 / len(numeric_columns))) - 1
geometric_annualized_return = geometric_mean_return * 12

# Step 4: Sharpe Ratio
risk_free_rate = 0.02  # Assume 2% annual risk-free rate
sharpe_ratios = (annualized_return - risk_free_rate) / annualized_volatility

# Step 5: Value at Risk (VaR) at 95% confidence level
z_score_95 = 1.645
var_95 = annualized_return - z_score_95 * annualized_volatility

# Step 6: Maximum Drawdown
cumulative_returns = (1 + numeric_columns).cumprod()
rolling_max = cumulative_returns.cummax()
drawdown = cumulative_returns / rolling_max - 1
max_drawdown = drawdown.min()

# Combine all metrics into a DataFrame
risk_metrics = pd.DataFrame({
    "Annualized Return": annualized_return,
    "Geometric Annualized Return": geometric_annualized_return,
    "Annualized Volatility": annualized_volatility,
    "Sharpe Ratio": sharpe_ratios,
    "Value at Risk (95%)": var_95,
    "Max Drawdown": max_drawdown
})










#Factor_data_loading 

factor_data_columns = ['Economic Factors', 'RBA Cash Rate', 'RBA Yield Curve - 3Yr Gov Bond',
       'US Federal Funds Rate', 'US - 2-Year Yield (%)',
       'US - 5-Year Yield (%)', 'US - 10-Year Yield (%)',
       'US - 30-Year Yield (%)', 'Australia', 'China', 'Japan', 'USA',
       'South Korea', 'India', 'EU', 'Australia.1', 'China.1', 'Japan.1',
       'USA.1', 'South Korea.1', 'India.1', 'EU.1',
       'Australia - Unemployment Rate',
       'Australia - Labour Force Participation Rate',
       'Australia - Employment - to - Population Ratio',
       'Australia - Consumer Sentiment Index',
       'Australia - ANZ-Roy Morgan Consumer Confidence Index',
       'US - Unemployment Rate (%)', 'US - Labor Force Participation Rate (%)',
       'US - Non-Farm Payroll Employment (Thousands)',
       'US - Initial Jobless Claims (Thousands)',
       'US - University of Michigan Consumer Sentiment Index',
       'US - Retail Sales (Monthly % Change)', 'AUD/USD', 'AUD/EUR', 'AUD/JPY',
       'AUD/RMB', 'AUS - Corporate Earnings Growth (%)',
       'AUS - Revenue Growth (%)', 'AUS - P/E Ratio', 'AUS - P/B Ratio',
       'US - Corporate Earnings Growth (%)', 'US - Revenue Growth (%)',
       'US - P/E Ratio', 'US - P/B Ratio', 'Crude Oil (USD/barrel)',
       'Natural Gas (USD/MMBtu)', 'Iron Ore (USD/tonne)', 'Gold (USD/oz)',
       'Copper (USD/tonne)', 'Wheat (USD/bushel)', 'Beef (USD/cwt)',
       'Lithium (USD/kg)', 'Nickel (USD/tonne)', 'Cobalt (USD/tonne)',
       'Sugar (USD/lb)', 'Cotton (USD/lb)', 'Wool (USD/kg)',
       'Carbon Credits (USD/tonne)', 'VIX',
       'AUS - Corporate Bond Spread (BBB, %)', 'AUS - 5-Year CDS Spread (%)',
       'AUS - Interest Rate Spread (%: Lending - Deposit Rate)',
       'US - Corporate Bond Spread (BBB, %)', 'US - 5-Year CDS Spread (%)',
       'US - Interest Rate Spread (%: Lending - Deposit Rate)',
       'AUS - Monetary Stance', 'AUS - Fiscal Stance', 'US - Monetary Stance',
       'US - Fiscal Stance']

factor_data = [['30/9/05', '5.50%',  '2.79%',  '3.75%',  '4.20%',  '4.00%',  '4.33%',  '4.57%',  '3.00%',  '1.80%',  '-0.30%',  '3.40%',  '2.80%',  '4.20%',  '2.20%',  '2.80%',  '11.30%',  '1.30%',  '3.30%',  '4.00%',  '9.30%',  '2.00%',  '5.0%',  '64.5%',  '61.3%',  101.8,  113.5,  '5.0%',  '66.1%',  ' 134,000 ',  335,  76.9,  '0.5%',  0.753,  0.625,  85.5,  6.2,  '12.5%',  '8.0%',  '16.20x',  '2.10x',  '15.0%',  '10.0%',  '17.50x',  '2.80x',  '$57',  '$9',  '$28',  '$445',  '$3,684',  '$3',  '$90',  '$6',  '$14,744',  '$12,000',  '$0',  '$1',  '$6',  '$7',  12.29,  '1.20%',  '0.10%',  '3.50%',  '1.50%',  '0.15%',  '3.00%',  'Tightening',  'Neutral',  'Tightening',  'Neutral'], ['30/9/06',  '6.00%',  '2.79%',  '5.25%',  '4.68%',  '4.55%',  '4.63%',  '4.76%',  '3.90%',  '1.50%',  '0.30%',  '2.10%',  '2.20%',  '6.20%',  '2.20%',  '2.90%',  '12.70%',  '1.70%',  '2.70%',  '5.20%',  '9.60%',  '3.30%',  '4.8%',  '64.8%',  '61.7%',  106.4,  115.2,  '4.5%',  '66.2%',  ' 136,000 ',  310,  85.4,  '0.7%',  0.746,  0.588,  88.0,  5.95,  '10.8%',  '7.5%',  '15.80x',  '2.00x',  '12.0%',  '8.5%',  '16.80x',  '2.70x',  '$66',  '$7',  '$65',  '$603',  '$6,731',  '$4',  '$85',  '$7',  '$24,290',  '$15,000',  '$0',  '$1',  '$6',  '$8',  11.64,  '1.10%',  '0.09%',  '3.48%',  '1.30%',  '0.12%',  '3.02%',  'Tightening',  'Neutral',  'Tightening',  'Neutral'], ['30/9/07',  '6.50%',  '2.79%',  '4.75%',  '3.97%',  '4.18%',  '4.59%',  '4.84%',  '1.90%',  '6.20%',  '0.00%',  '2.80%',  '2.50%',  '6.40%',  '2.30%',  '4.30%',  '14.20%',  '2.20%',  '1.90%',  '5.50%',  '9.80%',  '3.10%',  '4.3%',  '65.0%',  '62.2%',  110.3,  118.7,  '4.7%',  '66.0%',  ' 137,000 ',  320,  83.4,  '0.4%',  0.887,  0.625,  101.5,  6.65,  '9.5%',  '6.8%',  '17.00x',  '2.20x',  '9.0%',  '6.0%',  '17.00x',  '2.60x',  '$72',  '$7',  '$70',  '$697',  '$7,126',  '$6',  '$90',  '$7',  '$37,230',  '$32,000',  '$0',  '$1',  '$7',  '$10',  18.47,  '1.30%',  '0.12%',  '3.45%',  '1.40%',  '0.25%',  '3.00%',  'Tightening',  'Neutral',  'Loosening',  'Expansionary'], ['30/9/08',  '7.00%',  '2.79%',  '2.00%',  '1.97%',  '2.99%',  '3.83%',  '4.33%',  '5.00%',  '4.60%',  '1.40%',  '4.90%',  '4.70%',  '8.30%',  '3.70%',  '2.70%',  '9.60%',  '-1.00%',  '-0.10%',  '2.30%',  '3.90%',  '0.50%',  '4.2%',  '65.2%',  '62.5%',  97.8,  109.4,  '6.1%',  '66.0%',  ' 136,000 ',  450,  70.3,  '-1.2%',  0.796,  0.565,  84.0,  5.45,  '-5.0%',  '-2.0%',  '12.50x',  '1.80x',  '-20.0%',  '-5.0%',  '12.00x',  '1.80x',  '$100',  '$9',  '$140',  '$872',  '$6,956',  '$9',  '$85',  '$8',  '$21,111',  '$40,000',  '$0',  '$1',  '$7',  '$20',  39.73,  '4.00%',  '0.85%',  '3.55%',  '6.00%',  '1.60%',  '3.10%',  'Loosening',  'Expansionary',  'Loosening',  'Expansionary'], ['30/9/09',  '3.00%',  '2.79%',  '0.25%',  '0.95%',  '2.31%',  '3.31%',  '4.04%',  '1.30%',  '-0.80%',  '-1.30%',  '-1.30%',  '2.80%',  '10.80%',  '0.30%',  '1.40%',  '9.40%',  '-5.40%',  '-2.50%',  '0.80%',  '8.50%',  '-4.30%',  '5.7%',  '65.4%',  '61.7%',  85.6,  100.2,  '9.8%',  '65.0%',  ' 130,000 ',  550,  73.5,  '0.5%',  0.878,  0.6,  78.5,  6.0,  '-8.2%',  '-4.5%',  '14.00x',  '1.90x',  '-10.0%',  '-2.0%',  '15.00x',  '2.00x',  '$62',  '$4',  '$80',  '$972',  '$5,150',  '$5',  '$80',  '$9',  '$14,655',  '$30,000',  '$0',  '$1',  '$7',  '$15',  26.01,  '3.00%',  '0.75%',  '3.60%',  '4.50%',  '1.20%',  '3.15%',  'Loosening',  'Expansionary',  'Loosening',  'Expansionary'], ['30/9/10',  '4.50%',  '2.79%',  '0.25%',  '0.43%',  '1.26%',  '2.51%',  '3.69%',  '2.80%',  '3.60%',  '-0.70%',  '1.10%',  '3.00%',  '12.10%',  '2.10%',  '2.30%',  '10.60%',  '4.20%',  '2.60%',  '6.50%',  '10.30%',  '2.10%',  '5.1%',  '65.6%',  '62.3%',  103.9,  112.8,  '9.6%',  '64.7%',  ' 130,000 ',  450,  68.2,  '0.7%',  0.966,  0.7,  81.0,  6.5,  '6.0%',  '3.5%',  '15.00x',  '2.00x',  '8.0%',  '5.0%',  '16.00x',  '2.20x',  '$80',  '$4',  '$145',  '$1,225',  '$7,535',  '$6',  '$95',  '$9',  '$21,809',  '$40,000',  '$0',  '$1',  '$8',  '$12',  23.7,  '1.80%',  '0.65%',  '3.58%',  '3.50%',  '0.95%',  '3.05%',  'Tightening',  ' Contractionary',  'Neutral',  'Expansionary'], ['30/9/11',  '4.75%',  '2.79%',  '0.25%',  '0.25%',  '0.95%',  '1.92%',  '2.91%',  '3.50%',  '6.10%',  '-0.30%',  '3.90%',  '4.00%',  '8.90%',  '3.10%',  '1.90%',  '9.50%',  '-0.10%',  '1.60%',  '3.70%',  '6.60%',  '1.70%',  '5.3%',  '65.5%',  '62.1%',  96.9,  105.7,  '9.0%',  '64.2%',  ' 131,000 ',  400,  59.4,  '0.6%',  0.97,  0.72,  74.5,  6.2,  '4.5%',  '2.8%',  '14.50x',  '1.90x',  '6.0%',  '4.0%',  '15.50x',  '2.10x',  '$95',  '$4',  '$168',  '$1,572',  '$8,828',  '$7',  '$100',  '$10',  '$22,890',  '$34,000',  '$0',  '$1',  '$8',  '$15',  42.96,  '2.20%',  '0.95%',  '3.54%',  '2.80%',  '0.85%',  '3.08%',  'Tightening',  'Neutral',  'Neutral',  'Contractionary'], ['30/9/12',  '3.25%',  '2.79%',  '0.25%',  '0.23%',  '0.63%',  '1.64%',  '2.82%',  '2.00%',  '2.60%',  '-0.10%',  '2.00%',  '2.20%',  '9.30%',  '2.60%',  '3.70%',  '7.90%',  '1.50%',  '2.20%',  '2.30%',  '5.50%',  '-0.40%',  '5.4%',  '65.2%',  '61.7%',  98.2,  106.3,  '7.8%',  '63.6%',  ' 133,000 ',  360,  78.3,  '1.1%',  1.037,  0.8,  80.0,  6.6,  '3.0%',  '2.0%',  '15.20x',  '2.00x',  '5.0%',  '3.5%',  '16.20x',  '2.30x',  '$94',  '$3',  '$128',  '$1,669',  '$7,950',  '$8',  '$110',  '$10',  '$17,527',  '$27,000',  '$0',  '$1',  '$8',  '$8',  15.73,  '2.00%',  '0.80%',  '3.50%',  '2.50%',  '0.75%',  '3.12%',  'Loosening',  'Contractionary',  'Neutral',  'Contractionary'], ['30/9/13',  '2.50%',  '2.79%',  '0.25%',  '0.33%',  '1.39%',  '2.61%',  '3.68%',  '2.20%',  '2.60%',  '0.40%',  '1.20%',  '1.30%',  '9.40%',  '1.50%',  '2.40%',  '7.80%',  '2.00%',  '1.80%',  '2.90%',  '6.40%',  '0.30%',  '5.7%',  '64.9%',  '61.2%',  97.1,  105.0,  '7.2%',  '63.2%',  ' 135,000 ',  310,  77.5,  '0.3%',  0.93,  0.69,  92.5,  5.7,  '5.5%',  '3.5%',  '16.00x',  '2.10x',  '7.0%',  '4.5%',  '17.00x',  '2.40x',  '$98',  '$4',  '$135',  '$1,411',  '$7,322',  '$7',  '$120',  '$11',  '$15,018',  '$27,000',  '$0',  '$1',  '$7',  '$5',  14.97,  '1.80%',  '0.70%',  '3.52%',  '2.10%',  '0.65%',  '3.07%',  'Loosening',  'Neutral',  'Neutral',  'Contractionary'], ['30/9/14',  '2.50%',  '2.79%',  '0.25%',  '0.57%',  '1.76%',  '2.52%',  '3.21%',  '2.30%',  '1.50%',  '2.70%',  '1.70%',  '1.30%',  '6.70%',  '0.40%',  '2.70%',  '7.30%',  '0.40%',  '2.50%',  '3.30%',  '7.40%',  '1.80%',  '6.1%',  '64.7%',  '60.8%',  94.8,  103.2,  '5.9%',  '62.7%',  ' 139,000 ',  290,  84.6,  '0.4%',  0.876,  0.69,  95.0,  5.4,  '6.0%',  '4.0%',  '16.50x',  '2.20x',  '6.5%',  '4.0%',  '17.50x',  '2.50x',  '$93',  '$4',  '$97',  '$1,266',  '$6,863',  '$6',  '$130',  '$11',  '$16,865',  '$32,000',  '$0',  '$1',  '$7',  '$6',  16.31,  '1.50%',  '0.60%',  '3.51%',  '1.80%',  '0.50%',  '3.10%',  'Neutral',  'Contractionary',  'Neutral',  'Expansionary'], ['30/9/15',  '2.00%',  '1.86%',  '0.25%',  '0.64%',  '1.36%',  '2.06%',  '2.87%',  '1.50%',  '1.40%',  '0.80%',  '0.00%',  '0.70%',  '4.90%',  '0.00%',  '2.40%',  '6.90%',  '1.20%',  '2.90%',  '2.80%',  '8.00%',  '2.30%',  '6.0%',  '64.9%',  '61.0%',  101.5,  109.7,  '5.1%',  '62.4%',  ' 142,000 ',  270,  87.2,  '0.1%',  0.702,  0.625,  84.5,  4.5,  '4.0%',  '2.5%',  '15.80x',  '2.10x',  '4.0%',  '2.5%',  '18.00x',  '2.60x',  '$49',  '$3',  '$55',  '$1,160',  '$5,495',  '$5',  '$140',  '$12',  '$11,862',  '$30,000',  '$0',  '$1',  '$6',  '$7',  24.5,  '1.60%',  '0.70%',  '3.50%',  '2.00%',  '0.60%',  '3.15%',  'Loosening',  'Expansionary',  'Tightening',  'Expansionary'], ['30/9/16',  '1.50%',  '1.56%',  '0.50%',  '0.77%',  '1.14%',  '1.60%',  '2.32%',  '1.30%',  '2.00%',  '-0.10%',  '1.50%',  '1.00%',  '4.50%',  '0.20%',  '2.80%',  '6.70%',  '0.50%',  '1.60%',  '2.90%',  '8.20%',  '2.00%',  '5.6%',  '65.0%',  '61.4%',  102.4,  110.5,  '4.9%',  '62.8%',  ' 145,000 ',  250,  91.2,  '0.6%',  0.765,  0.68,  77.0,  5.1,  '3.5%',  '2.0%',  '16.20x',  '2.00x',  '3.5%',  '2.0%',  '18.50x',  '2.70x',  '$43',  '$3',  '$58',  '$1,252',  '$4,863',  '$5',  '$150',  '$12',  '$9,595',  '$25,000',  '$0',  '$1',  '$7',  '$5',  13.29,  '1.70%',  '0.75%',  '3.55%',  '2.20%',  '0.70%',  '3.18%',  'Loosening',  'Contractionary',  'Neutral',  'Neutral'], ['30/9/17',  '1.50%',  '2.06%',  '1.25%',  '1.47%',  '1.92%',  '2.33%',  '2.86%',  '1.80%',  '1.60%',  '0.50%',  '2.20%',  '1.90%',  '3.30%',  '1.50%',  '2.40%',  '6.90%',  '1.70%',  '2.40%',  '3.20%',  '7.00%',  '2.70%',  '5.5%',  '65.2%',  '61.7%',  101.0,  109.2,  '4.2%',  '63.1%',  ' 147,000 ',  230,  95.1,  '0.5%',  0.783,  0.665,  88.0,  5.2,  '7.0%',  '4.5%',  '17.50x',  '2.30x',  '9.0%',  '5.5%',  '19.00x',  '2.80x',  '$51',  '$3',  '$72',  '$1,257',  '$6,166',  '$4',  '$160',  '$13',  '$10,407',  '$30,000',  '$0',  '$1',  '$7',  '$6',  9.51,  '1.60%',  '0.60%',  '3.53%',  '1.90%',  '0.50%',  '3.12%',  'Neutral',  'Neutral',  'Tightening',  'Expansionary'], ['30/9/18',  '1.50%',  '2.06%',  '2.25%',  '2.82%',  '2.95%',  '3.05%',  '3.19%',  '1.90%',  '2.10%',  '1.00%',  '2.30%',  '1.50%',  '3.90%',  '1.80%',  '2.80%',  '6.60%',  '0.30%',  '2.90%',  '2.90%',  '6.10%',  '2.00%',  '5.0%',  '65.5%',  '62.3%',  100.5,  108.7,  '3.7%',  '62.7%',  ' 150,000 ',  210,  100.1,  '0.1%',  0.722,  0.62,  82.5,  4.95,  '6.5%',  '4.0%',  '18.00x',  '2.40x',  '8.0%',  '5.0%',  '19.50x',  '2.90x',  '$65',  '$3',  '$69',  '$1,268',  '$6,530',  '$5',  '$170',  '$13',  '$13,114',  '$35,000',  '$0',  '$1',  '$8',  '$8',  12.12,  '1.70%',  '0.65%',  '3.51%',  '2.00%',  '0.55%',  '3.11%',  'Neutral',  'Expansionary',  'Tightening',  'Neutral'], ['30/9/19',  '1.00%',  '0.78%',  '2.00%',  '1.62%',  '1.55%',  '1.68%',  '2.13%',  '1.70%',  '2.90%',  '0.50%',  '1.70%',  '0.40%',  '3.70%',  '1.20%',  '1.80%',  '6.00%',  '0.70%',  '2.30%',  '2.00%',  '4.20%',  '1.50%',  '5.2%',  '65.9%',  '62.5%',  98.2,  106.5,  '3.5%',  '63.2%',  ' 152,000 ',  200,  93.2,  '0.3%',  0.675,  0.615,  72.5,  4.8,  '5.0%',  '3.0%',  '17.00x',  '2.20x',  '5.5%',  '3.5%',  '20.00x',  '3.00x',  '$57',  '$3',  '$85',  '$1,393',  '$6,000',  '$5',  '$180',  '$14',  '$14,200',  '$30,000',  '$0',  '$1',  '$7',  '$7',  16.24,  '1.40%',  '0.55%',  '3.54%',  '1.80%',  '0.45%',  '3.14%',  'Loosening',  'Neutral',  'Loosening',  'Expansionary'], ['30/9/20',  '0.25%',  '0.23%',  '0.25%',  '0.13%',  '0.28%',  '0.68%',  '1.45%',  '0.70%',  '2.40%',  '-0.50%',  '1.40%',  '0.50%',  '6.60%',  '0.30%',  '-3.80%',  '2.30%',  '-4.80%',  '-3.40%',  '-0.90%',  '-7.30%',  '-6.00%',  '6.9%',  '64.8%',  '60.3%',  75.0,  92.3,  '7.9%',  '61.4%',  ' 141,000 ',  800,  80.4,  '1.6%',  0.716,  0.61,  75.0,  4.9,  '-10.0%',  '-6.0%',  '20.00x',  '2.50x',  '-15.0%',  '-7.0%',  '22.00x',  '3.20x',  '$40',  '$2',  '$108',  '$1,771',  '$6,173',  '$5',  '$190',  '$15',  '$13,700',  '$33,000',  '$0',  '$1',  '$8',  '$12',  26.37,  '2.00%',  '1.00%',  '3.50%',  '3.20%',  '1.20%',  '3.10%',  'Loosening',  'Expansionary',  'Loosening',  'Expansionary'], ['30/9/21',  '0.10%',  '0.19%',  '0.25%',  '0.28%',  '0.97%',  '1.52%',  '2.08%',  '3.00%',  '0.90%',  '-0.20%',  '5.40%',  '2.50%',  '5.10%',  '3.40%',  '4.20%',  '8.10%',  '1.70%',  '5.70%',  '4.10%',  '8.90%',  '5.40%',  '4.6%',  '65.2%',  '62.2%',  104.6,  113.0,  '4.8%',  '61.6%',  ' 147,000 ',  360,  72.8,  '0.7%',  0.722,  0.62,  80.0,  4.95,  '8.0%',  '5.0%',  '19.00x',  '2.40x',  '10.0%',  '6.0%',  '21.00x',  '3.10x',  '$68',  '$4',  '$162',  '$1,800',  '$9,315',  '$7',  '$200',  '$15',  '$18,000',  '$52,000',  '$0',  '$1',  '$8',  '$20',  23.14,  '1.80%',  '0.90%',  '3.52%',  '2.40%',  '0.90%',  '3.12%',  'Loosening',  'Expansionary',  'Neutral',  'Expansionary'], ['30/9/22',  '2.35%',  '3.41%',  '3.25%',  '4.22%',  '4.06%',  '3.83%',  '3.78%',  '7.30%',  '2.80%',  '2.50%',  '8.20%',  '5.10%',  '6.70%',  '9.10%',  '5.90%',  '3.00%',  '1.00%',  '2.60%',  '2.60%',  '6.80%',  '3.50%',  '3.5%',  '66.6%',  '64.2%',  106.3,  114.5,  '3.5%',  '62.3%',  ' 152,000 ',  220,  58.6,  '0.0%',  0.65,  0.6,  92.0,  4.6,  '6.0%',  '4.0%',  '18.50x',  '2.30x',  '7.0%',  '4.5%',  '20.50x',  '3.00x',  '$94',  '$6',  '$123',  '$1,802',  '$8,400',  '$8',  '$215',  '$16',  '$22,000',  '$51,000',  '$0',  '$1',  '$9',  '$28',  31.62,  '2.20%',  '1.20%',  '3.56%',  '2.60%',  '1.10%',  '3.20%',  'Tightening',  'Neutral',  'Tightening',  'Neutral'], ['30/9/23',  '4.10%',  '3.91%',  '5.25%',  '5.05%',  '4.65%',  '4.58%',  '4.70%',  '5.40%',  '1.60%',  '3.30%',  '3.70%',  '3.60%',  '5.90%',  '4.30%',  '2.10%',  '5.20%',  '1.90%',  '2.50%',  '1.30%',  '6.10%',  '0.80%',  '4.1%',  '67.1%',  '64.3%',  102.7,  110.8,  '3.8%',  '62.6%',  ' 154,000 ',  210,  68.1,  '0.2%',  0.64,  0.59,  95.0,  4.55,  '4.5%',  '3.0%',  '18.00x',  '2.20x',  '6.0%',  '4.0%',  '20.00x',  '2.90x',  '$80',  '$4',  '$120',  '$1,850',  '$8,050',  '$8',  '$220',  '$16',  '$20,500',  '$50,000',  '$0',  '$1',  '$8',  '$26',  17.5,  '2.40%',  '1.25%',  '3.58%',  '2.80%',  '1.15%',  '3.18%',  'Tightening',  'Contractionary',  'Tightening',  'Contractionary'], ['30/9/24',  '4.35%',  '3.50%',  '4.75%',  '4.75%',  '4.50%',  '4.30%',  '4.47%',  '3.80%',  '2.20%',  '2.70%',  '2.60%',  '2.90%',  '4.80%',  '3.10%',  '2.50%',  '4.60%',  '0.90%',  '1.90%',  '2.00%',  '6.80%',  '1.70%',  '4.2%',  '67.2%',  '64.4%',  101.5,  109.6,  '4.1%',  '62.6%',  ' 155,000 ',  230,  65.6,  '0.3%',  0.633,  0.585,  97.0,  4.5,  '5.0%',  '3.5%',  '18.20x',  '2.30x',  '5.5%',  '3.5%',  '19.80x',  '2.80x',  '$85',  '$5',  '$130',  '$1,900',  '$8,500',  '$8',  '$225',  '$16',  '$21,000',  '$52,000',  '$0',  '$1',  '$9',  '$30',  15.8,  '2.00%',  '1.15%',  '3.55%',  '2.50%',  '1.00%',  '3.16%',  'Tightening',  'Contractionary',  'Loosening',  'Contractionary'], ['30/9/25',  '3.85%',  '3.81%',  '4.25%',  '4.50%',  '4.50%',  '4.30%',  '4.50%',  '3.60%',  '2.00%',  '2.50%',  '2.50%',  '2.75%',  '4.50%',  '2.75%',  '1.50%',  '4.20%',  '0.70%',  '2.20%',  '2.30%',  '6.50%',  '1.60%',  '4.25%',  '67.0%',  '64.0%',  101.0,  109.0,  '4.50%',  '62.0%',  '155000',  230,  70.0,  '0.5%',  0.6,  0.575,  97.0,  4.5,  '4.5%',  '3.0%',  '18.00x',  '2.20x',  '6.0%',  '4.0%',  '19.00x',  '2.80x',  '$80',  '$5',  '$120',  '$1,900',  '$9,000',  '$8',  '$225',  '$16',  '$21,000',  '$52,000',  '$0',  '$1',  '$9',  '$35',  17.5,  '2.20%',  '1.15%',  '3.55%',  '2.50%',  '1.00%',  '3.15%',  'Tightening',  'Expansionary',  'Loosening',  'Expansionary'], ['30/9/26',  '3.85%',  '3.81%',  '4.25%',  '4.50%',  '4.50%',  '4.30%',  '4.50%',  '3.60%',  '2.00%',  '2.50%',  '2.50%',  '2.75%',  '4.50%',  '2.75%',  '1.50%',  '4.20%',  '0.70%',  '2.20%',  '2.30%',  '6.50%',  '1.60%',  '4.25%',  '67.0%',  '64.0%',  101.0,  109.0,  '4.50%',  '62.0%',  '155000',  230,  70.0,  '0.5%',  0.6,  0.575,  97.0,  4.5,  '4.5%',  '3.0%',  '18.00x',  '2.20x',  '6.0%',  '4.0%',  '19.00x',  '2.80x',  '$80',  '$5',  '$120',  '$1,900',  '$9,000',  '$8',  '$225',  '$16',  '$21,000',  '$52,000',  '$0',  '$1',  '$9',  '$35',  17.5,  '2.20%',  '1.15%',  '3.55%',  '2.50%',  '1.00%',  '3.15%',  'Tightening',  'Expansionary',  'Loosening',  'Expansionary'], ['30/9/27',  '3.85%',  '3.81%',  '4.25%',  '4.50%',  '4.50%',  '4.30%',  '4.50%',  '3.60%',  '2.00%',  '2.50%',  '2.50%',  '2.75%',  '4.50%',  '2.75%',  '1.50%',  '4.20%',  '0.70%',  '2.20%',  '2.30%',  '6.50%',  '1.60%',  '4.25%',  '67.0%',  '64.0%',  101.0,  109.0,  '4.50%',  '62.0%',  '155000',  230,  70.0,  '0.5%',  0.6,  0.575,  97.0,  4.5,  '4.5%',  '3.0%',  '18.00x',  '2.20x',  '6.0%',  '4.0%',  '19.00x',  '2.80x',  '$80',  '$5',  '$120',  '$1,900',  '$9,000',  '$8',  '$225',  '$16',  '$21,000',  '$52,000',  '$0',  '$1',  '$9',  '$35',  17.5,  '2.20%',  '1.15%',  '3.55%',  '2.50%',  '1.00%',  '3.15%',  'Tightening',  'Expansionary',  'Loosening',  'Expansionary'], ['30/9/28',  '3.85%',  '3.81%',  '4.25%',  '4.50%',  '4.50%',  '4.30%',  '4.50%',  '3.60%',  '2.00%',  '2.50%',  '2.50%',  '2.75%',  '4.50%',  '2.75%',  '1.50%',  '4.20%',  '0.70%',  '2.20%',  '2.30%',  '6.50%',  '1.60%',  '4.25%',  '67.0%',  '64.0%',  101.0,  109.0,  '4.50%',  '62.0%',  '155000',  230,  70.0,  '0.5%',  0.6,  0.575,  97.0,  4.5,  '4.5%',  '3.0%',  '18.00x',  '2.20x',  '6.0%',  '4.0%',  '19.00x',  '2.80x',  '$80',  '$5',  '$120',  '$1,900',  '$9,000',  '$8',  '$225',  '$16',  '$21,000',  '$52,000',  '$0',  '$1',  '$9',  '$35',  17.5,  '2.20%',  '1.15%',  '3.55%',  '2.50%',  '1.00%',  '3.15%',  'Tightening',  'Expansionary',  'Loosening',  'Expansionary'], ['30/9/29',  '3.85%',  '3.81%',  '4.25%',  '4.50%',  '4.50%',  '4.30%',  '4.50%',  '3.60%',  '2.00%',  '2.50%',  '2.50%',  '2.75%',  '4.50%',  '2.75%',  '1.50%',  '4.20%',  '0.70%',  '2.20%',  '2.30%',  '6.50%',  '1.60%',  '4.25%',  '67.0%',  '64.0%',  101.0,  109.0,  '4.50%',  '62.0%',  '155000',  230,  70.0,  '0.5%',  0.6,  0.575,  97.0,  4.5,  '4.5%',  '3.0%',  '18.00x',  '2.20x',  '6.0%',  '4.0%',  '19.00x',  '2.80x',  '$80',  '$5',  '$120',  '$1,900',  '$9,000',  '$8',  '$225',  '$16',  '$21,000',  '$52,000',  '$0',  '$1',  '$9',  '$35',  17.5,  '2.20%',  '1.15%',  '3.55%',  '2.50%',  '1.00%',  '3.15%',  'Tightening',  'Expansionary',  'Loosening',  'Expansionary']]
factor_df = pd.DataFrame(factor_data, columns=factor_data_columns)





def mean_absolute_error(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


import warnings

warnings.filterwarnings('ignore')
sarima_data = data[data.columns]
sarima_forecast_data = pd.DataFrame()
sarima_mae = []
sarima_rmse = []
for i in range(len(data.columns)):
    name = data.columns[i]
    model = sm.tsa.statespace.SARIMAX(data[data.columns[i]], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12), freq='M')
    results = model.fit(maxiter=100, method='powell')
    sarima_data[name] = results.predict(start=30, end=70, dynamic=True, freq='M')
    sarima_mae.append(mean_absolute_error(data[data.columns[i]][30:70], sarima_data[name][30:70]))
    sarima_rmse.append(root_mean_squared_error(data[data.columns[i]][30:70], sarima_data[name][30:70]))
    from pandas.tseries.offsets import DateOffset

    future_dates = [sarima_data.index[-1] + DateOffset(months=x) for x in range(0, 60)]
    future_datest_df = pd.DataFrame(index=future_dates[1:], columns=sarima_data.columns)
    # Future Forecasting
    Forecasted = results.predict(start=70, end=129, dynamic=True)
    sarima_forecast_data[name] = Forecasted

# In[7]:


# FB Prophet
p_data = data[data.columns]
p_data.reset_index(inplace=True)
fbprophet_mae = []
fbprophet_rmse = []
fbprophet_forecast_data = pd.DataFrame()
for i in range(1, 32):
    x = p_data[[p_data.columns[0], p_data.columns[i]]]
    x.columns = ['ds', 'y']
    # Fitting our FB Prophet model
    fb_model = Prophet()
    fb_model.fit(x)
    future = fb_model.make_future_dataframe(periods=60, freq='M')
    prediction = fb_model.predict(future)
    actual_values = x['y'][-len(prediction):]
    fbprophet_mae.append(mean_absolute_error(actual_values, prediction['yhat'][0:70]))
    fbprophet_rmse.append(np.sqrt(mean_squared_error(actual_values, prediction['yhat'][0:70])))

    fig = fb_model.plot(prediction)
    fbprophet_forecast_data[p_data.columns[i]] = prediction['yhat'][-60:]

fbprophet_forecast_data['Dates'] = sarima_forecast_data.index
fbprophet_forecast_data.set_index('Dates', inplace=True)

# In[8]:


# Lstm
lstm_data = data[data.columns]
lstm_rmse = []
lstm_mae = []


# convert an array of values into a dataset matrix
def create_dataset(dataset, time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - time_step - 1):
        a = dataset[i:(i + time_step), 0]  ###i=0, 0,1,2,3-----99   100
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return np.array(dataX), np.array(dataY)


lstm_forecast_data = pd.DataFrame()

# Iterate over each column for forecasting
for i in range(len(data.columns)):
    # Transforming the data using MinMaxScaler
    x = data.iloc[:, i:i + 1].values
    scaler = MinMaxScaler(feature_range=(0, 1))
    x = scaler.fit_transform(x)

    # Splitting the dataset into training and testing sets
    training_size = int(len(x) * 0.8)
    test_size = len(x) - training_size
    train_data, test_data = x[0:training_size, :], x[training_size:len(x), :]

    # Reshape into X=t, t+1, t+2, t+3 and Y=t+4
    time_step = 3
    X_train, y_train = create_dataset(train_data, time_step)
    X_test, y_test = create_dataset(test_data, time_step)

    # Reshape input to be [samples, time steps, features] which is required for LSTM
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    # Create the LSTM model
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(time_step, 1)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(50, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')

    #     # Display model summary
    #     print('==============================================================================')
    #     print(data.columns[i])
    #     print('==============================================================================')
    model.summary()

    # Train the model
    lstm = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=1)

    # Perform predictions and evaluate performance metrics
    train_predict = model.predict(X_train)
    test_predict = model.predict(X_test)
    train_predict = scaler.inverse_transform(train_predict)
    test_predict = scaler.inverse_transform(test_predict)
    look_back = 3
    trainPredictPlot = np.empty_like(x)
    trainPredictPlot[:, :] = np.nan
    trainPredictPlot[look_back:len(train_predict) + look_back, :] = train_predict
    # shift test predictions for plotting
    testPredictPlot = np.empty_like(x)
    testPredictPlot[:, :] = np.nan
    testPredictPlot[len(train_predict) + (look_back * 2) + 1:len(x) - 1, :] = test_predict
    # plot baseline and predictions
    plt.plot(scaler.inverse_transform(x))  # label = Sectors_legends['Legends'][i])
    plt.plot(trainPredictPlot, label="Training data")
    plt.plot(testPredictPlot, color="red", label="Testingn data")
    # plt.title(Sectors_legends['Sectors'][i])
    plt.legend(loc='best')
    # plt.show()

    # Concatenate predictions
    whole_predict = np.concatenate((train_predict, test_predict))
    whole_predict = whole_predict.tolist()

    # Store predictions in Lstm_forecast DataFrame
    predict = [item[0] for item in whole_predict]
    lstm_forecast_data[data.columns[i]] = predict[0:60]

    # RMSE
    lstm_rmse1 = math.sqrt(mean_squared_error(y_test, test_predict))

    # MAE
    lstm_mae1 = mean_absolute_error(y_test, test_predict)

    lstm_mae.append(lstm_mae1)
    lstm_rmse.append(lstm_rmse1)
# Setting indexes
lstm_forecast_data['Dates'] = sarima_forecast_data.index
lstm_forecast_data.set_index('Dates', inplace=True)






import pandas as pd
import xgboost as xgb

xg_data = data[data.columns]

# Forecast function
def forecast_next_values(model, last_date, num_values=60):
    # Create a DataFrame with the same structure as your original data
    forecast_data = pd.DataFrame(index=pd.date_range(start=last_date + pd.Timedelta(days=1), periods=num_values, freq='M'))
    
    # Create features for the forecast data
    X_forecast = create_features(forecast_data)
    
    # Predict using the trained model
    forecast_values = model.predict(X_forecast)
    
    # Add the predicted values to the forecast_data DataFrame
    forecast_data['Prediction'] = forecast_values
    
    return forecast_data

# Create an empty DataFrame to store forecasted values for each column
xgboost_forecast_data = pd.DataFrame()
xgboost_mae = []
xgboost_rmse = []
# Loop through each column in the original dataset
for i in range(data.shape[1]):
    # Extract the column for forecasting
    xg_data = data.iloc[:, i:i+1]  # Select the i-th column

    # Create features function
    def create_features(df):
        df['date'] = df.index
        df['hour'] = df['date'].dt.hour
        df['dayofweek'] = df['date'].dt.dayofweek
        df['quarter'] = df['date'].dt.quarter
        df['month'] = df['date'].dt.month
        df['year'] = df['date'].dt.year
        df['dayofyear'] = df['date'].dt.dayofyear
        df['dayofmonth'] = df['date'].dt.day
        df['weekofyear'] = df['date'].dt.weekofyear

        X = df[['hour', 'dayofweek', 'quarter', 'month', 'year',
                'dayofyear', 'dayofmonth', 'weekofyear']]

        return X

    # Splitting the dataset into train and test split (70% to training and 30% to testing)
    training_size = int(len(xg_data) * 0.7)
    train_data, test_data = xg_data.iloc[:training_size], xg_data.iloc[training_size:]

    # X and Y of training set
    X_train = create_features(train_data)
    y_train = train_data.iloc[:, 0]  # Assuming only one column in the training set

    # X and Y of testing set
    X_test = create_features(test_data)
    y_test = test_data.iloc[:, 0]  # Assuming only one column in the testing set

    # XGBoost model
    model = xgb.XGBRegressor(objective='reg:squarederror', learning_rate=0.6, n_estimators=1000)

    # Model fitting on the test data
    model.fit(X_train, y_train,
              eval_set=[(X_train, y_train), (X_test, y_test)],
              early_stopping_rounds=10,
              verbose=False)
    y_pred = model.predict(X_test)
    xgboost_mae.append(mean_absolute_error(y_test, y_pred))
    
    xgboost_rmse.append(np.sqrt(mean_squared_error(y_test, y_pred)))
    # Joining training and test data
    data_all = pd.concat([test_data, train_data], sort=False)

    # Forecast next 60 values
    forecast_df = forecast_next_values(model,data.index[-1], num_values=60)

    # Add forecasted values to the xgboost_forecast DataFrame
    xgboost_forecast_data[data.columns[i]] = forecast_df['Prediction']

    
    
    
    
    

    


# Combine the datasets into a multi-level column format
all_combined_data = pd.concat(
    [data.T, fbprophet_forecast_data.T, lstm_forecast_data.T, sarima_forecast_data.T, xgboost_forecast_data.T],
    keys=["Actual Data", "FBProphet", "LSTM", "SARIMA", "XGBoost"],
    axis=1
)

all_combined_data = all_combined_data.T
all_combined_data.reset_index(inplace= True)
all_combined_data.rename(columns={
    'level_0': 'Model / Historical',
    'level_1': 'Dates'
}, inplace=True)

# Format float values to 2 decimal places
formatted_data = all_combined_data.copy()
for col in formatted_data.columns:
    if formatted_data[col].dtype in ['float64', 'int64']:
        formatted_data[col] = formatted_data[col].apply(lambda x: f"{x:.2f}")

# Replace all column names at once
asset_name_data =formatted_data.copy()
new_column_names = list(asset_name_data.columns[:2]) + list(asset_names)
asset_name_data.columns = new_column_names




#import ace_tools as tools; tools.display_dataframe_to_user(name="RMSE and MAE Results DataFrame", dataframe=rmse_mae_result_dataframe)


#Rmse and Mae
rmse_mae_result_dataframe = pd.DataFrame()
rmse_mae_result_dataframe["Sectors"] = data.columns
rmse_mae_result_dataframe['Lstm_RMSE'] = lstm_rmse
rmse_mae_result_dataframe['Lstm_MAE'] = lstm_mae
rmse_mae_result_dataframe['Sarima_RMSE'] = sarima_rmse
rmse_mae_result_dataframe['Sarima_MAE'] = sarima_mae
rmse_mae_result_dataframe['Xgboost_RMSE'] = xgboost_rmse
rmse_mae_result_dataframe['Xgboost_MAE'] = xgboost_mae
rmse_mae_result_dataframe['Fbprophet_RMSE'] = fbprophet_rmse
rmse_mae_result_dataframe['Fbprophet_MAE'] = fbprophet_mae





#Discrete Dataframe function
def calculate_discrete_dataframe(df):
    Discrete = pd.DataFrame({'Discrete Return': ['Year1', 'Year2', 'Year3', 'Year4', 'Year5']})

    for i in range(df.shape[1]):
        temp_list = []
        val = df.iloc[:, i].to_list()

        year1 = np.sum(np.array(val[0:12]))
        temp_list.append(year1)

        year2 = np.sum(np.array(val[12:24]))
        temp_list.append(year2)

        year3 = np.sum(np.array(val[24:36]))
        temp_list.append(year3)

        year4 = np.sum(np.array(val[36:48]))
        temp_list.append(year4)

        year5 = np.sum(np.array(val[48:60]))
        temp_list.append(year5)
        Discrete[df.columns[i]] = temp_list
    
    # Transpose, reset index, and rename columns
    Discrete = Discrete.T.reset_index().rename(columns={'index': 'Discrete Return', 0: 'Year1', 1: 'Year2', 2: 'Year3', 3: 'Year4', 4: 'Year5'})
    Discrete.drop([0], axis=0, inplace=True)
    Discrete = Discrete.reset_index(drop=True)
    
    return Discrete



#Unit Dataframe
def calculate_unit_dataframe(df):
    Unit = pd.DataFrame({'Unit Price': ['Year1', 'Year2', 'Year3', 'Year4', 'Year5']})

    for i in range(df.shape[1]):
        temp_list = []
        val = df.iloc[:, i].to_list()

        year1 = 1 * (100 + np.sum(np.array(val[0:12])))
        year1 = year1 / 100
        temp_list.append(year1)

        year2 = 1 * (100 + np.sum(np.array(val[12:24])))
        year2 = year2 / 100
        temp_list.append(year2)

        year3 = 1 * (100 + np.sum(np.array(val[24:36])))
        year3 = year3 / 100
        temp_list.append(year3)

        year4 = 1 * (100 + np.sum(np.array(val[36:48])))
        year4 = year4 / 100
        temp_list.append(year4)

        year5 = 1 * (100 + np.sum(np.array(val[48:60])))
        year5 = year5 / 100
        temp_list.append(year5)

        Unit[df.columns[i]] = temp_list
    
        # Transpose, reset index, and rename columns
    Unit = Unit.T.reset_index().rename(columns={'index': 'Unit', 0: 'Year1', 1: 'Year2', 2: 'Year3', 3: 'Year4', 4: 'Year5'})
    Unit.drop([0], axis=0, inplace=True)
    Unit = Unit.reset_index(drop=True)
    return Unit

#Cumulative Return Function
def calculate_cumulative_returns(discrete_df, unit_df):
    cumulative_returns = discrete_df[[discrete_df.columns[0], discrete_df.columns[1]]]
    cumulative_returns['Year2'] = ((unit_df['Year2']/1) ** (0.5) - 1) * 100
    cumulative_returns['Year3'] = ((unit_df['Year3']/1) ** (0.33) - 1) * 100
    cumulative_returns['Year4'] = ((unit_df['Year4']/1) ** (0.25) - 1) * 100
    cumulative_returns['Year5'] = ((unit_df['Year5']/1) ** (0.20) - 1) * 100
    cumulative_returns.rename(columns={discrete_df.columns[0]: 'Cumulative Returns %'}, inplace=True)
    return cumulative_returns


def covariance_matrix_table(data):
    # Calculate covariance matrix using NumPy's cov function
    covariance_matrix = np.cov(data, rowvar=False)
    # Convert the covariance matrix to a pandas DataFrame
    columns = [data.columns[i] for i in range(covariance_matrix.shape[0])]
    covariance_df = pd.DataFrame(covariance_matrix, columns=columns, index=columns)
    return covariance_df







import numpy as np
import matplotlib.pyplot as plt
import gurobipy as gp
from gurobipy import GRB

def pre_optimize_weights(df, minimum_weight,maximum_weight, total_investment,volitility_value):

    # Define asset names list
    asset_names = ['Alternatives', 'Alternatives', 'Cash and Term Deposits', 'Cash and Term Deposits', 'Fixed Income', 'Commodities', 'International Securities', 'International Securities', 'Australian Securities', 'Australian Securities', 'Australian Securities', 'Australian Securities', 'International Securities', 'International Securities', 'International Securities', 'International Securities', 'International Securities', 'Infrastructure', 'International Securities', 'Private Equity', 'Fixed Income', 'Fixed Income', 'Fixed Income', 'Fixed Income', 'Fixed Income', 'Fixed Income', 'Fixed Income', 'Alternatives', 'Property', 'Property', 'Property']
    df = df.astype(float)
    # Define minimum and maximum allocations for each asset class
    min_allocations = {
        'Cash and Term Deposits': 10.0,
        'Fixed Income': 20.0,
        'Australian Securities': 10.0,
        'International Securities': 0.0,
        'Private Equity': 0.0,
        'Property': 0.0,
        'Infrastructure': 0.0,
        'Commodities': 0.0,
        'Alternatives': 0.0
    }
    max_allocations = {
        'Cash and Term Deposits': 60.0,
        'Fixed Income': 60.0,
        'Australian Securities': 50.0,
        'International Securities': 25.0,
        'Private Equity': 20.0,
        'Property': 20.0,
        'Infrastructure': 20.0,
        'Commodities': 10.0,
        'Alternatives': 20.0
    }

    # Store risk values and portfolio returns
    risk_values = []
    portfolio_returns = []

    optimal_model = pd.DataFrame()
    optimal_model['Asset'] = asset_names
    num_total_assets = len(df)

    for year in df.columns:
        # Create a model
        m = gp.Model()

        # Create variables
        weights = m.addVars(num_total_assets, lb=0, ub=20, vtype=gp.GRB.CONTINUOUS, name="weights")
        incidator = m.addVars(num_total_assets, vtype=gp.GRB.BINARY)
        volatility = m.addVar(name="volatility")  # Remove lb=0 from here
        


        for asset in set(asset_names):
            indices = [i for i, name in enumerate(asset_names) if name == asset]
            min_allocation = min_allocations[asset]
            max_allocation = max_allocations[asset]
            m.addConstr(gp.quicksum(weights[i] for i in indices) >= min_allocation, name=f"min_allocation_{asset}")
            m.addConstr(gp.quicksum(weights[i] for i in indices) <= max_allocation, name=f"max_allocation_{asset}")

        m.addConstr(gp.quicksum(weights[i] for i in range(num_total_assets)) == 100, name="total_weights")
        m.addConstrs(weights[i] >= minimum_weight * incidator[i] for i in range(num_total_assets))
        m.addConstrs(weights[i] <= maximum_weight * incidator[i] for i in range(num_total_assets))
        m.addConstr(gp.quicksum(incidator[i] for i in range(num_total_assets)) >= total_investment)

        m.setObjective(gp.quicksum(weights[i] * df[year][i] for i in range(num_total_assets)), GRB.MAXIMIZE)
        cov_matrix = data.cov()
        portfolio_variance = gp.quicksum(weights[i] * weights[j] * cov_matrix.iloc[i, j] for i in range(num_total_assets) for j in range(num_total_assets))
        m.addConstr(volatility * volatility <= volitility_value**2, name="volatility_constraint")
        m.optimize()
        if m.status == GRB.OPTIMAL:
            optimal_weights = np.array([weights[i].x for i in range(num_total_assets)])
            optimal_model[f'Optimal_Weights_{year}'] = optimal_weights
            sumproduct_year = optimal_weights* df[year].values
            optimal_model[f'Sumproduct_{year}'] = sumproduct_year
            
            # Calculate portfolio risk (volatility)
            cov_matrix = (df/100).T.cov()
            portfolio_risk = np.sqrt(np.dot(optimal_weights.T, np.dot(cov_matrix, optimal_weights)))
            risk_values.append(portfolio_risk)

            # Calculate portfolio return
            portfolio_return = np.dot(optimal_weights, df[year].values)
            portfolio_returns.append(portfolio_return)

            # Simulate portfolios for efficient frontier
            num_portfolios = 1000
            simulated_portfolio_returns = []
            simulated_portfolio_risks = []
            for _ in range(num_portfolios):
                random_weights = np.random.random(num_total_assets)
                random_weights /= np.sum(random_weights)
                simulated_portfolio_return = np.dot(random_weights, df[year].values)
                simulated_portfolio_std_dev = np.sqrt(np.dot(random_weights.T, np.dot(cov_matrix, random_weights)))
                simulated_portfolio_returns.append(simulated_portfolio_return)
                simulated_portfolio_risks.append(simulated_portfolio_std_dev)
            # Calculate scaling factors
            risk_scaling_factor = np.max(risk_values) / np.max(simulated_portfolio_risks)
            return_scaling_factor = np.max(portfolio_returns) / np.max(simulated_portfolio_returns)

            # Rescale simulated portfolio risks and returns
            scaled_simulated_portfolio_risks = [risk * risk_scaling_factor for risk in simulated_portfolio_risks]
            scaled_simulated_portfolio_returns = [return_ * return_scaling_factor for return_ in simulated_portfolio_returns]





    return optimal_model, risk_values ,portfolio_returns,scaled_simulated_portfolio_risks,scaled_simulated_portfolio_returns

# Example usage:
# optimal_model, risk_values, portfolio_returns = optimize_weights(df, minimum_weight, total_investment, cov_matrix, maximum_volatility)








def pre_optimize_weights_max_sharp(df, minimum_weight, maximum_weight, total_investment, volitility_value,
                                   risk_measure_type, risk_measure_period,
                                   max_risk_tolerance, min_investment_size, max_investment_size):
    # Define asset names list
    asset_names = ['Alternatives', 'Alternatives', 'Cash and Term Deposits', 'Cash and Term Deposits', 'Fixed Income',
                   'Commodities', 'International Securities', 'International Securities', 'Australian Securities',
                   'Australian Securities', 'Australian Securities', 'Australian Securities',
                   'International Securities', 'International Securities', 'International Securities',
                   'International Securities', 'International Securities', 'Infrastructure', 'International Securities',
                   'Private Equity', 'Fixed Income', 'Fixed Income', 'Fixed Income', 'Fixed Income', 'Fixed Income',
                   'Fixed Income', 'Fixed Income', 'Alternatives', 'Property', 'Property', 'Property']
    df = df.astype(float)
    # Define minimum and maximum allocations for each asset class
    min_allocations = {
        'Cash and Term Deposits': 10.0,
        'Fixed Income': 20.0,
        'Australian Securities': 10.0,
        'International Securities': 0.0,
        'Private Equity': 0.0,
        'Property': 0.0,
        'Infrastructure': 0.0,
        'Commodities': 0.0,
        'Alternatives': 0.0
    }
    max_allocations = {
        'Cash and Term Deposits': 60.0,
        'Fixed Income': 60.0,
        'Australian Securities': 50.0,
        'International Securities': 25.0,
        'Private Equity': 20.0,
        'Property': 20.0,
        'Infrastructure': 20.0,
        'Commodities': 10.0,
        'Alternatives': 20.0
    }

    # Store risk values and portfolio returns
    risk_values = []
    portfolio_returns = []

    optimal_model = pd.DataFrame()
    optimal_model['Asset'] = asset_names
    num_total_assets = len(df)

    for year in df.columns:
        # Create a model
        m = gp.Model()

        # Create variables
        weights = m.addVars(num_total_assets, lb=0, ub=20, vtype=gp.GRB.CONTINUOUS, name="weights")
        incidator = m.addVars(num_total_assets, vtype=gp.GRB.BINARY)
        volatility = m.addVar(name="volatility")  # Remove lb=0 from here

        excess_returns = np.array([df[year][i] for i in range(num_total_assets)])  # Assuming expected returns are in df
        risk_free_rate = 0.02  # Example risk-free rate

        for asset in set(asset_names):
            indices = [i for i, name in enumerate(asset_names) if name == asset]
            min_allocation = min_allocations[asset]
            max_allocation = max_allocations[asset]
            m.addConstr(gp.quicksum(weights[i] for i in indices) >= min_allocation, name=f"min_allocation_{asset}")
            m.addConstr(gp.quicksum(weights[i] for i in indices) <= max_allocation, name=f"max_allocation_{asset}")

        m.addConstr(gp.quicksum(weights[i] for i in range(num_total_assets)) == 100, name="total_weights")
        m.addConstrs(weights[i] >= minimum_weight * incidator[i] for i in range(num_total_assets))
        m.addConstrs(weights[i] <= maximum_weight * incidator[i] for i in range(num_total_assets))
        m.addConstr(gp.quicksum(incidator[i] for i in range(num_total_assets)) >= total_investment)

        # Reformulate the objective function
        large_constant = 1e6
        m.setObjective((gp.quicksum(weights[i] * excess_returns[i] for i in range(
            num_total_assets)) - risk_free_rate * large_constant * volatility) / large_constant, GRB.MAXIMIZE)

        cov_matrix = data.cov()
        portfolio_variance = gp.quicksum(
            weights[i] * weights[j] * cov_matrix.iloc[i, j] for i in range(num_total_assets) for j in
            range(num_total_assets))
        m.addConstr(volatility * volatility <= volitility_value ** 2, name="volatility_constraint")
        m.optimize()
        if m.status == GRB.OPTIMAL:
            optimal_weights = np.array([weights[i].x for i in range(num_total_assets)])
            optimal_model[f'Optimal_Weights_{year}'] = optimal_weights
            sumproduct_year = optimal_weights * df[year].values
            optimal_model[f'Sumproduct_{year}'] = sumproduct_year

            # Calculate portfolio risk (volatility)
            cov_matrix = (df / 100).T.cov()
            portfolio_risk = np.sqrt(np.dot(optimal_weights.T, np.dot(cov_matrix, optimal_weights)))
            risk_values.append(portfolio_risk)

            # Calculate portfolio return
            portfolio_return = np.dot(optimal_weights, df[year].values)
            portfolio_returns.append(portfolio_return)

            # Simulate portfolios for efficient frontier
            num_portfolios = 1000
            simulated_portfolio_returns = []
            simulated_portfolio_risks = []
            for _ in range(num_portfolios):
                random_weights = np.random.random(num_total_assets)
                random_weights /= np.sum(random_weights)
                simulated_portfolio_return = np.dot(random_weights, df[year].values)
                simulated_portfolio_std_dev = np.sqrt(np.dot(random_weights.T, np.dot(cov_matrix, random_weights)))
                simulated_portfolio_returns.append(simulated_portfolio_return)
                simulated_portfolio_risks.append(simulated_portfolio_std_dev)

            # Calculate scaling factors
            risk_scaling_factor = np.max(risk_values) / np.max(simulated_portfolio_risks)
            return_scaling_factor = np.max(portfolio_returns) / np.max(simulated_portfolio_returns)

            # Rescale simulated portfolio risks and returns
            scaled_simulated_portfolio_risks = [risk * risk_scaling_factor for risk in simulated_portfolio_risks]
            scaled_simulated_portfolio_returns = [return_ * return_scaling_factor for return_ in
                                                  simulated_portfolio_returns]

    return optimal_model, risk_values, portfolio_returns, scaled_simulated_portfolio_risks, scaled_simulated_portfolio_returns




import numpy as np
import matplotlib.pyplot as plt
import gurobipy as gp
from gurobipy import GRB

def optimize_weights(df,target_objective, tolerance):

    # Define asset names list
    
    asset_names = ['Alternatives', 'Alternatives', 'Cash and Term Deposits', 'Cash and Term Deposits', 'Fixed Income', 'Commodities', 'International Securities', 'International Securities', 'Australian Securities', 'Australian Securities', 'Australian Securities', 'Australian Securities', 'International Securities', 'International Securities', 'International Securities', 'International Securities', 'International Securities', 'Infrastructure', 'International Securities', 'Private Equity', 'Fixed Income', 'Fixed Income', 'Fixed Income', 'Fixed Income', 'Fixed Income', 'Fixed Income', 'Fixed Income', 'Alternatives', 'Property', 'Property', 'Property']
    df = df [df.columns[1:6]]
    df = df.astype(float)
   
    # Define minimum and maximum allocations for each asset class
    min_allocations = {
        'Cash and Term Deposits': 10.0,
        'Fixed Income': 20.0,
        'Australian Securities': 10.0,
        'International Securities': 0.0,
        'Private Equity': 0.0,
        'Property': 0.0,
        'Infrastructure': 0.0,
        'Commodities': 0.0,
        'Alternatives': 0.0
    }
    max_allocations = {
        'Cash and Term Deposits': 60.0,
        'Fixed Income': 60.0,
        'Australian Securities': 50.0,
        'International Securities': 25.0,
        'Private Equity': 20.0,
        'Property': 20.0,
        'Infrastructure': 20.0,
        'Commodities': 10.0,
        'Alternatives': 20.0
    }

    # Store risk values and portfolio returns
    risk_values = []
    portfolio_returns = []

    optimal_model = pd.DataFrame()
    optimal_model['Asset'] = asset_names
    num_total_assets = len(df)

    for year in df.columns:
        # Create a model
        m = gp.Model()

        # Create variables
        weights = m.addVars(num_total_assets, lb=0, ub=20, vtype=gp.GRB.CONTINUOUS, name="weights")
        incidator = m.addVars(num_total_assets, vtype=gp.GRB.BINARY)
        volatility = m.addVar(name="volatility")  # Remove lb=0 from here
        
        # Add constraint to ensure the objective is within the desired range
        m.addConstr(gp.quicksum(weights[i] * df[year][i] for i in range(num_total_assets)) >= target_objective - tolerance,
                    name="objective_lower_bound")
        m.addConstr(gp.quicksum(weights[i] * df[year][i] for i in range(num_total_assets)) <= target_objective + tolerance,
                    name="objective_upper_bound")

        for asset in set(asset_names):
            indices = [i for i, name in enumerate(asset_names) if name == asset]
            min_allocation = min_allocations[asset]
            max_allocation = max_allocations[asset]
            m.addConstr(gp.quicksum(weights[i] for i in indices) >= min_allocation, name=f"min_allocation_{asset}")
            m.addConstr(gp.quicksum(weights[i] for i in indices) <= max_allocation, name=f"max_allocation_{asset}")

        m.addConstr(gp.quicksum(weights[i] for i in range(num_total_assets)) == 100, name="total_weights")

        

        m.setObjective(gp.quicksum(weights[i] * df[year][i] for i in range(num_total_assets)), GRB.MAXIMIZE)

        m.optimize()
        if m.status == GRB.OPTIMAL:
            optimal_weights = np.array([weights[i].x for i in range(num_total_assets)])
            optimal_model[f'Optimal_Weights_{year}'] = optimal_weights
            sumproduct_year = optimal_weights*df[year].values
            optimal_model[f'Sumproduct_{year}'] = sumproduct_year
            
            # Calculate portfolio risk (volatility)
            cov_matrix = (df/100).T.cov()
            portfolio_risk = np.sqrt(np.dot(optimal_weights.T, np.dot(cov_matrix, optimal_weights)))
            risk_values.append(portfolio_risk)

            # Calculate portfolio return
            portfolio = np.dot(optimal_weights, df[year].values)
            portfolio_returns.append(portfolio)

            # Simulate portfolios for efficient frontier
            num_portfolios = 1000
            simulated_portfolio_returns = []
            simulated_portfolio_risks = []
            for _ in range(num_portfolios):
                random_weights = np.random.random(num_total_assets)
                random_weights /= np.sum(random_weights)
                simulated_portfolio_return = np.dot(random_weights, df[year].values)
                simulated_portfolio_std_dev = np.sqrt(np.dot(random_weights.T, np.dot(cov_matrix, random_weights)))
                simulated_portfolio_returns.append(simulated_portfolio_return)
                simulated_portfolio_risks.append(simulated_portfolio_std_dev)
            # Calculate scaling factors
            risk_scaling_factor = np.max(risk_values) / np.max(simulated_portfolio_risks)
            return_scaling_factor = np.max(portfolio_returns) / np.max(simulated_portfolio_returns)

            # Rescale simulated portfolio risks and returns
            scaled_simulated_portfolio_risks = [risk * risk_scaling_factor for risk in simulated_portfolio_risks]
            scaled_simulated_portfolio_returns = [return_ * return_scaling_factor for return_ in simulated_portfolio_returns]





    return optimal_model, risk_values

# Example usage:
# optimal_model, risk_values, portfolio_returns = optimize_weights(df, minimum_weight, total_investment, cov_matrix, maximum_volatility)







import io
import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import plotly.express as px
import pandas as pd


# Importing required functions, assuming they are defined elsewhere
def correlate_table(data):
    
    #Now printing how much each sector has correlating with other sectors.
    null_list = []
    for i in range(31):
        null_list_name  = 'list' + str(i)
        null_list.append(null_list_name)
        null_list[i] = data.corr().values[i]
    index_list=[]
    for i in range(31):
        name  = 'Column' + str(i)
        index_list.append(name)
    for i in range(31):
        index_list[i]  = []
        for x in range(31):

            if null_list[i][x] >= 0.7:
                index_list[i].append(x)
    number_corr = []
    for i in range(31):
        x = (len(index_list[i]))
        number_corr.append(x)
    correlation_table=pd.DataFrame({
        'Sectors':data.columns,
        'Correlation Number': number_corr})
    return correlation_table
def generate_dataframe(dataframe):

    df = calculate_discrete_dataframe(dataframe)
    for i in range(1,6):
        df[f'Categorize_Year{i}'] = "Neutral"
    return df
def create_bar_chart(metric, color, title):
    return {
        "data": [
            go.Bar(
                y=asset_names,
                x=risk_metrics[metric],
                orientation='h',
                marker=dict(color=color),
            )
        ],
        "layout": go.Layout(
            title=title,
            xaxis=dict(title='Percentage'),
            yaxis=dict(title='Sectors', automargin=True),
            height=900,  # Increase height to fit all sectors
            margin=dict(l=150, r=50, t=50, b=50)  # Adjust left margin to accommodate longer sector names
        )
    }
# Check if the columns are MultiIndex and flatten accordingly
if isinstance(all_combined_data.columns, pd.MultiIndex):
    flattened_columns = [
        {"name": list(map(str, col)), "id": " - ".join(map(str, col))}
        for col in all_combined_data.columns
    ]
    all_combined_data.columns = [" - ".join(map(str, col)) for col in all_combined_data.columns]
else:
    flattened_columns = [{"name": col, "id": col} for col in all_combined_data.columns]
    
# Make column names unique
def make_unique_columns(columns):
    seen = {}
    unique_columns = []
    for col in columns:
        if col not in seen:
            seen[col] = 0
        else:
            seen[col] += 1
            col = f"{col}_{seen[col]}"
        unique_columns.append(col)
    return unique_columns

asset_name_data.columns = make_unique_columns(asset_name_data.columns)
# Initialize optimal_model as an empty DataFrame
optimal_model = pd.DataFrame()

app = dash.Dash(__name__, external_stylesheets=['assets/all.css'])

asset_names = ['Alternatives', 'Alternatives', 'Cash and Term Deposits', 'Cash and Term Deposits', 'Fixed Income', 'Commodities', 'International Securities', 'International Securities', 'Australian Securities', 'Australian Securities', 'Australian Securities', 'Australian Securities', 'International Securities', 'International Securities', 'International Securities', 'International Securities', 'International Securities', 'Infrastructure', 'International Securities', 'Private Equity', 'Fixed Income', 'Fixed Income', 'Fixed Income', 'Fixed Income', 'Fixed Income', 'Fixed Income', 'Fixed Income', 'Alternatives', 'Property', 'Property', 'Property']


historical = calculate_discrete_dataframe(data)
covariance_table = covariance_matrix_table(data)
# Adding 5 new columns with the default value 'Neutral'
lstm_forecast = generate_dataframe(lstm_forecast_data)
sarima_forecast = generate_dataframe(sarima_forecast_data)
xgboost_forecast = generate_dataframe(xgboost_forecast_data)
fbprophet_forecast = generate_dataframe(fbprophet_forecast_data)
df = generate_dataframe(lstm_forecast_data)
df_historical_camulative = calculate_cumulative_returns(calculate_discrete_dataframe(data), calculate_unit_dataframe(data))
df2_lstm = calculate_cumulative_returns(calculate_discrete_dataframe(lstm_forecast_data),
                                   calculate_unit_dataframe(lstm_forecast_data))
df2_sarima = calculate_cumulative_returns(calculate_discrete_dataframe(sarima_forecast_data),
                                     calculate_unit_dataframe(sarima_forecast_data))
df2_xgboost = calculate_cumulative_returns(calculate_discrete_dataframe(xgboost_forecast_data),
                                      calculate_unit_dataframe(xgboost_forecast_data))
df2_fbprophet = calculate_cumulative_returns(calculate_discrete_dataframe(fbprophet_forecast_data),
                                        calculate_unit_dataframe(fbprophet_forecast_data))
df_historical_cumulative_asset = df_historical_camulative
df_historical_cumulative_asset['Asset'] =  asset_names
df_historical_cumulative_asset = df_historical_cumulative_asset.pivot_table(index='Asset', aggfunc='mean')
df_historical_cumulative_asset = df_historical_cumulative_asset.reset_index()
truncated_sector_names = [sector[13:-8] for sector in rmse_mae_result_dataframe['Sectors']]
change_df = sarima_forecast_data
forecast_compare_options = {
    'SARIMA': df2_sarima,
    'XGBoost': df2_xgboost,
    'FBProphet': df2_fbprophet,
    'LSTM':df2_lstm
}
forecast_options = {
    'SARIMA': df2_sarima,
    'XGBoost': df2_xgboost,
    'FBProphet': df2_fbprophet,
    'LSTM':df2_lstm
}

dropdown_options = {
    'LSTM Forecast': lstm_forecast,
    'SARIMA Forecast': sarima_forecast,
    'XGBoost Forecast': xgboost_forecast,
    'FBProphet Forecast': fbprophet_forecast
}

data_insight_sectors = [col for col in data.columns if col.startswith('Sector')]
app.layout = html.Div([# Site Header
html.Div([
    html.Div([
        html.A([
            html.Img(src="https://cashelfo.com/wp-content/themes/cashel/img/CFO_symbol_white.png",
                     alt="CashelFamilyOffice Logo", className="logo-icon"),
            html.Img(src="https://cashelfo.com/wp-content/themes/cashel/img/CFO_symbol_white.png",
                     alt="CashelFamilyOffice logo", className="logo-full desktop-only"),
            html.Img(src="https://cashelfo.com/wp-content/themes/cashel/img/CFO_symbol_white.png",
                     alt="CashelFamilyOffice logo", className="logo-full mobile-only"),
        ], href="/", className="logo"),
        html.A("03 9209 9000", href="tel:0392099000", className="get-in-touch"),
        html.A([
            html.Img(src="https://cashelfo.com/wp-content/themes/cashel/img/icon_form_white.png",
                     alt="CashelFamilyOffice form icon", title="Forms", className="header-icon icon"),
            html.Img(src="https://cashelfo.com/wp-content/themes/cashel/img/icon_form_white.png",
                     alt="CashelFamilyOffice form icon", title="Forms", className="header-icon icon-full"),
        ], href="/forms", className=""),
        html.Span([
            html.Img(src="https://cashelfo.com/wp-content/themes/cashel/img/icon_client_white.png",
                     alt="CashelFamilyOffice client icon", title="Client Login", className="header-icon two icon"),
            html.Img(src="https://cashelfo.com/wp-content/themes/cashel/img/icon_client_white.png",
                     alt="CashelFamilyOffice client", title="Client Login", className="header-icon two icon-full"),
            html.Div([
                html.Div([
                    html.Ul([
                        # Uncomment and add list items for client login links if needed
                        # html.Li(html.A("Xplore", href="https://portal.xplorewealthservices.com.au/linearBpms/loginCashelHouse.jsp", target="_blank")),
                        # html.Li(html.A("Preamium", href="https://login.onpraemium.com/CashelFS/", target="_blank")),
                        # html.Li(html.A("Hub24", href="https://my.hub24.com.au/Hub24/Login.aspx", target="_blank")),
                    ])
                ], className="inner")
            ], className="dropdown-content")
        ], className="dropdown"),
        html.Span([
            html.Span("", className=""),
            html.Span("", className="")
        ], className="menu-button"),
    ], className="container")
], id="site-header"),

# Menu Box
html.Div([
    html.Div([
        html.Span([
            html.Span(""),
            html.Span("")
        ], className="menu-button")
    ], className="container"),
    html.Div([
        # Mobile Menu
        html.Div([
            html.Ul([
                html.Li(html.A("Home", href="https://cashelfo.com/")),
                html.Li([
                    html.A("Wealth Services"),
                    html.Ul([
                        html.Li(html.A("Plan", href="https://cashelfo.com/wealth/plan/")),
                        html.Li(html.A("Protect", href="https://cashelfo.com/wealth/protect/")),
                        html.Li(html.A("Invest", href="https://cashelfo.com/wealth/invest/")),
                        html.Li(html.A("Transform", href="https://cashelfo.com/wealth/transform/")),
                        html.Li(html.A("Organise", href="https://cashelfo.com/wealth/organise/")),
                    ], className="sub-menu")
                ]),
                html.Li([
                    html.A("Family Office"),
                    html.Ul([
                        html.Li(html.A("Our Services", href="https://cashelfo.com/our-services/")),
                        html.Li(html.A("Global Product Range", href="https://cashelfo.com/wealth/global-product-range/")),
                        html.Li(html.A("Investment Platforms", href="https://cashelfo.com/wealth/investment-platforms/")),
                        html.Li(html.A("Join Cashel Family", href="https://cashelfo.com/join-cashelfo/")),
                    ], className="sub-menu")
                ]),
                html.Li(html.A("Insights", href="https://cashelfo.com/wealth-insights/")),
                html.Li(html.A("Contact Us", href="https://cashelfo.com/contact-us/")),
                html.Li(html.A("Client Login", href="https://portal.xplorewealthservices.com.au/linearBpms/loginCashelHouse.jsp", target="_blank", rel="noopener")),
            ], id="main-menu", className="navbar-nav")
        ], className="mobile-menu"),

        # Inner
        html.Div([
            # Menu Left
            html.Div([
                html.Ul([
                    html.Li(html.A("Home", href="https://cashelfo.com/")),
                    html.Li(html.A("Wealth Services", href="https://cashelfo.com/wealth/plan/", className="wealth-services")),
                    html.Li(html.A("Family Office", href="https://cashelfo.com/our-services/", className="family-office")),
                    html.Li(html.A("About Us", href="https://cashelfo.com/about-us/", className="about-us")),
                    html.Li(html.A("Insights", href="https://cashelfo.com/wealth-insights/")),
                    html.Li(html.A("Contact Us", href="https://cashelfo.com/contact-us/")),
                ], id="main-menu-desktop", className="navbar-nav")
            ], id="menu-left"),

            # Menu Right
            html.Div([
                html.Ul([
                    html.Li(html.A("Plan", href="https://cashelfo.com/wealth/plan/", id="menu-item-807")),
                    html.Li(html.A("Protect", href="https://cashelfo.com/wealth/protect/", id="menu-item-808")),
                    html.Li(html.A("Invest", href="https://cashelfo.com/wealth/invest/", id="menu-item-804")),
                    html.Li(html.A("Transform", href="https://cashelfo.com/wealth/transform/", id="menu-item-809")),
                    html.Li(html.A("Borrow", href="https://cashelfo.com/wealth/borrow/", id="menu-item-6201")),
                    html.Li(html.A("Organise", href="https://cashelfo.com/wealth/organise/", id="menu-item-806")),
                ], id="wealth-services", className="navbar-nav"),
                html.Ul([
                    html.Li(html.A("About Us", href="https://cashelfo.com/about-us/", id="menu-item-913")),
                    html.Li(html.A("Community", href="https://cashelfo.com/about-us/community/", id="menu-item-911")),
                    html.Li(html.A("Global Partnerships", href="https://cashelfo.com/about-us/global-partnerships/", id="menu-item-912")),
                ], id="about-us", className="navbar-nav"),
                html.Ul([
                    html.Li(html.A("About Us", href="https://cashelfo.com/about-us/")),
                    html.Li(html.A("Community", href="https://cashelfo.com/about-us/community/")),
                    html.Li(html.A("Global Partnerships", href="https://cashelfo.com/about-us/global-partnerships/")),
                ], id="superannuation", className="navbar-nav"),
                html.Ul([
                    html.Li(html.A("Our Services", href="https://cashelfo.com/our-services/", id="menu-item-815")),
                    html.Li(html.A("Global Product Range", href="https://cashelfo.com/wealth/global-product-range/", id="menu-item-817")),
                    html.Li(html.A("Investment Platforms", href="https://cashelfo.com/wealth/investment-platforms/", id="menu-item-818")),
                    html.Li(html.A("Join Cashel Family Office", href="https://cashelfo.com/join-cashelfo/", id="menu-item-5154")),
                ], id="family-office", className="navbar-nav"),
            ], id="menu-right"),
        ], className="inner"),
    ], className="container"),
], id="menu-box"),    # Button to trigger file upload
    
    html.H1("Time Series Data"),
    html.Button("Upload File", id="upload-button", n_clicks=0, style={
        'margin': '10px',
        'padding': '10px',
        'fontSize': '16px'
    }),
    
    # Upload component hidden initially
    dcc.Upload(
        id='upload-data',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select Files')
        ]),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px',
            'display': 'none'  # Hidden initially
        },
        multiple=False  # Single file upload
    ),

    # Toggle button to show/hide the table
    html.Button("Toggle Table", id="toggle-button", n_clicks=0, style={
        'margin': '10px',
        'padding': '10px',
        'fontSize': '16px'
    }),

    # Div to display table
    html.Div(id='output-table', style={'display': 'none'}),
        html.H2("Economic actuals and forecasts"),
    dash_table.DataTable(
        id='economic-actuals-table',
        columns=[
            {"name": col, "id": col} for col in factor_df.columns
        ],
        data=factor_df.to_dict('records'),
                        style_table={'overflowX': 'auto', 'height': '400px', 'overflowY': 'scroll'},
                style_cell={
                    'textAlign': 'left',
                    'padding': '5px',
                    'fontSize': '12px',  # Decrease font size
                    'whiteSpace': 'normal',
                    'height': 'auto'  # Adjust row height
                },
                style_data={
                    'whiteSpace': 'normal',
                },
                style_header={
                    'backgroundColor': 'lightgrey',
                    'fontWeight': 'bold',
                    'fontSize': '14px'  # Slightly larger font for headers
                }
    ),# Hidden initially
    
    
        html.H2("Sector class historical returns and forecast"),
    dash_table.DataTable(
        id='multi-level-table',
        columns=flattened_columns,
        data=formatted_data.to_dict('records'),
        merge_duplicate_headers=True,
        style_table={'overflowX': 'auto'},
        style_cell={
            'textAlign': 'center',
            'whiteSpace': 'normal',
            'height': 'auto',
            'fontFamily': 'Arial',
            'fontSize': '14px',
        },
        style_header={
            'backgroundColor': 'rgb(230, 230, 230)',
            'fontWeight': 'bold',
        }
    ),
    
    
            html.H2("Asset class historical returns and forecast"),
    dash_table.DataTable(
        id='multi-level-table2',
        columns=[{"name": col, "id": col} for col in asset_name_data.columns],
        data=asset_name_data.to_dict('records'),  
        merge_duplicate_headers=True,
        style_table={'overflowX': 'auto'},
        style_cell={
            'textAlign': 'center',
            'whiteSpace': 'normal',
            'height': 'auto',
            'fontFamily': 'Arial',
            'fontSize': '14px',
        },
        style_header={
            'backgroundColor': 'rgb(230, 230, 230)',
            'fontWeight': 'bold',
        }
    ),
    
                        html.H1("Data Analysis"),    html.Div([
                            dcc.Graph(figure=create_bar_chart('Annualized Return', '#6baed6', "Annualized Return")),
                            dcc.Graph(figure=create_bar_chart('Geometric Annualized Return', '#74c476', "Geometric Annualized Return")),
                            dcc.Graph(figure=create_bar_chart('Annualized Volatility', '#fc9272', "Annualized Volatility")),
                            dcc.Graph(figure=create_bar_chart('Sharpe Ratio', '#fdae61', "Sharpe Ratio")),
                            dcc.Graph(figure=create_bar_chart('Value at Risk (95%)', '#9e9ac8', "Value at Risk (95%)")),
                            dcc.Graph(figure=create_bar_chart('Max Drawdown', '#ef3b2c', "Maximum Drawdown")),
                        ], style={
                            'display': 'grid',
                            'gridTemplateColumns': 'repeat(3, 1fr)',
                            'gridGap': '5px',
                            'padding': '5px'
                        }),
                     # Dropdown to select multiple columns for plotting
                       html.Label("Select Column(s):"),
                       dcc.Dropdown(
                           id='column-dropdown',
                           style = {'padding':'10px'},
                           options=[{'label': col, 'value': col} for col in data.columns],
                           value=[data.columns[0]],  # Default value (can be a list of columns)
                           multi=True
                       ),

                       # Line chart to display the selected column(s) over time
                       dcc.Graph(id='line-chart'),

                       # Dropdown to select multiple forecast methods
                       html.Label("Select Forecast Method(s):"),
                       dcc.Dropdown(
                           id='forecast-method-dropdown',
                           options=[
                               {'label': 'Original Data', 'value': 'original'},
                               {'label': 'LSTM Forecast', 'value': 'lstm'},
                               {'label': 'XGBoost Forecast', 'value': 'xgboost'},
                               {'label': 'SARIMA Forecast', 'value': 'sarima'},
                               {'label': 'FBProphet Forecast', 'value': 'fbprophet'},
                           ],
                           value=['original'],  # Default value (can be a list of methods)
                           multi=True
                       ),

                       # Table to display the data or forecast
                       dash_table.DataTable(
                           id='data-table1',
                           columns=[{'name': col, 'id': col} for col in data.columns],
                           data=data.applymap(lambda x: f"{x:.2f}%").to_dict('records'),
                           style_table={'height': '400px', 'overflowY': 'auto'},
                           style_cell={'textAlign': 'center', 'fontSize': 12},
                           style_header={'backgroundColor': 'lightgrey', 'fontWeight': 'bold'}
                       ),   
                            #Data insight and key matrics
    html.Div([
    html.H4("Data Insights and Key Matrics"),

    html.Div([
        html.Label("Select a Sector:", style={"fontSize": "18px", "marginRight": "10px"}),
        dcc.Dropdown(
            id='sector-dropdown',
            options=[{'label': sector, 'value': sector} for sector in data_insight_sectors],
            value=data_insight_sectors[0],
            style={"width": "50%", "marginBottom": "20px"}
        )
    ], style={"padding": "20px", "textAlign": "center"}),

    dcc.Tabs([
        dcc.Tab(label='Sector Overview', children=[
            html.Div([
                dcc.Graph(id='sector-graph_log', style={"marginBottom": "20px"}),
                dcc.Graph(id='sector-graph_vol', style={"marginBottom": "20px"}),
                dcc.Graph(id='sector-graph_sharpe', style={"marginBottom": "20px"}),
            ], style={"padding": "10px"})
        ]),

        dcc.Tab(label='Key Metrics', children=[
            html.Div([
                html.Div([
                    html.H4("Risk and Returns", style={"textAlign": "center"}),
                    html.Ul(id='stats-box-risk', style={
                        "backgroundColor": "#f9f9f9",
                        "padding": "15px",
                        "border": "1px solid #ddd",
                        "borderRadius": "5px",
                        "boxShadow": "0 2px 4px rgba(0, 0, 0, 0.1)"
                    })
                ], style={"width": "30%", "display": "inline-block", "padding": "10px"}),

                html.Div([
                    html.H4("Volatility and Drawdowns", style={"textAlign": "center"}),
                    html.Ul(id='stats-box-volatility', style={
                        "backgroundColor": "#f9f9f9",
                        "padding": "15px",
                        "border": "1px solid #ddd",
                        "borderRadius": "5px",
                        "boxShadow": "0 2px 4px rgba(0, 0, 0, 0.1)"
                    })
                ], style={"width": "30%", "display": "inline-block", "padding": "10px"}),

                html.Div([
                    html.H4("Performance Metrics", style={"textAlign": "center"}),
                    html.Ul(id='stats-box-performance', style={
                        "backgroundColor": "#f9f9f9",
                        "padding": "15px",
                        "border": "1px solid #ddd",
                        "borderRadius": "5px",
                        "boxShadow": "0 2px 4px rgba(0, 0, 0, 0.1)"
                    })
                ], style={"width": "30%", "display": "inline-block", "padding": "10px"}),
            ], style={"display": "flex", "justifyContent": "center", "flexWrap": "wrap"})
        ]),

        dcc.Tab(label='Advanced Metrics', children=[
            html.Div([
                html.Div([
                    html.H4("Advanced Metrics", style={"textAlign": "center"}),
                    html.Ul(id='stats-box-advance', style={
                        "backgroundColor": "#f9f9f9",
                        "padding": "15px",
                        "border": "1px solid #ddd",
                        "borderRadius": "5px",
                        "boxShadow": "0 2px 4px rgba(0, 0, 0, 0.1)"
                    })
                ], style={"width": "45%", "display": "inline-block", "padding": "10px"}),

                html.Div([
                    html.H4("Stats Metrics", style={"textAlign": "center"}),
                    html.Ul(id='stats-box-base', style={
                        "backgroundColor": "#f9f9f9",
                        "padding": "15px",
                        "border": "1px solid #ddd",
                        "borderRadius": "5px",
                        "boxShadow": "0 2px 4px rgba(0, 0, 0, 0.1)"
                    })
                ], style={"width": "45%", "display": "inline-block", "padding": "10px"}),
            ], style={"display": "flex", "justifyContent": "space-between", "padding": "20px"})
        ]),

        dcc.Tab(label='Additional Insights', children=[
            html.Div([
                dcc.Graph(id='return-histogram', style={"marginBottom": "20px"}),
                dcc.Graph(id='sector-graph-drawdown', style={"marginBottom": "20px"}),
                dcc.Graph(id='sector-graph-underdown', style={"marginBottom": "20px"}),
                dcc.Graph(id='sector-return-quantiles', style={"marginBottom": "20px"}),
                dcc.Graph(id='sector-eoy-returns', style={"marginBottom": "20px"}),
            ], style={"padding": "10px"})
        ])
    ], style={"padding": "10px"})
]),

     
                     
    
    
    
    
    
    html.H4("Evaluation Matric"),
                               dcc.Dropdown(
                                id='metric-dropdown',
                                style = {'padding':'10px'},
                                options=[{'label': col, 'value': col} for col in rmse_mae_result_dataframe.columns[1:]],
                                value='Lstm_RMSE',  # Default value
                                multi=False
                            ),
                            dcc.Graph(id='bar-chart-ev'),
                        html.H1("Consolidation"),
                         html.H2("Customize Dataframe"),
                        html.Label("Select Dataframe to Customize:"),
                        dcc.Dropdown(
                            id='dataframe-dropdown',
                            style = {'padding':'10px'},
                            options=[
                                {'label': 'SARIMA DataFrame', 'value': 'sarima_forecast_data'},
                                {'label': 'LSTM DataFrame', 'value': 'lstm_forecast_data'},
                                {'label': 'XGBoost DataFrame', 'value': 'xgboost_forecast_data'},
                                {'label': 'FBProphet DataFrame', 'value': 'fbprophet_forecast_data'}
                            ],
                            value='sarima_forecast_data'
                        ),
                        html.Label("Select Column to Change:"),
                        dcc.Dropdown(
                            id='column-dropdown1',
                            style = {'padding':'10px'},
                            multi=False
                        ),
                        html.Label("Select DataFrame to Change With:"),
                        dcc.Dropdown(
                            id='change-dataframe-dropdown',
                            style = {'padding':'10px'},
                            options=[
                                {'label': 'SARIMA DataFrame', 'value': 'sarima_forecast_data'},
                                {'label': 'LSTM DataFrame', 'value': 'lstm_forecast_data'},
                                {'label': 'XGBoost DataFrame', 'value': 'xgboost_forecast_data'},
                                {'label': 'FBProphet DataFrame', 'value': 'fbprophet_forecast_data'}
                            ],
                            value='lstm_forecast_data'
                        ),
                        html.Button('Change Column', id='change-column-button', className='btn btn-default', n_clicks=0),
                        html.Div(id='dataframe-output'),
                        html.A('Download Modified Data', id='download-link', download="modified_data.csv", href="", target="_blank",
                              style= {
                        'background-color':'white',
                        'color': 'black',
                        'textAlign': 'center',
                        'display': 'inline-block',
                        'lineHeight': '60px',
                        'padding': '0 30px',
                        'textDecoration': 'none',
                        'transition': 'all 0.3s ease-in-out',
                        'outline': 'none !important',
                        'borderRadius': '0px',
                        'fontSize': '13px',
                        'fontWeight': '500',
                        'border': '2px solid light-grey',
                        'minWidth': '220px',
                        'height': '60px',
                        'width': '220px'
                    }),

                       html.H3("Historical Data"),
                       dash_table.DataTable(
                           id='historical_data-table',
                           columns=[{'name': col, 'id': col} for col in historical.columns],
                           data=historical.applymap(
        lambda x: f"{float(x):.2f}%" if str(x).replace('.', '', 1).isdigit() else x  # Format numeric columns and leave strings as is
    ).to_dict('records'),  # Use historical data here
                           style_table={'height': '400px', 'overflowY': 'auto'},
                           style_cell={'textAlign': 'center', 'fontSize': 12},
                           style_header={'backgroundColor': 'lightgrey', 'fontWeight': 'bold'}
                       ),
                       html.H3("Covariance table Data"),
                       dash_table.DataTable(
                           id='covariance_data-table',
                           columns=[{'name': col, 'id': col} for col in covariance_table.columns],
                           data=covariance_table.to_dict('records'),  # Use Covariance data here
                           style_table={'height': '400px', 'overflowY': 'auto'},
                           style_cell={'textAlign': 'center', 'fontSize': 12},
                           style_header={'backgroundColor': 'lightgrey', 'fontWeight': 'bold'}
                       ),
                       
                             dcc.Graph(
                            id='bar-chart',
                            figure={
                                'data': [
                                    {'x': correlate_table(data)['Correlation Number'], 'y': correlate_table(data)['Sectors'], 'type': 'bar', 'orientation': 'h'}
                                ],
                                'layout': {
                                'title': 'Correlation Number by Sectors',
                                'xaxis': {'title': 'Correlation Number', 'automargin': True},
                                'yaxis': {'title': 'Sectors', 'automargin': True, 'tickfont': {'size': 10}},
                                'margin': {'l': 150, 'r': 50, 't': 70, 'b': 70},
                                }
                            }
                        ),
                                        html.Div([         dcc.Dropdown(
                                    id='forecast-compare-dropdown',
                                     style = {'padding':'10px'},
                                    options=[{'label': key, 'value': key} for key in forecast_compare_options.keys()],
                                    value='SARIMA'  # Default value
                                ),
                                html.H4("Historical Cumulative Return"),  # Title for the first table

                                dash_table.DataTable(
                                    id='table1',
                                    columns=[{"name": i, "id": i} for i in df_historical_camulative.columns],
                                    data=df_historical_camulative.applymap(
        lambda x: f"{float(x):.2f}%" if str(x).replace('.', '', 1).isdigit() else x  # Format numeric columns and leave strings as is
    ).to_dict('records'),
                                    style_cell={'font_size': '6.5pt'},  # Adjust font size
                                    style_table={'overflowX': 'scroll', 'maxWidth': '100%'}  # Adjust table size
                                )
                            ], style={'width': '55%', 'display': 'inline-block', 'margin-right': '10px'}),

                            html.Div([
                                html.H4("Forecasted Cumulative Return"),  # Title for the second table

                                html.Div(id='table-container')
                            ], style={'width': '35%', 'display': 'inline-block'}),
                                   html.Div([         dcc.Dropdown(
                    id='forecast-compare-dropdown_asset',
                    style = {'padding':'10px'},
                    options=[{'label': key, 'value': key} for key in forecast_compare_options.keys()],
                    value='SARIMA'  # Default value
                ),
                html.H3("Historical Cumulative Return Asset Wise"),  # Title for the first table

                dash_table.DataTable(
                    id='table1_asset',
                    columns=[{"name": i, "id": i} for i in df_historical_cumulative_asset.columns],
                    data=df_historical_cumulative_asset.applymap(
        lambda x: f"{float(x):.2f}%" if str(x).replace('.', '', 1).isdigit() else x  # Format numeric columns and leave strings as is
    ).to_dict('records'),
                    style_cell={'font_size': '6.5pt'},  # Adjust font size
                    style_table={'overflowX': 'scroll', 'maxWidth': '100%'}  # Adjust table size
                )
            ], style={'width': '55%', 'display': 'inline-block', 'margin-right': '10px'}),

            html.Div([
                html.H3("Forecasted Cumulative Return"),  # Title for the second table

                html.Div(id='table-container_asset')
            ], style={'width': '35%', 'display': 'inline-block'}),
                       html.Hr(style = {'height':'2px','background-color':'black'}),
                       

                       html.H2("HandCraft Method"),

                       dbc.Row([
                           dbc.Col(html.Label("Select Forecast Model:"), width=2),
                           dbc.Col(dcc.Dropdown(
                               id='forecast-model-dropdown',
                               style = {'padding':'10px'},
                               options=[
                                   {'label': 'SARIMA', 'value': 'sarima'},
                                   {'label': 'FB PROPHET', 'value': 'fbprophet'},
                                   {'label': 'LSTM', 'value': 'lstm'},
                                   {'label': 'XGBOOST', 'value': 'xgboost'},
                                   {'label': 'CUSTOMIZE TABLE', 'value': 'customize_table'},
                               ],
                               value='lstm',
                               placeholder='Select Forecast Model'
                           ), width=10),
                       ], className="mb-3"),

                       dbc.Row([
                           dbc.Col(html.Label("Select Column:"), width=2),
                           dbc.Col(dcc.Dropdown(
                               id='column-input',
                               style = {'padding':'10px'},
                               options=[{'label': col, 'value': col} for col in df.columns[1:6]],
                               value='Year1',
                               placeholder='Select Column'
                           ), width=10),
                       ], className="mb-3"),

                       dbc.Row([
                           dbc.Col(html.Label("Row Number:"), width=2),
                           dbc.Col(dcc.Input(
                               id='row-input',
                               className='form-control',
                               style = {'padding':'10px'},
                               type='number',
                               value=0,
                               placeholder='Row Number',
                               min=0,
                               max=len(df) - 1
                           ), width=10),
                       ], className="mb-3"),

                       dbc.Row([
                           dbc.Col(html.Label("Select Multiplier:"), width=2),
                           dbc.Col(dcc.Dropdown(
                               id='multiplier-dropdown',
                               style = {'padding':'10px'},
                               options=[{'label': val, 'value': val} for val in
                                        ['Very strong buy', 'Strong buy', 'Buy', 'Weak buy', 'Neutral', 'Weak sell',
                                         'Sell',
                                         'Strong sell', 'Very strong sell', 'Exclude']],
                               value='Neutral',
                               placeholder='Select Multiplier'
                           ), width=10),
                       ], className="mb-3"),

                       dbc.Row([
                           dbc.Col(html.Label("Input Value (or NaN):"), width=2),
                           dbc.Col(dcc.Input(
                               id='input-value',
                               className='form-control',
                               style = {'padding':'10px'},
                               type='number',
                               value=float('nan'),
                               placeholder='Input Value (or NaN)'
                           ), width=10),
                       ], className="mb-3"),
                       html.Br(),
                       dbc.Row([
                           dbc.Col(html.Button('Update Value', id='update-button', className='btn btn-default'), width=12),
                       ], className="mb-3"),

                       html.Br(),

                       html.Div(id='update-output'),

                       html.Br(),

                       html.H2("Updated DataFrame Values"),

                       dcc.Download(id="download-updated-csv"),  # Add this line

                       dash_table.DataTable(
                           id='data-table',
                           columns=[{'name': col, 'id': col} for col in df.columns],
                           data=df.to_dict('records'),
                           style_table={'overflowX': 'auto', 'height': '400px'},
                           style_cell={'textAlign': 'center', 'fontSize': 12},
                           style_header={'backgroundColor': 'lightgrey', 'fontWeight': 'bold'}
                       ),

                       html.Button('Download Updated CSV', id='download-updated-csv-button', className='btn btn-default'),  # Add this line
                       html.Hr(style = {'height':'2px','background-color':'black'}),
                       html.Br(),

                     html.H1("Optimization"),

    html.Div([
        dcc.Dropdown(
            id='optimize-dropdown',
            options=[
                {'label': 'Maximize Return', 'value': 'maximize_return'},
                {'label': 'Maximize Sharp', 'value': 'maximize_sharp'}
            ],
            value='maximize_return',  # Set default value
            clearable=False,
            style={'width': '100%', 'margin': 'auto', 'padding': '5px 10px'}
        )
    ]),

    # Add a placeholder div for showing "Sharp is on" when selected
    html.Div(id='sharp-placeholder'),

    # Modify your existing layout to include conditional rendering
    html.Div(id='optimal-model-section'),

    html.Div([
        html.H3("Set Objective "),
        html.Label('Select DataFrame:'),
        dcc.Dropdown(
            id='set_objective_dropdown',
            style={'padding': '2px'},
            options=[{'label': key, 'value': key} for key in dropdown_options.keys()],
            value='LSTM Forecast'
        ),
        html.Label('Set Objective:'), html.Br(),
        dcc.Input(id='set_objective', type='number', value=0, className='form-control'),
        html.Br(),
        html.Label('Set Tolerance:'), html.Br(),
        dcc.Input(id='set_tolerance', type='number', value=0, className='form-control'),
        html.Br(),
        html.Button('Optimize', id='set_optimize_button', className='btn btn-default', n_clicks=0),
        dcc.Graph(id='set_objective_line-chart'),
        dash_table.DataTable(id='set_objective_table',
                             style_table={'overflowX': 'auto', 'height': '400px'},
                             style_cell={'textAlign': 'center', 'fontSize': 12},
                             style_header={'backgroundColor': 'lightgrey', 'fontWeight': 'bold'}),
                        
                    ]),
    html.Footer(style = {'background-color':'whilte'},children=[
    html.Div([
        html.Div([
            html.Img(src="https://cashelfo.com/wp-content/themes/cashel/img/CFO_logo_white.png", alt="CFO Logo")
        ], className="col-lg-6 col-sm-12"),

        html.Div([
            html.H5([
                html.A('[email protected]', href="mailto: [email protected]", className="_cf_email_", 
                       **{'data-cfemail': 'd1bbbeb8bf91b2b0a2b9b4bdb7beffb2bebc'}),
                html.A("03 9209 9000", href="tel:0392099000")
            ])
        ], className="col-lg-6 col-sm-12")
    ], className="container top-container"),

    # Navigation Links
    html.Div([
        html.Div([
            html.H4("Wealth Services"),
            html.Ul([
                html.Li(html.A("Plan", href="/wealth/plan/")),
                html.Li(html.A("Protect", href="/wealth/protect/")),
                html.Li(html.A("Invest", href="/wealth/invest/")),
                html.Li(html.A("Borrow", href="/wealth/borrow/")),
                html.Li(html.A("Transform", href="/wealth/transform/")),
                html.Li(html.A("Organise", href="/wealth/organise/")),
            ])
        ], id="text-2", className="widget widget_text"),

        html.Div([
            html.H4("Family Office"),
            html.Ul([
                html.Li(html.A("Our Services", href="/our-services/")),
                html.Li(html.A("Global Product Range", href="/wealth/global-product-range/")),
                html.Li(html.A("Investment Platforms", href="/wealth/investment-platforms/")),
                html.Li(html.A("Join Cashel Family Office", href="/join-cashelfo/")),
            ])
        ], id="text-4", className="widget widget_text"),

        html.Div([
            html.H4("About Us"),
            html.Ul([
                html.Li(html.A("About Us", href="/about-us/about-us/")),
                html.Li(html.A("Global Partnerships", href="/about-us/global-partnerships/")),
                html.Li(html.A("Community", href="/about-us/community/")),
            ])
        ], id="text-5", className="widget widget_text"),

        html.Div([
            html.H4("Useful links"),
            html.Ul([
                html.Li(html.A("Privacy Policy", href="/privacy-policy/")),
                html.Li(html.A("Conflicts of Interest policy", href="/conflicts-of-interest-policy/")),
                html.Li(html.A("Forms", href="/forms")),
            ])
        ], id="text-6", className="widget widget_text"),

        html.Div([
            html.Div([
                html.A(html.Img(src="https://cashelfo.com/wp-content/uploads/2023/12/apple_app_store.png", alt=""),
                       href="https://apps.apple.com/au/app/cashel-family-office/id6473882152", target="_blank",
                       rel="noopener", style={"max-width": "128px !important", "max-height": "min-content"}),
                html.Br(),
                html.A(html.Img(src="https://cashelfo.com/wp-content/uploads/2023/12/google_play_badge.png", alt=""),
                       href="https://play.google.com/store/apps/details?id=com.softobiz.casher_family_office&hl=en_US&gl=US",
                       target="_blank", rel="noopener", style={"max-width": "128px !important", "max-height": "min-content"}),
            ], className="textwidget active")
        ], className="widget widget_text", style={"padding": "0px", "padding-top": "0px"})
    ], className="container", style={"display": "flex", "justify-content": "space-around"})
                ]),
                        ])
@app.callback(
    [Output('optimal-model-section', 'children'),
     Output('sharp-placeholder', 'children'),
     Output('optimize-dropdown', 'value')],  # Add this output
    [Input('optimize-dropdown', 'value')]
)
def update_optimal_model_section(value):
    if value == 'maximize_return':
        optimal_layout = html.Div([
            dcc.Download(id="download-optimal-csv"),  # Add this line
            dash_table.DataTable(
                id='optimal-table',
                columns=[],  # Initially empty, will be dynamically updated
                data=[],
                style_table={'overflowX': 'auto', 'height': '400px'},
                style_cell={'textAlign': 'center', 'fontSize': 12},
                style_header={'backgroundColor': 'lightgrey', 'fontWeight': 'bold'}
            ),
            html.Button('Download Optimal CSV', id='download-optimal-csv-button', className='btn btn-default'),
            # Corrected line
            html.Br(),
            dbc.Row([
                dbc.Col(html.Label("Minimum Weights:"), width=2),
                dbc.Col(dcc.Input(
                    id='min-weight-input',
                    className='form-control',
                    type='number',
                    value=5,
                    placeholder='Enter Minimum Weights'
                ), width=10),
            ], className="mb-3"),
            html.Br(),
            dbc.Row([
                dbc.Col(html.Label("Maximum Weights:"), width=2),
                dbc.Col(dcc.Input(
                    id='max-weight-input',
                    className='form-control',
                    type='number',
                    value=15,
                    placeholder='Enter Maximum Weights'
                ), width=10),
            ], className="mb-3"),
            html.Br(),
            dbc.Row([
                dbc.Col(html.Label("Total Investments:"), width=2),
                dbc.Col(dcc.Input(
                    id='total-investment-input',
                    className='form-control',
                    type='number',
                    value=12,
                    placeholder='Enter Total Investments'
                ), width=10),
            ], className="mb-3"),
            html.Br(),
            dbc.Row([
                dbc.Col(html.Label("Maximum Volatility:"), width=2),
                dbc.Col(dcc.Input(
                    id='volatility-input',
                    className='form-control',
                    type='number',
                    value=13.26,
                    placeholder='Enter Maximum Volatility'
                ), width=10),
            ], className="mb-3"),
            html.Br(),
            dbc.Row([
                dbc.Col(html.Button('Optimize', id='optimize-button', className='btn btn-default'), width=12),
            ], className="mb-3"),
            dcc.Store(id='user-modified-data', data={}),
            dcc.Interval(id='interval-component', interval=1000, n_intervals=0),
            html.H1("Optimization Results"),
            html.H3("Distributions of Weights Table"),
            dash_table.DataTable(
                id='total-table',
                columns=[],  # Initially empty, will be dynamically updated
                data=[],
                style_table={'overflowX': 'auto', 'height': '400px'},
                style_cell={'textAlign': 'center', 'fontSize': 12},
                style_header={'backgroundColor': 'lightgrey', 'fontWeight': 'bold'}
            ),
            html.Br(),
            html.H3("Model Optimize Statics Data"),
            html.Div(className='Model_optimize', children=[
                dcc.Graph(
                    id='discrete-period-return',
                    figure={
                        'data': [
                            {'x': ['Year1', 'Year2', 'Year3', 'Year4', 'Year5'], 'y': [], 'name': 'Data 1'},
                            # Add more traces with names as needed
                        ],
                        'layout': {
                            'title': 'Discrete Period Return',
                            'legend': {'x': 0, 'y': 1},
                        }
                    }
                ),
                dcc.Graph(
                    id='cumulative-return',
                    figure={
                        'data': [
                            {'x': ['Year1', 'Year2', 'Year3', 'Year4', 'Year5'], 'y': [], 'name': 'Data 2'},
                            # Add more traces with names as needed
                        ],
                        'layout': {
                            'title': 'Cumulative Return',
                            'legend': {'x': 0, 'y': 1},
                        }
                    }
                ),
                dcc.Graph(
                    id='portfolio-value',
                    figure={
                        'data': [
                            {'x': ['Year1', 'Year2', 'Year3', 'Year4', 'Year5'], 'y': [], 'name': 'Data 3'},
                            # Add more traces with names as needed
                        ],
                        'layout': {
                            'title': 'Portfolio Value',
                            'legend': {'x': 0, 'y': 1},
                        }
                    }
                ),
                dcc.Graph(
                    id='risk',
                    figure={
                        'data': [
                            {'x': ['Year1', 'Year2', 'Year3', 'Year4', 'Year5'], 'y': [], 'name': 'Data 4'},
                            # Add more traces with names as needed
                        ],
                        'layout': {
                            'title': 'Risk',
                            'legend': {'x': 0, 'y': 1},
                        }
                    }
                ),
            ]),
            html.Br(),

            html.H3("Historical Optimize Statics Data"),
            html.Div(className='Historical_optimize', children=[
                dcc.Graph(
                    id='Hdiscrete-period-return',
                    figure={
                        'data': [
                            {'x': ['Year1', 'Year2', 'Year3', 'Year4', 'Year5'], 'y': [], 'name': 'Historical Data 1'},
                            # Add more traces with names as needed
                        ],
                        'layout': {
                            'title': 'Historical Discrete Period Return',
                            'legend': {'x': 0, 'y': 1},
                        }
                    }
                ),
                dcc.Graph(
                    id='Hcumulative-return',
                    figure={
                        'data': [
                            {'x': ['Year1', 'Year2', 'Year3', 'Year4', 'Year5'], 'y': [], 'name': 'Historical Data 2'},
                            # Add more traces with names as needed
                        ],
                        'layout': {
                            'title': 'Historical Cumulative Return',
                            'legend': {'x': 0, 'y': 1},
                        }
                    }
                ),
                dcc.Graph(
                    id='Hportfolio-value',
                    figure={
                        'data': [
                            {'x': ['Year1', 'Year2', 'Year3', 'Year4', 'Year5'], 'y': [], 'name': 'Historical Data 3'},
                            # Add more traces with names as needed
                        ],
                        'layout': {
                            'title': 'Historical Portfolio Value',
                            'legend': {'x': 0, 'y': 1},
                        }
                    }
                ),
                dcc.Graph(
                    id='historical_risk',
                    figure={
                        'data': [
                            {'x': ['Year1', 'Year2', 'Year3', 'Year4', 'Year5'], 'y': [], 'name': 'Historical Data 4'},
                            # Add more traces with names as needed
                        ],
                        'layout': {
                            'title': 'Historical Risk',
                            'legend': {'x': 0, 'y': 1},
                        }
                    }
                )
            ]), html.H2("Optimal Weights by Year - Pie Charts"),
    html.Div(id='pie-charts-container', style={'columnCount': 2}) ,
        html.H2("Sector Risk Analysis"),
    html.Div(id='metrics-container'),  # Placeholder for risk metrics table
    html.Div([
        dcc.Graph(id='risk-bar-chart'),
        dcc.Graph(id='sharpe-scatter-plot')
    ]),
    
    html.Div(dcc.Graph(id='efficient-frontier-plot')), 
        ])

        return optimal_layout, None, value
    elif value == 'maximize_sharp':
        sharp_layout = html.Div([
            dcc.Download(id="sharp-download-optimal-csv"),  # Add this line
            dash_table.DataTable(
                id='sharp-optimal-table',
                columns=[],  # Initially empty, will be dynamically updated
                data=[],
                style_table={'overflowX': 'auto', 'height': '400px'},
                style_cell={'textAlign': 'center', 'fontSize': 12},
                style_header={'backgroundColor': 'lightgrey', 'fontWeight': 'bold'}
            ),
            # html.Button('Download Optimal CSV', id='sharp-download-optimal-csv', className='btn btn-default'),
            html.Br(),
            dcc.Dropdown(
                id='risk-measure-dropdown',
                options=[
                    {'label': 'Variance', 'value': 'variance'},
                    {'label': 'Semi-Variance', 'value': 'semi_variance'},
                    {'label': 'Conditional Value-at-Risk', 'value': 'conditional_var'},
                    {'label': 'Conditional Drawdown-at-Risk', 'value': 'conditional_drawdown'}
                ],
                value='variance',  # Set default value
                clearable=False,
                style={'width': '100%', 'margin': 'auto', 'padding': '5px 10px'}
            ),
            dcc.Dropdown(
                id='risk-period-dropdown',
                options=[
                    {'label': '3 Year Average', 'value': '3_year_avg'},
                    {'label': 'CAPM Forecasting Return', 'value': 'capm_return'},
                    {'label': 'Exponential Moving Average', 'value': 'ema'}
                ],
                value='3_year_avg',  # Set default value
                clearable=False,
                style={'width': '100%', 'margin': 'auto', 'padding': '5px 10px'}
            ),
            dbc.Row([
                dbc.Col(html.Label("Maximum Risk Tolerance Value:"), width=2),
                dbc.Col(dcc.Input(
                    id='max-risk-input',
                    className='form-control',
                    type='number',
                    value=0,  # Set default value as necessary
                    placeholder='Enter Maximum Risk Tolerance Value'
                ), width=10),
            ], className="mb-3"),
            html.Br(),
            dbc.Row([
                dbc.Col(html.Label("Minimum Investment Size:"), width=2),
                dbc.Col(dcc.Input(
                    id='min-investment-input',
                    className='form-control',
                    type='number',
                    value=0,  # Set default value as necessary
                    placeholder='Enter Minimum Investment Size'
                ), width=10),
            ], className="mb-3"),
            html.Br(),
            dbc.Row([
                dbc.Col(html.Label("Maximum Investment Size:"), width=2),
                dbc.Col(dcc.Input(
                    id='max-investment-input',
                    className='form-control',
                    type='number',
                    value=0,  # Set default value as necessary
                    placeholder='Enter Maximum Investment Size'
                ), width=10),
            ], className="mb-3"),
            html.Br(),

            dbc.Row([
                dbc.Col(html.Label("Minimum Weights:"), width=2),
                dbc.Col(dcc.Input(
                    id='sharp-min-weight-input',
                    className='form-control',
                    type='number',
                    value=5,
                    placeholder='Enter Minimum Weights'
                ), width=10),
            ], className="mb-3"),
            html.Br(),
            dbc.Row([
                dbc.Col(html.Label("Maximum Weights:"), width=2),
                dbc.Col(dcc.Input(
                    id='sharp-max-weight-input',
                    className='form-control',
                    type='number',
                    value=15,
                    placeholder='Enter Maximum Weights'
                ), width=10),
            ], className="mb-3"),
            html.Br(),
            dbc.Row([
                dbc.Col(html.Label("Total Investments:"), width=2),
                dbc.Col(dcc.Input(
                    id='sharp-total-investment-input',
                    className='form-control',
                    type='number',
                    value=12,
                    placeholder='Enter Total Investments'
                ), width=10),
            ], className="mb-3"),
            html.Br(),
            dbc.Row([
                dbc.Col(html.Label("Maximum Volatility:"), width=2),
                dbc.Col(dcc.Input(
                    id='sharp-volatility-input',
                    className='form-control',
                    type='number',
                    value=13.26,
                    placeholder='Enter Maximum Volatility'
                ), width=10),
            ], className="mb-3"),
            html.Br(),
            dbc.Row([
                dbc.Col(html.Button('Optimize', id='sharp-optimize-button', className='btn btn-default'), width=12),
            ], className="mb-3"),
            dcc.Store(id='user-modified-data', data={}),
            dcc.Interval(id='interval-component', interval=1000, n_intervals=0),
            html.H4("Distributions of Weights"),
            dash_table.DataTable(
                id='sharp-total-table',
                columns=[],  # Initially empty, will be dynamically updated
                data=[],
                style_table={'overflowX': 'auto', 'height': '400px'},
                style_cell={'textAlign': 'center', 'fontSize': 12},
                style_header={'backgroundColor': 'lightgrey', 'fontWeight': 'bold'}
            ),
            html.Br(),
            html.H2("Model Optimize Statics Data"),
            html.Div(className='Model_optimize', children=[
                dcc.Graph(
                    id='sharp-discrete-period-return',
                    figure={
                        'data': [
                            {'x': ['Year1', 'Year2', 'Year3', 'Year4', 'Year5'], 'y': [], 'name': 'Data 1'},
                            # Add more traces with names as needed
                        ],
                        'layout': {
                            'title': 'Discrete Period Return',
                            'legend': {'x': 0, 'y': 1},
                        }
                    }
                ),
                dcc.Graph(
                    id='sharp-cumulative-return',
                    figure={
                        'data': [
                            {'x': ['Year1', 'Year2', 'Year3', 'Year4', 'Year5'], 'y': [], 'name': 'Data 2'},
                            # Add more traces with names as needed
                        ],
                        'layout': {
                            'title': 'Cumulative Return',
                            'legend': {'x': 0, 'y': 1},
                        }
                    }
                ),
                dcc.Graph(
                    id='sharp-portfolio-value',
                    figure={
                        'data': [
                            {'x': ['Year1', 'Year2', 'Year3', 'Year4', 'Year5'], 'y': [], 'name': 'Data 3'},
                            # Add more traces with names as needed
                        ],
                        'layout': {
                            'title': 'Portfolio Value',
                            'legend': {'x': 0, 'y': 1},
                        }
                    }
                ),
                dcc.Graph(
                    id='sharp-risk',
                    figure={
                        'data': [
                            {'x': ['Year1', 'Year2', 'Year3', 'Year4', 'Year5'], 'y': [], 'name': 'Data 4'},
                            # Add more traces with names as needed
                        ],
                        'layout': {
                            'title': 'Risk',
                            'legend': {'x': 0, 'y': 1},
                        }
                    }
                ),
            ]),
            html.Br(),

            html.H2("Historical Optimize Statics Data"),
            html.Div(className='Historical_optimize', children=[
                dcc.Graph(
                    id='sharp-Hdiscrete-period-return',
                    figure={
                        'data': [
                            {'x': ['Year1', 'Year2', 'Year3', 'Year4', 'Year5'], 'y': [], 'name': 'Historical Data 1'},
                            # Add more traces with names as needed
                        ],
                        'layout': {
                            'title': 'Historical Discrete Period Return',
                            'legend': {'x': 0, 'y': 1},
                        }
                    }
                ),
                dcc.Graph(
                    id='sharp-Hcumulative-return',
                    figure={
                        'data': [
                            {'x': ['Year1', 'Year2', 'Year3', 'Year4', 'Year5'], 'y': [], 'name': 'Historical Data 2'},
                            # Add more traces with names as needed
                        ],
                        'layout': {
                            'title': 'Historical Cumulative Return',
                            'legend': {'x': 0, 'y': 1},
                        }
                    }
                ),
                dcc.Graph(
                    id='sharp-Hportfolio-value',
                    figure={
                        'data': [
                            {'x': ['Year1', 'Year2', 'Year3', 'Year4', 'Year5'], 'y': [], 'name': 'Historical Data 3'},
                            # Add more traces with names as needed
                        ],
                        'layout': {
                            'title': 'Historical Portfolio Value',
                            'legend': {'x': 0, 'y': 1},
                        }
                    }
                ),
                dcc.Graph(
                    id='sharp-historical_risk',
                    figure={
                        'data': [
                            {'x': ['Year1', 'Year2', 'Year3', 'Year4', 'Year5'], 'y': [], 'name': 'Historical Data 4'},
                            # Add more traces with names as needed
                        ],
                        'layout': {
                            'title': 'Historical Risk',
                            'legend': {'x': 0, 'y': 1},
                        }
                    }
                )
            ]),  html.H2("Optimal Weights by Year - Pie Charts"),
    html.Div(id='sharp-pie-charts-container', style={'columnCount': 2}) ,
        html.H2("Sector Risk Analysis"),
    html.Div(id='sharp-metrics-container'),  # Placeholder for risk metrics table
    html.Div([
        dcc.Graph(id='sharp-risk-bar-chart'),
        dcc.Graph(id='sharp-sharpe-scatter-plot')
    ]),
    
    html.Div(dcc.Graph(id='sharp-efficient-frontier-plot')), 
        ])
        return None, sharp_layout, value
    else:
                # Prevent callback execution if value is invalid
        raise PreventUpdate
    
def parse_contents(contents, filename):
    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            # Assume that the user uploaded a CSV file
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        elif 'xls' in filename:
            # Assume that the user uploaded an Excel file
            df = pd.read_excel(io.BytesIO(decoded))
        else:
            return html.Div([
                'Unsupported file format. Please upload a CSV or Excel file.'
            ])

        return html.Div([
            html.H5(f"Uploaded File: {filename}"),
            
            # Display the table
            dash_table.DataTable(
                data=df.to_dict('records'),
                columns=[{'name': col, 'id': col} for col in df.columns],
                editable=True,  # Make the table editable
                style_table={'overflowX': 'auto', 'height': '400px', 'overflowY': 'scroll'},
                style_cell={
                    'textAlign': 'left',
                    'padding': '5px',
                    'fontSize': '12px',  # Decrease font size
                    'whiteSpace': 'normal',
                    'height': 'auto'  # Adjust row height
                },
                style_data={
                    'whiteSpace': 'normal',
                },
                style_header={
                    'backgroundColor': 'lightgrey',
                    'fontWeight': 'bold',
                    'fontSize': '14px'  # Slightly larger font for headers
                }
            )
        ])
    except Exception as e:
        return html.Div([
            f"There was an error processing this file: {str(e)}"
        ])


@app.callback(
    Output('upload-data', 'style'),
    Input('upload-button', 'n_clicks'),
    prevent_initial_call=True
)
def show_upload(n_clicks):
    return {
        'width': '100%',
        'height': '60px',
        'lineHeight': '60px',
        'borderWidth': '1px',
        'borderStyle': 'dashed',
        'borderRadius': '5px',
        'textAlign': 'center',
        'margin': '10px',
        'display': 'block'  # Show the upload component
    }

@app.callback(
    Output('output-table', 'style'),
    Input('toggle-button', 'n_clicks'),
    State('output-table', 'style')
)
def toggle_table(n_clicks, current_style):
    if n_clicks % 2 == 1:
        return {'display': 'block'}  # Show the table
    return {'display': 'none'}  # Hide the table

@app.callback(
    Output('output-table', 'children'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename')
)
def update_table(contents, filename):
    if contents is not None:
        return parse_contents(contents, filename)
    return html.Div("Upload a file to see its content.")
# Callbacks to update the data insight graph and statistics
@app.callback(
    [Output('sector-graph_log', 'figure'),
     Output('sector-graph_vol', 'figure'),
     Output('sector-graph_sharpe', 'figure'),
     Output('stats-box-risk', 'children'),
     Output('stats-box-volatility', 'children'),
     Output('stats-box-performance', 'children'),
     Output('stats-box-advance', 'children'),
     Output('stats-box-base', 'children'),
     Output('return-histogram', 'figure'),
     Output('sector-graph-drawdown', 'figure'),
     Output('sector-graph-underdown', 'figure'),
     Output('sector-return-quantiles', 'figure'),
     Output('sector-eoy-returns', 'figure'),
    ],
    [Input('sector-dropdown', 'value')]
)
def update_dashboard(selected_sector):
    if selected_sector:
        sector_data = data[[selected_sector]].dropna()
        sector_data = data[[selected_sector]]

        # Line chart for sector performance
        fig_line_log = px.line(
            sector_data.cumsum().apply(np.exp),
            x=sector_data.index,
            y=selected_sector,
            title=f"Cumulative Return in Log Scale {selected_sector}",
            labels={"value": "Cumulative Return (Log Scale)", "index": "Date"}
        )
        
        fig_line_vol = px.line(
            sector_data.rolling(window=6).std() * np.sqrt(12),
            x=sector_data.index,
            y=selected_sector,
            title=f"Rolling Volatility {selected_sector}",
            labels={"value": "Rolling Volatility (6-Month Window)", "index": "Date"}
        )
        fig_line_sharpe = px.line(
            sector_data.rolling(window=6).mean() / sector_data.rolling(window=6).std(),
            x=sector_data.index,
            y=selected_sector,
            title=f"Rolling Sharpe Ratio (6-Month Window) {selected_sector}",
            labels={"value": "Rolling Sharpe (6-Month Window)", "index": "Date"}
        )    
       # Calculate the statistics
        risk_free_rate = 0.02  # Assuming a risk-free rate of 2% annually
        cumulative_return = (1 + sector_data[selected_sector]).prod() - 1
        start_value = 1
        end_value = (1 + cumulative_return)
        years = 5
        CAGR = (end_value / start_value) ** (1 / years) - 1
        mean_return = sector_data[selected_sector].mean()
        std_dev = sector_data[selected_sector].std()
        sharpe_ratio = (mean_return - risk_free_rate) / std_dev
        downside_returns = data[sector_data[selected_sector] < 0][selected_sector]
        downside_std_dev = downside_returns.std()
        sortino_ratio = (mean_return - risk_free_rate) / downside_std_dev
        # Key statistics
        avg_return = sector_data[selected_sector].mean()
        max_return = sector_data[selected_sector].max()
        min_return = sector_data[selected_sector].min()
        volatility = sector_data[selected_sector].std()
        cumulative_return = (sector_data[selected_sector].add(1).prod() - 1) * 100
        positive_return_rate = (sector_data[selected_sector] > 0).mean() * 100
        # Calculate additional statistics
        max_drawdown = (1 - sector_data[selected_sector].cumsum().apply(np.exp)).max()
        longest_dd_days = len(sector_data[selected_sector].cumsum().apply(np.exp)) - np.argmax(np.maximum.accumulate(sector_data[selected_sector].cumsum().apply(np.exp)) - sector_data[selected_sector].cumsum().apply(np.exp))
        volatility_annualized = sector_data[selected_sector].std() * np.sqrt(12)
        r_squared = np.square(sector_data[selected_sector].cumsum().apply(np.exp).pct_change()).mean()
        calmar_ratio = (sector_data[selected_sector].cumsum().apply(np.exp).pct_change().mean() * 12) / max_drawdown
        skewness = skew(sector_data[selected_sector])
        kurt = kurtosis(sector_data[selected_sector])
        # Calculate the annualized volatility of the original dataset
        volatility_annualized = sector_data[selected_sector].std() * np.sqrt(12)
        # Desired volatility (hypothetical value, adjust as needed)
        desired_volatility = 0.15
        # Create volatility-matched returns
        volatility_matched_return = sector_data[selected_sector] * (desired_volatility / volatility_annualized)
        #data['Volatility_Matched_Return'] = sector_data[selected_sector] * (desired_volatility / volatility_annualized)
        # Calculate the expected daily, monthly, and yearly percentages
        expected_daily_pct = volatility_matched_return.mean()
        expected_monthly_pct = expected_daily_pct * 30  # Assuming 30 trading days per month
        expected_yearly_pct = expected_daily_pct * 252  # Assuming 252 trading days per year
        # Calculate Kelly Criterion
        win_rate = (volatility_matched_return > 0).mean()
        loss_rate = 1 - win_rate
        kelly_criterion = (win_rate * volatility_matched_return.mean() - loss_rate) / (volatility_matched_return.std() ** 2)
        # Calculate Risk of Ruin
        risk_of_ruin = (loss_rate / win_rate) ** kelly_criterion
        # Calculate Daily Value-at-Risk (VaR) at 95% confidence level
        daily_var = norm.ppf(0.05, expected_daily_pct, volatility_matched_return.std())
        # Calculate Expected Shortfall (Conditional Value-at-Risk) at 95% confidence level
        cvar = -norm.pdf(norm.ppf(0.05)) * volatility_matched_return.std() / 0.05
        
        # Calculate cumulative returns
        sector_data['Cumulative_Return'] = (1 + sector_data[selected_sector] / 100).cumprod() - 1
        
        # Calculate drawdowns
        sector_data['Peak'] = sector_data['Cumulative_Return'].cummax()
        sector_data['Drawdown'] = sector_data['Cumulative_Return'] - sector_data['Peak']
        # Calculate average drawdown
        average_drawdown = sector_data['Drawdown'].mean()

        # Find top 5 drawdown periods
        top_drawdowns = sector_data.nlargest(5, 'Drawdown')

        # Calculate additional statistics
        start_date = sector_data.index[0]
        end_date = sector_data.index[-1]
        mtd_return = sector_data[sector_data.index.month == end_date.month][selected_sector].sum()
        three_month_return = sector_data[(end_date - pd.DateOffset(months=3) <= sector_data.index)]['Cumulative_Return'].iloc[-1]
        six_month_return = sector_data[(end_date - pd.DateOffset(months=6) <= sector_data.index)]['Cumulative_Return'].iloc[-1]
        ytd_return = sector_data[sector_data.index.year == end_date.year]['Cumulative_Return'].iloc[-1]
        one_year_return = sector_data[(end_date - pd.DateOffset(years=1) <= sector_data.index)]['Cumulative_Return'].iloc[-1]
        three_year_annualized_return = ((1 + sector_data['Cumulative_Return'].iloc[-1]) ** (1/3)) - 1
        five_year_annualized_return = ((1 + sector_data['Cumulative_Return'].iloc[-1]) ** (1/5)) - 1
        ten_year_annualized_return = ((1 + sector_data['Cumulative_Return'].iloc[-1]) ** (1/10)) - 1
        all_time_annualized_return = ((1 + sector_data['Cumulative_Return'].iloc[-1]) ** (1/(end_date.year - start_date.year))) - 1
        best_day_return = sector_data[selected_sector].max()
        worst_day_return = sector_data[selected_sector].min()
        best_month_return = sector_data.groupby(sector_data.index.to_period('M'))['Cumulative_Return'].sum().max()
        worst_month_return = sector_data.groupby(sector_data.index.to_period('M'))['Cumulative_Return'].sum().min()
        best_year_return = sector_data.groupby(sector_data.index.year)['Cumulative_Return'].sum().max()
        worst_year_return = sector_data.groupby(sector_data.index.year)['Cumulative_Return'].sum().min()
        avg_drawdown = (sector_data['Cumulative_Return'] - sector_data['Cumulative_Return'].cummax()).mean()
        recovery_factor = (sector_data['Cumulative_Return'].iloc[-1] - sector_data['Cumulative_Return'].iloc[0]) / abs(avg_drawdown)
        ulcer_index = np.sqrt((1 / len(sector_data)) * ((sector_data['Cumulative_Return'] - sector_data['Cumulative_Return'].cummax()) ** 2).sum())
        avg_up_month = sector_data[sector_data[selected_sector] > 0][selected_sector].mean()
        avg_down_month = sector_data[sector_data[selected_sector] < 0][selected_sector].mean()
        win_days_percent = (sector_data[selected_sector] > 0).mean() * 100
        win_month_percent = (sector_data[sector_data[selected_sector] > 0].shape[0] / sector_data.shape[0]) * 100

         # Calculate returns for different intervals
        daily_returns = sector_data[selected_sector]
        weekly_returns = sector_data[selected_sector].resample('W').sum()
        monthly_returns = sector_data[selected_sector].resample('M').sum()
        quarterly_returns = sector_data[selected_sector].resample('Q').sum()
        yearly_returns = sector_data[selected_sector].resample('Y').sum()
        # Calculate return quantiles for different intervals
        daily_quantiles = daily_returns.quantile([0.25, 0.5, 0.75])
        weekly_quantiles = weekly_returns.quantile([0.25, 0.5, 0.75])
        monthly_quantiles = monthly_returns.quantile([0.25, 0.5, 0.75])
        quarterly_quantiles = quarterly_returns.quantile([0.25, 0.5, 0.75])
        yearly_quantiles = yearly_returns.quantile([0.25, 0.5, 0.75])
        
       
         # Risk and Returns Stats
        risk_stats = [
            html.Li(f"CAGR: {CAGR:.2f}%"),
            html.Li(f"Mean Return: {mean_return:.2f}%"),
            html.Li(f"Standard Deviation: {std_dev:.2f}%"),
            html.Li(f"Sharpe Ratio: {sharpe_ratio:.2f}"),
            html.Li(f"Sortino Ratio: {sortino_ratio:.2f}"),
            html.Li(f"Positive Return Rate: {positive_return_rate:.2f}"),
            html.Li(f"Expected Daily: {expected_daily_pct:.0f}"),
            html.Li(f"Expected Yearly: {expected_yearly_pct:.0f}"),
            html.Li(f"Kelly Criterion: {kelly_criterion:.2f}"),
            html.Li(f"Win Rate: {win_rate:.2f}"),
            html.Li(f"Loss Rate: {loss_rate:.2f}"),
        ]
         # Volatility and Drawdown Stats
        volatility_stats = [
            html.Li(f"Volatility (Annualized): {volatility_annualized:.2f}%"),
            html.Li(f"Max Drawdown: {max_drawdown:.2f}%"),
            html.Li(f"Average Drawdown: {average_drawdown:.2f}%"),
            html.Li(f"Longest Drawdown Days: {longest_dd_days:.0f}"),
            html.Li(f"Average Drawdown: {average_drawdown:.2f}%"),
            html.Li(f"Skewness: {skewness:.2f}%"),
            html.Li(f"Kurtosis: {kurt:.2f}%"),
            html.Li(f"Conditional Value-at-Risk: {cvar:.2f}%"),
            html.Li(f"Value-at-Risk: {daily_var:.2f}%"),
            html.Li(f"Ulcer Index: {ulcer_index:.2f}%"),
            html.Li(f"Desired Volatility: {desired_volatility:.2f}%"),
            html.Li(f"Volatility-Matched Return: {volatility_matched_return.mean():.2f}%"),
        ]
        
                # Performance Metrics
        performance_stats = [
            html.Li(f"Start Date: {start_date.strftime('%Y-%m-%d')}"),
            html.Li(f"End Date: {end_date.strftime('%Y-%m-%d')}"),
            html.Li(f"Month-to-Date Return (MTD): {mtd_return:.2f}%"),
            html.Li(f"3-Month Return: {three_month_return:.2f}%"),
            html.Li(f"6-Month Return: {six_month_return:.2f}%"),
            html.Li(f"Year-to-Date Return (YTD): {ytd_return:.2f}%"),
            html.Li(f"1-Year Return: {one_year_return:.2f}%"),
            html.Li(f"3-Year Annualized Return: {three_year_annualized_return:.2f}%"),
            html.Li(f"5-Year Annualized Return: {five_year_annualized_return:.2f}%"),
            html.Li(f"10-Year Annualized Return: {ten_year_annualized_return:.2f}%"),
            html.Li(f"All-Time Annualized Return: {all_time_annualized_return:.2f}%"),
            html.Li(f"Best Day Return: {best_day_return:.2f}%"),
            html.Li(f"Worst Day Return: {worst_day_return:.2f}%"),
            html.Li(f"Best Month Return: {best_month_return:.2f}%"),
            html.Li(f"Worst Month Return: {worst_month_return:.2f}%"),
            html.Li(f"Best Year Return: {best_year_return:.2f}%"),
            html.Li(f"Worst Year Return: {worst_year_return:.2f}%"),
        ]

        
        
                        # Tail Risk and Advanced Metrics
        advance_performance_stats = [
            html.Li(f"R-Squared: {r_squared:.2f}%"),
            html.Li(f"Calmar Ratio: {calmar_ratio:.2f}%"),
            #html.Li(f"Expected Shortfall: {expected_shortfall:.2f}%"),
            html.Li(f"Recovery Factor: {recovery_factor:.2f}%"),
            #html.Li(f"Drawdown Details: {drawdown_details:.2f}%"),
        ]
        
                                  # Interval-Based Returns
        based_return_stats = [
            html.Li(f"Daily Returns (Mean): {daily_returns.mean():.2f}%"),
            html.Li(f"Weekly Returns: {weekly_returns.mean():.2f}%"),
            html.Li(f"Monthly Returns: {monthly_returns.mean():.2f}%"),
            html.Li(f"Quarterly Returns: {quarterly_returns.mean():.2f}%"),
            html.Li(f"Yearly Returns: {yearly_returns.mean():.2f}%"),
        ]
    
        


        
        
        
       

        # Histogram for return distribution with KDE and average line
        avg_value = sector_data[selected_sector].mean()
        x_vals = sector_data[selected_sector]
        kde = gaussian_kde(x_vals)
        x_range = np.linspace(x_vals.min(), x_vals.max(), 100)
        kde_vals = kde(x_range)

        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(
            x=x_vals,
            nbinsx=20,
            name="Histogram",
            marker_color='skyblue',
            opacity=0.7
        ))
        fig_hist.add_trace(go.Scatter(
            x=x_range,
            y=kde_vals * len(x_vals) / 10,  # Scale KDE to histogram
            mode="lines",
            line=dict(color="green", width=2),
            name="KDE"
        ))
        fig_hist.add_trace(go.Scatter(
            x=[avg_value, avg_value],
            y=[0, max(kde_vals) * len(x_vals) / 10],
            mode="lines",
            line=dict(color="red", dash="dash"),
            name="Average Return"
        ))
        fig_hist.update_layout(
            title="Return Distribution with KDE and Average Line",
            xaxis_title="Return",
            yaxis_title="Frequency",
            barmode="overlay",
            template="plotly_white"
        )
          # Highlight drawdown periods
        fig_line_draw_down = go.Figure()

        # Add cumulative return line
        fig_line_draw_down.add_trace(
            go.Scatter(
                x=sector_data.index,
                y=sector_data['Cumulative_Return'],
                mode='lines',
                name='Cumulative Return',
                line=dict(color='blue')
            )
        )

       
        fig_line_draw_down.add_trace(
            go.Scatter(
                x=sector_data.index,
                y=sector_data['Drawdown'],
                mode='lines',
                name='Drawdown',
                line=dict(color='red', dash='dot')
            )
        )

        # Layout customization
        fig_line_draw_down.update_layout(
            title="Cumulative Returns and Drawdowns",
            xaxis_title="Date",
            yaxis_title="Value",
            template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        # Create the underwater plot
        fig_underwater = go.Figure()

        # Add cumulative return line
        fig_underwater.add_trace(
            go.Scatter(
                x=sector_data.index,
                y=sector_data['Cumulative_Return'],
                mode='lines+markers',
                name='Cumulative Return',
                line=dict(color='blue'),
                marker=dict(size=4, color='blue', symbol='circle')
            )
        )

        # Add shaded area for underwater periods
        underwater_condition = sector_data['Cumulative_Return'] < sector_data['Peak']
        fig_underwater.add_trace(
            go.Scatter(
                x=sector_data.index,
                y=np.where(underwater_condition, sector_data['Cumulative_Return'], sector_data['Peak']),
                fill='tonexty',
                fillcolor='rgba(255, 0, 0, 0.3)',
                name='Underwater',
                mode='lines',
                line=dict(width=0)  # No border line
            )
        )

        # Add horizontal line for average drawdown
        fig_underwater.add_trace(
            go.Scatter(
                x=[sector_data.index.min(), sector_data.index.max()],
                y=[average_drawdown, average_drawdown],
                mode='lines',
                line=dict(color='red', dash='dash'),
                name='Average Drawdown'
            )
        )

        # Add markers for underwater points
        fig_underwater.add_trace(
            go.Scatter(
                x=sector_data.index[underwater_condition],
                y=sector_data['Cumulative_Return'][underwater_condition],
                mode='markers',
                marker=dict(size=6, color='red', symbol='circle'),
                name='Underwater Points'
            )
        )

        # Layout customization
        fig_underwater.update_layout(
            title="Underwater Plot with Average Drawdown",
            xaxis_title="Date",
            yaxis_title="Cumulative Return",
            template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            xaxis=dict(showgrid=True),
            yaxis=dict(showgrid=True)
        )

         # Create the Box figure
        fig_box = go.Figure()

        # Daily Return Quantiles
        fig_box.add_trace(
            go.Box(
                y=daily_returns,
                name='Daily',
                boxpoints='outliers',
                marker_color='blue'
            )
        )
        for q, v in daily_quantiles.items():
            fig_box.add_trace(
                go.Scatter(
                    x=['Daily'],
                    y=[v],
                    mode='markers',
                    marker=dict(color='red', size=8),
                    name=f'Daily Quantile {q:.2f}'
                )
            )

        # Weekly Return Quantiles
        fig_box.add_trace(
            go.Box(
                y=weekly_returns,
                name='Weekly',
                boxpoints='outliers',
                marker_color='green'
            )
        )
        for q, v in weekly_quantiles.items():
            fig_box.add_trace(
                go.Scatter(
                    x=['Weekly'],
                    y=[v],
                    mode='markers',
                    marker=dict(color='red', size=8),
                    name=f'Weekly Quantile {q:.2f}'
                )
            )

        # Monthly Return Quantiles
        fig_box.add_trace(
            go.Box(
                y=monthly_returns,
                name='Monthly',
                boxpoints='outliers',
                marker_color='orange'
            )
        )
        for q, v in monthly_quantiles.items():
            fig_box.add_trace(
                go.Scatter(
                    x=['Monthly'],
                    y=[v],
                    mode='markers',
                    marker=dict(color='red', size=8),
                    name=f'Monthly Quantile {q:.2f}'
                )
            )

        # Quarterly Return Quantiles
        fig_box.add_trace(
            go.Box(
                y=quarterly_returns,
                name='Quarterly',
                boxpoints='outliers',
                marker_color='purple'
            )
        )
        for q, v in quarterly_quantiles.items():
            fig_box.add_trace(
                go.Scatter(
                    x=['Quarterly'],
                    y=[v],
                    mode='markers',
                    marker=dict(color='red', size=8),
                    name=f'Quarterly Quantile {q:.2f}'
                )
            )

        # Yearly Return Quantiles
        fig_box.add_trace(
            go.Box(
                y=yearly_returns,
                name='Yearly',
                boxpoints='outliers',
                marker_color='cyan'
            )
        )
        for q, v in yearly_quantiles.items():
            fig_box.add_trace(
                go.Scatter(
                    x=['Yearly'],
                    y=[v],
                    mode='markers',
                    marker=dict(color='red', size=8),
                    name=f'Yearly Quantile {q:.2f}'
                )
            )

        # Layout customization
        fig_box.update_layout(
            title=f'Return Quantiles for {selected_sector}',
            yaxis_title='Return',
            xaxis_title='Time Interval',
            template='plotly_white',
            showlegend=True
        )
        
       # Create the figure
        fig_line_yearly = go.Figure()

        # Add the EOY returns line
        fig_line_yearly.add_trace(
            go.Scatter(
                x=yearly_returns.index,
                y=yearly_returns,
                mode='lines+markers',
                name='EOY Returns',
                line=dict(color='blue'),
                marker=dict(size=8, color='blue', symbol='circle')
            )
        )

        # Layout customization
        fig_line_yearly.update_layout(
            title=f'End of Year (EOY) Returns for {selected_sector}',
            xaxis_title='Year',
            yaxis_title='Cumulative Return',
            template='plotly_white',
            xaxis=dict(tickformat='%Y', showgrid=True),
            yaxis=dict(showgrid=True),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )


        return fig_line_log,fig_line_vol,fig_line_sharpe, risk_stats,volatility_stats,performance_stats,advance_performance_stats,based_return_stats , fig_hist , fig_line_draw_down , fig_underwater , fig_box, fig_line_yearly

    return {}, "", {}
# Callback to update the bar chart based on user selection
@app.callback(
    Output('bar-chart-ev', 'figure'),
    [Input('metric-dropdown', 'value')]
)
def update_chart(selected_metric):
    # Prepare data for plotting
    bar_data = []
    for sector, truncated_sector in zip(rmse_mae_result_dataframe['Sectors'], truncated_sector_names):
        bar_data.append(go.Bar(
            x=[truncated_sector],
            y=[rmse_mae_result_dataframe.loc[rmse_mae_result_dataframe['Sectors'] == sector, selected_metric].iloc[0]],
            name=f'{truncated_sector}'
        ))

    # Define layout
    layout = go.Layout(
        title=f'Bar Chart of {selected_metric} for Each Sector',
        xaxis={'title': 'Sectors'},
        yaxis={'title': selected_metric}
    )

    # Return the figure
    return {'data': bar_data, 'layout': layout}

# Define callback to update column dropdown based on selected dataframe
@app.callback(
    Output('column-dropdown1', 'options'),
    [Input('dataframe-dropdown', 'value')]
)
def update_column_dropdown(selected_dataframe):
    dataframe = globals()[selected_dataframe]  # Get the selected dataframe
    options = [{'label': col, 'value': col} for col in dataframe.columns]
    return options

# Define callback to display customized dataframe
@app.callback(
    Output('dataframe-output', 'children'),
    [Input('change-column-button', 'n_clicks')],
    [State('dataframe-dropdown', 'value'),
     State('column-dropdown1', 'value'),
     State('change-dataframe-dropdown', 'value')]
)
def change_column(n_clicks, selected_dataframe, column, change_dataframe):
    if n_clicks > 0 and column is not None:
        global change_df 
        change_df = pd.DataFrame(globals()[selected_dataframe])
        change_df[column] = globals()[change_dataframe][column]
        return dash_table.DataTable(
            id='table',
            columns=[{'name': col, 'id': col} for col in change_df.columns],
            data=change_df.to_dict('records'),
            style_table={'height': '400px', 'overflowY': 'auto'},
            style_cell={'textAlign': 'center', 'fontSize': 12},
            style_header={'backgroundColor': 'lightgrey', 'fontWeight': 'bold'}
        )

# Define callback to download modified data
@app.callback(
    Output('download-link', 'href'),
    [Input('change-column-button', 'n_clicks')],
    [State('dataframe-dropdown', 'value')]
)
def download_modified_data(n_clicks, selected_dataframe):
    if n_clicks > 0:
        change_df = pd.DataFrame(globals()[selected_dataframe])
        csv_string = change_df.to_csv(index=False, encoding='utf-8')
        csv_string = "data:text/csv;charset=utf-8," + urllib.parse.quote(csv_string)
        return csv_string
    else:
        return ""

# Define callback to update the line chart and table based on the selected column and forecast method
@app.callback(
    [Output('line-chart', 'figure'),
     Output('data-table1', 'data')],
    [Input('column-dropdown', 'value'),
     Input('forecast-method-dropdown', 'value')]
)
def update_chart_and_table(selected_columns, selected_forecast_methods):
    # Create an empty figure
    fig = px.line(title="Selected Columns over Time")

    # Define a list of colors for the lines
    line_colors = px.colors.qualitative.Set1

    for idx, selected_column in enumerate(selected_columns):
        for forecast_method in selected_forecast_methods:
            line_color = line_colors[idx % len(line_colors)]  # Cycle through colors for each selected column

            if forecast_method == 'original':
                # Plot the original data
                fig.add_trace(px.line(data, x=data.index, y=selected_column, title=f'{selected_column} over time').update_traces(line=dict(color=line_color)).data[0])
            elif forecast_method == 'lstm':
                # Assuming you have 'lstm_forecast' DataFrame
                forecast_data = lstm_forecast_data[selected_column]

                fig.add_trace(px.line(lstm_forecast_data, x=lstm_forecast_data.index, y=selected_column, title=f'LSTM Forecast: {selected_column} (RMSE: {2.3:.2f}, MAE: {1.4:.2f})').update_traces(line=dict(color=line_color)).data[0])
            elif forecast_method == 'xgboost':
                # Assuming you have 'xgboost_forecast' DataFrame
                forecast_data = xgboost_forecast_data[selected_column]

                fig.add_trace(px.line(xgboost_forecast_data, x=xgboost_forecast_data.index, y=selected_column, title=f'XGBoost Forecast: {selected_column} (RMSE: {4:.2f}, MAE: {2:.2f})').update_traces(line=dict(color=line_color)).data[0])
            elif forecast_method == 'sarima':
                # Assuming you have 'sarima_forecast' DataFrame
                forecast_data = sarima_forecast_data[selected_column]

                fig.add_trace(px.line(sarima_forecast_data, x=sarima_forecast_data.index, y=selected_column, title=f'SARIMA Forecast: {selected_column} (RMSE: {6:.2f}, MAE: {1:.2f})').update_traces(line=dict(color=line_color)).data[0])
            elif forecast_method == 'fbprophet':
                # Assuming you have 'fbprophet_forecast' DataFrame
                forecast_data = fbprophet_forecast_data[selected_column]

                fig.add_trace(px.line(fbprophet_forecast_data, x=fbprophet_forecast_data.index, y=selected_column, title=f'FBProphet Forecast: {selected_column} (RMSE: {7:.2f}, MAE: {4.53:.2f})').update_traces(line=dict(color=line_color)).data[0])

    return fig, data.to_dict('records')
# Define callback to update forecast table based on dropdown selection
@app.callback(
    Output('table-container', 'children'),
    [Input('forecast-compare-dropdown', 'value')]
)
def update_forecast_table(selected_option):
    selected_df = forecast_options[selected_option]
    return dash_table.DataTable(
        id='table2',
        columns=[{"name": i, "id": i} for i in selected_df.columns[1:]],
        data=selected_df.to_dict('records'),
        style_cell={'font_size': '6.5pt'},  # Adjust font size
        style_table={'overflowX': 'scroll', 'maxWidth': '100%'}  # Adjust table size
    )
# Define callback to update forecast table based on dropdown selection
@app.callback(
    Output('table-container_asset', 'children'),
    [Input('forecast-compare-dropdown_asset', 'value')]
)
def update_forecast_table_asset(selected_option):
    selected_df = forecast_options[selected_option]
    selected_df["Asset"] = asset_names
    selected_df = selected_df.pivot_table(index='Asset', aggfunc='mean')
    selected_df = selected_df.reset_index()
    return dash_table.DataTable(
        id='table2',
        columns=[{"name": i, "id": i} for i in selected_df.columns[1:]],
        
        data=selected_df.to_dict('records'),
        style_cell={'font_size': '6.5pt'},  # Adjust font size
        style_table={'overflowX': 'scroll', 'maxWidth': '100%'}  # Adjust table size
    )


# Function to handle "Update Value" button click
@app.callback(
    [Output('update-output', 'children'),
     Output('data-table', 'columns'),
     Output('data-table', 'data')],
    [Input('update-button', 'n_clicks')],
    [State('forecast-model-dropdown', 'value'),
     State('column-input', 'value'),
     State('row-input', 'value'),
     State('multiplier-dropdown', 'value'),
     State('input-value', 'value')]
)
def update_value(n_clicks, selected_forecast_model, selected_column, selected_row, multiplier_str, input_value):
    if n_clicks is None:
        raise dash.exceptions.PreventUpdate
    global change_df
    customize_table = generate_dataframe(change_df)
    # Get the selected forecast model dataframe
    if selected_forecast_model == 'sarima':
        df = sarima_forecast
    elif selected_forecast_model == 'xgboost':
        df = xgboost_forecast
    elif selected_forecast_model == 'fbprophet':
        df = fbprophet_forecast
    elif selected_forecast_model == 'customize table':
        df = customize_table
    else:  # Default to LSTM
        df = lstm_forecast

    # Update the dataframe based on the user input
    if pd.isna(input_value):
        # If input_value is NaN, update using multiplier
        updated_value = df.at[selected_row, selected_column] * get_multiplier_value(multiplier_str)
    else:
        # If input_value is provided, use it
        updated_value = input_value

    # Update the dataframe
    df.at[selected_row, selected_column] = updated_value

    # Update the 'Categorize' column
    categorize_column = f'Categorize_{selected_column}'
    df.at[selected_row, categorize_column] = multiplier_str

    # Prepare data for DataTable
    columns = [{'name': col, 'id': col} for col in df.columns]
    data = df.to_dict('records')

    return f"Successfully Updated On {selected_column} row {selected_row}", columns, data


def get_multiplier_value(multiplier_str):
    multiplier_mapping = {
        'Very strong buy': 1.2,
        'Strong buy': 1.15,
        'Buy': 1.1,
        'Weak buy': 1.05,
        'Neutral': 1,
        'Weak sell': 0.95,
        'Sell': 0.9,
        'Strong sell': 0.85,
        'Very strong sell': 0.8,
        'Exclude': 0,
    }
    return multiplier_mapping.get(multiplier_str, 1)


# Callback to download the updated CSV
@app.callback(
    Output('download-updated-csv', 'data'),
    [Input('download-updated-csv-button', 'n_clicks')],
    [State('forecast-model-dropdown', 'value')]
)
def download_updated_csv(n_clicks, selected_forecast_model):
    if n_clicks is None:
        raise dash.exceptions.PreventUpdate
    
    buffer = io.StringIO()
    if selected_forecast_model == 'sarima':
        sarima_forecast.to_csv(buffer, index=False)
    elif selected_forecast_model == 'xgboost':
        xgboost_forecast.to_csv(buffer, index=False)
    elif selected_forecast_model == 'fbprophet':
        fbprophet_forecast.to_csv(buffer, index=False)
    elif selected_forecast_model == 'customize table':
        customize_table.to_csv(buffer,index=False)
    else:  # Default to LSTM
        lstm_forecast.to_csv(buffer, index=False)
    buffer.seek(0)

    return dict(content=buffer.getvalue(), filename='updated_dataframe.csv')


# Callback to download the sharp optimal CSV
@app.callback(
    Output('sharp-download-optimal-csv', 'data'),
    [Input('sharp-download-optimal-csv-button', 'n_clicks')],
    prevent_initial_call=True
)
def download_optimal_csv(n_clicks):
    if n_clicks is None:
        raise dash.exceptions.PreventUpdate

    buffer = io.StringIO()
    optimal_model.to_csv(buffer, index=False)  # Replace df_optimal with your optimal DataFrame
    buffer.seek(0)

    return dict(content=buffer.getvalue(), filename='optimal_model.csv')


# Callback to download the optimal CSV
@app.callback(
    Output('download-optimal-csv', 'data'),
    [Input('download-optimal-csv-button', 'n_clicks')],
    prevent_initial_call=True
)
def download_optimal_csv(n_clicks):
    if n_clicks is None:
        raise dash.exceptions.PreventUpdate

    buffer = io.StringIO()
    optimal_model.to_csv(buffer, index=False)  # Replace df_optimal with your optimal DataFrame
    buffer.seek(0)

    return dict(content=buffer.getvalue(), filename='optimal_model.csv')

@app.callback(
    [Output('set_objective_line-chart', 'figure'),
     Output('set_objective_table', 'data')],
    [Input('set_optimize_button', 'n_clicks')],
    [dash.dependencies.State('set_objective_dropdown', 'value'),
     dash.dependencies.State('set_objective', 'value'),
     dash.dependencies.State('set_tolerance', 'value')]
)
def update_output(n_clicks, selected_df, objective, tolerance):
    if n_clicks > 0:
        # Assuming you have the optimize_weights function defined above
        df = dropdown_options[selected_df]
        
        df.index = pd.date_range(start=pd.Timestamp.now().date(), periods=len(df), freq='D')
        set_optimal_model, set_risk_values = optimize_weights(df, objective, tolerance)

        # Create line chart
        line_chart = go.Figure()
        line_chart.add_trace(go.Scatter(x=['Year1','Year2','Year3','Year4','Year5'], y=set_risk_values, mode='lines', name='Risk Values'))
        line_chart.update_layout(title='Risk Values Over Time')

        # Prepare data for table
        table_data = set_optimal_model.to_dict('records')

        return line_chart, table_data
    else:
        return {}, [] 


@app.callback(
    [Output('optimal-table', 'columns'),
     Output('optimal-table', 'data'),
     Output('total-table', 'columns'),
     Output('total-table', 'data'),
     Output('discrete-period-return', 'figure'),
     Output('cumulative-return', 'figure'),
     Output('portfolio-value', 'figure'),
     Output('Hdiscrete-period-return', 'figure'),
     Output('Hcumulative-return', 'figure'),
     Output('Hportfolio-value', 'figure'),
     Output('risk', 'figure'),
     Output('historical_risk', 'figure'),
     Output('efficient-frontier-plot', 'figure'),
     Output('pie-charts-container', 'children'),
     Output('metrics-container', 'children'),
     Output('risk-bar-chart', 'figure'),
     Output('sharpe-scatter-plot', 'figure')],
    [Input('optimize-button', 'n_clicks')],
    [State('data-table', 'data'),
     State('min-weight-input', 'value'),
     State('max-weight-input', 'value'),
     State('total-investment-input', 'value'),
     State('volatility-input', 'value'),
     State('optimize-dropdown', 'value')],
    prevent_initial_call=True
)
def optimize_button_click(n_clicks, data_table_data, min_weight, max_weight, total_investment, volatility, dropdown_value):
    global optimal_model, historical_optimize

    # Extract updated data from data_table_data
    updated_df = pd.DataFrame(data_table_data)
    updated_df = updated_df.iloc[:, 1:6]  # Assuming relevant columns are from index 1 to 5

    # Optimize the weights using the updated data
    optimal_model, risk, portfolio_returns, simulated_portfolio_risks, simulated_portfolio_return = pre_optimize_weights(
        updated_df, minimum_weight=min_weight, maximum_weight=max_weight, total_investment=total_investment,
        volitility_value=volatility
    )

    historical_optimize_model, historical_risk, p, sr, spr = pre_optimize_weights(
        historical.iloc[:, 1:6],
        minimum_weight=min_weight, maximum_weight=max_weight, total_investment=total_investment,
        volitility_value=volatility
    )

    # Generate optimal table columns and data
    optimal_columns = [{'name': col, 'id': col} for col in optimal_model.columns]

    # Calculate discrete and cumulative returns
    discreate_period_return = [optimal_model[f'Sumproduct_Year{i+1}'].sum() / 100 for i in range(5)]
    cumulative_return = [100 + sum(discreate_period_return[:i+1]) for i in range(5)]
    portfolio_value = [1000000 + sum([(discreate_period_return[j] * 100000) for j in range(i+1)]) for i in range(5)]

    # Calculate historical returns
    Hdiscrete_period_return = [historical_optimize_model[f'Sumproduct_Year{i+1}'].sum() / 100 for i in range(5)]
    Hcumulative_return = [100 + sum(Hdiscrete_period_return[:i+1]) for i in range(5)]
    Hportfolio_value = [1000000 + sum([(Hdiscrete_period_return[j] * 100000) for j in range(i+1)]) for i in range(5)]

    # Generate figures
    discrete_fig = {
        'data': [{'x': [f'Year{i+1}' for i in range(5)], 'y': discreate_period_return, 'type': 'bar', 'name': 'Discrete Period Return'}],
        'layout': {'title': 'Discrete Return', 'height': 400, 'width': 400}
    }

    cumulative_fig = {
        'data': [{'x': [f'Year{i+1}' for i in range(5)], 'y': cumulative_return, 'type': 'line', 'name': 'Cumulative Return'}],
        'layout': {'title': 'Cumulative Return', 'height': 400, 'width': 400}
    }

    portfolio_fig = {
        'data': [{'x': [f'Year{i+1}' for i in range(5)], 'y': portfolio_value, 'type': 'line', 'name': 'Portfolio Value'}],
        'layout': {'title': 'Portfolio Value', 'height': 400, 'width': 400}
    }

    Hdiscrete_fig = {
        'data': [{'x': [f'Year{i+1}' for i in range(5)], 'y': Hdiscrete_period_return, 'type': 'bar', 'name': 'Historical Discrete Period Return'}],
        'layout': {'title': 'Historical Discrete Return', 'height': 400, 'width': 400}
    }

    Hcumulative_fig = {
        'data': [{'x': [f'Year{i+1}' for i in range(5)], 'y': Hcumulative_return, 'type': 'line', 'name': 'Historical Cumulative Return'}],
        'layout': {'title': 'Historical Cumulative Return', 'height': 400, 'width': 400}
    }

    Hportfolio_fig = {
        'data': [{'x': [f'Year{i+1}' for i in range(5)], 'y': Hportfolio_value, 'type': 'line', 'name': 'Historical Portfolio Value'}],
        'layout': {'title': 'Historical Portfolio Value', 'height': 400, 'width': 400}
    }

    forecast_volatility = {
        'data': [{'x': [f'Year{i+1}' for i in range(5)], 'y': risk, 'type': 'line', 'name': 'Forecast Volatility'}],
        'layout': {'title': 'Forecast Volatility', 'height': 400, 'width': 400}
    }

    historical_volatility = {
        'data': [{'x': [f'Year{i+1}' for i in range(5)], 'y': historical_risk, 'type': 'line', 'name': 'Historical Volatility'}],
        'layout': {'title': 'Historical Volatility', 'height': 400, 'width': 400}
    }

    # Total table columns and data
    total_table = optimal_model.pivot_table(index='Asset', aggfunc='sum').reset_index()
    total_table_columns = [{'name': col, 'id': col} for col in total_table.columns]

    
    # Scatter plot for efficient frontier
    trace_simulated = go.Scatter(
        x=simulated_portfolio_risks,
        y=simulated_portfolio_return,
        mode='markers',
        marker=dict(color='blue', opacity=0.2),
        name='Simulated Portfolios'
    )

    trace_optimal = go.Scatter(
        x=risk,
        y=portfolio_returns,
        mode='markers',
        marker=dict(color='red', symbol='star'),
        name='Optimal Portfolios'
    )

    layout = go.Layout(
        title='Efficient Frontier',
        xaxis=dict(title='Volatility'),
        yaxis=dict(title='Return'),
        legend=dict(x=0, y=1)
    ) 
    charts = []
    for year in range(1, 6):
        column_name = f'Optimal_Weights_Year{year}'
        pie_fig = go.Figure(
            data=[go.Pie(
                labels=optimal_model['Asset'],
                values=optimal_model[column_name],
                hole=0.4,
                hoverinfo='label+percent+value',  # Show label, percent, and value on hover
                textinfo='label+percent+value',  # Display label, percent, and value on chart
                texttemplate='%{label}<br>%{value} (%{percent})'  # Format text to include label, value, and percent
            )]
        )
        pie_fig.update_layout(
            title_text=f'Optimal Weights Distribution for Year {year}',
            annotations=[dict(text=f'Year {year}', x=0.5, y=0.5, font_size=20, showarrow=False)],
            showlegend=True
        )
        charts.append(dcc.Graph(figure=pie_fig, id=f'pie-chart-year{year}'))
        
    #Risk Charts According to weights
    sector_name = []
    for i in range(len(data.columns)):
        sector_name.append(data.columns[i][13:-9])
    
    optimal_model1 = optimal_model
    optimal_model1['Sector']=  sector_name
    risk_free_rate = 0.02
    weight_columns = [col for col in optimal_model1.columns if 'Optimal_Weights_Year' in col]
    
    # Standard Deviation
    optimal_model1['Risk (Standard Deviation)'] = optimal_model1[weight_columns].std(axis=1)
    
    # Average Weight
    optimal_model1['Average Weight'] = optimal_model1[weight_columns].mean(axis=1)
    
    # Sharpe Ratio
    optimal_model1['Sharpe Ratio'] = (optimal_model1['Average Weight'] - risk_free_rate) / optimal_model1['Risk (Standard Deviation)']
    
    # Max Drawdown
    max_drawdowns = []
    for _, row in optimal_model1[weight_columns].iterrows():
        peak = row.max()
        trough = row.min()
        max_drawdown = (trough - peak) / peak if peak != 0 else 0
        max_drawdowns.append(max_drawdown)
    optimal_model1['Max Drawdown'] = max_drawdowns

# Step 2: Create a table to display metrics
    table = html.Table([
        html.Thead(
            html.Tr([html.Th(col) for col in ['Asset', 'Risk (Standard Deviation)', 'Sharpe Ratio', 'Max Drawdown']])
        ),
        html.Tbody([
            html.Tr([
                html.Td(optimal_model1.iloc[i]['Asset']),
                html.Td(optimal_model1.iloc[i]['Sector']),
                html.Td(f"{optimal_model1.iloc[i]['Risk (Standard Deviation)']:.2f}"),
                html.Td(f"{optimal_model1.iloc[i]['Sharpe Ratio']:.2f}"),
                html.Td(f"{optimal_model1.iloc[i]['Max Drawdown']:.2f}")
            ]) for i in range(len(optimal_model1))
        ])
    ])
    
    # Step 3: Create a bar chart for risk (standard deviation)
    bar_chart = go.Figure()
    bar_chart.add_trace(go.Bar(
        x=optimal_model1['Asset'],
        y=optimal_model1['Risk (Standard Deviation)'],
        name='Risk (Standard Deviation)'
    ))
    bar_chart.update_layout(
        title='Sector Risk (Volatility)',
        xaxis_title='Asset',
        yaxis_title='Risk (Standard Deviation)',
        showlegend=False
    )
    
    # Step 4: Create a scatter plot for Sharpe Ratio vs. Risk
    scatter_plot = go.Figure()
    scatter_plot.add_trace(go.Scatter(
        x=optimal_model1['Risk (Standard Deviation)'],
        y=optimal_model1['Sharpe Ratio'],
        mode='markers+text',
        text=optimal_model1['Sector'],
        textposition='top center',
        marker=dict(size=10, color='blue')
    ))
    scatter_plot.update_layout(
        title='Risk vs. Reward (Sharpe Ratio)',
        xaxis_title='Risk (Standard Deviation)',
        yaxis_title='Sharpe Ratio',
        showlegend=False
    )


    return optimal_columns, optimal_model.to_dict('records'), total_table_columns, total_table.to_dict('records'), discrete_fig, cumulative_fig, portfolio_fig, Hdiscrete_fig, Hcumulative_fig, Hportfolio_fig, forecast_volatility, historical_volatility, {'data': [trace_simulated, trace_optimal], 'layout': layout},charts,table, bar_chart, scatter_plot

# Define the callback function outside of the layout definition
@app.callback(
    [Output('sharp-optimal-table', 'columns'),
     Output('sharp-optimal-table', 'data'),
     Output('sharp-total-table', 'columns'),
     Output('sharp-total-table', 'data'),
     Output('sharp-discrete-period-return', 'figure'),
     Output('sharp-cumulative-return', 'figure'),
     Output('sharp-portfolio-value', 'figure'),
     Output('sharp-Hdiscrete-period-return', 'figure'),
     Output('sharp-Hcumulative-return', 'figure'),
     Output('sharp-Hportfolio-value', 'figure'),
     Output('sharp-risk', 'figure'),
     Output('sharp-historical_risk', 'figure'),
     Output('sharp-efficient-frontier-plot', 'figure'),
     Output('sharp-pie-charts-container', 'children'),
     Output('sharp-metrics-container', 'children'),
     Output('sharp-risk-bar-chart', 'figure'),
     Output('sharp-sharpe-scatter-plot', 'figure')],
    [Input('sharp-optimize-button', 'n_clicks')],
    [State('data-table', 'data'),
     State('sharp-min-weight-input', 'value'),
     State('sharp-max-weight-input', 'value'),
     State('sharp-total-investment-input', 'value'),
     State('sharp-volatility-input', 'value'),
     State('risk-measure-dropdown', 'value'),
     State('risk-period-dropdown', 'value'),
     State('max-risk-input', 'value'),
     State('min-investment-input', 'value'),
     State('max-investment-input', 'value')],
    prevent_initial_call=True
)
def sharp_optimize_button_click(n_clicks, data_table_data, minimum_weight, maximum_weight, total_investment,
                                volitility_value,
                                risk_measure_type, risk_measure_period, max_risk_tolerance, min_investment_size,
                                max_investment_size):
    global optimal_model, historical_optimize

    # Extract updated data from data_table_data
    updated_df = pd.DataFrame(data_table_data)
    updated_df = updated_df[updated_df.columns[1:6]]  # Assuming the relevant columns are from index 1 to 5

    # Optimize the weights using the updated data
    optimal_model, risk, portfolio_returns, simulated_portfolio_risks, simulated_portfolio_return = pre_optimize_weights_max_sharp(
        updated_df, minimum_weight, maximum_weight, total_investment, volitility_value, risk_measure_type,
        risk_measure_period, max_risk_tolerance, min_investment_size, max_investment_size)
    historical_optimize_model, historical_risk, p, sr, spr = pre_optimize_weights_max_sharp(
        historical[historical.columns[1:6]], minimum_weight, maximum_weight, total_investment, volitility_value,
        risk_measure_type, risk_measure_period, max_risk_tolerance, min_investment_size, max_investment_size)

    # Process and calculate figures
    optimal_columns = [{'name': col, 'id': col} for col in optimal_model.columns]

    discreate_period_return = [optimal_model['Sumproduct_Year1'].sum() / 100,
                               optimal_model['Sumproduct_Year2'].sum() / 100,
                               optimal_model['Sumproduct_Year3'].sum() / 100,
                               optimal_model['Sumproduct_Year4'].sum() / 100,
                               optimal_model['Sumproduct_Year5'].sum() / 100]

    # Optimize table
    cumulative_return = []
    v = 100
    for i in range(5):
        v = v + discreate_period_return[i]
        cumulative_return.append(v)
    portfolio_value = []
    v = 1000000
    for i in range(5):
        v = (v + (discreate_period_return[i] * 100) * 100)
        portfolio_value.append(v)

    # Historical return
    Hdiscreate_period_return = [historical_optimize_model['Sumproduct_Year1'].sum() / 100,
                                historical_optimize_model['Sumproduct_Year2'].sum() / 100,
                                historical_optimize_model['Sumproduct_Year3'].sum() / 100,
                                historical_optimize_model['Sumproduct_Year4'].sum() / 100,
                                historical_optimize_model['Sumproduct_Year5'].sum() / 100]
    Hcumulative_return = []
    v = 100
    for i in range(5):
        v = v + Hdiscreate_period_return[i]
        Hcumulative_return.append(v)
    Hportfolio_value = []
    v = 1000000
    for i in range(5):
        v = (v + (Hdiscreate_period_return[i] * 100) * 100)
        Hportfolio_value.append(v)
    # Update the optimize model figures
    discrete_fig = {
        'data': [{'x': ['Year1', 'Year2', 'Year3', 'Year4', 'Year5'], 'y': discreate_period_return, 'type': 'bar',
                  'name': 'Discrete Period Return'}],
        'layout': {'title': 'Discrete Return', 'height': 400, 'width': 400}
    }

    cumulative_fig = {
        'data': [{'x': ['Year1', 'Year2', 'Year3', 'Year4', 'Year5'], 'y': cumulative_return, 'type': 'line',
                  'name': 'Cumulative Return'}],
        'layout': {'title': 'Cumulative Return', 'height': 400, 'width': 400}
    }

    portfolio_fig = {
        'data': [{'x': ['Year1', 'Year2', 'Year3', 'Year4', 'Year5'], 'y': portfolio_value, 'type': 'line',
                  'name': 'Portfolio Value'}],
        'layout': {'title': 'Portfolio Value', 'height': 400, 'width': 400}
    }

    # Update the historical model figures
    Hdiscrete_fig = {
        'data': [{'x': ['Year1', 'Year2', 'Year3', 'Year4', 'Year5'], 'y': Hdiscreate_period_return, 'type': 'bar',
                  'name': 'Historical Discrete Period Return'}],
        'layout': {'title': 'Historical Discrete Return', 'height': 400, 'width': 400}
    }

    Hcumulative_fig = {
        'data': [{'x': ['Year1', 'Year2', 'Year3', 'Year4', 'Year5'], 'y': Hcumulative_return, 'type': 'line',
                  'name': 'Historical Cumulative Return'}],
        'layout': {'title': 'Historical Cumulative Return', 'height': 400, 'width': 400}
    }

    Hportfolio_fig = {
        'data': [{'x': ['Year1', 'Year2', 'Year3', 'Year4', 'Year5'], 'y': Hportfolio_value, 'type': 'line',
                  'name': 'Historical Portfolio Value'}],
        'layout': {'title': 'Historical Portfolio Value', 'height': 400, 'width': 400}
    }

    forecast_volatility = {
        'data': [{'x': ['Year1', 'Year2', 'Year3', 'Year4', 'Year5'], 'y': risk, 'type': 'line',
                  'name': 'Forecast Volatility'}],
        'layout': {'title': 'Forecast Volatility', 'height': 400, 'width': 400}
    }

    historical_volatility = {
        'data': [{'x': ['Year1', 'Year2', 'Year3', 'Year4', 'Year5'], 'y': historical_risk, 'type': 'line',
                  'name': 'Historical Volatility'}],
        'layout': {'title': 'Historical Volatility', 'height': 400, 'width': 400}
    }

    total_table = optimal_model.pivot_table(index='Asset', aggfunc='sum')
    total_table.reset_index(inplace=True)
    total_table_columns = [{'name': col, 'id': col} for col in total_table.columns[0:6]]
    # Scatter plot for efficient frontier
    trace_simulated = go.Scatter(
        x=simulated_portfolio_risks,
        y=simulated_portfolio_return,
        mode='markers',
        marker=dict(color='blue', opacity=0.2),
        name='Simulated Portfolios'
    )

    trace_optimal = go.Scatter(
        x=risk,
        y=portfolio_returns,
        mode='markers',
        marker=dict(color='red', symbol='star'),
        name='Optimal Portfolios'
    )

    layout = go.Layout(
        title='Efficient Frontier',
        xaxis=dict(title='Volatility'),
        yaxis=dict(title='Return'),
        legend=dict(x=0, y=1)
    ) 
    charts = []
    for year in range(1, 6):
        column_name = f'Optimal_Weights_Year{year}'
        pie_fig = go.Figure(
            data=[go.Pie(
                labels=optimal_model['Asset'],
                values=optimal_model[column_name],
                hole=0.4,
                hoverinfo='label+percent+value',  # Show label, percent, and value on hover
                textinfo='label+percent+value',  # Display label, percent, and value on chart
                texttemplate='%{label}<br>%{value} (%{percent})'  # Format text to include label, value, and percent
            )]
        )
        pie_fig.update_layout(
            title_text=f'Optimal Weights Distribution for Year {year}',
            annotations=[dict(text=f'Year {year}', x=0.5, y=0.5, font_size=20, showarrow=False)],
            showlegend=True
        )
        charts.append(dcc.Graph(figure=pie_fig, id=f'pie-chart-year{year}'))
        
    #Risk Charts According to weights
    sector_name = []
    for i in range(len(data.columns)):
        sector_name.append(data.columns[i][13:-9])
    
    optimal_model1 = optimal_model
    optimal_model1['Sector']=  sector_name
    risk_free_rate = 0.02
    weight_columns = [col for col in optimal_model1.columns if 'Optimal_Weights_Year' in col]
    
    # Standard Deviation
    optimal_model1['Risk (Standard Deviation)'] = optimal_model1[weight_columns].std(axis=1)
    
    # Average Weight
    optimal_model1['Average Weight'] = optimal_model1[weight_columns].mean(axis=1)
    
    # Sharpe Ratio
    optimal_model1['Sharpe Ratio'] = (optimal_model1['Average Weight'] - risk_free_rate) / optimal_model1['Risk (Standard Deviation)']
    
    # Max Drawdown
    max_drawdowns = []
    for _, row in optimal_model1[weight_columns].iterrows():
        peak = row.max()
        trough = row.min()
        max_drawdown = (trough - peak) / peak if peak != 0 else 0
        max_drawdowns.append(max_drawdown)
    optimal_model1['Max Drawdown'] = max_drawdowns

# Step 2: Create a table to display metrics
    table = html.Table([
        html.Thead(
            html.Tr([html.Th(col) for col in ['Asset', 'Risk (Standard Deviation)', 'Sharpe Ratio', 'Max Drawdown']])
        ),
        html.Tbody([
            html.Tr([
                html.Td(optimal_model1.iloc[i]['Asset']),
                html.Td(optimal_model1.iloc[i]['Sector']),
                html.Td(f"{optimal_model1.iloc[i]['Risk (Standard Deviation)']:.2f}"),
                html.Td(f"{optimal_model1.iloc[i]['Sharpe Ratio']:.2f}"),
                html.Td(f"{optimal_model1.iloc[i]['Max Drawdown']:.2f}")
            ]) for i in range(len(optimal_model1))
        ])
    ])
    
    # Step 3: Create a bar chart for risk (standard deviation)
    bar_chart = go.Figure()
    bar_chart.add_trace(go.Bar(
        x=optimal_model1['Asset'],
        y=optimal_model1['Risk (Standard Deviation)'],
        name='Risk (Standard Deviation)'
    ))
    bar_chart.update_layout(
        title='Sector Risk (Volatility)',
        xaxis_title='Asset',
        yaxis_title='Risk (Standard Deviation)',
        showlegend=False
    )
    
    # Step 4: Create a scatter plot for Sharpe Ratio vs. Risk
    scatter_plot = go.Figure()
    scatter_plot.add_trace(go.Scatter(
        x=optimal_model1['Risk (Standard Deviation)'],
        y=optimal_model1['Sharpe Ratio'],
        mode='markers+text',
        text=optimal_model1['Sector'],
        textposition='top center',
        marker=dict(size=10, color='blue')
    ))
    scatter_plot.update_layout(
        title='Risk vs. Reward (Sharpe Ratio)',
        xaxis_title='Risk (Standard Deviation)',
        yaxis_title='Sharpe Ratio',
        showlegend=False
    )


    return optimal_columns, optimal_model.to_dict('records'), total_table_columns, total_table.to_dict('records'), discrete_fig, cumulative_fig, portfolio_fig, Hdiscrete_fig, Hcumulative_fig, Hportfolio_fig, forecast_volatility, historical_volatility, {'data': [trace_simulated, trace_optimal], 'layout': layout},charts,table, bar_chart, scatter_plot


   

if __name__ == '__main__':
    app.run_server(debug=True)


# In[ ]:




