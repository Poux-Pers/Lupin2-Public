# -------------------- 
# INTRODUCTION
# -------------------- 
# Author: Poux Louis
# Description: TODO
# Python version: 3.9


# -------------------- 
# IMPORTS
# -------------------- 

# ---- LIBRARIES -----
import os
import sys
import time
import json
import keras
import warnings
import datetime
import math as ma
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm.auto import tqdm
from keras.models import Sequential
from keras.layers import Dense

# ---- FUNCTIONS -----
from functions.Portfolio_Class import Portfolio_class
from functions.BuySell_Models import BuySell
from functions.Dataset_Class import Dataset
from functions.ES_interaction import Elasticsearch_class
from functions.Plot_Class import Plot

# Desactivate warnings
warnings.filterwarnings("ignore")

# -------------------- 
# PARAMETERS
# -------------------- 

with open(os.getcwd()+'\\parameters\\Parameters.json', 'r') as json_file:
    Parameters = json.load(json_file)

# Opening the log file
sys.stdout = open(os.getcwd()+'\\models\\logs\\Model_Eval_'+str(Parameters['trend_length'])+'-'+str(Parameters['ML_trend_length'])+'-'+str(Parameters['study_length'])+'.txt', 'w')


# -------------------- 
# MAIN
# -------------------- 

# ------- LOAD -------
# Loading history csv
full_hist = Dataset(Parameters)
full_hist.load()

# Setting back the timestamp format
if full_hist.date_name != '1m':
    full_hist.hist[full_hist.date_name] = full_hist.hist[full_hist.date_name].astype('datetime64[ns]')
else:
    full_hist.hist[full_hist.date_name] = full_hist.hist[full_hist.date_name].dt.floor('min')

# Load the porfolio
Portfolio = Portfolio_class(Parameters).reset()

# ------ UPDATE ------
# Update Companies list if needed
companies_list = pd.read_csv(os.getcwd() +Parameters['Companies_list_path'])['Companies'].to_list()

# Using only the symbols of the company list
full_hist.hist[full_hist.hist['Company'].isin(companies_list)]

# ----- DATASET ------
# Reduce the dataset size to the period studied
dataset = full_hist.new_format(Parameters['study_length']).fillna(0)

# ------- RUN -------- 
# Print parameters
print('##### Trend length: '+str(Parameters['trend_length'])+' #####')
print('##### ML Trend length: '+str(Parameters['ML_trend_length'])+' #####')
print('##### Study length: '+str(Parameters['study_length'])+' #####')

# Trend
# Reset other models
for model in Parameters['Models_to_use']:
    Parameters['Models_to_use'][model] = False

Parameters['Models_to_use']['trend'] = True
Portfolio = Portfolio_class(Parameters)
Portfolio.reset()
last_portfolio, Portfolio_history_list, R2, deals_history_dict = Portfolio.simulation(dataset, Portfolio.portfolio)

print('----- Trend -----')
print('----- Relative ROI: '+str(int((Portfolio_history_list[-1]['ROI']/(sum(dataset[dataset.columns[-1]])/sum(dataset[dataset.columns[0]])))*1000)/1000)+' -----')
print('----- Average R2: '+str(R2)+' -----')

# Graduated wheights
# Reset other models
for model in Parameters['Models_to_use']:
    Parameters['Models_to_use'][model] = False

Parameters['Models_to_use']['graduated_weights'] = True
Portfolio = Portfolio_class(Parameters)
Portfolio.reset()
last_portfolio, Portfolio_history_list, R2, deals_history_dict = Portfolio.simulation(dataset, Portfolio.portfolio)

print('----- Graduated wheights -----')
print('----- Relative ROI: '+str(int((Portfolio_history_list[-1]['ROI']/(sum(dataset[dataset.columns[-1]])/sum(dataset[dataset.columns[0]])))*1000)/1000)+' -----')
print('----- Average R2: '+str(R2)+' -----')

# Trend + Graduated wheights
# Reset other models
for model in Parameters['Models_to_use']:
    Parameters['Models_to_use'][model] = False

Parameters['Models_to_use']['trend'] = True
Parameters['Models_to_use']['graduated_weights'] = True
Portfolio = Portfolio_class(Parameters)
Portfolio.reset()
last_portfolio, Portfolio_history_list, R2, deals_history_dict = Portfolio.simulation(dataset, Portfolio.portfolio)

print('----- Trend + Graduated wheights -----')
print('----- Relative ROI: '+str(int((Portfolio_history_list[-1]['ROI']/(sum(dataset[dataset.columns[-1]])/sum(dataset[dataset.columns[0]])))*1000)/1000)+' -----')
print('----- Average R2: '+str(R2)+' -----')

# Random walks
# Reset other models
for model in Parameters['Models_to_use']:
    Parameters['Models_to_use'][model] = False

Parameters['Models_to_use']['random_walks'] = True
Portfolio = Portfolio_class(Parameters)
Portfolio.reset()
last_portfolio, Portfolio_history_list, R2, deals_history_dict = Portfolio.simulation(dataset, Portfolio.portfolio)

print('----- Random walks -----')
print('----- Relative ROI: '+str(int((Portfolio_history_list[-1]['ROI']/(sum(dataset[dataset.columns[-1]])/sum(dataset[dataset.columns[0]])))*1000)/1000)+' -----')
print('----- Average R2: '+str(R2)+' -----')

# 3 in a row
# Reset other models
for model in Parameters['Models_to_use']:
    Parameters['Models_to_use'][model] = False

Parameters['Models_to_use']['random_walks'] = True
Portfolio = Portfolio_class(Parameters)
Portfolio.reset()
last_portfolio, Portfolio_history_list, R2, deals_history_dict = Portfolio.simulation(dataset, Portfolio.portfolio)

print('----- 3 in a row -----')
print('----- Relative ROI: '+str(int((Portfolio_history_list[-1]['ROI']/(sum(dataset[dataset.columns[-1]])/sum(dataset[dataset.columns[0]])))*1000)/1000)+' -----')
print('----- Average R2: '+str(R2)+' -----')

# ARIMA
# Reset other models
for model in Parameters['Models_to_use']:
    Parameters['Models_to_use'][model] = False

Parameters['Models_to_use']['ARIMA'] = True
Portfolio = Portfolio_class(Parameters)
Portfolio.reset()
last_portfolio, Portfolio_history_list, R2, deals_history_dict = Portfolio.simulation(dataset, Portfolio.portfolio)

print('----- ARIMA -----')
print('----- Relative ROI: '+str(int((Portfolio_history_list[-1]['ROI']/(sum(dataset[dataset.columns[-1]])/sum(dataset[dataset.columns[0]])))*1000)/1000)+' -----')
print('----- Average R2: '+str(R2)+' -----')

# NN
# Reset other models
for model in Parameters['Models_to_use']:
    Parameters['Models_to_use'][model] = False

Parameters['Models_to_use']['NN'] = True
Portfolio = Portfolio_class(Parameters)
Portfolio.reset()
last_portfolio, Portfolio_history_list, R2, deals_history_dict = Portfolio.simulation(dataset, Portfolio.portfolio)

print('----- NN -----')
print('----- Relative ROI: '+str(int((Portfolio_history_list[-1]['ROI']/(sum(dataset[dataset.columns[-1]])/sum(dataset[dataset.columns[0]])))*1000)/1000)+' -----')
print('----- Average R2: '+str(R2)+' -----')

# TCN
# Reset other models
for model in Parameters['Models_to_use']:
    Parameters['Models_to_use'][model] = False

Parameters['Models_to_use']['TCN'] = True
Portfolio = Portfolio_class(Parameters)
Portfolio.reset()
last_portfolio, Portfolio_history_list, R2, deals_history_dict = Portfolio.simulation(dataset, Portfolio.portfolio)

print('----- TCN -----')
print('----- Relative ROI: '+str(int((Portfolio_history_list[-1]['ROI']/(sum(dataset[dataset.columns[-1]])/sum(dataset[dataset.columns[0]])))*1000)/1000)+' -----')
print('----- Average R2: '+str(R2)+' -----')

# LSTM won't be trated in this file as it would need too much computational power with this configuration

# Close the log file
sys.stdout.close()