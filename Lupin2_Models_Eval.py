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
import time
import json
import datetime
import math as ma
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm.auto import tqdm

# ---- FUNCTIONS -----
from functions.Portfolio_Class import Portfolio_class
from functions.BuySell_Models import BuySell
from functions.Dataset_Class import Dataset
from functions.ES_interaction import Elasticsearch_class
from functions.Plot_Class import Plot


# -------------------- 
# PARAMETERS
# -------------------- 

with open(os.getcwd()+'\\parameters\\Parameters.json', 'r') as json_file:
    Parameters = json.load(json_file)


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
# Fetching Data to complete history
if Parameters['Update_Values'] == True:
    full_hist.update('7d')

# Update Companies list if needed
companies_list = pd.read_csv(os.getcwd() +Parameters['Companies_list_path'])['Companies'].to_list()

# Using only the symbols of the company list
full_hist.hist[full_hist.hist['Company'].isin(companies_list)]

# ----- DATASET ------
# Reduce the dataset size to the period studied
dataset = full_hist.new_format(Parameters['study_length'])

# ------- RUN --------
for 
Portfolio = Portfolio_class(Parameters)
Portfolio.reset()
last_portfolio, Portfolio_history_list, R2 = Portfolio.simulation(dataset, Portfolio.portfolio)

print('----- Average R²: '+str(R2)+' -----')