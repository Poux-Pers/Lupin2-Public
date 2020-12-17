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
from functions.BuySell_Trend import BuySellTrend
from functions.Dataset_Class import Dataset
from functions.ES_interaction import Elasticsearch_class


# -------------------- 
# PARAMETERS
# -------------------- 

with open(os.getcwd()+'\\parameters\\Parameters.json', 'r') as json_file:
    Parameters = json.load(json_file)

# Overwrite mesh
Parameters['Mesh'] = '1m'


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

# Create the porfolio
Portfolio = Portfolio_class(Parameters)
Portfolio.create()

# ------ UPDATE ------
# Fetching Data to complete history
if Parameters['Update_Values'] == True:
    full_hist.update('7d')

# Update Companies list if needed
companies_list = full_hist.companies_list

# Using only the symbols of the company list
full_hist.hist[full_hist.hist['Company'].isin(companies_list)]

# ----- DATASET ------
# Reduce the dataset size to the period studied
dataset = full_hist.new_format(Parameters['study_length'])

# Loop initialisation
t0 = time.time()
i = 0


while True:
    print(str(int(time.time()-t0)) + ' seconds to perform a loop')
    # Reset timer to measure the loop time
    t0 = time.time()

    full_hist.update('10min')
    Elasticsearch_class(Parameters).upload_df(full_hist.hist)

    # Reduce the dataset size to the period studied
    dataset = full_hist.new_format(Parameters['trend_length'] + 1)
    
    # Portfolio calculation
    Portfolio.simulation(dataset, Portfolio.portfolio)
    #Portfolio.save()
    
    # Update the Json file - indexed since the begining
    i += 1
    Elasticsearch_class(Parameters).upload_dict(Portfolio.portfolio, i)