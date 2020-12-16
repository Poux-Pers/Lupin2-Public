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

# ---- PORTFOLIO  ----
#Portfolio_class().simulation(dataset, Portfolio, trend_length, companies_list, initial_investment, ratio_of_gain_to_save, ratio_max_investment_per_value, BS_deals_print)


if __name__ == "__main__":
    if Parameters['Optimization_run']:
        Savings= {}
        for trend_length in tqdm([5, 10, 15]):
            Savings[trend_length] = {}
            for ratio_of_gain_to_save in tqdm([0.1, 0.25, 0.5]):
                Savings[trend_length][ratio_of_gain_to_save] = {}
                for ratio_max_investment_per_value in tqdm([0.01, 0.1, 0.25, 0.5]):
                    Portfolio = Portfolio_class(Parameters).reset(Parameters['initial_investment'])
                    Portfolio, Portfolio_history = Portfolio_class().simulation(dataset, Portfolio)

                    Savings[trend_length][ratio_of_gain_to_save][ratio_max_investment_per_value] = Portfolio_class(Parameters).value(Portfolio, dataset)
        
            with open(os.getcwd() + '\\resources\\Savings.json', 'w') as f:
                json.dump(Savings, f)

    else:
        # Initialisation
        list_Total_Value = []
        list_ROI = []
        list_Cash = []
        list_Savings = []
        list_Shares_value = []
        Portfolio_history = []

        Portfolio = Portfolio_class(Parameters).reset()
        Portfolio, Portfolio_history = Portfolio_class(Parameters).simulation(dataset, Portfolio)

        # List construction
        for Savings in Portfolio_history:
            list_Total_Value.append(Savings['Total value'])
            list_ROI.append(Savings['ROI'])
            list_Cash.append(Savings['Spending Money'])
            list_Savings.append(Savings['Savings'])
            list_Shares_value.append(Savings['Shares value'])            
            
        # Plot savings
        fig, axs = plt.subplots(3)
        fig.suptitle('Savings over time')

        x_list = np.arange(0,len(dataset.columns.to_list()),1)

        # Sub plot for the sum of the symbols
        axs[0].plot(x_list, dataset.iloc[-1].to_list(), 'purple', label='NASDAQ')
        
        axs[1].plot(x_list[:-Parameters['trend_length']], list_Cash, 'r', label='Cash')
        axs[1].plot(x_list[:-Parameters['trend_length']], list_Savings, 'b', label='Bank')
        axs[1].plot(x_list[:-Parameters['trend_length']], list_Total_Value, 'black', label='Total')
        axs[1].plot(x_list[:-Parameters['trend_length']], list_Shares_value, 'orange', label='Shares Value')

        # Sub plot for ROI
        axs[2].plot(x_list[:-Parameters['trend_length']], list_ROI)
        plt.show()

        print(Portfolio_history)

# -------------------- 
# COMMENTS
# -------------------- 
# TODO
# Optimisation des paremètre avec affichage graphique
# Autres fonctions B/S
# Deals audit trail - Best deals, worst deals
# Selling staging values after 5 days or 2 days
# Combining different B/S algo
# Inflation - Bank %
# Holding shares cost
# Give parameters in BS functions like selling after high increase
# If you ever do a prod file for 1d actualization with a dashboard, have a list of the B/S functions and their profitability over the preivous x days
# Comparainson to rating agencies

# Further TODO
# Place companies on the map: color countries by medium company price/number of companies
# Include volume
# Inclue Companies info
# (Further dev) Dashboard d'évolution des fonds avec une simulation 1min = 1 sec (plotly ?) 
# Calculate trend compared to industry trend