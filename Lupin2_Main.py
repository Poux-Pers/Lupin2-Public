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
# Fetching Data to complete nhistory
if Parameters['Update_Values'] == True:
    full_hist.update('max')

# Update Companies list if needed
companies_list = pd.read_csv(os.getcwd() +Parameters['Companies_list_path'])['Companies'].to_list()

# Using only the symbols of the company list
full_hist.hist = full_hist.hist[full_hist.hist['Company'].isin(companies_list)]

# ----- DATASET ------
# Reduce the dataset size to the period studied
dataset = full_hist.new_format(Parameters['study_length']).fillna(0)

if __name__ == "__main__":
    if Parameters['Optimization_run']:
        # Initialization
        trend_length_list = [2, 3, 4, 5, 8, 10, 12, 15]
        trend_length_relative_ROI_list = []
        trend_length_R2_list = []
        ratio_of_gain_to_save_list = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75]
        ratio_of_gain_to_save_relative_ROI_list = []        
        ratio_max_investment_per_value_list = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9]
        ratio_max_investment_per_value_relative_ROI_list = []
        initial_investment_list = [5*10**i for i in range(2,6)]
        initial_investment_relative_ROI_list = []

        fig, axs = plt.subplots(5)
        fig.suptitle('Optimization run')        

        # Optimizatrion run will study different parameters separately to save time
        # Trend length
        for trend_length in tqdm(trend_length_list):
            Parameters['trend_length'] = trend_length
            dataset = full_hist.new_format(Parameters['study_length'])
            Portfolio = Portfolio_class(Parameters)
            Portfolio.reset()
            Portfolio_history_list = []
            last_portfolio, Portfolio_history_list, R2, deals_history_dict = Portfolio.simulation(dataset, Portfolio.portfolio)

            trend_length_relative_ROI_list.append(int((Portfolio_history_list[-1]['ROI']/(sum(dataset[dataset.columns[-1]])/sum(dataset[dataset.columns[0]])))*1000)/1000)
            trend_length_R2_list.append(R2)

        # Sub plot 
        axs[0].plot(trend_length_list, trend_length_relative_ROI_list, label='Trend length - Relative ROI')
        axs[1].plot(trend_length_list, trend_length_R2_list, 'r', label='Trend length - R2')

        # Putting back the original parameter
        Parameters['trend_length'] = 5
        dataset = full_hist.new_format(Parameters['study_length'])

        # 
        for ratio_of_gain_to_save in tqdm(ratio_of_gain_to_save_list):
            Parameters['ratio_of_gain_to_save'] = ratio_of_gain_to_save
            Portfolio = Portfolio_class(Parameters)
            Portfolio.reset()
            Portfolio_history_list = []
            last_portfolio, Portfolio_history_list, R2, deals_history_dict = Portfolio.simulation(dataset, Portfolio.portfolio)

            ratio_of_gain_to_save_relative_ROI_list.append(int((Portfolio_history_list[-1]['ROI']/(sum(dataset[dataset.columns[-1]])/sum(dataset[dataset.columns[0]])))*1000)/1000)

        # Sub plot
        axs[2].plot(ratio_of_gain_to_save_list, ratio_of_gain_to_save_relative_ROI_list, label='ratio_of_gain_to_save - Relative ROI')

        # Putting back the original parameter
        Parameters['ratio_of_gain_to_save'] = 0.1

        #
        for ratio_max_investment_per_value in tqdm(ratio_max_investment_per_value_list):
            Parameters['ratio_max_investment_per_value'] = ratio_max_investment_per_value
            Portfolio = Portfolio_class(Parameters)
            Portfolio.reset()
            Portfolio_history_list = []
            last_portfolio, Portfolio_history_list, R2, deals_history_dict = Portfolio.simulation(dataset, Portfolio.portfolio)

            ratio_max_investment_per_value_relative_ROI_list.append(int((Portfolio_history_list[-1]['ROI']/(sum(dataset[dataset.columns[-1]])/sum(dataset[dataset.columns[0]])))*1000)/1000)

        # Sub plot
        axs[3].plot(ratio_max_investment_per_value_list, ratio_max_investment_per_value_relative_ROI_list, label='ratio_max_investment_per_value - Relative ROI')

        # Putting back the original parameter
        Parameters['ratio_max_investment_per_value'] = 0.25

        #
        for initial_investment in tqdm(initial_investment_list):
            Parameters['initial_investment'] = initial_investment
            Portfolio = Portfolio_class(Parameters)
            Portfolio.reset()
            Portfolio_history_list = []
            last_portfolio, Portfolio_history_list, R2, deals_history_dict = Portfolio.simulation(dataset, Portfolio.portfolio)

            initial_investment_relative_ROI_list.append(int((Portfolio_history_list[-1]['ROI']/(sum(dataset[dataset.columns[-1]])/sum(dataset[dataset.columns[0]])))*1000)/1000)

        # Sub plot
        axs[4].plot(initial_investment_list, initial_investment_relative_ROI_list, label='initial_investment - Relative ROI')

        # Putting back the original parameter
        Parameters['initial_investment'] = 5000

        plt.show()
        


        initial_investment

    else:
        # ------- RUN --------
        Portfolio = Portfolio_class(Parameters)
        Portfolio.reset()
        last_portfolio, Portfolio_history_list, R2, deals_history_dict = Portfolio.simulation(dataset, Portfolio.portfolio)

        # ------- PLOT -------
        if Parameters['Plot_Graph']:
            Plot(Parameters).portfolio_history(Portfolio_history_list, dataset, deals_history_dict)

        # Sending Portfolio to ES
        i = 0
        if Parameters['Send_to_ES']:
            Elasticsearch_class(Parameters).upload_hist()
            Elasticsearch_class(Parameters).reset_portfolio_index()
            for portfolio in tqdm(Portfolio_history_list):
                i += 1
                Elasticsearch_class(Parameters).upload_dict(portfolio, i)
        
        # ------- KPI --------
        # Print last Savings
        print(Portfolio_history_list[-1])

        # Print relative ROI        
        print('----- Relative ROI: '+str(int((Portfolio_history_list[-1]['ROI']/(sum(dataset[dataset.columns[-1]])/sum(dataset[dataset.columns[0]])))*1000)/1000)+' -----')
        print('----- Average R²: '+str(R2)+' -----')

        # Last show
        if Parameters['Plot_Graph']:
            plt.show()

# -------------------- 
# COMMENTS
# -------------------- 
# TODO
# Optimisation des paremètre avec affichage graphique
# Deals audit trail - Best deals, worst deals
# Inflation - Bank % - Holding shares cost
# If you ever do a prod file for 1d actualization with a dashboard, have a list of the B/S functions and their profitability over the preivous x days
# Comparainson to rating agencies
# save B/S dict 
# Autres fonctions B/S
# - Trend identification
# - TCN (would need to differentiate trend_length and the stat))set size)
# - Buy for 3 increase in a row, sell for 3 loss in a row
# - Elliott waves rules
# - ARMA
# - GARCH
# https://commodity.com/technical-analysis/
# Quantdom project
# Identification de valeurs refuge
# trust indice based on the historical R² of the company (mainly for the full)
# predict multiple companiesd at the same time
# Tidy the parameters file

# Further TODO
# Place companies on the map: color countries by medium company price/number of companies
# Include volume
# (Further dev) Dashboard d'évolution des fonds avec une simulation 1min = 1 sec (plotly ?) 
# Calculate trend compared to industry trend
# Save in the trend the best model to describe the company stock variation