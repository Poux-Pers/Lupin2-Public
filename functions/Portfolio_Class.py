# -------------------- 
# IMPORTS
# -------------------- 

# ---- LIBRARIES -----
import os
import json
import keras
import datetime
import math as ma
import numpy as np
import pandas as pd

from tqdm.auto import tqdm

# ---- FUNCTIONS -----
from functions.BuySell_Models import BuySell
from functions.ML_Models import ML_Models


# -------------------- 
# PARAMETERS
# -------------------- 

with open(os.getcwd()+'\\parameters\\Parameters.json', 'r') as json_file:
    Parameters = json.load(json_file)


# -------------------- 
# MAIN
# -------------------- 

class Portfolio_class():

    def __init__(self, Parameters):
        self.path = os.getcwd() + '\\resources\\Portfolio.json'
        self.portfolio = {}
        self.parameters = Parameters
        self.mesh = Parameters['Mesh']
        self.trend_length = Parameters['trend_length']
        self.initial_investment = Parameters['initial_investment']
        self.ratio_of_gain_to_save = Parameters['ratio_of_gain_to_save']
        self.ratio_max_investment_per_value = Parameters['ratio_max_investment_per_value']
        self.BS_deals_print = Parameters['BS_deals_print']
        self.transaction_fees_percentage = Parameters['transaction_fees_percentage']
        self.companies_list = pd.read_csv(os.getcwd() +Parameters['Companies_list_path'])['Companies'].to_list()
        self.models_to_use = Parameters['Models_to_use']
        self.allow_loss = Parameters['allow_loss']
        self.NN_model_path = Parameters['NN_model_path']
        self.TCN_model_path = Parameters['TCN_model_path']
        self.ML_trend_length = Parameters['ML_trend_length']
        self.allow_loss_imputation_on_savings = Parameters['allow_loss_imputation_on_savings']

    def create(self):
        # This function will create a portfolio structure if it doesn't already exists
        if os.path.exists(self.path):
            with open(self.path) as json_file:
                self.portfolio = json.load(json_file)

        else:
            self.portfolio = {}
            self.portfolio['Money'] = {'Spending Money': self.initial_investment, 'Savings': 0, 'Transaction_fees_paid': 0}
            self.portfolio['Shares'] = {'Count': {}, 'Buy_Value': {}, 'Current_Value':{}}
            self.portfolio['Timestamp'] = datetime.datetime.now()

            for company in self.companies_list:
                self.portfolio['Shares']['Count'][company] = 0
                self.portfolio['Shares']['Buy_Value'][company] = 0
                self.portfolio['Shares']['Current_Value'][company] = 0

            with open(self.path, 'w') as f:
                json.dump(self.portfolio, f, default=str)

        return(self.portfolio)

    def reset(self):
        # This function will remove the existing portfolio to create it again withthe current structure
        os.remove(self.path)
        return(self.create())

    def update(self, new_portfolio):
        self.portfolio = new_portfolio
        with open(self.path, 'w') as f:
            json.dump(self.portfolio, f)

    def save(self):
        with open(self.path, 'w') as f:
            json.dump(self.portfolio, f)

    def value(self, portfolio, dataset):
        ROI = (portfolio['Money']['Spending Money'] + portfolio['Money']['Savings'] + sum([portfolio['Shares']['Count'][x]*int(dataset.loc[x,:].to_list()[-1]*1000)/1000 for x in self.companies_list]))/self.initial_investment

        Savings = {
            'Total value': ROI * self.initial_investment,
            'ROI': ROI,
            'Spending Money': portfolio['Money']['Spending Money'],
            'Savings': portfolio['Money']['Savings'],
            'Shares value':sum([portfolio['Shares']['Count'][x] * int(dataset.loc[x,:].to_list()[-1] * 1000) / 1000 for x in self.companies_list]),
            'Fees paid': int(portfolio['Money']['Transaction_fees_paid'] * 1000) / 1000
        }

        return(Savings)

    def simulation(self, dataset, Portfolio):
        # Initialisation
        Portfolio_history = []
        prediction_dict = {}        
        sum_prediction_deviation = {}
        sum_mean_deviation = {}
        list_R2 = []
        deals_history_dict = {}

        # Dict initialization
        for company in self.companies_list:
            sum_prediction_deviation[company] = 0
            sum_mean_deviation[company] = 0

        # Reduce the dataset only to the companies in the list
        if len(self.companies_list) > 1:
            dataset = dataset[dataset.index.isin(self.companies_list+['NASDAQ'])]
        else:
            dataset = dataset[dataset.index.isin(self.companies_list)]
        
        # Train the models if needed and load it
        if self.models_to_use['NN']:
            ML_Models(self.parameters).verify_train_NN()
            pre_loaded_model = keras.models.load_model(os.getcwd()+self.NN_model_path+str(self.ML_trend_length))
        elif self.models_to_use['TCN']:
            ML_Models(self.parameters).verify_train_TCN()
            pre_loaded_model = keras.models.load_model(os.getcwd()+self.TCN_model_path+str(self.ML_trend_length))

        # Visual feedback
        print('Portfolio simulation in progress')

        for day in tqdm(range(self.ML_trend_length, len(dataset.columns.to_list()))):
            # Reinitialization
            BS_dict_list = []
            prediction_dict_list = []
            deals_history_dict[day] = []

            # Reducing the dataset to the trend period studied and the companies in the companies list
            if self.models_to_use['NN'] or self.models_to_use['TCN']:
                small_dataset = dataset[dataset.columns[day-self.ML_trend_length:day]].fillna(0)
                str_model = 'pre_loaded_model'
            else:
                small_dataset = dataset[dataset.columns[day-self.trend_length:day]].fillna(0)
                str_model = ''

            # Accuracy measurment (R²)
            if day != self.ML_trend_length:
                for company in self.companies_list:
                    values_list = small_dataset.loc[company,:].to_list()
                    sum_prediction_deviation[company] += (prediction_dict[company] - values_list[-1])**2
                    sum_mean_deviation[company] += (np.mean(values_list) - values_list[-1])**2

            # Getting the list of the values to buy and their prediction
            for model in self.models_to_use:
                if self.models_to_use[model]:
                    BS_dict, prediction_dict, next_variation_dict = eval('BuySell(small_dataset, Parameters).'+model+'('+str_model+')')

                    # Add the results to the lists if we want to combine some models
                    BS_dict_list.append(BS_dict)
                    prediction_dict_list.append(prediction_dict)

            if len(BS_dict_list) > 1:
                BS_dict = {}
                prediction_dict = {}
                for company in self.companies_list:
                    values_list = small_dataset.loc[company,:].to_list()
                    
                    prediction_dict[company] = np.mean([pred_dict[company] for pred_dict in prediction_dict_list])
                    
                    # Condition to buy or Sell
                    if prediction_dict[company] > values_list[-1]:
                        BS_dict[company] = "Buy"

                    else:
                        BS_dict[company] = "Sell"

            else:
                BS_dict = BS_dict_list[0]
                prediction_dict = prediction_dict_list[0]

            # Sorting the trend dict in reverse to get the best relative trends first
            sorted_next_variation_dict = {k: v  for k, v in sorted(next_variation_dict.items(), key=lambda item: item[1] , reverse=True)}
            if 'NASDAQ' in sorted_next_variation_dict:
                sorted_next_variation_dict.pop('NASDAQ')

            # Register the date in the portfolio
            Portfolio['Timestamp'] = datetime.datetime.now()

            # Current value update/buy/sell all in a loop to evaluate the current price only once since you can't buy and sell on at the same time
            for company in sorted_next_variation_dict:
                symbol_current_value = int(small_dataset.loc[company,:].to_list()[-1] * 1000) / 1000
                Portfolio['Shares']['Current_Value'][company] = Portfolio['Shares']['Count'][company] * symbol_current_value

                # ----- BUYING ------
                if BS_dict[company] == 'Buy' and symbol_current_value > 0:
                    
                    # Buy as much as possible
                    nb_symbols_to_buy = int((Portfolio['Money']['Spending Money']*self.ratio_max_investment_per_value)//symbol_current_value)
                    buy_value = int(nb_symbols_to_buy*symbol_current_value*1000)/1000
                    
                    if nb_symbols_to_buy >=1:
                        # Portfolio update
                        Portfolio['Money']['Spending Money'] -= buy_value * (100 + self.transaction_fees_percentage) / 100
                        Portfolio['Shares']['Count'][company] += nb_symbols_to_buy
                        Portfolio['Shares']['Buy_Value'][company] += buy_value
                        Portfolio['Money']['Transaction_fees_paid'] += buy_value * self.transaction_fees_percentage / 100
                        
                        # Print deal details
                        deal_str = str(int(nb_symbols_to_buy))+' x '+company+' bought at '+str(symbol_current_value)+' (Total: '+str(buy_value)+'$ - Cash: '+str(int(Portfolio['Money']['Spending Money']*1000)/1000)+'$'
                        deals_history_dict[day].append(deal_str)
                        if self.BS_deals_print:
                            print(deal_str)

                # ----- SELLING ------
                nb_symbols_to_sell = ma.ceil(Portfolio['Shares']['Count'][company]*self.ratio_max_investment_per_value)

                if BS_dict[company] == 'Sell' and nb_symbols_to_sell >= 1 and symbol_current_value > 0:
                    # Sell all
                    
                    sell_value = int(nb_symbols_to_sell*symbol_current_value * 1000) / 1000
                    profit = sell_value - Portfolio['Shares']['Buy_Value'][company] * nb_symbols_to_sell / Portfolio['Shares']['Count'][company]
                    
                    # Portfolio update
                    if sell_value*(100-self.transaction_fees_percentage)/100 > Portfolio['Shares']['Buy_Value'][company] or self.allow_loss:
                        # If wanted, negatif profit will be charged on spending money
                        if profit > 0 or self.allow_loss_imputation_on_savings:
                            Portfolio['Money']['Spending Money'] += sell_value * ( 100 - self.transaction_fees_percentage) / 100 - profit * self.ratio_of_gain_to_save
                            Portfolio['Money']['Savings'] += profit * self.ratio_of_gain_to_save

                        else:
                            Portfolio['Money']['Spending Money'] += sell_value * ( 100 - self.transaction_fees_percentage) / 100

                        # Simulate the rest of the portfolio
                        Portfolio['Shares']['Buy_Value'][company] -= Portfolio['Shares']['Buy_Value'][company] * (nb_symbols_to_sell / Portfolio['Shares']['Count'][company])
                        Portfolio['Shares']['Count'][company] -= nb_symbols_to_sell
                        Portfolio['Money']['Transaction_fees_paid'] += sell_value * self.transaction_fees_percentage / 100

                        # Print deal details
                        deal_str = str(int(nb_symbols_to_sell))+' x '+company+' sold at '+str(symbol_current_value)+' (Profit: '+str(int(profit*1000)/1000)+'$ - Cash: '+str(int(Portfolio['Money']['Spending Money']*1000)/1000)+'$'
                        deals_history_dict[day].append(deal_str)
                        if self.BS_deals_print:
                            print(deal_str)
            
            # Portfolio audit trail creation plus enrichment
            Portfolio['Shares']['B/S_advice'] = BS_dict
            Portfolio['Shares']['Predicted_Variation'] = next_variation_dict

            Savings = self.value(Portfolio, small_dataset)
            Portfolio_history.append({**Portfolio, **Savings}) # python 3.8 solution
            #Portfolio_history.append(Portfolio | Savings) # ALERT only in python 3.9

            self.portfolio = Portfolio
    
        # NaN handling
        for x in sum_prediction_deviation:
            if ma.isnan(sum_prediction_deviation[x]):
                sum_prediction_deviation[x] = 0
            if ma.isnan(sum_mean_deviation[x]):
                sum_mean_deviation[x] = 0.00000000000000001

        R2 = int(np.mean([1 - sum_prediction_deviation[x]/(sum_mean_deviation[x]+0.00000000000000001) for x in sum_prediction_deviation])*1000)/1000 # temp fix to avoid /O
        
        return(Portfolio, Portfolio_history, R2, deals_history_dict)


if __name__ == "__main__":
    # Test lines, executed only when the file is executed as main
    Portfolio_class(Parameters).create()
    #Portfolio_class(Parameters).reset()