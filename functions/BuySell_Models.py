# -------------------- 
# IMPORTS
# -------------------- 

# ---- LIBRARIES -----
import os
import json
import keras
import random
import datetime
import numpy as np
import pandas as pd

from keras.models import Sequential
from keras.layers import Dense

# ---- FUNCTIONS -----
from functions.Dataset_Class import Dataset
from functions.ML_Models import ML_Models

# -------------------- 
# PARAMETERS
# -------------------- 

with open(os.getcwd()+'\\parameters\\Parameters.json', 'r') as json_file:
    Parameters = json.load(json_file)


# -------------------- 
# LOCAL PARAMETERS
# -------------------- 
add_specific_rules = True
sell_after_stagnation = {'y/n': True, 'nb_days': 4}
sell_after_high_rise_ratio = 2
sell_after_high_loss_ratio = 1.5


# -------------------- 
# MAIN
# -------------------- 

class BuySell():
    
    def __init__(self, short_hist_dataframe, Parameters):
        self.df = short_hist_dataframe.fillna(0)
        self.mesh = Parameters['Mesh']
        self.trend_length = Parameters['trend_length']
        self.ML_trend_length = Parameters['ML_trend_length']
        self.initial_investment = Parameters['initial_investment']
        self.ratio_of_gain_to_save = Parameters['ratio_of_gain_to_save']
        self.ratio_max_investment_per_value = Parameters['ratio_max_investment_per_value']
        self.BS_deals_print = Parameters['BS_deals_print']
        self.transaction_fees_percentage = Parameters['transaction_fees_percentage']
        self.companies_list = pd.read_csv(os.getcwd() +Parameters['Companies_list_path'])['Companies'].to_list()
        self.models_to_use = Parameters['Models_to_use']
        self.ML_dataset_parameters_path = Parameters['ML_dataset_parameters_path']
        self.NN_model_path = Parameters['NN_model_path']
        self.TCN_model_path = Parameters['TCN_model_path']

    def trend(self):
        # Creation of the dictionary to calculate trends
        next_variation_dict = {}
        prediction_dict = {}

        # Trends dictionary filling
        for company in self.df.index:
            values_list = self.df.loc[company,:].to_list()

            # Linear Regression
            list_x = np.arange(0,len(values_list))
            values_array = np.array(values_list)
            a,b = np.polyfit(list_x,values_array,1)

            # Calculate next value for accuracy 
            prediction_dict[company] = len(values_list) * a + b

            if values_list[-1] > 0:
                next_variation_dict[company] = prediction_dict[company] / values_list[-1]
            else:
                next_variation_dict[company] = 1

        # Creation of the dictionary to advise Buy or Sell
        BS_dict = {}
        if len(self.companies_list) > 1:
            avg_next_variation = np.mean([next_variation_dict[x] for x in next_variation_dict])
        else:
            avg_next_variation = 1

        # Buy or Sell dictionary filling
        for company in self.companies_list:
            # Condition to buy or Sell
            if next_variation_dict[company] > avg_next_variation:
                BS_dict[company] = "Buy"

            elif next_variation_dict[company] < avg_next_variation:
                BS_dict[company] = "Sell"

            else:
                BS_dict[company] = "Hold"

            # Specific rules
            if add_specific_rules:
                # Sell after a high rise
                if values_list[-1] > sell_after_high_rise_ratio * values_list[-2]:
                    BS_dict[company] = "Sell"

                # Sell after a high loss
                if values_list[-1] < values_list[-2]/sell_after_high_loss_ratio:
                    BS_dict[company] = "Sell"

                # Stagnation
                if sell_after_stagnation['y/n'] and sum(values_list[-sell_after_stagnation['nb_days']-1:-1])/sell_after_stagnation['nb_days']+1 == values_list[-1]:
                    BS_dict[company] = "Sell"
        
        return(BS_dict, prediction_dict, next_variation_dict)

    def graduated_weights(self):
        # Creation of the dictionary to calculate next values
        prediction_dict = {}
        next_variation_dict = {}

        # Creation of the dictionary to advise Buy or Sell
        BS_dict = {}

        # Prediction dictionary filling
        for company in self.df.index:
            values_list = self.df.loc[company,:].to_list()

            # calculate next value for accuracy 
            prediction_dict[company] = sum([values_list[x]/2**(len(values_list)-x) for x in range(len(values_list))])/sum([1/2**(len(values_list)-x) for x in range(len(values_list))])
            if values_list[-1] > 0:
                next_variation_dict[company] = prediction_dict[company] / values_list[-1]
            else:
                next_variation_dict[company] = 1

        # Creation of the dictionary to advise Buy or Sell
        if len(self.companies_list) > 1:
            avg_next_variation = np.mean([next_variation_dict[x] for x in next_variation_dict])
        else:
            avg_next_variation = 1   
        
        # Buy or Sell dictionary filling
        for company in self.companies_list:
            # Condition to buy or Sell            
            if next_variation_dict[company] > avg_next_variation:
                BS_dict[company] = "Buy"

            elif next_variation_dict[company] < avg_next_variation:
                BS_dict[company] = "Sell"

            else:
                BS_dict[company] = "Hold"

            # Specific rules
            if add_specific_rules:
                # Sell after a high rise
                if values_list[-1] > sell_after_high_rise_ratio * values_list[-2]:
                    BS_dict[company] = "Sell"

                # Sell after a high loss
                if values_list[-1] < values_list[-2]/sell_after_high_loss_ratio:
                    BS_dict[company] = "Sell"

                # Stagnation
                if sell_after_stagnation['y/n'] and sum(values_list[-sell_after_stagnation['nb_days']-1:-1])/sell_after_stagnation['nb_days']+1 == values_list[-1]:
                    BS_dict[company] = "Sell"
        
        return(BS_dict, prediction_dict, next_variation_dict)

    def NN(self, model):
        # Creation of the dictionary to calculate next values
        prediction_dict = {}
        next_variation_dict = {}

        # Creation of the dictionary to advise Buy or Sell
        BS_dict = {}
        
        # Get max value by list
        maxes_list = self.df.max(axis=1).multiply(2)

        # Batch prediction
        predict_list = model.predict(self.df.div(maxes_list.to_list(), axis=0))

        # Flatten the prediction
        flat_predict_list = [x[0] for x in predict_list]

        # Multiply the prediction result
        prediction_dict = maxes_list.multiply(flat_predict_list, axis=0)

        # Last values
        last_values = self.df.loc[:,self.df.columns.to_list()[-1]]

        # Calculate the predicted variation 
        next_variation_df = prediction_dict.div(last_values.to_list(), axis=0).fillna(1)

        # Creation of the dictionary to advise Buy or Sell
        if len(self.companies_list) > 1:
            avg_next_variation = np.mean(next_variation_df)
        else:
            avg_next_variation = 1
        
        # Transform df in dict
        next_variation_dict = next_variation_df.to_dict()
        
        # Buy or Sell dictionary filling
        for company in self.companies_list:
            # Condition to buy or Sell
            if next_variation_dict[company] > avg_next_variation:
                BS_dict[company] = "Buy"

            elif next_variation_dict[company] < avg_next_variation:
                BS_dict[company] = "Sell"

            else:
                BS_dict[company] = "Hold"
                
            # Specific rules
            if add_specific_rules:
                # Sell after a high rise
                if prediction_dict[company] > sell_after_high_rise_ratio * last_values.loc[company]:
                    BS_dict[company] = "Sell"

                # Sell after a high loss
                if prediction_dict[company] < last_values.loc[company] / sell_after_high_loss_ratio:
                    BS_dict[company] = "Sell"

        return(BS_dict, prediction_dict, next_variation_dict)

    def random_walks(self):
        # Creation of the dictionary to calculate next values
        prediction_dict = {}
        next_variation_dict = {}

        # Creation of the dictionary to advise Buy or Sell
        BS_dict = {}

        # Prediction dictionary filling
        for company in self.df.index:
            values_list = self.df.loc[company,:].to_list()

            # calculate next value
            variance = np.var(values_list)
            mean = np.mean(values_list)

            prediction_dict[company] = (random.choice([-1, 0, 1])) * variance + mean
            if values_list[-1] > 0:
                next_variation_dict[company] = prediction_dict[company] / values_list[-1]
            else:
                next_variation_dict[company] = 1

        # Creation of the dictionary to advise Buy or Sell
        if len(self.companies_list) > 1:
            avg_next_variation = np.mean([next_variation_dict[x] for x in next_variation_dict])
        else:
            avg_next_variation = 1        
        
        # Buy or Sell dictionary filling
        for company in self.companies_list:
            # Condition to buy or Sell            
            if next_variation_dict[company] > avg_next_variation:
                BS_dict[company] = "Buy"

            elif next_variation_dict[company] < avg_next_variation:
                BS_dict[company] = "Sell"

            else:
                BS_dict[company] = "Hold"

            # Specific rules
            if add_specific_rules:
                # Sell after a high rise
                if values_list[-1] > sell_after_high_rise_ratio * values_list[-2]:
                    BS_dict[company] = "Sell"

                # Sell after a high loss
                if values_list[-1] < values_list[-2]/sell_after_high_loss_ratio:
                    BS_dict[company] = "Sell"

                # Stagnation
                if sell_after_stagnation['y/n'] and sum(values_list[-sell_after_stagnation['nb_days']-1:-1])/sell_after_stagnation['nb_days']+1 == values_list[-1]:
                    BS_dict[company] = "Sell"
        
        return(BS_dict, prediction_dict, next_variation_dict)

    def three_in_row(self):
        # Creation of the dictionary to calculate next values
        prediction_dict = {}
        next_variation_dict = {}

        # Creation of the dictionary to advise Buy or Sell
        BS_dict = {}

        # Prediction dictionary filling
        for company in self.df.index:
            values_list = self.df.loc[company,:].to_list()

            # calculate next value
            variance = np.var(values_list[-3:])

            # Decide the next value based on the last 3 values
            if values_list[-3] < values_list[-2] < values_list[-1]:
                prediction_dict[company] = values_list[-1] + variance
            elif values_list[-3] > values_list[-2] > values_list[-1]:
                prediction_dict[company] = values_list[-1] - variance
            else:
                prediction_dict[company] = values_list[-1]

            if values_list[-1] > 0:
                next_variation_dict[company] = prediction_dict[company] / values_list[-1]
            else:
                next_variation_dict[company] = 1

        # Creation of the dictionary to advise Buy or Sell
        if len(self.companies_list) > 1:
            avg_next_variation = np.mean([next_variation_dict[x] for x in next_variation_dict])
        else:
            avg_next_variation = 1        
        
        # Buy or Sell dictionary filling
        for company in self.companies_list:
            # Condition to buy or Sell            
            if next_variation_dict[company] > avg_next_variation:
                BS_dict[company] = "Buy"

            elif next_variation_dict[company] < avg_next_variation:
                BS_dict[company] = "Sell"

            else:
                BS_dict[company] = "Hold"

            # Specific rules
            if add_specific_rules:
                # Sell after a high rise
                if values_list[-1] > sell_after_high_rise_ratio * values_list[-2]:
                    BS_dict[company] = "Sell"

                # Sell after a high loss
                if values_list[-1] < values_list[-2]/sell_after_high_loss_ratio:
                    BS_dict[company] = "Sell"

                # Stagnation
                if sell_after_stagnation['y/n'] and sum(values_list[-sell_after_stagnation['nb_days']-1:-1])/sell_after_stagnation['nb_days']+1 == values_list[-1]:
                    BS_dict[company] = "Sell"
        
        return(BS_dict, prediction_dict, next_variation_dict)

    def TCN(self, model):
        # Same as NN
        # Creation of the dictionary to calculate next values
        prediction_dict = {}
        next_variation_dict = {}

        # Creation of the dictionary to advise Buy or Sell
        BS_dict = {}

        # Get max value by list
        maxes_list = self.df.max(axis=1).multiply(2)

        # Batch prediction
        predict_list = model.predict(self.df.div(maxes_list.to_list(), axis=0))

        # Flatten the prediction
        flat_predict_list = [x[0] for x in predict_list]

        # Multiply the prediction result
        prediction_dict = maxes_list.multiply(flat_predict_list, axis=0)

        # Last values
        last_values = self.df.loc[:,self.df.columns.to_list()[-1]]

        # Calculate the predicted variation 
        next_variation_df = prediction_dict.div(last_values.to_list(), axis=0).fillna(0)

        # Creation of the dictionary to advise Buy or Sell
        if len(self.companies_list) > 1:
            avg_next_variation = np.mean(next_variation_df)
        else:
            avg_next_variation = 1
        
        # Transform df in dict
        next_variation_dict = next_variation_df.to_dict()

        # Buy or Sell dictionary filling
        for company in self.companies_list:
            # Condition to buy or Sell
            if next_variation_dict[company] > avg_next_variation:
                BS_dict[company] = "Buy"

            elif next_variation_dict[company] < avg_next_variation:
                BS_dict[company] = "Sell"

            else:
                BS_dict[company] = "Hold"
                
            # Specific rules
            if add_specific_rules:
                # Sell after a high rise
                if prediction_dict[company] > sell_after_high_rise_ratio * last_values.loc[company]:
                    BS_dict[company] = "Sell"

                # Sell after a high loss
                if prediction_dict[company] < last_values.loc[company] / sell_after_high_loss_ratio:
                    BS_dict[company] = "Sell"
        
        return(BS_dict, prediction_dict, next_variation_dict)


if __name__ == "__main__":
    study_length = 10

    # Loading history csv
    full_hist = pd.read_csv(os.getcwd() + '\\resources\\full_NASDAQ_history.csv', usecols=['Close', 'Company', 'Date', 'Dividends', 'High', 'Low', 'Open', 'Stock Splits', 'Volume'])
    # Setting back the timestamp format
    full_hist['Date'] = full_hist['Date'].astype('datetime64[ns]')
    
    dataset = Dataset(full_hist).new_format(study_length)

    BS_dict, Trend_dict = BuySell(dataset, Parameters).trend()
    print(BS_dict, Trend_dict)

