# -------------------- 
# IMPORTS
# -------------------- 

# ---- LIBRARIES -----
import os
import json
import keras
import random
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
sell_after_high_rise_ratio = 1.5
sell_after_high_loss_ratio = 1.5


# -------------------- 
# MAIN
# -------------------- 

class BuySell():
    
    def __init__(self, short_hist_dataframe, Parameters):
        self.df = short_hist_dataframe
        self.mesh = Parameters['Mesh']
        self.trend_length = Parameters['trend_length']
        self.initial_investment = Parameters['initial_investment']
        self.ratio_of_gain_to_save = Parameters['ratio_of_gain_to_save']
        self.ratio_max_investment_per_value = Parameters['ratio_max_investment_per_value']
        self.BS_deals_print = Parameters['BS_deals_print']
        self.transaction_fees_percentage = Parameters['transaction_fees_percentage']
        self.companies_list = pd.read_csv(os.getcwd() +Parameters['Companies_list_path'])['Companies'].to_list()
        self.models_to_use = Parameters['Models_to_use']
        self.ML_dataset_parameters_path = Parameters['ML_dataset_parameters_path']

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

            else:
                BS_dict[company] = "Sell"

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

            else:
                BS_dict[company] = "Sell"

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

    def NN(self):
        # Creation of the dictionary to calculate next values
        prediction_dict = {}
        next_variation_dict = {}

        # Creation of the dictionary to advise Buy or Sell
        BS_dict = {}

        # Load parameters used during model training and if trend different than current param re-create model and train and save param
        with open(os.getcwd()+self.ML_dataset_parameters_path, 'r') as json_file:
            ML_Parameters = json.load(json_file)

        if ML_Parameters['trend_length'] != self.trend_length:
            # Hist loading and dataset creation
            my_hist = Dataset(Parameters)
            my_hist.load()
            ML_dataset = my_hist.new_format(len(my_hist.hist))

            # Create the dataset
            ML_dataset = my_hist.create_ML_dataset(ML_dataset)

            # Train the dataset
            ML_Models(Parameters).train_NN(ML_dataset)

        # Load model
        model = keras.models.load_model(os.getcwd()+'\\models\\NN')

        # Prediction dictionary filling
        for company in self.companies_list:
            values_list = self.df.loc[company,:].to_list()

            # Normalize
            normalizer = max(values_list) * 2

            # calculate next value for accuracy 
            prediction_dict[company] = model.predict(np.array([[x/normalizer for x in values_list]], dtype=float)) * normalizer
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

            else:
                BS_dict[company] = "Sell"

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

            else:
                BS_dict[company] = "Sell"

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

if __name__ == "__main__":
    study_length = 10

    # Loading history csv
    full_hist = pd.read_csv(os.getcwd() + '\\resources\\full_NASDAQ_history.csv', usecols=['Close', 'Company', 'Date', 'Dividends', 'High', 'Low', 'Open', 'Stock Splits', 'Volume'])
    # Setting back the timestamp format
    full_hist['Date'] = full_hist['Date'].astype('datetime64[ns]')
    
    dataset = Dataset(full_hist).new_format(study_length)

    BS_dict, Trend_dict = BuySell(dataset, Parameters).trend()
    print(BS_dict, Trend_dict)

