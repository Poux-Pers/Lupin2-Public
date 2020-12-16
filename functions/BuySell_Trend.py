# -------------------- 
# IMPORTS
# -------------------- 

# ---- LIBRARIES -----
import os
import numpy as np
import pandas as pd

# ---- FUNCTIONS -----
from functions.Dataset_Class import Dataset


# -------------------- 
# LOCAL PARAMETERS
# -------------------- 
add_specific_rules = True
sell_after_stagnation = {'y/n': True, 'nb_days': 4}
sell_after_high_rise_ratio = 1.5


# -------------------- 
# MAIN
# -------------------- 

class BuySellTrend():

    def __init__(self, short_hist_dataframe):
        self.df = short_hist_dataframe

    def run(self):
        # Creation of the dictionary to calculate trends
        trends_dict = {}

        # Trends dictionary filling
        for company in self.df.index:
            values_list = self.df.loc[company,:].to_list()

            # Linear Regression
            list_x = np.arange(0,len(values_list))
            values_array = np.array(values_list)
            a,b = np.polyfit(list_x,values_array,1)

            # Calculate relative trend
            if np.mean(values_list) != 0:
                trends_dict[company] = a / np.mean(values_list)
            else:
                trends_dict[company] = 0

            # Specific rules
            if add_specific_rules:
                # Sell after a high rise
                if values_list[-1] > sell_after_high_rise_ratio * values_list[-2]:
                    trends_dict[company] = 0

                # Stagnation
                if sell_after_stagnation['y/n'] and sum(values_list[-sell_after_stagnation['nb_days']-1:-1])/sell_after_stagnation['nb_days']+1 == values_list[-1]:
                    trends_dict[company] = 0


        # List of companies without the total ('NASDAQ')
        companies_list = self.df.index[:-1].to_list()

        # Creation of the dictionary to advise Buy or Sell
        BS_dict = {}
        Trend_dict={}

        # Buy or Sell dictionary filling
        for company in companies_list:
            trend = trends_dict[company]
            general_trend = trends_dict['NASDAQ']

            BS_dict[company] = {}

            # Condition to buy or Sell
            if trend > 0 and trend/general_trend >= 1:
                BS_dict[company] = "Buy"
                Trend_dict[company] = trend
            else:
                BS_dict[company] = "Sell"
                Trend_dict[company] = trend
        
        return(BS_dict, Trend_dict)

if __name__ == "__main__":
    study_length = 10

    # Loading history csv
    full_hist = pd.read_csv(os.getcwd() + '\\resources\\full_NASDAQ_history.csv', usecols=['Close', 'Company', 'Date', 'Dividends', 'High', 'Low', 'Open', 'Stock Splits', 'Volume'])
    # Setting back the timestamp format
    full_hist['Date'] = full_hist['Date'].astype('datetime64[ns]')
    
    dataset = Dataset(full_hist).new_format(study_length)

    BS_dict, Trend_dict = BuySellTrend(dataset).run()
    print(BS_dict, Trend_dict)

