# -------------------- 
# IMPORTS
# -------------------- 

# ---- LIBRARIES -----
import os
import json
import datetime
import pandas as pd

from tqdm.auto import tqdm

# ---- FUNCTIONS -----
from functions.BuySell_Trend import BuySellTrend


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
        self.mesh = Parameters['Mesh']
        self.trend_length = Parameters['trend_length']
        self.initial_investment = Parameters['initial_investment']
        self.ratio_of_gain_to_save = Parameters['ratio_of_gain_to_save']
        self.ratio_max_investment_per_value = Parameters['ratio_max_investment_per_value']
        self.BS_deals_print = Parameters['BS_deals_print']
        self.transaction_fees_percentage = Parameters['transaction_fees_percentage']
        self.companies_list = pd.read_csv(os.getcwd() +Parameters['Companies_list_path'])['Companies'].to_list()

    def create(self):
        # This function will create a portfolio structure if it doesn't already exists
        if os.path.exists(self.path):
            with open(self.path) as json_file:
                self.portfolio = json.load(json_file)

        else:
            self.portfolio = {}
            self.portfolio['Money'] = {'Spending Money': self.initial_investment, 'Savings': 0}
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
        ROI = (portfolio['Money']['Spending Money'] + portfolio['Money']['Savings'] + sum([portfolio['Shares']['Count'][x]*int(dataset.loc[x,:].to_list()[-1]*1000)/1000 for x in companies_list]))/self.initial_investment

        Savings = {
            'Total value': ROI *self.initial_investment,
            'ROI': ROI,
            'Spending Money': portfolio['Money']['Spending Money'],
            'Savings': portfolio['Money']['Savings'],
            'Shares value':sum([portfolio['Shares']['Count'][x]*int(dataset.loc[x,:].to_list()[-1]*1000)/1000 for x in companies_list])
        }

        #TODO
        # Add relative ROI (ROI / NASDAQ evolution)

        return(Savings)

    def simulation(self, dataset, Portfolio):
        # Initialisation
        Portfolio_history = []
        
        # Visual feedback
        print('Portfolio simulation in progress')

        for day in tqdm(range(self.trend_length, len(dataset.columns.to_list()))):
            # Reducing the dataset to the trend period studied
            small_dataset = dataset[dataset.columns[day-self.trend_length:day]].fillna(0)

            # Getting the list of the values to buy and their actual trend
            BS_dict, Trend_dict = BuySellTrend(small_dataset).run()

            # Sorting the trend dict in reverse to get the best relative trends first
            Sorted_Trend_dict = {k: v for k, v in sorted(Trend_dict.items(), key=lambda item: item[1] , reverse=True)}

            # Register the date in the portfolio
            Portfolio['Timestamp'] = datetime.datetime.now()

            # ----- BUYING ------ 
            for company in self.companies_list:
                symbol_current_value = int(small_dataset.loc[company,:].to_list()[-1]*1000)/1000

                if BS_dict[company] == 'Buy' and symbol_current_value != 0:
                    
                    # Buy as much as possible
                    nb_symbols_to_buy = Portfolio['Money']['Spending Money']*self.ratio_max_investment_per_value//symbol_current_value
                    buy_value = int(nb_symbols_to_buy*symbol_current_value*1000)/1000
                    
                    # Portfolio update
                    Portfolio['Money']['Spending Money'] -= buy_value
                    Portfolio['Shares']['Count'][company] += nb_symbols_to_buy
                    Portfolio['Shares']['Buy_Value'][company] += buy_value
                    
                    # Print deal details
                    if nb_symbols_to_buy >= 1 and self.BS_deals_print:
                        print(str(int(nb_symbols_to_buy))+' x '+company+' bought at '+str(symbol_current_value)+' ('+str(buy_value)+'$ Total - Cash: '+str(int(Portfolio['Money']['Spending Money']*1000)/1000)+'$'+' - Savings: '+str(int(Portfolio['Money']['Savings']*1000)/1000)+'$')

            # ----- SELLING ------
            for company in self.companies_list:
                symbol_current_value = int(small_dataset.loc[company,:].to_list()[-1]*1000)/1000

                if BS_dict[company] == 'Sell' and Portfolio['Shares']['Count'][company] > 0 and symbol_current_value != 0:
                    # Sell all
                    nb_symbols_to_sell = Portfolio['Shares']['Count'][company]
                    sell_value = nb_symbols_to_sell*symbol_current_value
                    
                    profit = sell_value - Portfolio['Shares']['Buy_Value'][company]
                    
                    # Portfolio update
                    if profit > 0 and symbol_current_value > 0:
                        Portfolio['Money']['Spending Money'] += sell_value*(100-self.transaction_fees_percentage)/100 - profit * self.ratio_of_gain_to_save
                        Portfolio['Money']['Savings'] += profit*self.ratio_of_gain_to_save
                        Portfolio['Shares']['Count'][company] = 0
                        Portfolio['Shares']['Buy_Value'][company] = 0

                        if nb_symbols_to_sell >= 1 and self.BS_deals_print:
                            print(str(int(nb_symbols_to_sell))+' x '+company+' sold at '+str(symbol_current_value)+' (Profit: '+str(profit)+'$ - Cash: '+str(int(Portfolio['Money']['Spending Money']*1000)/1000)+'$'+' - Savings: '+str(int(Portfolio['Money']['Savings']*1000)/1000)+'$')

            # Current value update
            for company in self.companies_list:
                symbol_current_value = int(small_dataset.loc[company,:].to_list()[-1]*1000)/1000
                Portfolio['Shares']['Current_Value'][company] += Portfolio['Shares']['Count'][company] * symbol_current_value

            # Portfolio audit trail creation
            Savings = self.value(Portfolio, dataset, self.companies_list)
            Portfolio_history.append(Savings)  

            self.portfolio = Portfolio  

        return(Portfolio, Portfolio_history)


if __name__ == "__main__":
    # Test lines, executed only when the file is executed as main
    Portfolio_class(Parameters).create()
    #Portfolio_class(Parameters).reset()