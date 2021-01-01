# -------------------- 
# IMPORTS
# -------------------- 

# ---- LIBRARIES -----
import os
import csv
import json
import cryptocompare
import numpy as np
import pandas as pd
import yfinance as yf

from tqdm.auto import tqdm


# -------------------- 
# PARAMETERS
# -------------------- 

Parameters = {
    # Hisotry parameters
    'Update_Values': True, 
    'Mesh':  '1d', #Can be '1d' or '1m'

    # Model parameters definition
    'trend_length': 10, # Number of days to calculate a trend
    'study_length': 500, # Last days/minutes on which it will iterate

    # Financial parameters
    'initial_investment': 5000, #$ - Initial sum of money to invest
    'transaction_fees_percentage': 0.1, #%

    # Investment parameters
    'ratio_of_gain_to_save': 0.1, # Ratio of the gain we save up
    'ratio_max_investment_per_value': 0.5, # Ratio of the max investment that can be invested in a value per day
    #max_simultaneous_investment: 10, # Max percentage of the overall portfolio on the same value to minimize risks # To be implemùented in a much more conservative scenario

    # Simulation parameters
    'BS_deals_print': False,
    'Optimization_run': False
}

with open(os.getcwd()+'\\parameters\\Parameters.json', 'r') as json_file:
    Parameters = json.load(json_file)


# -------------------- 
# MAIN
# -------------------- 

class Dataset():

    def __init__(self, Parameters):
        self.mesh = Parameters['Mesh']
        self.hist = pd.DataFrame([])
        self.hist_path = Parameters['hist_path']
        self.ML_dataset_path = Parameters['ML_dataset_path']
        self.trend_length = Parameters['trend_length']
        self.ML_trend_length = Parameters['ML_trend_length']
        self.parameters = Parameters
        self.date_name = ''
        self.companies_list_path = Parameters['Companies_list_path']
        self.companies_list = pd.read_csv(os.getcwd() +Parameters['Companies_list_path'])['Companies'].to_list()
        

    def load(self):
        
        if self.mesh == '1m':
            self.path = os.getcwd() + '\\resources\\full_NASDAQ_history_1m.csv'
            self.date_name = 'Datetime'
            self.hist = pd.read_csv(self.path, usecols=['Close', 'Company', 'Datetime', 'Dividends', 'High', 'Low', 'Open', 'Stock Splits', 'Volume'])

        else:
            self.path = os.getcwd() + self.hist_path
            self.date_name = 'Date'
            self.hist = pd.read_csv(self.path, usecols=['Close', 'Company', 'Date', 'Dividends', 'High', 'Low', 'Open', 'Stock Splits', 'Volume'])
        
        # Date formating
        return(self.hist)

    def update(self, update_period='max'):
        # Initialization
        list_of_df_to_merge = [self.hist]

        for company in tqdm(self.companies_list):
            # Get history Ignore timezone
            hist = yf.Ticker(company).history(period=update_period, interval=self.mesh)

            if hist.empty != True:
                # Reset index, add company name and format date
                hist = hist.reset_index()
                hist['Company'] = [company]*len(hist)

                if self.date_name == '1m':                
                    hist[self.date_name] = hist[self.date_name].dt.strftime('%Y-%m-%d %H:%M:%S')
                    hist[self.date_name] = hist[self.date_name].dt.floor('min')
                else:
                    hist[self.date_name] = hist[self.date_name].astype('datetime64[ns]')
                
                hist[self.date_name] = pd.to_datetime(hist[self.date_name])
                
                # Add hist to the list of dict to merge
                list_of_df_to_merge.append(hist)

            else:
                # We remove companies whose data was not available
                self.companies_list.remove(company)

        # Concat and remove duplicates
        new_hist = pd.concat(list_of_df_to_merge)[self.hist.columns]
        new_hist = new_hist.drop_duplicates(subset=[self.date_name, 'Company'], keep='last')

        # reset index for the new dataframe
        new_hist = new_hist.reset_index(drop=True)
        new_hist

        # Update Company list
        with open(os.getcwd() + self.companies_list_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(zip(['Companies']+self.companies_list))

        self.hist = new_hist
        self.save()

        return(self.hist)

    def update_crypto(self, cryptoname):
        # Fetch info
        btc_hist = cryptocompare.get_historical_price_day(cryptoname, curr='USD', limit=2000)

        # Load
        df_hist = pd.DataFrame(btc_hist)

        # Time format
        df_hist['time'] = pd.to_datetime(df_hist['time'], unit='s')

        # Rename
        df_hist = df_hist.rename(columns={"time": "Date", "open": "Open", "high": "High", "low": "Low", "close": "Close", "volumeto": "Volume"})

        # Reorder columns
        df_hist = df_hist[['Date','Open','High','Low','Close','Volume']]

        # Completion with fake values
        df_hist['Dividends'] = len(df_hist) * [0]
        df_hist['Stock Splits'] = len(df_hist) * [0]
        df_hist['Company'] = len(df_hist) * [cryptoname]

        # Concat and remove duplicates
        new_hist = pd.concat([self.hist, df_hist])[self.hist.columns]
        new_hist = new_hist.drop_duplicates(subset=[self.date_name, 'Company'], keep='last')

        # reset index for the new dataframe
        new_hist = new_hist.reset_index(drop=True)
        new_hist

        # Update Company list
        with open(os.getcwd() + self.companies_list_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(zip(['Companies']+self.companies_list))

        self.hist = new_hist
        self.save()

        return(self.hist)

    def save(self):
        self.hist.to_csv(self.path)

    def new_format(self, study_length):
        # TCD to set date in columns, have a sum of the companies
        TCD = pd.pivot_table(self.hist, 'Open', index=['Company'], columns=[self.date_name], aggfunc=np.sum, margins=True, margins_name='NASDAQ').fillna(method='ffill', axis=1)
        
        # Keeping only the NASDAQ row
        TCD = TCD.drop(columns=['NASDAQ'])

        # Sorting columns
        TCD = TCD.reindex(TCD.columns.tolist().sort(), axis=1)
        
        # Replacing remaining NaN by 0 
        #TCD.fillna(0)

        # Reshaping
        TCD.columns.name = None
        TCD = TCD.reset_index().rename_axis(None, axis=1).set_index('Company')

        # Resizing
        dataset = TCD[TCD.columns[-study_length:]]

        return(dataset)

    def create_ML_dataset(self, dataset):
        datasets_lists = []
        # Reducin the dataset only to the companies in the list
        dataset = dataset[dataset.index.isin(self.companies_list)]
        dataset = dataset.reset_index()

        # Dataset enrichment
        #supplement_df = self.enrich_symbol(['sector', 'country', 'shortName'])
        #supplement_df = supplement_df.reset_index()
        
        # Visual feedback
        print('Portfolio simulation in progress')

        # Columns creation
        columns = []
        for i in range(self.ML_trend_length):
            columns.append('Day_'+str(i+1))
        columns.append('prediction')

        for day in tqdm(range(self.ML_trend_length+1, len(dataset.columns.to_list())-1)):
            # Reinitialization
            BS_dict_list = []
            prediction_dict_list = []

            # Reducing the dataset to the trend period studied and the companies in the companies list TODO add string integration for ML 
            #small_dataset = dataset[['Company']+dataset.columns[day-self.trend_length:day+1].to_list()].fillna(0)
            small_dataset = dataset[dataset.columns[day-self.ML_trend_length:day+1].to_list()]

            # Rename columns
            small_dataset.columns = columns

            datasets_lists.append(small_dataset)

        # Add columns TODO add string integration for ML 
        #columns += ['sector', 'country', 'shortName']

        # Enrich the data frame with feature
        ML_dataset = pd.concat(datasets_lists).dropna()
        #ML_dataset = ML_dataset.join(supplement_df.set_index('index'), on='Company')

        # Remove row with a 0 as value²
        ML_dataset = ML_dataset.loc[ML_dataset.Day_1 > 0]

        # Normalize lines
        ML_dataset = ML_dataset.div(ML_dataset.max(axis=1)*2, axis=0)
        
        # Save dataframe and the parameters used
        ML_dataset.to_csv(os.getcwd() + self.ML_dataset_path)

        return(ML_dataset)

    def enrich_symbol(self, info_to_gather_list):
        # Initialization
        enriched_dict = {}
        
        # Get info for each symbol of the list
        for symbol in tqdm(self.companies_list):
            enriched_dict[symbol] = []
            full_info = yf.Ticker(symbol).info

            for info in info_to_gather_list:
                enriched_dict[symbol].append(full_info[info])

        # Transform it into a dataframe
        enriched_df = pd.DataFrame.from_dict(enriched_dict, orient='index', columns=info_to_gather_list)

        return(enriched_df)

if __name__ == "__main__":
    # Test lines, executed only when the file is executed as main
    full_hist = Dataset(Parameters)
    full_hist.load()
    
    if full_hist.date_name != '1m':
        full_hist.hist[full_hist.date_name] = full_hist.hist[full_hist.date_name].astype('datetime64[ns]')
    else:
        full_hist.hist[full_hist.date_name] = full_hist.hist[full_hist.date_name].dt.floor('min')
 
    full_hist.update_crypto('ETH')
    full_hist.save()
    #print(full_hist.hist)

    dataset = full_hist.new_format(1000)

    #print(full_hist.enrich_symbol(['sector', 'country', 'shortName']))

    new_hist = full_hist.create_ML_dataset(dataset)
    new_hist = new_hist.reset_index(drop=True)
    #print(new_hist)