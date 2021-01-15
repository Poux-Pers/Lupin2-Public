# -------------------- 
# IMPORTS
# -------------------- 

# ---- LIBRARIES -----
import os
import json
import pandas as pd

from tqdm.auto import tqdm
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk

# ---- FUNCTIONS -----
from functions.Dataset_Class import Dataset


# -------------------- 
# PARAMETERS
# -------------------- 

with open(os.getcwd()+'\\parameters\\Parameters.json', 'r') as json_file:
    Parameters = json.load(json_file)


# -------------------- 
# MAIN
# -------------------- 

class Elasticsearch_class():

    def __init__(self, Parameters):
        self.es = Elasticsearch([{'host': 'localhost', 'port': 9200}], timeout=30, max_retries=10, retry_on_timeout=True)
        self.mesh = Parameters['Mesh']
        self.hist = pd.DataFrame([])        
        self.date_name = ''
        self.json = {}
        self.parameters = Parameters
        
        if Parameters['Crypto?']:
            self.hist_path = Parameters['Source_path']['Crypto_hist_path']            
            self.companies_list_path = Parameters['Source_path']['Crypto_list_path']
            self.companies_list = pd.read_csv(os.getcwd() +Parameters['Source_path']['Crypto_list_path'])['Companies'].to_list()
        else:
            self.hist_path = Parameters['Source_path']['Companies_hist_path']            
            self.companies_list_path = Parameters['Source_path']['Companies_list_path']
            self.companies_list = pd.read_csv(os.getcwd() +Parameters['Source_path']['Companies_list_path'])['Companies'].to_list()

    def upload_hist(self):
        # Clean index
        self.es.indices.delete(index='hist_'+self.mesh, ignore=[400, 404])

        # Create index
        self.es.indices.create(index='hist_'+self.mesh,body={})

        # Recover data
        if self.mesh == '1m':
            self.path = os.getcwd() + '\\resources\\full_NASDAQ_history_1m.csv'
            self.date_name = 'Datetime'
            self.hist = pd.read_csv(self.path, usecols=['Close', 'Company', 'Datetime', 'Dividends', 'High', 'Low', 'Open', 'Stock Splits', 'Volume'])

        else:
            self.path = os.getcwd() + self.hist_path
            self.date_name = 'Date'
            self.hist = pd.read_csv(self.path, usecols=['Close', 'Company', 'Date', 'Dividends', 'High', 'Low', 'Open', 'Stock Splits', 'Volume'])

        self.companies_list = list(set(self.hist['Company'].to_list()))

        # Dropping null value columns to avoid errors 
        self.hist.dropna(inplace = True) 

        # Initialisation
        self.dict = {}
        
        for column in self.hist.columns:
            self.dict[str(column)] = self.hist[column]

        # Bulk loading
        df = pd.DataFrame(self.dict)

        # Adding info - BE CAREFUL: Adding any new information requires to recreate the index in kibana
        #supplement_df = Dataset(self.parameters).enrich_symbol(['sector', 'country', 'shortName'])
        #supplement_df = supplement_df.reset_index()
        #df = df.join(supplement_df.set_index('index'), on='Company')

        documents = df.to_dict(orient='records')
        bulk(self.es, documents, index='hist_'+self.mesh,doc_type='foo', raise_on_error=True)

        self.es.indices.refresh(index='hist_'+self.mesh)

    def upload_df(self, df_to_upload):
        self.companies_list = list(set(df_to_upload['Company'].to_list()))

        # Dropping null value columns to avoid errors 
        df_to_upload.dropna(inplace = True) 

        # Initialisation
        self.dict = {}
        
        for column in df_to_upload.columns:
            self.dict[str(column)] = df_to_upload[column]

        # Bulk loading
        df = pd.DataFrame(self.dict)
        documents = df.to_dict(orient='records')
        bulk(self.es, documents, index='hist_'+self.mesh,doc_type='foo', raise_on_error=True)

        self.es.indices.refresh(index='hist_'+self.mesh)

    def upload_dict(self, my_dict, es_id):
        # Portfolio enrichment
        my_dict['id'] = es_id

        # Simple loading
        self.es.index(index='portfolio',doc_type='portfolio',body=my_dict, id=str(es_id))

        self.es.indices.refresh(index='portfolio')

        # Loading in bulk companies information
        df = pd.DataFrame(my_dict['Shares'])
        df['id'] = [es_id] * len(df)
        df.dropna(inplace = True)
        df = df.reset_index()
        documents = df.to_dict(orient='records')
        bulk(self.es, documents, index='my_portfolio', raise_on_error=True)

    def reset_portfolio_index(self):        
        # Clean index
        self.es.indices.delete(index='portfolio', ignore=[400, 404])
        self.es.indices.delete(index='my_portfolio', ignore=[400, 404])

        # Create index
        self.es.indices.create(index='portfolio',body={})
        self.es.indices.create(index='my_portfolio',body={})
       
    
if __name__ == "__main__":
    es_dict = Elasticsearch_class(Parameters)
    #es_dict.df_to_dict()
    es_dict.upload_hist()
