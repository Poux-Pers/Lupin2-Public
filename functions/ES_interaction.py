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
        self.hist_path = Parameters['hist_path']
        self.date_name = ''
        self.json = {}

    def upload_hist(self):
        # Create index
        #self.es.indices.create(index='hist_'+self.mesh,body={})

        # TODO Only add what is not already on ES

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
        
    
if __name__ == "__main__":
    es_dict = Elasticsearch_class(Parameters)
    #es_dict.df_to_dict()
    es_dict.upload_hist()
