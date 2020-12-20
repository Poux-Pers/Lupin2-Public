# -------------------- 
# IMPORTS
# -------------------- 

# ---- LIBRARIES -----
import os
import csv
import json
import numpy as np
import pandas as pd
import yfinance as yf

from tqdm.auto import tqdm
from keras.models import Sequential
from keras.layers import Dense


# -------------------- 
# PARAMETERS
# -------------------- 

with open(os.getcwd()+'\\parameters\\Parameters.json', 'r') as json_file:
    Parameters = json.load(json_file)


# -------------------- 
# MAIN
# -------------------- 

class ML_Models():
    def __init__(self, Parameters):        
        self.mesh = Parameters['Mesh']
        self.hist = pd.DataFrame([])
        self.hist_path = Parameters['hist_path']
        self.ML_dataset_path = Parameters['ML_dataset_path']    
        self.trend_length = Parameters['trend_length']
        self.date_name = ''
        self.companies_list_path = Parameters['Companies_list_path']
        self.companies_list = pd.read_csv(os.getcwd() +Parameters['Companies_list_path'])['Companies'].to_list()

    def train_NN(self, dataset):
        # Define input and output
        X = dataset.loc[:, dataset.columns != 'prediction']
        y = dataset['prediction']
        
        # Define the keras model
        model = Sequential()
        model.add(Dense(12, input_dim=self.trend_length, activation='relu'))
        model.add(Dense(8, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))

        # Compile the keras model
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        # Fit the keras model on the dataset
        model.fit(X, y, epochs=150, batch_size=10)

        # evaluate the keras model
        _, accuracy = model.evaluate(X, y)
        print('Accuracy: %.2f' % (accuracy*100))

        return()


if __name__ == "__main__":
    # Columns creation
    columns = []
    for i in range(Parameters['trend_length']):
        columns.append('Day_'+str(i+1))
    columns.append('prediction')
    # Add columns
    #columns += ['sector', 'country', 'shortName']

    print(columns)
    # Load Dataset
    dataset = pd.read_csv(os.getcwd()+Parameters['ML_dataset_path'], usecols=columns)

    print(dataset)

    ML_Models(Parameters).train_NN(dataset)

# ----- COMMENTS -----
# https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/
