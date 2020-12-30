# -------------------- 
# IMPORTS
# -------------------- 

# ---- LIBRARIES -----
import os
import csv
import json
import keras
import numpy as np
import pandas as pd
import yfinance as yf
import tensorflow as tf

from tqdm.auto import tqdm
from keras.models import Model, Sequential
from keras.layers import add,Input,Conv1D,Activation,Flatten,Dense


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
        self.NN_model_path = Parameters['NN_model_path']
        self.TCN_model_path = Parameters['TCN_model_path']
        self.trend_length = Parameters['trend_length']
        self.ML_trend_length = Parameters['ML_trend_length']
        self.date_name = ''
        self.companies_list_path = Parameters['Companies_list_path']
        self.companies_list = pd.read_csv(os.getcwd() +Parameters['Companies_list_path'])['Companies'].to_list()        
        self.parameters = Parameters

    def train_NN(self, dataset):
        # Define input and output
        X = dataset.loc[:, dataset.columns != 'prediction']
        y = dataset['prediction']
        
        # Define the keras model
        model = Sequential()
        model.add(Dense(12, input_dim=self.ML_trend_length, activation='relu'))
        model.add(Dense(8, activation='relu'))
        model.add(Dense(8, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))

        # Compile the keras model
        model.compile(loss=tf.losses.MeanSquaredError(), optimizer='adam', metrics=[tf.metrics.MeanAbsoluteError()])

        # Fit the keras model on the dataset
        model.fit(X, y, epochs=15, batch_size=10, verbose=2)

        # evaluate the keras model
        _, accuracy = model.evaluate(X, y)
        print('Accuracy: %.2f' % (accuracy*100))

        # Save the model and the parameters used
        model.save(os.getcwd()+self.NN_model_path+str(self.ML_trend_length))
        with open(os.getcwd()+self.parameters['ML_dataset_parameters_path'], 'w') as json_file:
            json.dump(self.parameters, json_file)

        return()
    
    def ResBlock(self, x, filters, kernel_size, dilation_rate):
        # Residual block
        r=Conv1D(filters,kernel_size,padding='same',dilation_rate=dilation_rate,activation='relu')(x) #first convolution
        r=Conv1D(filters,kernel_size,padding='same',dilation_rate=dilation_rate)(r) #Second convolution
        if x.shape[-1]==filters:
            shortcut=x
        else:
            shortcut=Conv1D(filters,kernel_size,padding='same')(x) #shortcut (shortcut)
        o=add([r,shortcut])
        o=Activation('relu')(o) #Activation function
        return(o)
    
    def train_TCN(self, dataset):
        # Columns creation
        columns = []
        for i in range(self.ML_trend_length):
            columns.append('Day_'+str(i+1))

        # Define input and output
        X = dataset.loc[:, columns]
        y = dataset['prediction']
        #dict_slices = tf.data.Dataset.from_tensor_slices((X.to_dict('list'), y.values)).batch(self.ML_trend_length)
        

        # Define the nb of layers to adapt to the parameters        
        nb_layers = int(np.log(self.ML_trend_length)/np.log(2))

        # Define the tcn model
        inputs = Input(shape=(self.ML_trend_length,1))

        x = self.ResBlock(inputs,filters=2**nb_layers,kernel_size=3,dilation_rate=1)
        for i in range(nb_layers-3):
            x = self.ResBlock(x,filters=2**(nb_layers-i),kernel_size=3,dilation_rate=2**(i+1))
        x = Flatten()(x)
        x = Dense(10,activation='softmax')(x)
        model = Model(inputs, x)

        # Compile the tcn model
        model.compile(loss=tf.losses.MeanSquaredError(), optimizer='adam', metrics=[tf.metrics.MeanAbsoluteError()])

        # Train the tcn model on the dataset
        model.fit(X, y, epochs=15, batch_size=100)

        # evaluate the keras model
        _, accuracy = model.evaluate(X, y)
        print('Accuracy: %.2f' % (accuracy*100))

        # Save the model and the parameters used
        model.save(os.getcwd()+self.TCN_model_path+str(self.ML_trend_length))
        with open(os.getcwd()+self.parameters['ML_dataset_parameters_path'], 'w') as json_file:
            json.dump(self.parameters, json_file)

        return()


if __name__ == "__main__":
    # Columns creation
    columns = []
    for i in range(Parameters['ML_trend_length']):
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
# TCN model is a 
