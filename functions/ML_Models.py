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
from keras.layers import add,Input,Conv1D,Activation,Flatten,Dense, LSTM

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

class ML_Models():
    
    def __init__(self, Parameters):        
        self.mesh = Parameters['Mesh']
        self.hist = pd.DataFrame([])
        self.hist_path = Parameters['hist_path']
        self.ML_dataset_path = Parameters['ML_dataset_path']   
        self.NN_model_path = Parameters['NN_model_path']
        self.TCN_model_path = Parameters['TCN_model_path']        
        self.LSTM_model_path = Parameters['LSTM_model_path']
        self.trend_length = Parameters['trend_length']
        self.ML_trend_length = Parameters['ML_trend_length']
        self.date_name = ''
        self.companies_list_path = Parameters['Companies_list_path']
        self.companies_list = pd.read_csv(os.getcwd() +Parameters['Companies_list_path'])['Companies'].to_list()        
        self.parameters = Parameters
        self.ML_dataset_parameters_path = Parameters['ML_dataset_parameters_path']

    def train_NN(self, dataset):
        # Columns creation
        columns = ['Company_' + company for company in self.companies_list]
        for i in range(self.ML_trend_length):
            columns.append('Day_'+str(i+1))

        # Define input and output
        X = dataset.loc[:, columns]
        y = dataset['prediction']
        
        # Define the keras model
        model = Sequential()
        model.add(Dense(12, input_dim=len(columns), activation='relu'))
        model.add(Dense(8, activation='relu'))
        model.add(Dense(8, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))

        # Compile the keras model
        model.compile(loss=tf.losses.MeanSquaredError(), optimizer='adam', metrics=[tf.metrics.MeanAbsoluteError()])

        # Fit the keras model on the dataset
        model.fit(X, y, epochs=15, batch_size=10, verbose=2)

        # evaluate the keras model
        _, accuracy = model.evaluate(X, y)
        print('Average error: %.2f' % (accuracy*100))

        # Save the model
        model.save(os.getcwd()+self.NN_model_path+str(self.ML_trend_length))

        return()
    
    def verify_train_NN(self):
        # Load parameters used during model training and if trend different than current param re-create model and train and save param
        with open(os.getcwd()+self.ML_dataset_parameters_path, 'r') as json_file:
            ML_Parameters = json.load(json_file)

        # Verify if the dataset has to be redone
        if ML_Parameters['ML_trend_length'] != self.ML_trend_length:
            # Hist loading and dataset creation
            my_hist = Dataset(Parameters)
            my_hist.load()
            ML_dataset = my_hist.new_format(len(my_hist.hist))

            # Create the dataset
            ML_dataset = my_hist.create_ML_dataset(ML_dataset)

            # Save dataset parameters
            with open(os.getcwd()+self.parameters['ML_dataset_parameters_path'], 'w') as json_file:
                json.dump(self.parameters, json_file)

        else:
            ML_dataset = pd.read_csv(os.getcwd()+Parameters['ML_dataset_path'])

        if not(os.path.exists(os.getcwd()+self.NN_model_path+str(self.ML_trend_length))):
            # Train the dataset
            self.train_NN(ML_dataset)

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
        
        # Define the nb of layers to adapt to the parameters        
        nb_layers = int(np.log(self.ML_trend_length)/np.log(2))

        # Define the tcn model
        inputs = Input(shape=(self.ML_trend_length,1))

        x = self.ResBlock(inputs,filters=2**nb_layers,kernel_size=3,dilation_rate=1)
        for i in range(nb_layers-3):
            x = self.ResBlock(x,filters=2**(nb_layers-i),kernel_size=3,dilation_rate=2**(i+1))
        x = Flatten()(x)
        x = Dense(10,activation='softmax')(x)
        x = Dense(1, activation='sigmoid')(x)
        model = Model(inputs, x)

        # Compile the tcn model
        model.compile(loss=tf.losses.MeanSquaredError(), optimizer='adam', metrics=[tf.metrics.MeanAbsoluteError()])

        # Train the tcn model on the dataset
        model.fit(X, y, epochs=15, batch_size=100)

        # evaluate the keras model
        _, accuracy = model.evaluate(X, y)
        print('Average error: %.2f' % (accuracy*100))

        # Save the model 
        model.save(os.getcwd()+self.TCN_model_path+str(self.ML_trend_length))

        return()

    def verify_train_TCN(self):
        # Load parameters used during model training and if trend different than current param re-create model and train and save param
        with open(os.getcwd()+self.ML_dataset_parameters_path, 'r') as json_file:
            ML_Parameters = json.load(json_file)

        # Verify if the dataset has to be redone
        if ML_Parameters['ML_trend_length'] != self.ML_trend_length:
            # Hist loading and dataset creation
            my_hist = Dataset(Parameters)
            my_hist.load()
            ML_dataset = my_hist.new_format(len(my_hist.hist))

            # Create the dataset
            ML_dataset = my_hist.create_ML_dataset(ML_dataset)

            # Save dataset parameters
            with open(os.getcwd()+self.parameters['ML_dataset_parameters_path'], 'w') as json_file:
                json.dump(self.parameters, json_file)
            
        else:
            ML_dataset = pd.read_csv(os.getcwd()+Parameters['ML_dataset_path'])

        if not(os.path.exists(os.getcwd()+self.TCN_model_path+str(self.ML_trend_length))):
            # Train the dataset
            self.train_TCN(ML_dataset)

    def create_dataset(self, dataset, look_back=1):
        dataX, dataY = [], []
        for i in range(len(dataset)-look_back-1):
            a = dataset[i:(i+look_back), 0]
            dataX.append(a)
            dataY.append(dataset[i + look_back, 0])
        return(np.array(dataX), np.array(dataY))

    def train_LSTM(self, dataset):
        # Columns creation
        columns = []
        for i in range(self.ML_trend_length):
            columns.append('Day_'+str(i+1))

        # Define input, output and memory
        X = dataset.loc[:, columns]
        y = dataset['prediction']
        look_back = 1
        
                
        # create and fit the LSTM network
        model = Sequential()
        model.add(LSTM(4, input_shape=(1, look_back)))
        model.add(Dense(1))

        # Compile the keras model
        model.compile(loss=tf.losses.MeanSquaredError(), optimizer='adam', metrics=[tf.metrics.MeanAbsoluteError()])

        # Fit the keras model on the dataset        
        model.fit(X, y, epochs=100, batch_size=1, verbose=2)

        # evaluate the keras model
        _, accuracy = model.evaluate(X, y)
        print('Average error: %.2f' % (accuracy*100))

        # Save the model
        model.save(os.getcwd()+self.LSTM_model_path+str(self.ML_trend_length))

        return()

    def verify_train_LSTM(self):
        # Load parameters used during model training and if trend different than current param re-create model and train and save param
        with open(os.getcwd()+self.ML_dataset_parameters_path, 'r') as json_file:
            ML_Parameters = json.load(json_file)

        # Verify if the dataset has to be redone
        if ML_Parameters['ML_trend_length'] != self.ML_trend_length:
            # Hist loading and dataset creation
            my_hist = Dataset(Parameters)
            my_hist.load()
            ML_dataset = my_hist.new_format(len(my_hist.hist))

            # Create the dataset
            ML_dataset = my_hist.create_ML_dataset(ML_dataset)

            # Save dataset parameters
            with open(os.getcwd()+self.parameters['ML_dataset_parameters_path'], 'w') as json_file:
                json.dump(self.parameters, json_file)

        else:
            ML_dataset = pd.read_csv(os.getcwd()+Parameters['ML_dataset_path'])

        if not(os.path.exists(os.getcwd()+self.LSTM_model_path+str(self.ML_trend_length))):
            # Train the dataset
            self.train_LSTM(ML_dataset)

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
# TCN model is inspired of https://github.com/philipperemy/keras-tcn and https://www.programmersought.com/article/13674618779/
