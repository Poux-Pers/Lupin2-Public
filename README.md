# Lupin2

The Lupin2 project is a simple algo-trading simulation.
It has been designed to have a Kibana dashboard.

Folder organisation is as follows:
- functions -- contains all the functions needed to create datasets, simulate a portfolio, predict symbols value etc?
- kibana -- contains the json files needed to reconstruct the dashboard
- models -- contains the model.fit for keras models and the results of the evaluation model python script (Lupin2_Models_Evals) 
- parameters -- contains the parameters for the program and the parameters used to create the dataset (used for dataset automatic recreation if needed)
- resources -- contains all the dataset (NASDAQ history, list of NASDAQ companies, etc.)

Models used in this program
- trend -- Simple trend calculation qith a linear regression
- graduated_weights -- Weighted average of the last values. Weights for step i is 1/(2^(n+1-i)) where n is the last step
- random_walks -- Next value is a random value in the range [last_value - standard_deviation, last_value + standard_deviation]
- three_in_row -- Buy if 3 increases in a row, sell if 3 losses in a row else hold (Associated previsions are as follow last_value + standard_deviation, last_value - standard_deviation, last_value)
- ARIMA -- AutoRegressive Integrated Moving Average
- NN -- Convolutional Neural Network
- TCN -- Temporal Convolutional Neural Network
- LSTM -- Long Short-Term Memory (Recurent Neural Network)