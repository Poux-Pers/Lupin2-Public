# Lupin2

The Lupin2 project is a simple algo-trading simulation.
It has been designed to have a Kibana dashboard.

Folder organisation is as follows:
- functions -- contains all the functions needed to create datasets, simulate a portfolio, predict symbols value etc?
- kibana -- contains the json files needed to reconstruct the dashboard
- models -- contains the model.fit for CNN and TCN and the results of the evaluation model python script (Lupin2_Models_Evals) 
- parameters -- contains the parameters for the program and the parameters used to create the dataset (used for dataset automatic recreation if needed)
- resources -- contains all the dataset (NASDAQ history, list of NASDAQ companies, etc.)

Models used in this program
- trend --
- graduated_weights --
- random_walks --
- three_in_row --
- NN --
- TCN --