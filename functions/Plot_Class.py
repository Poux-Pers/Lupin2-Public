# -------------------- 
# IMPORTS
# -------------------- 

# ---- LIBRARIES -----
import matplotlib.pyplot as plt


# -------------------- 
# MAIN
# -------------------- 

class Plot():
    def __init__(self, Parameters):
        # Initialisation
        self.Parameters = Parameters
        self.list_Total_Value = []
        self.list_ROI = []
        self.list_Cash = []
        self.list_Savings = []
        self.list_Shares_value = []
        self.Portfolio_history = []

    def portfolio_history(self, Portfolio_history, dataset):
        # List construction
        for Savings in Portfolio_history:
            self.list_Total_Value.append(Savings['Total value'])
            self.list_ROI.append(Savings['ROI'])
            self.list_Cash.append(Savings['Spending Money'])
            self.list_Savings.append(Savings['Savings'])
            self.list_Shares_value.append(Savings['Shares value'])            
            
        # Plot savings
        fig, axs = plt.subplots(3, 1)
        fig.suptitle('Savings over time')

        x_list = dataset.columns.to_list()[self.Parameters['trend_length']:]

        # Sub plot for the sum of the symbols
        axs[0].plot(x_list, dataset.iloc[-1].to_list()[self.Parameters['trend_length']:], 'purple', label='NASDAQ')
        
        axs[1].plot(x_list, self.list_Cash, 'r', label='Cash')
        axs[1].plot(x_list, self.list_Savings, 'b', label='Bank')
        axs[1].plot(x_list, self.list_Total_Value, 'black', label='Total')
        axs[1].plot(x_list, self.list_Shares_value, 'orange', label='Shares Value')

        # Sub plot for ROI
        axs[2].plot(x_list, self.list_ROI)
        plt.show(block = False)