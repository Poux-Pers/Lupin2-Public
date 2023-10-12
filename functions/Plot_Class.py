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

    def portfolio_history(self, Portfolio_history, dataset, deals_history_dict):
        # List construction
        for Savings in Portfolio_history:
            self.list_Total_Value.append(Savings['Total value'])
            self.list_ROI.append(Savings['ROI'])
            self.list_Cash.append(Savings['Spending Money'])
            self.list_Savings.append(Savings['Savings'])
            self.list_Shares_value.append(Savings['Shares value'])            
            
        # Plot savings
        fig, axs = plt.subplots(4, 1)
        fig.suptitle('Savings over time')

        x_list = dataset.columns.to_list()[self.Parameters['ML_trend_length']:]

        # Sub plot for the sum of the symbols
        axs[0].plot(x_list, dataset.iloc[-1].to_list()[self.Parameters['ML_trend_length']:], 'purple', label='NASDAQ')
        axs[0].legend(loc='upper left')
        
        axs[1].plot(x_list, self.list_Cash, 'r', label='Cash')
        axs[1].plot(x_list, self.list_Savings, 'b', label='Bank')
        axs[1].plot(x_list, self.list_Total_Value, 'black', label='Total')
        axs[1].plot(x_list, self.list_Shares_value, 'orange', label='Shares Value')
        axs[1].legend(loc='upper left')

        # Sub plot for ROI
        axs[2].plot(x_list, self.list_ROI, label='ROI')
        axs[2].legend(loc='upper left')

        # Sub plot for deals volume
        deals_count_list = [len(deals_history_dict[x]) for x in deals_history_dict]
        axs[3].plot(x_list, deals_count_list, label='Deals daily volume (# of actions bought)')
        axs[3].legend(loc='upper left')
        plt.show(block = False)