""" This file contains auxiliary functions that are used in the student-project-DaLueke.py file.""" 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from localreg import *
from statsmodels.nonparametric.smoothers_lowess import lowess

def rdd_plot(data, x_variable, y_variable, nbins=20, ylimits=None, frac=None, width=20.1, deg=1):
    """ Plots a Regression Discontinouity Design graph. For this, binned observations are portrayed in a scatter plot. 
    Uses non-parametric regression (local polynomial estimation) to fit a curve on the original observations.
    
    Args: 
        data: contains DataFrame that contains the data to plot (DataFrame)
        x_var: determines variable on x-axis, passed as the column name (string)
        y_var: determines variable on y_axis, passed as the column name (string)
        nbins: defines number of equally sized bins (int)
        ylimits: A tuple specifying the limits of the y axis (tuple)
        width: Bandwidth for the local polynomial estimation
        deg: degree of the polynomial to be estimated


    Returns:
        Returns the RDD Plot
    """
    
    # Find min and max of the running variable
    x_var, y_var = data.loc[:,x_variable], data.loc[:,y_variable]
    x_min = int(round(x_var.min()))
    x_max = int(round(x_var.max()))
    x_width = int(round(((abs(x_min) + abs(x_max)) / nbins)))
    
    # Get a list of t uples with borders of each bin. 
    bins = []
    for b in range(x_min, x_max, x_width):
        bins.append((b, b + x_width))
        
    # Find bin for a given value
    def find_bin(value, bins):
        for count, b in enumerate(bins):
            # Bins generally follow the structure [lower_bound, upper_bound), thus do not include the upper bound.
            if (count < len(bins)-1):
                if (value >= bins[count][0]) & (value < bins[count][1]): 
                    bin_number = count
            # The last bin, however, includes its upper bound.
            else:
                if (value >= bins[count][0]) & (value <= bins[count][1]): 
                    bin_number = count 
        return bin_number
    
    # Sort running data into bins
    x_bin = np.zeros(len(x_var))
    i=0
    for value in x_var.values:
        x_bin[i] = find_bin(value, bins)
        i+=1
    
    # Write data needed for the plot into a DataFrame
    df = pd.DataFrame(data = {'x_variable': x_var, 
                              'y_variable': y_var,
                              'x_bin': x_bin
                             }
                     )
    # For each bin calculate the mean of affiliated values on the y-axis.
    y_bin_mean = np.zeros(len(bins))
    for n, b in enumerate(bins):
        affiliated_y_values = df.loc[x_bin == n]
        y_bin_mean[n] = affiliated_y_values.y_variable.mean()
    
    # For the x-axis take the mean of the bounds of each bin.
    x_bin_mean = np.zeros(len(bins))
    i=0
    for e, t in enumerate(bins):
        x_bin_mean[i] = (bins[e][0]+bins[e][1])/2
        i+=1
    
    # Draw the actual plot for all bins of the running variable and their affiliated mean in the y-variable.
    plt.scatter(x=x_bin_mean,
                y=y_bin_mean, 
                s=50, 
                c='black', 
               alpha=1)
    plt.axvline(x=0)
    if ~(ylimits == None): 
        plt.ylim(ylimits)
    plt.grid()
    
    #### TODO: As a validity test: see if x_bin is as long as x_var is!
    
    # Implement local polynomial regression 
    # This is estimated seperatly for the untreadted state (0) and the treated state (1)

    df0 = pd.DataFrame(data={
        "x0":data.loc[data[x_variable]<0][x_variable], 
        "y0":data.loc[data[x_variable]<0][y_variable],
    }).sort_values(by="x0")
    df0["y0_hat"] = localreg(x=df0["x0"].to_numpy(), y=df0["y0"].to_numpy(), degree=deg, kernel=tricube, frac=frac, width=width)

    df1 = pd.DataFrame(data={
        "x1":data.loc[data[x_variable]>0][x_variable], 
        "y1":data.loc[data[x_variable]>0][y_variable],
    }).sort_values(by="x1")
    df1["y1_hat"] = localreg(x=df1["x1"].to_numpy(), y=df1["y1"].to_numpy(), degree=deg, kernel=tricube, frac=frac, width=width)
  
    # Plot the RDD-Graph
    # fittet lines
    plt.plot(df0.x0, df0.y0_hat, color='r')
    plt.plot(df1.x1, df1.y1_hat, color='r')
    
    # labels
    plt.title(label='Figure 3: Regression Discontinuity Design Plot')
    plt.xlabel('Binned margin of victory')
    plt.ylabel('Normalized rank improvement')
    plt.show
    
    
    return #pd.DataFrame(data=[x_bin_mean, y_bin_mean])