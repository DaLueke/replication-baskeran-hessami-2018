""" This file contains auxiliary functions that are used in the student-project-DaLueke.py file.""" 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess

def rdd_plot(data, x_variable, y_variable, nbins=20, ylimits=None, frac=0.75):
    """ Plots a Regression Discontinouity Design graph for binned observations. 
    Uses non-parametric regression (LOWESS) to fit a curve on binned data.
    
    Args: 
        data: contains DataFrame that contains the data to plot (DataFrame)
        x_var: determines variable on x-axis, passed as the column name (string)
        y_var: determines variable on y_axis, passed as the column name (string)
        nbins: defines number of equally sized bins (int)
        ylimits: A tuple specifying the limits of the y axis (tuple)
        frac: defines the fraction of observations used when estimating the LOWESS and confidence intervals


    Returns:
        pd.DataFrame with two columns: 
        1. "x_bin_mean" contains mean values for the margin of victory of a female in the mayor election.
        2. "y_bin_mean" contains mean values for rank improvements of females at the subsequent council election. 
        
    """
    
    # Find min and max of the running variable
    x_var, y_var = data.loc[:,x_variable], data.loc[:,y_variable]
    x_min = int(round(x_var.min()))
    x_max = int(round(x_var.max()))
    x_width = int(round(((abs(x_min) + abs(x_max)) / nbins)))
    
    # Get a list of tuples with borders of each bin. 
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
    
    #plt.ylim(-5, 5)
    #### TODO: As a validity test: see if x_bin is as long as x_var is!
    
    # Implement local polynomial regression 
    # This is estimated seperatly for the untreadted state (0) and the treated state (1)
    import statsmodels as sm
    untreated = x_bin_mean<0
    treated = x_bin_mean>0
    y_lowess_fit_0 = sm.nonparametric.smoothers_lowess.lowess(
        endog=y_bin_mean[0:int(nbins/2)], 
        exog=x_bin_mean[0:int(nbins/2)], 
        return_sorted=False, 
        frac=frac, 
        it=3,
        delta=0,
    )
    
    y_lowess_fit_1 = sm.nonparametric.smoothers_lowess.lowess(
        endog=y_bin_mean[int(nbins/2):int(nbins+1)], 
        exog=x_bin_mean[int(nbins/2):int(nbins+1)], 
        return_sorted=False, 
        frac=frac,
        it=3,
        delta=0,
    )
    
    #### TODO: Write this into a nice little loop over suffixes 0 and 1
    # Calculate conficence interval (95%)
    se_0 = _standard_deviations(obs=y_lowess_fit_0, frac=frac)
    se_1 = _standard_deviations(obs=y_lowess_fit_1, frac=frac)
    
    #se_0 = np.std(y_lowess_fit_0) ### old & depreciated
    ubound_0 = y_lowess_fit_0 + 1.96*se_0
    lbound_0 = y_lowess_fit_0 - 1.96*se_0
    
    #se_1 = np.std(y_lowess_fit_1)### old & depreciated
    ubound_1 = y_lowess_fit_1 + 1.96*se_1
    lbound_1 = y_lowess_fit_1 - 1.96*se_1
    
    
    # Plot the RDD-Graph
    # fittet lines
    plt.plot(x_bin_mean[0:int(nbins/2)], y_lowess_fit_0, color='r')
    plt.plot(x_bin_mean[int(nbins/2):int(nbins+1)], y_lowess_fit_1, color='r')
    
    # confidence intervals
    plt.plot(x_bin_mean[0:int(nbins/2)], ubound_0, color='black')
    plt.plot(x_bin_mean[0:int(nbins/2)], lbound_0, color='black')
    plt.plot(x_bin_mean[int(nbins/2):int(nbins+1)], ubound_1, color='black')
    plt.plot(x_bin_mean[int(nbins/2):int(nbins+1)], lbound_1, color='black')
    plt.fill_between(x=x_bin_mean[0:int(nbins/2)], y1=ubound_0, y2=lbound_0, alpha=0.3, color='grey')
    plt.fill_between(x=x_bin_mean[int(nbins/2):int(nbins+1)], y1=ubound_1, y2=lbound_1, alpha=0.3, color='grey')
    
    
    
    # labels
    plt.title(label='Figure 3: Regression Discontinuity Design Plot')
    plt.xlabel('Binned margin of victory')
    plt.ylabel('Normalized rank improvement')
    plt.show
    
    
    return #pd.DataFrame(data=[x_bin_mean, y_bin_mean])


def _standard_deviations(obs, frac):
    """ Calculates local standard deviations. For each observation the standard deviation of its 
    k-nearest neighbors is calculated.
    
    Args:
        frac: gives the share of observations that should be included in the neighborhood around each value.
        obs: contains the array of observations for which the local standard deviations should be returned.
        
    Returns:
        std: An array with the local standard deviation for each observation.
        
    """
    
    # calculate the number of neighboring observations that are used in each local estimation, floor if its a float
    n = len(obs)
    k = int(frac * n)
    std = np.zeros(n)

    # define borders for k 
    if k <= 1:
        k = 1
    if k >= n:
        k = n

    # While there is too few values on the left, consider first k elements.
    i=0

    while i - np.floor(k/2) < 0:
        left_border, right_border = 0, k
        neighborhood = obs[left_border:right_border+1]
        std[i] = np.std(neighborhood)
        i+=1

    # Each i's neighborhood is centered around i. For odd k, i is left from the center.
    while (i - np.floor(k/2) >= 0) & (i + np.ceil(k/2) < n-1):
        left_border = int(i-np.floor(k/2))
        right_border = int(i+np.ceil(k/2))
        neighborhood = obs[left_border:right_border+1]
        std[i] = np.std(neighborhood)
        i+=1


    # While there is too few values on the right, consider last k elements
    while (i + np.ceil(k/2) >= n-1) & (i<=n-1):
        left_border, right_border = n-k-1, n-1
        neighborhood = obs[left_border:right_border+1]
        std[i] = np.std(neighborhood)
        i+=1
    
    return std