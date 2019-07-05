""" This file contains functions used to create plots in the 
student-project-DaLueke jupyter notebook file.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_observations_municipalities_candidates(df):
    """ Plots bar charts that show:
    (1) The number of municipalities for which we have observations on council
    elections over the different mayor election periods.
    
    (2) The number of council candidates for which we have observations on the
    results of a council election. Split by gender of the candidate and again 
    over the different mayor election periods.
    """
    df_main_dataset = df
    width = 2.5
    nyears = np.arange(len(df_main_dataset.groupby("jahr").nunique()))
    # first subplot
    plt.subplot(1, 2, 1)
    plt.bar(x=nyears*3, 
            height=df_main_dataset.groupby("jahr").gkz.nunique(), 
            tick_label=df_main_dataset.jahr.unique().astype(int), 
            width=width,
    )
    # Write the exact values on top of each bar
    label=df_main_dataset.jahr.unique().astype(int)
    for i in range(len(nyears)):
        plt.text(x=nyears[i]*3 - width/4, y=df_main_dataset.groupby("jahr").gkz.nunique()[label[i]] + 5, s=df_main_dataset.groupby("jahr").gkz.nunique()[label[i]])
    plt.grid(True)

    # Second subplot, females
    plt.subplot(1, 2, 2)
    plt.bar(x=nyears*3 - width/4,
            height=df_main_dataset.loc[df_main_dataset["female"]==1].groupby("jahr").gewinn_norm.count(),
            width=width/2,
            label="females",
           )
    label=df_main_dataset.jahr.unique().astype(int)

    # Write exact values on top of each bar
    for i in range(len(nyears)):
        plt.text(x=nyears[i]*3 - width/1.5, 
                 y=df_main_dataset.loc[df_main_dataset["female"]==1].groupby("jahr").gewinn_norm.count()[label[i]] + 200, 
                 s=df_main_dataset.loc[df_main_dataset["female"]==1].groupby("jahr").gewinn_norm.count()[label[i]],
        )

    # Second subplot, males
    plt.bar(x=nyears*3 + width/4,
            height=df_main_dataset.loc[df_main_dataset["female"]==0].groupby("jahr").gewinn_norm.count(),
            width=width/2,
            tick_label=df_main_dataset.jahr.unique().astype(int),
            label="males"
           )

    # Write exact values on top of each bar
    for i in range(len(nyears)):
        plt.text(x=nyears[i]*3 - width/4, 
                 y=df_main_dataset.loc[df_main_dataset["female"]==0].groupby("jahr").gewinn_norm.count()[label[i]] + 200, 
                 s=df_main_dataset.loc[df_main_dataset["female"]==0].groupby("jahr").gewinn_norm.count()[label[i]],
        )

    # Adjust layout
    plt.grid(True)
    plt.tight_layout()
    plt.legend()
    
    return