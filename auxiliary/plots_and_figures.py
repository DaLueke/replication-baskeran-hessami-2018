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

    plt.figure().set_figheight(5)
    plt.figure().set_figwidth(8)
    
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

    # Title of table
    plt.title("Number of Municipalities with \nknown characteristitcs")
    
    # Adjust layout
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

    # Add grid, title and legend
    plt.grid(True)
    plt.title("Number of observed candidates per \nmayor election period, split by gender.")
    plt.legend()

    # Adjust layout
    plt.subplots_adjust(wspace=1)    
    # Store plot and prevent it from showing in code cell.
    plt.savefig("out/figure_1.png")
    plt.close()


def summary_stats(df):

    df_main_dataset = df

    variables = np.array(["gewinn_norm", 
    	"age",
    	"non_university_phd",
    	"university",
    	"phd",
    	"architect", 
    	"businessmanwoman", 
    	"engineer", 
    	"lawyer", 
    	"civil_administration", 
    	"teacher", 
    	"employed",
    	"selfemployed", 
    	"student",
    	"retired",
    	"housewifehusband"
    ])

    measures = np.array(["Count", 
    	"Mean",
    	"SD",
    	"Min",
    	"Max"    
    ])

    arrays = [np.repeat(a=np.array(["All candidates", "Female candidates"]), repeats=5), 
              np.tile(measures, 2)]

    summary_stat = pd.DataFrame(index=variables, columns=arrays)
    for category in ["All candidates", "Female candidates"]:
        if category == "Female candidates":
            relevant_slice = df_main_dataset["female"]==1
        else:
            relevant_slice = df_main_dataset.index

        for i, var in enumerate(variables):

            row = {}
            row["Count"] = df_main_dataset.loc[relevant_slice, var].count()
            row["Mean"] = df_main_dataset.loc[relevant_slice, var].mean()
            row["SD"] = df_main_dataset.loc[relevant_slice, var].std()
            row["Min"] = df_main_dataset.loc[relevant_slice, var].min()
            row["Max"] = df_main_dataset.loc[relevant_slice, var].max()

            for measure in measures:
                summary_stat[category][measure][var] = row[measure]

    return summary_stat


def plot_observations(df, s=1):
    """ Scatter plot for observed data on normalized rank improvement 
    of a female council candidate and the margin of victory of a female 
    mayor in that municipality.
    
    Args:
        - df: DataFrame that contains the observations (main_dataset.dta)
        - s: share of observations to be plotted
    
    """
    
    obs = df.sample(frac=s)
    plt.figure(figsize=(10,10))
    plt.scatter(x=obs['margin_1'], y=obs['gewinn_norm'], marker = 'x', s=25, color='k', linewidth=1)
    plt.title(label='Figure 2: Margin of victory of a female mayor and \n list rank improvements of females in subsequent council elections')
    plt.xlabel('Margin of Victory')
    plt.ylabel('Rank Improvements')
    plt.grid()
    plt.savefig("out/figure_2.png")
    plt.close()
    

def hist_council_sizes(df, bins):
    """ Plots a histogram for the number of seats in councils.
    """
    plt.hist(df.drop_duplicates(subset=['gkz_jahr'])['council_size'], bins=bins)
    plt.title(label='Figure 3: Distribution of council sizes')
    plt.xlabel('Number of seats')
    plt.ylabel('Number of councils')
    plt.grid()
    plt.savefig("out/figure_3.png")
    plt.close()