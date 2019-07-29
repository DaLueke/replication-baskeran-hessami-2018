""" This file contains auxiliary functions that are used in the
student-project-DaLueke jupyter notebook file."""
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from scipy.stats import ttest_ind
from auxiliary.localreg import *


def rdd_plot(
    data, x_variable, y_variable, nbins=20, ylimits=None, frac=None, width=20.1, deg=1
):
    """ Plots a Regression Discontinouity Design graph. For this, binned
    observations are portrayed in a scatter plot.
    Uses non-parametric regression (local polynomial estimation) to fit a curve
    on the original observations.

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
    x_var, y_var = data.loc[:, x_variable], data.loc[:, y_variable]
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
            # Bins generally follow the structure [lower_bound, upper_bound),
            # thus do not include the upper bound.
            if count < len(bins) - 1:
                if (value >= bins[count][0]) & (value < bins[count][1]):
                    bin_number = count
            # The last bin, however, includes its upper bound.
            else:
                if (value >= bins[count][0]) & (value <= bins[count][1]):
                    bin_number = count
        return bin_number

    # Sort running data into bins
    x_bin = np.zeros(len(x_var))
    i = 0
    for value in x_var.values:
        x_bin[i] = find_bin(value, bins)
        i += 1

    # Write data needed for the plot into a DataFrame
    df = pd.DataFrame(data={"x_variable": x_var, "y_variable": y_var, "x_bin": x_bin})
    # For each bin calculate the mean of affiliated values on the y-axis.
    y_bin_mean = np.zeros(len(bins))
    for n, b in enumerate(bins):
        affiliated_y_values = df.loc[x_bin == n]
        y_bin_mean[n] = affiliated_y_values.y_variable.mean()

    # For the x-axis take the mean of the bounds of each bin.
    x_bin_mean = np.zeros(len(bins))
    i = 0
    for e, t in enumerate(bins):
        x_bin_mean[i] = (bins[e][0] + bins[e][1]) / 2
        i += 1

    # Draw the actual plot for all bins of the running variable and their
    # affiliated mean in the y-variable.
    plt.scatter(x=x_bin_mean, y=y_bin_mean, s=50, c="black", alpha=1)
    plt.axvline(x=0)
    if ~(ylimits == None):
        plt.ylim(ylimits)
    plt.grid()

    # TODO: As a validity test: see if x_bin is as long as x_var is!

    # Implement local polynomial regression, calculate fitted values as well as
    # estimated betas.
    # This is estimated seperatly for the untreadted state 0 and the treated state 1

    df0 = pd.DataFrame(
        data={
            "x0": data.loc[data[x_variable] < 0][x_variable],
            "y0": data.loc[data[x_variable] < 0][y_variable],
        }
    ).sort_values(by="x0")
    df0["y0_hat"] = localreg(
        x=df0["x0"].to_numpy(),
        y=df0["y0"].to_numpy(),
        degree=deg,
        kernel=tricube,
        frac=frac,
        width=width,
    )["y"]
    for i in range(deg + 1):
        df0["beta_hat_" + str(i)] = localreg(
            x=df0["x0"].to_numpy(),
            y=df0["y0"].to_numpy(),
            degree=deg,
            kernel=tricube,
            frac=frac,
            width=width,
        )["beta"][:, i]

    df1 = pd.DataFrame(
        data={
            "x1": data.loc[data[x_variable] > 0][x_variable],
            "y1": data.loc[data[x_variable] > 0][y_variable],
        }
    ).sort_values(by="x1")
    df1["y1_hat"] = localreg(
        x=df1["x1"].to_numpy(),
        y=df1["y1"].to_numpy(),
        degree=deg,
        kernel=tricube,
        frac=frac,
        width=width,
    )["y"]
    for i in range(deg + 1):
        df1["beta_hat_" + str(i)] = localreg(
            x=df1["x1"].to_numpy(),
            y=df1["y1"].to_numpy(),
            degree=deg,
            kernel=tricube,
            frac=frac,
            width=width,
        )["beta"][:, i]

    # Calculate local standard errors
    y0_se = local_se(df=df0, kernel=tricube, deg=deg, width=width)
    y1_se = local_se(df=df1, kernel=tricube, deg=deg, width=width)

    # Calculate confidence intervals
    # TODO: This certainly would be faster if I would not use dictionaries!
    y0_upper_ci = np.empty(len(df0["y0"]))
    y0_lower_ci = np.empty(len(df0["y0"]))
    y1_upper_ci = np.empty(len(df1["y1"]))
    y1_lower_ci = np.empty(len(df1["y1"]))

    for count, element in enumerate(df0["x0"].array):
        y0_upper_ci[count] = df0["y0_hat"].iloc[count] + 1.96 * y0_se[str(element)]

    for count, element in enumerate(df0["x0"].array):
        y0_lower_ci[count] = df0["y0_hat"].iloc[count] - 1.96 * y0_se[str(element)]

    for count, element in enumerate(df1["x1"].array):
        y1_upper_ci[count] = df1["y1_hat"].iloc[count] + 1.96 * y1_se[str(element)]

    for count, element in enumerate(df1["x1"].array):
        y1_lower_ci[count] = df1["y1_hat"].iloc[count] - 1.96 * y1_se[str(element)]

    # Plot the RDD-Graph
    # fittet lines
    plt.plot(df0.x0, df0.y0_hat, color="r")
    plt.plot(df1.x1, df1.y1_hat, color="r")

    plt.plot(df0.x0, y0_upper_ci, color="black")
    plt.plot(df0.x0, y0_lower_ci, color="black")
    plt.plot(df1.x1, y1_upper_ci, color="black")
    plt.plot(df1.x1, y1_lower_ci, color="black")

    # Plot the RDD-Graph
    # fittet lines
    plt.plot(df0.x0, df0.y0_hat, color="r")
    plt.plot(df1.x1, df1.y1_hat, color="r")

    # labels
    plt.title(label="Figure 3: Regression Discontinuity Design Plot")
    plt.xlabel("Binned margin of victory")
    plt.ylabel("Normalized rank improvement")
    plt.show

    return


def local_se(df, kernel, deg, width):
    """ This function is used to calculate the local standard errors, based on
    estimation results of a local polynomial regression.
    """

    if deg != 1:
        print(
            "WARNING: function local_se is currently hard-coded for ",
            "polynomials of degree 1 and will deliver wrong results for ",
            "anything else!",
        )

    x = df[df.columns[0]].array
    cap_x = pd.DataFrame(data={"constant": np.ones(len(x)), "x": x})
    beta_hat_covariances = {}
    y_hat_se = {}
    for count, element in enumerate(x):

        # get weights from the local regression
        weights = kernel(np.abs(x - element) / width)

        # only consider values with a weight > 0
        inds = np.where(np.abs(weights) > 1e-10)[0]

        # CAUTION: This area is hard-coded for a polynomial of degree 1
        rel_data = df.iloc[inds]
        beta_cov = np.cov(m=rel_data[["beta_hat_0", "beta_hat_1"]], rowvar=0)
        beta_hat_covariances.update({str(element): beta_cov})

        se = np.dot(cap_x.loc[count, :].array, beta_hat_covariances[str(element)])
        se = np.dot(se, np.transpose(cap_x.loc[count, :].array))

        y_hat_se.update({str(element): np.sqrt(se)})
    return y_hat_se


def calculate_weights(df, bandwidth):
    """ This function calculates the weights for the weightes linear regression
    which is performed to estimate the effect of a female mayor around the
    cutoff in the regression discontinuity analysis.

    Calculations follow the code provided in the bandwidth_and_weights.ado file
    step by step.
    """
    temp1 = df["margin_1"] / bandwidth
    ind = abs(temp1) <= 1
    temp2 = 1 - abs(temp1)
    return temp2 * ind


def ttest_mean_characteristics(df, bw):
    """ Function used to calculate a t-test for difference in means of the
    characteristics of municipalities. The data merging process is left in
    the notebook on purpose so one can easily see and change the difference
    in the mergin process between the notebook and the original article.

    Input:
    df: DataFrame that contains the characteristics of all
        municipalities as well as data on  mayor elctions.
    bw: A list of bandwidths for that the t-test will be performed.

    Returns:
    rslt_dict:  A dictionary with one key for every bandwidth that is set
                in the inputs. Each value is a dataframe containing the
                charactersitics and the t-test result.

    """

    # Adjust name of input df
    df_pretreatment_characteristics = df
    bandwidths = bw

    rslt_dict = {}

    for bw in bandwidths:
        data = df_pretreatment_characteristics.loc[
            abs(df_pretreatment_characteristics["margin_1"]) < bw
        ]

        # Build a dataframe that resembles table A4: means, differences in means, se
        # and n for characterstics in munis where men won and those where women won.
        index = [
            "log_bevoelkerung",
            "log_flaeche",
            "log_debt_pc",
            "log_tottaxrev_pc",
            "log_tot_beschaeft_pc",
            "log_female_share_totbesch",
            "log_gemeinde_beschaef_pc",
            "log_female_sh_gem_besch",
            "log_prod_share_tot",
            "log_female_share_prod",
        ]
        gender = ["female_mayor", "male_mayor"]
        df = pd.DataFrame(index=index, columns=gender)

        # Calculate means, differences in means and number of obs.
        for i in gender:
            df[i] = data.loc[data[i] == 1, index].mean()

        df["Diff."] = df.loc[:, gender[0]] - df.loc[:, gender[1]]
        df["Obs."] = data[index].count()

        # Perform t-test and report p-values
        for i, v in enumerate(index):
            statistic, p_value = ttest_ind(
                data.loc[data["female_mayor"] == 1, index[i]].dropna(),
                data.loc[data["female_mayor"] == 0, index[i]].dropna(),
            )
            df.loc[v, "t_test_p"] = p_value.round(3)

        rslt_dict[bw] = df
    return rslt_dict


def party_affiliation(df, bw):

    # adjust name of input df
    df_mayor_election_data = df

    bandwidths = bw
    rslt_dict = {}
    parties = ["cdu", "spd", "other_party"]

    # loop over different bandwiths
    for bw in bandwidths:
        container = pd.DataFrame(index=parties)

        # Select data on party affiniation of mayors that won mixed-gender elections
        data = df_mayor_election_data.loc[
            (df_mayor_election_data["rdd_sample"] == 1)
            & (abs(df_mayor_election_data["margin_1"]) < bw)
        ]

        # loop over paties (only spd, cdu and "other" are available) and calculate the
        # shares of elected mayors that are affiliated with the particular party
        for p in parties:
            container.loc[p, "female_mayor_shares"] = np.round(
                data.loc[data["female_mayor"] == 1, p + "_winner"].sum()
                / data["female_mayor"].sum(),
                3,
            )
            container.loc[p, "male_mayor_shares"] = np.round(
                data.loc[data["female_mayor"] == 0, p + "_winner"].sum()
                / data["male_mayor"].sum(),
                3,
            )

            # Calculate t-test for difference in means
            statistic, p_value = ttest_ind(
                data.loc[data["female_mayor"] == 1, p + "_winner"],
                data.loc[data["male_mayor"] == 1, p + "_winner"],
            )
            container.loc[p, "p_value"] = p_value.round(3)

        # calculate differences in shares of party affiliation between
        # males and females.
        container["difference"] = (
            container["female_mayor_shares"] - container["male_mayor_shares"]
        )

        rslt_dict[bw] = container

    return rslt_dict
