import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import math
import re
import os

def readTargets():
    """
    Reads target values to be used in evaluating error of each optimization step.
    data should be in a csv file, two values per line: [time value]. this scheme
    can be modified easily.

    Returns
    ----------
    times : list
        The list of first elements of each pair in each line of the csv file
    targets : list
        The list of second elements of each pair in each line of the csv file
    """
    times = []
    targets = []
    with open("data.csv", 'r') as f:
        lines = f.readlines()
        for item in lines:
            times.append(float(item.split(',')[0]))
            targets.append(float(item.split(',')[1]))
    return targets, times

def rmse(predictions, targets):
    """
    Computes Root-Mean-Square Error of simulation output and target values.
    the input lists should have the same size.

    Parameters
    ----------
    predictions : list
        Simulation output values
    target : list
        Targets values obtained from external source file

    Returns
    -------
    output : float
        RMS error of two lists
    """
    return np.sqrt(((predictions - targets) ** 2).mean())

def prepareDataframe():
    """
    Extracts all the performed optimization steps (based on the saved simulation output
    files) and computes the error of each.

    Returns
    -------
    output : pandas dataframe
        A dataframe of all optimization steps ad their error and design parameter values
    """
    # defining the dataframe column names
    df = pd.DataFrame(columns = ['Run', 'K1', 'K2', 'DMg', 'DCl', 'Error', 'Output'])
    # assuming the output files are in 'output' directory with the pattern 'result-{}.txt'
    for file in os.listdir('output/'):
        if file.startswith('result-'):
            with open(f'output/{file}', 'r') as f:
                # run number is grabbed from the file name
                run = re.search('result-(.*).txt', file).group(1)
                # value of design parameters of each simulation are stored in first line
                values = f.readline().split('\t')
                # next lines are filled by saved time steps
                lines = f.readlines()
                output = []
                for item in lines:
                    output.append(float(item.split('\t')[1])) # second element of each line
                # if you interrupt the optimization, a simulation is also interrupted,
                # so we will end up with a simulation that has less time data points
                # than expected. so lets check every file
                if (len(output) == len(targetValues)):
                    error = rmse(np.array(output), np.array(targetValues))
                    # create a new row and append it to dataframe
                    df = df.append({'Run': int(run),
                                    'K1':float(values[0]),
                                    'K2':math.log10(float(values[1])), # extra postprocessing
                                    'DMg':float(values[2]),
                                    'DCl':float(values[3]),
                                    'Output': output,
                                    'Error': error}, ignore_index=True)
                else:
                    print("Mismatch detected in the number of simulated data at run #{0}".format(run))
    df = df.sort_values(by=['Run']) # sort the dataframe from the beginning of the optimization
    return df

if __name__ == "__main__":

    targetValues, timeValues = readTargets()
    df = prepareDataframe()

    # saving a separate plot for each optimization step in 'figs' directory.
    # every detail of the plots should be modified according to the application
    for index, row in df.iterrows():
        plt.plot(timeValues, targetValues, 'bo', label='Experimental data')
        plt.plot(timeValues, row['Output'], 'r*', label='Computational data')
        plt.xlabel('Immersion time ($hour$)')
        plt.ylabel('Evolved hydrogen ($ml.cm^{-2}$)')
        # putting the parameters values on the top of the chart
        #TODO: the display format of title should be improved
        plt.suptitle('{:4s} {:3d}   {:7s} {:9.6f}   {:4s} {:3.0f}   {:4s} {:3.0f}\n{:7s} {:9.7f}   {:7s} {:9.7f}'
                     .format("Run:", row['Run'], "Error:", row['Error'], "K1:", row['K1'],
                             "K2:", row['K2'], "DMg:", row['DMg'], "DCl:", row['DCl']),
                     fontsize=16, fontweight='bold', ha='left', x=0.05)
        plt.ylim(-0.1, 2) # adjust y axis range
        plt.legend(loc='upper left')
        plt.grid(True)
        plt.savefig(f"figs/fig-{row['Run']}.png", dpi=200)
        plt.clf()

    # creaing seaborn pair plots for the search space parameters
    sns.set(style="ticks", color_codes=True)
    p = sns.pairplot(df, vars=['K1', 'K2', 'DMg', 'DCl'])
    g = sns.PairGrid(df, vars=['K1', 'K2', 'DMg', 'DCl'])
    g = g.map_upper(plt.scatter, edgecolor="w")
    g = g.map_lower(sns.kdeplot, cmap="Blues_d")
    # g = g.map_diag(sns.kdeplot, shade=True)
    g = g.map_diag(plt.hist, histtype="step", linewidth=3)

    # we can adjust incorrect automatic axis limits here
    p.axes[2,0].set_ylim(-0.001, 0.011)
    g.axes[2,0].set_ylim(-0.001, 0.011)
    g.axes[0,2].set_xlim(-0.001, 0.011)

    # printing the dataframe to the screen as well as a CSV file
    print(df)
    df.to_csv('export_dataframe.csv', index = None, header=True)
    plt.show()
    #TODO: the empty useless figure window should be removed
