import numpy as np
from hyperopt import hp, tpe, fmin, Trials
import pickle
import process

targetValues = [] # target values. will be read from a CSV file

# the real range of the design parameters, which will be used to denormalize
# the normalized value of paramters. the names should match the space elements names
# defined in the 'space' variable. these names are used all over the code.
ranges = {
    'k1': (1, 10),
    'k2': (6, 15),
    'dmg': (1e-4, 1e-2),
    'dcl': (1e-3, 6e-1)
}

def normalize(name, value):
    """
    Normalizes input into [0 1] based on the real range defined

    Parameters
    ----------
    name : string
        The name of the variable. should match the name in 'ranges'
    value : float
        The value to be normalized

    Returns
    -------
    output : float
        Normalized value
    """
    return (value - ranges[name][0])/(ranges[name][1] - ranges[name][0])

def denormalize(name, value):
    """
    Denormalizes input from [0 1] based on the real range defined

    Parameters
    ----------
    name : string
        The name of the variable. should match the name in 'ranges'
    value : float
        The value to be denormalized

    Returns
    -------
    output : float
        Denormalized value
    """
    return value * (ranges[name][1] - ranges[name][0]) + ranges[name][0]

def readTargets():
    """
    Reads target values to be used in evaluating error of each optimization step.
    data should be in a CSV file, two values per line: [time value]. this scheme
    can be modified easily.
    """
    with open("data.csv", 'r') as f:
        lines = f.readlines()
        for item in lines:
            targetValues.append(float(item.split(',')[1]))

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

def objective(input):
    """
    Main objective function, constructed based on the RMS error between simulated values
    and target values. It first calls the finite element simulation and then computes the
    error based on the generated output. This error is used to choose the next suggest of
    the design parameters.

    Parameters
    ----------
    input : dictionary
        Current value of design paramters, determined by the bayesian algorithm

    Returns
    -------
    output : float
        RMS error of current evaluation of objective function
    """
    # names should match the names defined in 'space' variable
    output = process.call(input['k1'], input['k2'], input['dmg'], input['dcl'])
    return rmse(np.array(output), np.array(targetValues))

if __name__=="__main__":
    # the main search space. we use normalized and uniform values for all parameters,
    # but they can be any kind of supported spaces (such as choice or lognormal)
    # refer to (https://github.com/hyperopt/hyperopt/wiki/FMin) for more info on this.
    space = {'k1': hp.uniform('k1', 0, 1),
            'k2': hp.uniform('k2', 0, 1),
            'dmg': hp.uniform('dmg', 0, 1),
            'dcl': hp.uniform('dcl', 0, 1)}

    tpe_algo = tpe.suggest # it can be changed to stochastic random search

    # although we have our own result processing code, the trials are also saved in pickle format
    tpe_trials = Trials()

    readTargets()

    # max_evals should be tuned
    tpe_best = fmin(fn=objective, space=space, algo=tpe_algo, trials=tpe_trials,
                    max_evals=400)

    # dumping the pickles. they are not used in current state of the code
    pickle.dump(tpe_best, open("best.pickle", "wb"))
    pickle.dump(tpe_trials, open("trials.pickle", "wb"))
