import os
import time
from optim import denormalize

number = 0 # used to hold current optimization step in order to name output file

def call(k1, k2, DMg, DCl):
    """
    Calls the main simulation based on the input values and returns a list of
    simulation output. The list used to calculate the error of the objective
    function. In addition, this function records the runtime duration of
    each simulation into a text file (time.txt).

    Parameters
    ----------
    All the parameters we want to estimate. Here as an example, we have four parameters
    passed to the finite element code.

    Returns
    -------
    output : list
        A list containing all the output points of the performed simulation
    """
    global number
    # the parameters are normalized. we should denormalized them first
    k1 = denormalize('k1', k1)
    k2 = 10**denormalize('k2', k2) # any additional preprocessing can be performed here
    DMg = denormalize('dmg', DMg)
    DCl = denormalize('dcl', DCl)
    number += 1
    start = time.time()
    # here is the main call to the external simulation. it can be any internal or external code,
    # but it should store the output in an expected manner that can be read by processData().
    # the following line is an example of passing the parameters using command line arguments,
    # but it can be done in any other method such as writing the parameters to a file and asking
    # the simulation code to read them.
    os.system("mpirun -n 8 FreeFem++-mpi ff/mg_3D.edp -num {0} -k1 {1} -k2 {2} -dmg {3} -dcl {4} > output/output-{0}.txt".format(number, k1, k2, DMg, DCl))
    # record elapsed time
    f = open("output/time.txt", "a+")
    f.write("run={0}: {1:.0f} seconds\r\n".format(number, time.time() - start))
    f.close()
    # return simulation output values
    return processData()

def processData():
    """
    Reads simulation output file according to the current optimization step number
    and returns the values as a list. The files' name pattern should be adjusted.
    The first line of the file is also ignored because it contains the current value
    of our paramters. The next lines should be a pair of [time value], so the second
    element of each line will be read to be used in evaluation of objective function.

    Returns
    -------
    output : list
        A list containing all the output points of the performed simulation
    """
    global number
    items = []
    # adjust the file name pattern. here, each step is saved as result-{step}.txt
    # in our finite element code.
    with open("output/result-{0}.txt".format(number), 'r') as f:
        lines = f.readlines()
        for item in lines[1:]: # ignoring the first line
            # reading the second element of each line. the first one is time
            items.append(float(item.split()[1]))
    return items
