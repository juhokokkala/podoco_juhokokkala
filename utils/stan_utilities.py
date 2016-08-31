###############################################################################
# Copyright (C) 2016 Juho Kokkala
# This is part of Juho Kokkala's PoDoCo project.
#
# This file is licensed under the MIT License.
###############################################################################
"""
Tools for writing data into the format expected by the Stan models and for
reading CmdStan output.

NB: PyStan (which I have not tried) would probably be better for this and work
for more general purposes than only the present models. I wrote my own tools
only due to licensing issues (PyStan is GPL).

TODO: pandas dataframes would be useful for this...
"""

import csv
import numpy as np


def read_output(file, thin=1):
    """
    Parses variable names and samples from CmdStan output csv.

    Arguments:
    file -- string, the file name
    thin -- int, only every thin:th row is taken

    Returns:
    header -- list of strings containing the variable names
    data -- np.ndarray, #TODO: Check the dimensions of this

    """

    csvfile = open(file, 'rt')
    csvreader = csv.reader(csvfile, delimiter=',')

    headerfound = False
    data = []
    header = []

    iters = 0

    keepiterating = True
    while keepiterating:
        # print(str(iters)+", "+str(len(data)))
        try:
            row = csvreader.__next__()
            if len(row) == 0 or row[0][0] is '#':
                pass

            elif not headerfound:
                header = row
                headerfound = True

            else:
                iters += 1
                if iters % thin == 0:
                    data.append([float(i) for i in row])

        except StopIteration:
            keepiterating = False

        except:  # Some erroneous row in the file
            print("Ignoring an erroneous row")

    data = np.array(data)
    return header, data


def write_incoming_standata(outfilename, dt, y_ic):
    """Write stan-datadump for incoming data

    Arguments:
    outfilename -- string, the file name
    dt         .--.float, positive, the length of the time interval
....y_ic       .--.numpy array of ints, nonnegative, shape NxD, incoming
                   traffic counts for (interval,day)
    """

    N, D = y_ic.shape

    outfile = open(outfilename, 'w')

    outfile.write('N<-'+str(N)+'\n')
    outfile.write('D<-'+str(D)+'\n')
    outfile.write('dt<-'+str(dt)+'\n')

    outfile.write('y_ic<-structure(c(')
    for k in range(N):
        for d in range(D):
            outfile.write(str(y_ic[k, d]))
            if k+d is not N+D-2:
                outfile.write(',')
    outfile.write('),.Dim = c('+str(D)+','+str(N)+'))\n')

    outfile.close()


def write_3component_standata(outfilename, dt, y_ic, y_if, y_og):
    """Write stan-datadump for incoming+interfloor+outgoing data
    Arguments:
    outfilename -- string, the file name
    dt         .--.float, positive, the length of the time interval
....y_ic       .--.numpy array of ints, nonnegative, shape NxD, incoming
                   traffic counts for (interval,day)
....y_if       .--.numpy array of ints, nonnegative, shape NxD, interfloor
                   traffic counts for (interval,day)
....y_og       .--.numpy array of ints, nonnegative, shape NxD, outgoing
                   traffic counts for (interval,da
    """

    N, D = y_ic.shape

    outfile = open(outfilename, 'w')

    outfile.write('N<-'+str(N)+'\n')
    outfile.write('D<-'+str(D)+'\n')
    outfile.write('dt<-'+str(dt)+'\n')

    outfile.write('y_ic<-structure(c(')
    for k in range(N):
        for d in range(D):
            outfile.write(str(y_ic[k, d]))
            if k+d is not N+D-2:
                outfile.write(',')
    outfile.write('),.Dim = c('+str(D)+','+str(N)+'))\n')

    outfile.write('y_if<-structure(c(')
    for k in range(N):
        for d in range(D):
            outfile.write(str(y_if[k, d]))
            if k+d is not N+D-2:
                outfile.write(',')
    outfile.write('),.Dim = c('+str(D)+','+str(N)+'))\n')

    outfile.write('y_og<-structure(c(')
    for k in range(N):
        for d in range(D):
            outfile.write(str(y_og[k, d]))
            if k+d is not N+D-2:
                outfile.write(',')
    outfile.write('),.Dim = c('+str(D)+','+str(N)+'))\n')

    outfile.close()
