"""
  Copyright (c) 2016- by Dietmar W Weiss

  This is free software; you can redistribute it and/or modify it
  under the terms of the GNU Lesser General Public License as
  published by the Free Software Foundation; either version 3.0 of
  the License, or (at your option) any later version.
  This software is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
  Lesser General Public License for more details.
  You should have received a copy of the GNU Lesser General Public
  License along with this software; if not, write to the Free
  Software Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA
  02110-1301 USA, or see the FSF site: http://www.fsf.org.

  Version:
      2018-03-15 DWW
"""

import numpy as np
import psutil
import sys
import os
if os.name == 'posix':
    try:
        import mpi4py
        try:
            import mpi4py.MPI
        except AttributeError:
            print('!!! mpi4py.MPI import failed')
    except ImportError:
        print('!!! mpi4py import failed')


def communicator():
    """
    Gets communicator of the message passing interface (MPI)

    Returns:
        (int or None):
            None if MPI is not available, communicator otherwise
    """
    if os.name != 'posix' or 'mpi4py' not in sys.modules:
        return None
    return mpi4py.MPI.COMM_WORLD


def rank():
    """
    Gets process rank from communicator of message passing interface (MPI)

    Returns:
        (int or None):
            None if MPI is not available, process rank otherwise
    """
    comm = communicator()
    if comm is None:
        return None
    return comm.Get_rank()


def predict(f, x, **kwargs):
    """
    Parallelizes prediction of model y = f(x)

    Args:
        f (method):
            generic model f(x)

        x (2D or 1D array_like of float):
            input array, shape: (nPoint, nInp)

        kwargs (dict, optional):
            keyword arguments

    Returns:
        y (2D array of float):
            output array, shape: (nPoint, nOut)
            or np.array((None)) if no MPI
    """
    assert f is not None
    assert x is not None

    silent = kwargs.get('silent', False)
    kw = kwargs.copy()
    if 'x' in kw:
        del kw['x']

    comm = communicator()

    if comm is None:
        return np.atleast_2d(np.inf)

#        Computation on single core
#        y = []
#        for xPoint in x:
#            yPoint = f(xPoint) if xPoint[0] != np.inf else np.inf
#            y.append(np.atleast_1d(yPoint))
#        y = np.array(y)

    nProc = comm.Get_size()
    nCore = psutil.cpu_count(logical=False)

    if not silent:
        print('+++ predictParallel(), rank: ', rank(),
              ' (nProc: ', nProc, 'nCore: ', nCore, ')', sep='')

    # splits 2D input to 3D input and distributes 2D segments to multiple cores
    if rank() == 0:
        xAll = split(x, nProc)
    else:
        xAll = None

    # distributes 2D input segments to multiple cores
    xProc = comm.scatter(xAll, 0)

    yProc = []
    for xPoint in xProc:
        yPoint = f(xPoint, **kw) if xPoint[0] != np.inf else [np.inf]
        yProc.append(yPoint)

    # collects 2D output segments from multiple cores in 3D output array
    yAll = comm.gather(yProc, 0)

    # merges 3D output array to single 2D output array
    y = merge(yAll)
    return y


def split(x2D, nProc):
    """
    - Fills up the 2D array with 'np.inf' to a size of multiple of 'nProc'
    - Splits the 2D array into an array of 2D sub-arrays of equal division

    Args:
        x (2D or 1D array_like of float):
            input array, shape: (nProc, nInp) or (nInp)

        nProc (int):
            number of x-segments to be sent to multiple processes

    Returns:
        (3D array of float):
            array of 2D arrays, shape: (nProc, nPointPerProc, nInp)
    """
    if x2D is None:
        return np.atleast_3d(np.inf)

    x2D = np.atleast_2d(x2D)
    nPoint, nInp = x2D.shape
    if nPoint % nProc != 0:
        nFillUp = (nPoint // nProc + 1) * nProc - nPoint
        x2D = np.r_[x2D, [[np.inf] * nInp] * nFillUp]
    return np.array(np.split(x2D, nProc))


def merge(y3D):
    """
    - Merges output from predictions of all processes to single 2D output array
    - Excludes output points with first element equaling 'np.inf'

    Args:
        y3D (3D array of float):
            output array, shape: (nProc, nPointPerProc, nOut)

    Returns:
        (2D array of float):
            array of output, shape: (nPoint, nOut)
            or np.atleast_2d(np.inf) if y3D is None
    """
    if y3D is None:
        return np.atleast_2d(np.inf)

    y2D = []
    for yProc in y3D:
        for yPoint in yProc:
            if yPoint[0] != np.inf:
                y2D.append(yPoint)
    return np.array(y2D)


def x3D_str(data, indent='    '):
    """
    Creates string matrix with of MPI input or output

    Args:
        data (3D array_like of float):
            input or output array, shape: (nProc, nPointPerProc, nInp)

        indent (string or int):
            indent in string description as string or indent * ' '

    Returns:
        (string):
            string representation of segment of process and of filled up values
    """
    assert data is not None

    if isinstance(indent, int):
        indent = int * ' '
    s = ''
    for iProc, yProc in enumerate(data):
        s += indent + 'process ' + str(iProc) + ': '
        for yPoint in yProc:
            s += ' [ '
            for val in yPoint:
                s += '*' if val != np.inf else '-'
            s += ' ] '
        s += '\n'
    return s


def xDemo(nPoint=24, nInp=2):
    """
    Args:
        nPoint (int, optional):
            number of data points

        nInp (int, optional):
            number of input

    Returns:
        (2D array of float):
            demo input array, shape: (nPoint, nInp)
    """
    return np.array([[i+j for j in range(nInp)] for i in range(nPoint)])


# Examples ####################################################################

if __name__ == '__main__':
    def f(x, **kwargs):
        for i in range(10*1000):
            sum = 0
            for i in range(1000):
                sum += 0.001
        return [x[0] * 2, x[1]**2]

    if 1:
        nPoint, nInp = 500, 2
        x = xDemo(nPoint, nInp)
        print('x:', x.tolist())

        if communicator() is not None:
            print('+++ predict on muliple cores:', communicator().Get_size())
            y = predict(f=f, x=x)
        else:
            print('+++ predict on single core')
            y = []
            for xPoint in x:
                yPoint = f(x=xPoint) if xPoint[0] != np.inf else np.inf
                y.append(np.atleast_1d(yPoint))
            y = np.array(y)

        print('y:', y.tolist())
        
