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
      2018-05-21 DWW
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


"""
Note:
    Execute python script parallel:
        mpiexec -n 4 python3 foo.py     # ==> 4 processe

References:
    Example for Spawn():
        https://gist.github.com/donkirkby/eec37a7cf25f89c6f259

    Example of MpiPoolExecutor
        http://mpi4py.scipy.org/docs/usrman/mpi4py.futures.html
        http://mpi4py.readthedocs.io/en/stable/mpi4py.futures.html
"""


def mpi():
    """
    Gets message passing interface (MPI)

    Returns:
        (int or None):
            None if MPI is not available, MPI reference otherwise
    """
    if os.name != 'posix' or 'mpi4py' not in sys.modules:
        return None
    return mpi4py.MPI


def communicator():
    """
    Gets communicator of the message passing interface (MPI)

    Returns:
        (int or None):
            None if MPI is not available, communicator otherwise
    """
    x = mpi()
    if x is None:
        return None
    return x.COMM_WORLD


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


def predict_scatter(f, x, *args, **kwargs):
    """
    Parallelizes prediction of model y = f(x) employing scatter() and gather()

    Args:
        f (function):
            generic function f(x) without 'self' argument

        x (2D or 1D array_like of float):
            prediction input, shape: (nPoint, nInp)

        args (argument list, optional):
            positional arguments

        kwargs (dict, optional):
            keyword arguments

            silent (bool):
            If silent is True then supress printing

    Returns:
        y (2D array of float):
            output array, shape: (nPoint, nOut)
            or np.array((None)) if no MPI
    """
    assert os.name == 'posix', os.name
    assert f is not None
    assert x is not None

    silent = kwargs.get('silent', True)

    comm = communicator()

    if comm is None:
        return np.atleast_2d(np.inf)

    nProc = comm.Get_size()
    nCore = psutil.cpu_count(logical=False)

    if not silent:
        print('+++ predictParallel(), rank: ', rank(),
              ' (nProc: ', nProc, ', nCore: ', nCore, ')', sep='')

    # splits single 2D input array to groups of 2D inputs (to array of 2D arr.)
    if rank() == 0:
        xAll = split(x, nProc)
    else:
        xAll = None

    # distributes groups of 2D inputs to multiple cores
    xProc = comm.scatter(xAll, 0)

    yProc = []
    for xPoint in xProc:
        yPoint = np.atleast_1d(f(xPoint, *args)) if xPoint[0] != np.inf \
                                                 else [np.inf]
        yProc.append(yPoint)

    yAll = comm.gather(yProc, 0)

    # merges array of 2D group outputs to single 2D output array
    y = merge(yAll)
    return y


# def predict_subprocess(f, x, **kwargs):
#    """
#    Parallelizes prediction of model y = f(x) employing subprocess
#
#    Args:
#        f (file):
#            file with implementation of generic model f(x), see example
#
#        x (2D or 1D array_like of float):
#            input array, shape: (nPoint, nInp)
#
#        kwargs (dict, optional):
#            keyword arguments
#
#    Returns:
#        y (2D array of float):
#            output array, shape: (nPoint, nOut)
#            or np.array((None)) if no MPI
#
#    Example of file 'f':
#
#        if __name__ == '__main__':
#            xProc, kwargs = np.loads(sys.stdin.buffer.read())
#            for x in xProc:
#                if x[0] != np.inf:
#                    #
#                    # computation of y = f(x)
#                    y = x * 1.1 + 2
#                    #
#                yProc.append(y)
#            sys.stdout.buffer.write(pickle.dumps(y))
#    """
#    assert f is not None
#    assert x is not None
#
#    silent = kwargs.get('silent', False)
#    kw = kwargs.copy()
#    if 'x' in kw:
#        del kw['x']
#
#    nCore = psutil.cpu_count(logical=False)
#    nProc = nCore - 1
#
#    if not silent:
#        print('+++ predict_subprocess(), nCore:', nCore)
#
#    # splits 2D input to 3D input, distributes 2D segments to multiple cores
#    xAll = split(x, nProc)
#    print('xAll:', xAll.shape)
#
#    # distributes 2D input groups to multiple cores
#    yAll = []
#    for xProc in xAll:
#        # excutes for every point in group
#        print('xProc:', xProc)
#        yProc = loads(subprocess.check_output(
#                [sys.executable, f, 'subprocess'],
#                input=dumps([xProc, kwargs])))
#        print('yProc:', yProc)
#
#        # collects 2D output segments from multiple cores in 3D output array
#        yAll.append(yProc)
#
#    # merges 3D output array to single 2D output array
#    y = merge(yAll)
#    return y


def split(x2D, nProc):
    """
    - Fills up given 2D array with 'np.inf' to a size of multiple of 'nProc'
    - Splits the 2D array into an 3D array

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

    y3D = np.array(y3D)
    if y3D.ndim == 2:
        return y3D

    y2D = []
    for yProc in y3D:
        for yPoint in yProc:
            if yPoint[0] != np.inf:
                y2D.append(yPoint)
    return np.array(y2D)


def x3D2str(data, indent='    '):
    """
    Creates string matrix with of MPI input or output

    Args:
        data (3D array_like of float):
            input or output array, shape=(nProc, nPointPerProc, nInp)

        indent (str or int):
            indent in string description as string or indent * ' '

    Returns:
        (str):
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
    ALL = 0

    #    if 'worker' in sys.argv:
    #
    #        def f(x, **kwargs):
    #            for i in range(10*1000):
    #                sum = 0
    #                for i in range(1000):
    #                    sum += 0.001
    #            return [x[0] * 2, x[1]**2]
    #
    #
    #        xProc, kwargs = np.loads(sys.stdin.buffer.read())
    #        yProc = []
    #        for x in xProc:
    #            print('x:', x)
    #            if x[0] != np.inf:
    #
    #                # computation of y = f(x)
    #                y = x * 1.1 + 2
    #
    #            yProc.append(y)
    #        sys.stdout.buffer.write(dumps(y))
    #    else:
    #        x = xDemo()
    #        y = predict_subprocess(__file__, x)
    #
    #        print('x:', 'y:', y)

    if 1 or ALL:
        def f(x, *args, **kwargs):
            for i in range(10*1000):
                sum = 0
                for i in range(1000):
                    sum += 0.001
            return [x[0] * 2, x[1]**2]

        comm = communicator()
        if comm is None:
            nProc = 4
        else:
            nProc = comm.Get_size()
            nCore = psutil.cpu_count(logical=False)

        nPoint, nInp = 5, 2
        print('mpi():', mpi())
        print('communicator():', communicator())
        print('nCore nProc:', nCore, nProc)

    #######################################################################

    if 1 or ALL:
        x = xDemo(nPoint, nInp)
        print('x:', x)

        if communicator() is not None:
            print('+++ predict on muliple cores:',
                  communicator().Get_size())
            y = predict_scatter(f=f, x=x)
        else:
            print('+++ predict on single core')
            y = []
            for xPoint in x:
                yPoint = f(x=xPoint) if xPoint[0] != np.inf else np.inf
                y.append(np.atleast_1d(yPoint))
            y = np.array(y)

        print('x:', x.tolist())
        print('y:', y.tolist())

    #######################################################################

    # TODO remove code below after module test

    if 0 or ALL:
        s = 'Generates example input'
        print('-' * len(s) + '\n' + s + '\n' + '-' * len(s))
        x = xDemo(nPoint, nInp)
        if nPoint <= 20:
            print('x:', x.tolist(), '\n')

        s = 'Split input into sequence of input groups for multiple cores'
        print('-' * len(s) + '\n' + s + '\n' + '-' * len(s))
        xAll = split(x, nProc)
        print(x3D2str(xAll))
        if nPoint <= 20:
            print('xAll:', xAll.tolist())

        if communicator() is None:
            s = 'Computes output on single core'
            print('-' * len(s) + '\n' + s + '\n' + '-' * len(s))
            yAll = []
            for xProc in xAll:
                yProc = []
                for xPoint in xProc:
                    yPoint = f(xPoint) if xPoint[0] != np.inf else [np.inf]
                    yProc.append(yPoint)
                yAll.append(yProc)
            yAll = np.array(yAll)
            print(x3D2str(yAll))
            if nPoint <= 20:
                print('yAll:', yAll.tolist(), '\n')

            s = "Merges output from multiple cores and removes 'inf'-rows"
            print('-' * len(s) + '\n' + s + '\n' + '-' * len(s))
            y = merge(yAll)
            if nPoint <= 20:
                print('x:', x.tolist())
            if nPoint <= 20:
                print('y:', y.tolist())
        else:
            s = 'Computes output on multiple cores'
            print('-' * len(s) + '\n' + s + '\n' + '-' * len(s))
            y = predict_scatter(f, x, silent=False)

        if nPoint <= 20:
            print('x:', x.tolist())
            print('y:', y.tolist())

        """ Example output:
        -----------------------
        Generates example input
        -----------------------
        x: [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8],
            [8, 9], [9, 10]]
        ------------------------------------------------------------
        Split input into sequence of input groups for multiple cores
        ------------------------------------------------------------
            core 0:  [ ** ]  [ ** ]  [ ** ]  [ ** ]
            core 1:  [ ** ]  [ ** ]  [ ** ]  [ ** ]
            core 2:  [ ** ]  [ ** ]  [ -- ]  [ -- ]
        xAll: [[[0.0, 1.0], [1.0, 2.0], [2.0, 3.0], [3.0, 4.0]], [[4.0, 5.0],
                [5.0, 6.0], [6.0, 7.0], [7.0, 8.0]], [[8.0, 9.0], [9.0, 10.0],
                [inf, inf], [inf, inf]]]
        ---------------------------------
        Computes output on multiple cores
        ---------------------------------
            core 0:  [ ** ]  [ ** ]  [ ** ]  [ ** ]
            core 1:  [ ** ]  [ ** ]  [ ** ]  [ ** ]
            core 2:  [ ** ]  [ ** ]  [ - ]  [ - ]
        yAll: [[[0.0, 1.0], [2.0, 4.0], [4.0, 9.0], [6.0, 16.0]], [[8.0, 25.0],
                [10.0, 36.0], [12.0, 49.0], [14.0, 64.0]], [[16.0, 81.0],
                [18.0, 100.0], [inf], [inf]]]
        --------------------------------------------------------
        Merges output from multiple cores and removes 'inf'-rows
        --------------------------------------------------------
        x: [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8],
            [8, 9], [9, 10]]
        y: [[0.0, 1.0], [2.0, 4.0], [4.0, 9.0], [6.0, 16.0], [8.0, 25.0],
            [10.0, 36.0], [12.0, 49.0], [14.0, 64.0], [16.0, 81.0],
            [18.0, 100.0]]
        """
