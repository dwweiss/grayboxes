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
      2020-02-03 DWW
"""

__all__ = ['mpi', 'communicator', 'rank', 'predict_scatter', 'split', 'merge']

import numpy as np
import os
import psutil
import sys
from typing import Any, Callable, List, Optional, Union
if os.name == 'posix':
    try:
        import mpi4py
        try:
            import mpi4py.MPI
        except AttributeError:
            print('!!! module mpi4py.MPI not loaded')
    except ImportError:
        print('!!! module mpi4py.MPI not loaded')

try:
    from grayboxes.datatype import Float1D, Float2D, Float3D, None2D
except ImportError:
    try:
        from datatype import Float1D, Float2D, Float3D
    except ImportError:
        print('!!! module datatype not loaded')
        print('    continue with local definition of Float1D, ' +
              'Float2D, Float3D, None2D')        
        Float1D = Optional[np.ndarray]
        Float2D = Optional[np.ndarray]
        Float3D = Optional[np.ndarray]
        None2D = Optional[np.ndarray]
        

"""
    Tools for splitting & merging data sets, and execution of task on 
    multiple cores employing MPI (message passing interface)

    Note:
        Execute python script parallel:
            mpiexec -n 4 python3 foo.py     # ==> 4 processes
    
    References:
        Example for Spawn():
            https://gist.github.com/donkirkby/eec37a7cf25f89c6f259
    
        Example of MpiPoolExecutor
            http://mpi4py.scipy.org/docs/usrman/mpi4py.futures.html
            http://mpi4py.readthedocs.io/en/stable/mpi4py.futures.html
"""


def mpi() -> Optional[int]:
    """
    Gets message passing interface (MPI)

    Returns:
        None if MPI is not available, MPI reference otherwise
    """
    if os.name != 'posix' or 'mpi4py' not in sys.modules:
        return None
    return mpi4py.MPI


def communicator() -> Optional[int]:
    """
    Gets communicator of the message passing interface (MPI)

    Returns:
        None if MPI is not available, communicator otherwise
    """
    x = mpi()
    if x is None:
        return None
    return x.COMM_WORLD


def rank() -> Optional[int]:
    """
    Gets process rank from communicator of MPI

    Returns:
        None if MPI is not available, process rank otherwise
    """
    comm = communicator()
    if comm is None:
        return None
    return comm.Get_rank()


def predict_scatter(f: Callable[..., List[float]], 
                    x: Union[Float2D, Float1D], 
                    *args: float, 
                    **kwargs: Any) -> Union[Float2D, None2D]:
    """
    Parallelizes prediction of model y = f(x) employing scatter() and 
    gather()

    Args:
        f:
            generic function f(x) without 'self' argument

        x:
            prediction input, shape: (n_point, n_inp) or (n_inp,)

        args:
            positional arguments

    Kwargs:
        silent (bool):
            If silent is True then supress printing

    Returns:
        output array, shape: (n_point, n_out)
        or 
        np.array([[None]]) if no MPI
    """
    assert os.name == 'posix', os.name
    assert f is not None
    assert x is not None

    silent = kwargs.get('silent', True)

    comm = communicator()

    if comm is None:
        return np.atleast_2d(np.inf)

    n_proc = comm.Get_size()
    n_core = psutil.cpu_count(logical=False)

    if not silent:
        print('+++ predict_scatter(), rank: ', rank(),
              ' (n_proc: ', n_proc, ', n_core: ', n_core, ')', sep='')

    # splits single 2D input array to a list of 2D sub-arrays
    if rank() == 0:
        x_all = split(x, n_proc)
    else:
        x_all = None

    # distributes groups of 2D inputs to multiple cores
    x_proc = comm.scatter(x_all, 0)

    y_proc = []
    for x_point in x_proc:
        y_point = np.atleast_1d(f(x_point, *args)) if x_point[0] != np.inf \
                                                   else [np.inf]
        y_proc.append(y_point)

    y_all = comm.gather(y_proc, 0)

    # merges array of 2D group outputs to single 2D output array
    y = merge(y_all)
    
    return y


# def predict_subprocess(f: Callable, x: Union[Float2D, Float1D], 
#                        **kwargs: Any) -> Float2D:
#    """
#    Parallelizes prediction of model y = f(x) employing subprocess
#
#    Args:
#        f (file):
#            file with implementation of generic model f(x), see example
#
#        x (2D or 1D array_like of float):
#            input array, shape: (n_point, n_inp)
#
#        kwargs:
#            keyword arguments
#
#    Returns:
#        y (2D array of float):
#            output array, shape: (n_point, n_out)
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
#    # splits 2D input array to list of 2D sub-arrays, 
#    # distributes 2D segments to multiple cores
#    xAll = split(x, nProc)
#    print('xAll:', xAll.shape)
#
#    # distributes 2D input groups to multiple cores
#    yAll = []
#    for xProc in xAll:
#        # executes for every point in group
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


def split(x2d: Union[Float2D, Float1D], n_proc: int) -> Float3D:
    """
    - Fills up given 2D array with 'np.inf' to a size of multiple of 'nProc'
    - Splits the 2D array into an 3D array

    Args:
        x2d:
            input array, shape: (n_proc, n_inp) or (n_inp,)

        n_proc:
            number of x-segments to be sent to multiple processes

    Returns:
        array of 2D arrays, shape: (n_proc, n_point_per_proc, n_inp)
        or 
        np.array([[[np.inf]]]]) if x2d is None
    """
    if x2d is None:
        return np.atleast_3d(np.inf)

    x2d = np.atleast_2d(x2d)
    n_point, n_inp = x2d.shape
    
    if n_point % n_proc != 0:
        n_fill_up = (n_point // n_proc + 1) * n_proc - n_point
        x2d = np.r_[x2d, [[np.inf] * n_inp] * n_fill_up]
        
    return np.array(np.split(x2d, n_proc))


def merge(y3d: Float3D) -> Float2D:
    """
    - Merges output from predictions of all processes to single 2D output array
    - Excludes output points with first element equaling np.inf

    Args:
        y3d:
            Sequence of 2D output arrays, 
            shape: (n_proc, n_point_per_proc, n_out)

    Returns:
        merged output, shape: (n_point, n_out)
        or
        np.aray([[np.inf]]) if y3D is None, shape: (1, 1)
    """
    if y3d is None:
        return np.atleast_2d(np.inf)

    y3d = np.array(y3d)
    if y3d.ndim == 2:
        return y3d

    y2d = []
    for yProc in y3d:
        for yPoint in yProc:
            if yPoint[0] != np.inf:
                y2d.append(yPoint)
    return np.array(y2d)


def x3d_to_str(data: Float3D, indent: Union[str, int] = '    ') -> str:
    """
    Creates string matrix with of MPI input or output

    Args:
        data
            input or output array, shape: (n_proc, n_point_per_proc, n_inp)

        indent:
            indent in string representation if type of indent is string
            or number of spaces used as indent

    Returns:
       string representation of segments of processes and of filled
       up values
    """
    assert data is not None

    if isinstance(indent, int):
        indent = indent * ' '
    s = ''
    for i_proc, y_proc in enumerate(data):
        s += indent + 'process ' + str(i_proc) + ': '
        for y_point in y_proc:
            s += ' [ '
            for val in y_point:
                s += '*' if val != np.inf else '-'
            s += ' ] '
        s += '\n'
    return s


def x_demo(n_point: int = 24, n_inp: int = 2) -> Float2D:
    """
    Args:
        n_point:
            number of data points

        n_inp:
            number of input

    Returns:
        demo input array, shape: (n_point, n_inp)
    """
    return np.array([[i + j for j in range(n_inp)] for i in range(n_point)])
