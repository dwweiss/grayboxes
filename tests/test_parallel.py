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
      2018-08-16 DWW
"""

import unittest	
import sys
import os
import numpy as np
import psutil

sys.path.append(os.path.abspath('..'))
from grayboxes.parallel import mpi, communicator, predict_scatter, split, \
    merge, xDemo, x3d_to_str


def f(x, *args, **kwargs):
    for i in range(10*1000):
        sum = 0
        for i in range(1000):
            sum += 0.001
    return [x[0] * 2, x[1]**2]

comm = communicator()
if comm is None:
    nProc = 4
    nCore = 2
else:
    nProc = comm.Get_size()
    nCore = psutil.cpu_count(logical=False)

nPoint, nInp = 5, 2
print('mpi():', mpi())
print('communicator():', communicator())
print('nCore nProc:', nCore, nProc)


class TestUM(unittest.TestCase):
    def setUp(self):
        pass
 
    def tearDown(self):
        pass
 
    def test1(self):
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
        self.assertTrue(True)


    def test2(self):
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

        self.assertTrue(True)

    def test3(self):
        s = 'Generates example input'
        print('-' * len(s) + '\n' + s + '\n' + '-' * len(s))
        x = xDemo(nPoint, nInp)
        if nPoint <= 20:
            print('x:', x.tolist(), '\n')

        s = 'Split input into sequence of input groups for multiple cores'
        print('-' * len(s) + '\n' + s + '\n' + '-' * len(s))
        xAll = split(x, nProc)
        print(x3d_to_str(xAll))
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
            print(x3d_to_str(yAll))
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

        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
