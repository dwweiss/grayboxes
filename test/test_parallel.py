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
      2018-11-22 DWW
"""

import __init__
__init__.init_path()

import unittest
import numpy as np
import psutil
from typing import Any, List, Optional, Sequence

from grayboxes.parallel import mpi, communicator, predict_scatter, split, \
                               merge, x_demo, x3d_to_str


def f(x: Optional[Sequence[float]], *args: float, **kwargs: Any) \
        -> List[float]:
    for i in range(10*1000):
        _sum = 0
        for j in range(1000):
            _sum += 0.001
    return [x[0] * 2, x[1]**2]


class TestUM(unittest.TestCase):
    def setUp(self):
        pass

        comm = communicator()
        if comm is None:
            self.n_proc = 4
            self.n_core = 2
        else:
            self.n_proc = comm.Get_size()
            self.n_core = psutil.cpu_count(logical=False)
        
        self.n_point, self.nInp = 5, 2
        print('mpi():', mpi())
        print('communicator():', communicator())
        print('n_core n_proc:', self.n_core, self.n_proc)

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
        #        x_proc, kwargs = np.loads(sys.stdin.buffer.read())
        #        y_proc = []
        #        for x in x_proc:
        #            print('x:', x)
        #            if x[0] != np.inf:
        #
        #                # computation of y = f(x)
        #                y = x * 1.1 + 2
        #
        #            y_proc.append(y)
        #        sys.stdout.buffer.write(dumps(y))
        #    else:
        #        x = x_demo()
        #        y = predict_subprocess(__file__, x)
        #
        #        print('x:', 'y:', y)
        self.assertTrue(True)

    def test2(self):
        x = x_demo(self.n_point, self.nInp)
        print('x:', x)

        if communicator() is not None:
            print('+++ predict on muliple cores:',
                  communicator().Get_size())
            y = predict_scatter(f=f, x=x)
        else:
            print('+++ predict on single core')
            y = []
            for x_point in x:
                y_point = f(x=x_point) if x_point[0] != np.inf else np.inf
                y.append(np.atleast_1d(y_point))
            y = np.array(y)

        print('x:', x.tolist())
        print('y:', y.tolist())

        self.assertTrue(True)

    def test3(self):
        s = 'Generates example input'
        print('-' * len(s) + '\n' + s + '\n' + '-' * len(s))
        x = x_demo(self.n_point, self.nInp)
        if self.n_point <= 20:
            print('x:', x.tolist(), '\n')

        s = 'Split input into sequence of input groups for multiple cores'
        print('-' * len(s) + '\n' + s + '\n' + '-' * len(s))
        x_all = split(x, self.n_proc)
        print(x3d_to_str(x_all))
        if self.n_point <= 20:
            print('x_all:', x_all.tolist())

        if communicator() is None:
            s = 'Computes output on single core'
            print('-' * len(s) + '\n' + s + '\n' + '-' * len(s))
            n_all = []
            for x_proc in x_all:
                y_proc = []
                for x_point in x_proc:
                    y_point = f(x_point) if x_point[0] != np.inf else [np.inf]
                    y_proc.append(y_point)
                n_all.append(y_proc)
            n_all = np.array(n_all)
            print(x3d_to_str(n_all))
            if self.n_point <= 20:
                print('n_all:', n_all.tolist(), '\n')

            s = "Merges output from multiple cores and removes 'inf'-rows"
            print('-' * len(s) + '\n' + s + '\n' + '-' * len(s))
            y = merge(n_all)
            if self.n_point <= 20:
                print('x:', x.tolist())
            if self.n_point <= 20:
                print('y:', y.tolist())
        else:
            s = 'Computes output on multiple cores'
            print('-' * len(s) + '\n' + s + '\n' + '-' * len(s))
            y = predict_scatter(f, x, silent=False)

        if self.n_point <= 20:
            print('x:', x.tolist())
            print('y:', y.tolist())

        """ Example output:
        -----------------------
        Generates example input
        -----------------------
        x: [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7],
            [7, 8], [8, 9], [9, 10]]
        ------------------------------------------------------------
        Split input into sequence of input groups for multiple cores
        ------------------------------------------------------------
            core 0:  [ ** ]  [ ** ]  [ ** ]  [ ** ]
            core 1:  [ ** ]  [ ** ]  [ ** ]  [ ** ]
            core 2:  [ ** ]  [ ** ]  [ -- ]  [ -- ]
        x_all: [[[0., 1.], [1., 2.], [2., 3.], [3., 4.]], [[4., 5.],
                [5., 6.], [6., 7.], [7., 8.]], [[8., 9.], [9., 10.],
                [inf, inf], [inf, inf]]]
        ---------------------------------
        Computes output on multiple cores
        ---------------------------------
            core 0:  [ ** ]  [ ** ]  [ ** ]  [ ** ]
            core 1:  [ ** ]  [ ** ]  [ ** ]  [ ** ]
            core 2:  [ ** ]  [ ** ]  [ - ]  [ - ]
        n_all: [[[0., 1.], [2., 4.], [4., 9.], [6., 16.]], [[8., 25.],
                [10., 36.], [12., 49.], [14., 64.]], [[16., 81.],
                [18., 100.], [inf], [inf]]]
        --------------------------------------------------------
        Merges output from multiple cores and removes 'inf'-rows
        --------------------------------------------------------
        x: [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7],
            [7, 8], [8, 9], [9, 10]]
        y: [[0., 1.], [2., 4.], [4., 9.], [6., 16.], [8., 25.],
            [10.0, 36.0], [12.0, 49.0], [14.0, 64.0], [16.0, 81.0],
            [18.0, 100.0]]
        """

        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
