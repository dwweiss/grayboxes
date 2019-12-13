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
      2019-12-10 DWW
"""

import initialize
initialize.set_path()

import unittest
import numpy as np
from typing import List, Optional, Sequence

from grayboxes.sensitivity import Sensitivity
from grayboxes.array import cross
from grayboxes.white import White


def f(self, x: Optional[Sequence[float]], *c: float) -> List[float]:
    return np.sin(x[0]) + (x[1] - 1)**2


class TestUM(unittest.TestCase):
    """
    Test sensitivity matrix calculation
    """
    def setUp(self):
        pass


    def tearDown(self):
        pass


    def test1(self):
        s = 'Sensitivity with method f(self, x)'
        print('-' * len(s) + '\n' + s + '\n' + '-' * len(s))

        x_ref, dy_dx = Sensitivity(White(f))(x=cross((3, 3), [2, 3], [3, 4]))
        if dy_dx.shape[0] == 1 or dy_dx.shape[1] == 1:
            dy_dx = dy_dx.tolist()

        print('dy_dx:', dy_dx)
        print('x_ref:', x_ref)

        self.assertTrue(True)


    def test2(self):
        s = 'Sensitivity with demo function'
        print('-' * len(s) + '\n' + s + '\n' + '-' * len(s))

        operation = Sensitivity(White(f='demo'))
        x_ref, dy_dx = operation(x=cross((3, 3, 3), [2, 3], [3, 4], [4, 5]))
        if dy_dx.shape[0] == 1 or dy_dx.shape[1] == 1:
            dy_dx = dy_dx.tolist()
            
        print('dy_dx:', dy_dx)
        print('x_ref:', x_ref)

        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
