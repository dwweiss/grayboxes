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
      2019-12-09 DWW
"""

import initialize
initialize.set_path()

import numpy as np
from typing import List, Optional, Iterable
import unittest

from grayboxes.array import rand
from grayboxes.maximum import Maximum
from grayboxes.white import White


# user defined method with theoretical submodel
def f(self, x: Optional[Iterable[float]], *c: float) -> List[float]:
    c0, c1, c2 = c if len(c) >= 3 else 1., 1., 1.
    return -(np.sin(c0 * x[0]) + c1 * (x[1] - 1)**2 + c2)


class TestUM(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test1(self):
        s = 'Maximum, assigns series of initial x'
        print('-' * len(s) + '\n' + s + '\n' + '-' * len(s))

        op = Maximum(White(f), 'test1')
        x, y = op(x=rand(10, [-5, 5], [-7, 7]), optimizer='nelder-mead')

        op.plot()
        print('x:', x, 'y:', y, '\nop.x:', op.x, 'op.y:', op.y)

        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
