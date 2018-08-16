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
from grayboxes.plotarrays import plotIsoMap
from grayboxes.model import grid, cross
from grayboxes.forward import Forward

sys.path.append(os.path.abspath('..'))
from grayboxes.white import White


def fUser(self, x, *args):
    c0, c1, c2, c3 = args if len(args) > 0 else np.ones(4)
    x0, x1 = x[0], x[1]
    y0 = c0 + c1 * np.sin(c2 * x0) + c3 * (x1 - 1.5)**2
    return [y0]

x = grid((8, 8), [-1, 8], [0, 3])


class TestUM(unittest.TestCase):
    def setUp(self):
        pass
 
    def tearDown(self):
        pass
 
    def test1(self):
        s = 'White box (expanded)'
        print('-' * len(s) + '\n' + s + '\n' + '-' * len(s))

        model = White(fUser)
        y = model(x=x)

        plotIsoMap(x[:, 0], x[:, 1], y[:, 0])

        self.assertTrue(True)

    def test2(self):
        s = 'White box (compact)'
        print('-' * len(s) + '\n' + s + '\n' + '-' * len(s))

        y = White(fUser)(x=x)

        plotIsoMap(x[:, 0], x[:, 1], y[:, 0])

        self.assertTrue(True)

    def test3(self):
        s = 'Forward operator on White box model'
        print('-' * len(s) + '\n' + s + '\n' + '-' * len(s))

        model = White(fUser)
        x, y = Forward(model)(x=grid(8, [-1, 8], [0, 3]))

        plotIsoMap(x[:, 0], x[:, 1], y[:, 0])

        self.assertTrue(True)

    def test4(self):
        s = 'Forward operator on demo White box'
        print('-' * len(s) + '\n' + s + '\n' + '-' * len(s))

        x, y = Forward(White('demo'))(x=cross(9, [-1, 8], [0, 3]))

        plotIsoMap(x[:, 0], x[:, 1], y[:, 0])

        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
