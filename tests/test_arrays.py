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
      2019-06-04 DWW
"""

import unittest
import sys
import os

sys.path.append(os.path.abspath('..'))
from grayboxes.arrays import cross, grid, rand, xy_rand_split


class TestUM(unittest.TestCase):


    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test1(self):
        x = grid((3, 4), [0., 1.], [2., 3.])

        x_start = rand(3, [0., 1.], [2., 4.])
        
        x = cross((3, 4), [0., 1.], [2., 3.])

        self.assertTrue(True)

if __name__ == '__main__':
    unittest.main()
