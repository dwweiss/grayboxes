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
      2019-11-22 DWW
"""

import __init__
__init__.init_path()

import unittest

from grayboxes.xyz import xyz, xyzt


class TestUM(unittest.TestCase):
    """
    Test of point in 3D space
    """
    def setUp(self):
        pass


    def tearDown(self):
        pass


    def test1(self):
        P0 = xyz(2.2, -1)
        print('P0:', P0)
        P1 = xyz(x=1, z=4)
        print('P1:', P1)
        P2 = xyz(point=P1)
        print('P2:', P2)
        try:
            P3 = xyz(point=[])                        # invalid
        except:
            print('invalid P3:', P3)

        print('P0.at(1)=P0.y:', P0.at(1))
        print('P0, P1:', P0, P1)
        print('P0 + 1:', P0 + 1)
        print('P0 + P1:', P0 + P1)
        print('P0 - 1:', P0 - 1)
        print('P0 - P1:', P0 - P1)
        print('P0 * 2:', P0 * 2)
        print('P0 * (1/2.):', P0 * (1/2.))
        print('P0 * P1:', P0 * P1)

        self.assertTrue(True)


    def test2(self):
        P1 = xyz(x=1, z=4)
        P4 = xyzt(2.2, -1, t=7)
        print('P4:', P4)
        P5 = xyzt(point=P1)
        print('P5:', P5)
        P6 = xyzt(point=P4)
        print('P6:', P6)
        try:
            P7 = xyzt(point={'a': 1, 'b': 2})         # invalid
        except:
            print('Invalid P7:', P7)
        print('P7:', P7)

        self.assertTrue(P5 == P1)
        self.assertFalse(P5 != P1)
        self.assertTrue(P7.x is None)
   

if __name__ == '__main__':
    unittest.main()
