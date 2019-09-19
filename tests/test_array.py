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
      2019-09-18 DWW
"""

import __init__
__init__.init_path()

import unittest
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import grayboxes.array as arr


class TestUM(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test1(self):
        x = arr.grid((3, 4), [0., 1.], [2., 3.])
        print('x:', x)
        plt.title('array.grid()')
        plt.plot(x.T[0], x.T[1], 'o')
        plt.show()
        plt.title('array.grid()')
        plt.plot(x.T[0], x.T[1], '-o')
        plt.show()

        self.assertTrue(True)

    def test2(self):
        x = arr.rand(3, [0., 1.], [2., 4.])
        print('x:', x)
        plt.title('array.rand()')
        plt.plot(x.T[0], x.T[1], 'o')
        plt.show()
        
        self.assertTrue(True)

    def test3(self):
        x = arr.cross((3, 4), [0., 1.], [2., 3.])
        print('x:', x)
        plt.title('array.cross()')
        plt.plot(x.T[0], x.T[1], 'o')
        plt.show()
        plt.title('array.cross()')
        plt.plot(x.T[0], x.T[1], '-o')
        plt.show()

        self.assertTrue(True)

    def test4(self):
        x = arr.rand(100, [0., 1.], [2., 4.])
        x_split = arr.xy_rand_split(x, fractions=[0.8, 0.2])
        x_train = x_split[0][0]
        x_test = x_split[0][1]

        print('x_test:', x_test)
        print('x_train:', x_train)
        
        plt.title('array.xy_rand_split()')
        plt.plot(x.T[0], x.T[1], 'o', label='all')
        plt.legend()
        plt.show()        
        plt.title('array.xy_rand_split()')
        plt.plot(x_train.T[0], x_train.T[1], 'x', label='train')
        plt.plot(x_test.T[0], x_test.T[1], 'o', label='test')
        plt.legend()
        plt.show()
 
        self.assertTrue(True)

    def test5(self):
        x = np.linspace(0., 1., 100)
        y = np.sin(x * 2 * np.pi)
        
        print('x:', x, 'y:', y)
        
        x_thin, y_thin = arr.xy_thin_out(x, y, bins=32) 
        
        plt.title('array.xy_thin_out()')
        plt.plot(x, y, 'x', label='all')
        plt.plot(x_thin, y_thin, 'o', label='thin')
        plt.legend()
        plt.show()
        
        plt.title('array.xy_thin_out()')        
        plt.bar(x_thin, y_thin, width=(x_thin[1] - x_thin[0]) * 0.66, 
                align='edge')
        plt.plot(x, y, '-', label='all')
        plt.legend()
        plt.show()        
 
        self.assertTrue(True)

    def test6(self):
        print(arr.ensure2D(np.array([5])))
        print(arr.ensure2D(np.array([2, 3, 4, 5])))
        print(arr.ensure2D(np.array([[2, 3, 4, 5]])))
        print(arr.ensure2D(np.array([[2, 3, 4, 5]]).T))
        print('------')
        print(arr.ensure2D(np.array([2, 3, 4, 5]).T, np.array([2, 3, 4, 5])))
        print('------')
        print(arr.ensure2D(pd.core.series.Series([2, 3, 4, 5])))
        print(arr.ensure2D(pd.core.series.Series([2, 3, 4, 5]).T))
        print('------')
        print(arr.ensure2D(np.atleast_1d([2, 3, 4, 5])))
        print(arr.ensure2D(np.atleast_2d([2, 3, 4, 5])))
        print(arr.ensure2D(np.atleast_2d([[2, 3, 4, 5]])))
        print(arr.ensure2D(np.atleast_1d([2, 3, 4, 5]).T))
        print(arr.ensure2D(np.atleast_2d([2, 3, 4, 5]).T))
        print(arr.ensure2D(np.atleast_2d([[2, 3, 4, 5]]).T))

        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
