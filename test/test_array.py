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
      2019-12-04 DWW
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import unittest

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
        print('*' * 40)

        self.assertTrue(True)


    def test1b(self):
        x = arr.grid((4, 1), [0., 1.])
        print('x:', x)
        print('*' * 40)

        self.assertTrue(True)


    def test2(self):
        x = arr.rand(3, [0., 1.], [2., 4.])
        print('x:', x)
        plt.title('array.rand()')
        plt.plot(x.T[0], x.T[1], 'o')
        plt.show()
        print('*' * 40)
        
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
        print(arr.convert_to_2d(np.array([5])))
        print(arr.convert_to_2d(np.array([2, 3, 4, 5])))
        print(arr.convert_to_2d(np.array([[2, 3, 4, 5]])))
        print(arr.convert_to_2d(np.array([[2, 3, 4, 5]]).T))
        print('------')
        print(arr.convert_to_2d(np.array([2, 3, 4, 5]).T))
        print('------')
        print(arr.convert_to_2d(pd.core.series.Series([2, 3, 4, 5])))
        print(arr.convert_to_2d(pd.core.series.Series([2, 3, 4, 5]).T))
        print('------')
        print(arr.convert_to_2d(np.atleast_1d([2, 3, 4, 5])))
        print(arr.convert_to_2d(np.atleast_2d([2, 3, 4, 5])))
        print(arr.convert_to_2d(np.atleast_2d([[2, 3, 4, 5]])))
        print(arr.convert_to_2d(np.atleast_1d([2, 3, 4, 5]).T))
        print(arr.convert_to_2d(np.atleast_2d([2, 3, 4, 5]).T))
        print(arr.convert_to_2d(np.atleast_2d([[2, 3, 4, 5]]).T))

        self.assertTrue(True)


    def test7(self):
        """
        Tests array.smooth()
        """
        XY = [
1000.05,2687.92,
1000.33,2685.27,
1000.61,2676.16,
1000.89,2656.06,
1001.18,2660.08,
1001.46,2668.65,
1001.74,2650.91,
1002.02,2645.6,
1002.3,2638.01,
1002.59,2646.05,
1002.87,2628.83,
1003.15,2627.16,
1003.43,2621.7,
1003.71,2602.59,
1004,2612.83,
1004.28,2615.03,
1004.56,2590.98,
1004.84,2596.07,
1005.12,2584.69,
1005.4,2573.31,
1005.68,2568.08,
1005.96,2574.75,
1006.25,2584.99,
1006.53,2579.38,
1006.81,2552.83,
1007.09,2547.82,
1007.37,2535.84,
1007.65,2550.33,
1007.93,2533.11,
1008.21,2536.83,
1008.49,2518.24,
1008.77,2526.74,
1009.05,2509.37,
1009.33,2518.32,
1009.61,2511.34,
1009.89,2517.64,
1010.17,2497.99,
1010.45,2514.3,
1010.74,2499.36,
1011.02,2485.4,
1011.3,2495.03,
1011.58,2489.34,
            ]

        x = [XY[i*2] for i in range(len(XY) // 2)]
        y = [XY[i*2+1] for i in range(len(XY) // 2)]
                
        plt.title('original array')
        plt.plot(x, y, label='meas', linestyle='--')      
        plt.show()

        plt.title('smoothed arrays')
        plt.plot(x, y, label='meas', linestyle='--')      
        for frac in np.linspace(0.2, 0.4, 4):
            y_avg = arr.smooth(x=None, y=y, frac=frac)            
            plt.plot(x, y_avg, label='frac ' + str(round(frac, 2)))
        plt.legend()
        plt.show()

        plt.title('difference to smoothed arrays')
        for frac in np.linspace(0.1, 0.3, 3):
            y_avg = arr.smooth(x=None, y=y, frac=frac)            
            plt.plot(x, y_avg - y, label='frac ' + str(round(frac, 2)))
        plt.legend()
        plt.show()
        
        self.assertTrue(True)


    def test8(self):
        """
        Tests array.average_array()
        """
        x = np.linspace(0., 2 * np.pi, 100)
        y = np.sin(x)
        y_nse = arr.noise(y, absolute=0.5, relative=0e-2, uniform=True)

        # find boxcar with lowest difference between noisy and averaged array 
        fracs = np.linspace(0.025, 0.1, 4)
        frac_min_delta, min_delta = fracs[0], np.inf
        for frac in fracs:
            y_avg = arr.smooth(x=None, y=y, frac=frac)            
            dy = y_avg - y
            max_abs = np.absolute(dy).max()
            if min_delta > max_abs:
                frac_min_delta, min_delta = frac, max_abs

        plt.title('averaging sine curve 1 (2)')
        plt.plot(x, y_nse, linestyle=':', label='$y_{nse}$')
        for frac in fracs:
            y_avg = arr.smooth(x=None, y=y, frac=frac)            
            
            plt.plot(x, y_avg, label='box: ' + str(round(frac, 3)))
        plt.legend(bbox_to_anchor=(1.1, 1.05))
        plt.show()

        plt.title('averaging sine curve 2 (2)')
        plt.plot(x, y_nse - y, linestyle=':', label='$y_{nse} - y$')
        for frac in fracs:
            y_avg = arr.smooth(x=None, y=y, frac=frac)            
            dy = y_avg - y
            max_abs = np.absolute(dy).max()

            style = '--' if frac == frac_min_delta else '-' 
            comment = ' (best)' if frac == frac_min_delta else '' 
            plt.plot(x, dy, linestyle=style,
                     label='box: ' + str(round(frac, 3)) + 
                     r', $\Delta$: ' + str(round(max_abs, 3)) + comment)
        plt.legend(bbox_to_anchor=(1.1, 1.05))
        plt.show()

        self.assertTrue(True)


    def test9(self):
        
        data = {'ind': [1, 2, 3], 'val':[20, 21, 19], 'n': [3,4,7]} 
        df = pd.DataFrame(data)
        print('df:', df)
        print()
        print('ind in df:', 'ind' in df)
        print('val in df:', 'val' in df)
        print('n in df:', 'n' in df)
        print('abc in df:', 'abc' in df)
        print()
        
        try:
            X = arr.frame_to_arrays(df, 'ind', 'val', 'n', '?', '?', ' ', 
                                    '_', '.', ',')
        except:
            print('??? invalid keys')

        try:
            X = arr.frame_to_arrays(df, 'ind', 'val', 'n', 'n')
        except:
            print('??? redundant keys')
        print('\n X[0..n_key-1]:')        
        X = arr.frame_to_arrays(df, 'ind', 'val', 'n')
        for j, x in enumerate(X):
            print(x)

        print('\n X[0..n_key-1]:')        
        X = arr.frame_to_arrays(df, 'ind', 'val', 'n')
        for j, x in enumerate(X):
            print(x)

        self.assertTrue(True)

if __name__ == '__main__':
    unittest.main()
