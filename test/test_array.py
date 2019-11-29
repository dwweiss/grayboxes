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


    def test7(self):
        """
        Tests array.average_array()
        """
        XY = [
#950.008,4684.24,
#950.301,4661.63,
#950.593,4674.15,
#950.886,4655.64,
#951.178,4628.18,
#951.471,4605.88,
#951.763,4576.68,
#952.055,4576.22,
#952.348,4539.82,
#952.64,4542.47,
#952.932,4527.6,
#953.224,4516.98,
#953.516,4488.01,
#953.808,4466.24,
#954.1,4478.3,
#954.392,4443.18,
#954.684,4416.86,
#954.975,4410.72,
#955.267,4398.73,
#955.559,4401.16,
#955.85,4353.98,
#956.142,4352.09,
#956.433,4329.94,
#956.725,4296.87,
#957.016,4304.83,
#957.307,4289.89,
#957.598,4245.14,
#957.89,4244,
#958.181,4242.25,
#958.472,4230.72,
#958.763,4229.21,
#959.054,4184.61,
#959.345,4170.8,
#959.635,4155.03,
#959.926,4176.34,
#960.217,4129.24,
#960.507,4122.03,
#960.798,4097.08,
#961.089,4097.99,
#961.379,4080.54,
#961.669,4094.27,
#961.96,4051.87,
#962.25,4022.97,
#962.54,4015.92,
#962.83,4001.96,
#963.12,3998.47,
#963.411,3982.69,
#963.701,3975.64,
#963.99,3943.17,
#964.28,3907.07,
#964.57,3930.73,
#964.86,3921.78,
#965.15,3907.75,
#965.439,3876.58,
#965.729,3846.09,
#966.018,3833.04,
#966.308,3838.88,
#966.597,3827.12,
#966.886,3824.01,
#967.176,3799.13,
#967.465,3789.65,
#967.754,3782.75,
#968.043,3747.4,
#968.332,3748.69,
#968.621,3731.63,
#968.91,3724.34,
#969.199,3743.23,
#969.488,3726.32,
#969.777,3680.88,
#970.065,3669.73,
#970.354,3666.24,
#970.643,3646.98,
#970.931,3647.96,
#971.22,3617.17,
#971.508,3606.93,
#971.796,3586.07,
#972.085,3576.51,
#972.373,3572.34,
#972.661,3561.42,
#972.949,3547.69,
#973.237,3528.65,
#973.525,3529.79,
#973.813,3508.25,
#974.101,3477.3,
#974.389,3478.06,
#974.676,3462.58,
#974.964,3474.57,
#975.252,3453.94,
#975.539,3448.63,
#975.827,3442.56,
#976.114,3426.25,
#976.402,3391.21,
#976.689,3373.54,
#976.976,3375.74,
#977.263,3366.94,
#977.551,3357.38,
#977.838,3354.72,
#978.125,3352.3,
#978.412,3340.77,
#978.699,3304.36,
#978.986,3292.3,
#979.272,3310.73,
#979.559,3286.76,
#979.846,3271.06,
#980.132,3272.35,
#980.419,3256.35,
#980.705,3247.32,
#980.992,3247.62,
#981.278,3217.13,
#981.565,3202.26,
#981.851,3192.25,
#982.137,3183.76,
#982.423,3179.13,
#982.709,3177.92,
#982.995,3186.49,
#983.281,3149.17,
#983.567,3147.05,
#983.853,3135.14,
#984.139,3133.47,
#984.425,3113.75,
#984.71,3097.21,
#984.996,3101.08,
#985.281,3096,
#985.567,3098.65,
#985.852,3078.93,
#986.138,3032.36,
#986.423,3057.84,
#986.708,3039.72,
#986.993,3033.8,
#987.279,3027.2,
#987.564,3009.68,
#987.849,2997.47,
#988.134,2980.48,
#988.418,2971.68,
#988.703,2967.05,
#988.988,2957.34,
#989.273,2976.23,
#989.557,2944.6,
#989.842,2939.75,
#990.127,2924.58,
#990.411,2922.6,
#990.695,2916.69,
#990.98,2909.78,
#991.264,2892.26,
#991.548,2885.21,
#991.833,2889.53,
#992.117,2876.87,
#992.401,2860.25,
#992.685,2877.17,
#992.969,2852.06,
#993.253,2855.7,
#993.536,2834.84,
#993.82,2832.87,
#994.104,2835.68,
#994.387,2815.88,
#994.671,2798.81,
#994.955,2809.59,
#995.238,2790.32,
#995.521,2781.9,
#995.805,2771.43,
#996.088,2789.33,
#996.371,2771.96,
#996.654,2755.66,
#996.937,2744.43,
#997.22,2761.19,
#997.503,2748.68,
#997.786,2736.62,
#998.069,2727.67,
#998.352,2726.91,
#998.635,2715.53,
#998.917,2705.67,
#999.2,2703.85,
#999.483,2695.73,
#999.765,2693,
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
1011.86,2479.33,
1012.13,2482.52,
1012.41,2470.91,
1012.69,2474.02,
1012.97,2476.6,
1013.25,2463.02,
1013.53,2456.05,
1013.81,2461.58,
1014.09,2453.92,
1014.37,2450.66,
1014.65,2446.87,
1014.93,2448.46,
1015.21,2441.94,
1015.49,2444.59,
1015.77,2454.45,
1016.05,2430.18,
1016.32,2434.05,
1016.6,2426.69,
1016.88,2431.7,
1017.16,2426.77,
1017.44,2415.84,
1017.72,2409.78,
1018,2410.23,
1018.28,2403.94,
1018.55,2396.5,
1018.83,2404.32,
1019.11,2400.83,
1019.39,2385.81,
1019.67,2400.3,
1019.95,2388.23,
1020.22,2390.21,
1020.5,2390.21,
1020.78,2390.21,
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


if __name__ == '__main__':
    unittest.main()
