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
      2019-03-01 DWW
"""

import unittest
import sys
import os
from io import StringIO
import numpy as np
import pandas as pd

sys.path.append(os.path.abspath('..'))
sys.path.append(os.path.abspath('../grayboxes'))

from grayboxes.darkgray import DarkGray
from grayboxes.plotarrays import plot_isomap, plot_wireframe
from grayboxes.boxmodel import frame_to_arrays
from grayboxes.black import Black


df = pd.DataFrame({'x0': [2, 3, 4.5, 5, 9],
                   'x1': [3, 4, 5, 6, 7],
                   'x2': [4, 5, 6, 7, 8],
                   'x3': [5, 6, 7, 8, 9],
                   'x4': [6, 7, 8, 9, 10],
                   'y0': [7, 8, 9, 10, 12],
                   'y1': [8, 9, 10, 11, 12],
                   'y2': [9, 10, 11, 12, 14],
                   })


# anonymized data of an observation: A = mDotInd - mDot = F(mDot, p)
# E in [%], mDot and mDotInd normalized with min/max of both arrays
raw = StringIO("""mDot,p,E,A,mDotInd
    0.003393,  0.000,    NaN,  0.000154,  0.003547
    0.597247,  0.054, -0.785, -0.004662,  0.592586
    0.858215,  0.054, -0.334, -0.002855,  0.855360
    0.901367,  0.262, -0.621, -0.005576,  0.895790
    0.893147,  0.516, -0.857, -0.007625,  0.885522 ## outlier (regular)
    0.884928,  0.771, -0.879, -0.007749,  0.877179
    0.849995,  0.931, -0.865, -0.007323,  0.842672
    0.003393,  0.000,    NaN, -0.003391,  0.000002
    0.862324,  0.054, -0.687, -0.005901,  0.856423
    0.525327,  0.250, -0.962, -0.005021,  0.520306 ## outlier (extra)
    1.000000,  0.260, -0.616, -0.006139,  0.993861
    0.003393,  0.056,    NaN,  0.000616,  0.004009
    0.765746,  0.056, -0.249, -0.001898,  0.763848
    0.003393,  0.261,    NaN, -0.000411,  0.002982
    0.843831,  0.261, -0.471, -0.003958,  0.839872 ## outlier for 2D
    0.003393,  0.000,    NaN, -0.003156,  0.000236
    0.003393,  0.000,    NaN, -0.003386,  0.000006
    0.003393,  0.100,    NaN, -0.002885,  0.000508
    0.003393,  0.100,    NaN, -0.003319,  0.000074
    0.003393,  0.250,    NaN, -0.003393,  0.000000
    0.003393,  0.270,    NaN, -0.002817,  0.000575
    0.003393,  0.260,    NaN, -0.002860,  0.000532
    0.003393,  0.260,    NaN, -0.002922,  0.000471
    0.003393,  0.500,    NaN, -0.002774,  0.000619
    0.003393,  0.770,    NaN, -0.002710,  0.000682
    0.003393,  1.000,    NaN, -0.002770,  0.000623
    0.003393,  1.000,    NaN, -0.002688,  0.000705
    0.003393,  1.000,    NaN, -0.002686,  0.000707
""")


class TestUM(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test1(self):
        s = 'Dark gray box model 1'
        print('-' * len(s) + '\n' + s + '\n' + '-' * len(s))

        model = DarkGray('demo', 'test1')
        X, Y = frame_to_arrays(df, ['x0', 'x4'], ['y0'])
        y = model(X=X, Y=Y, x=X, silent=True, neurons=[10])
        plot_isomap(X[:, 0], X[:, 1], Y[:, 0], title='Y(X)')
        plot_isomap(X[:, 0], X[:, 1], y[:, 0], title='y(X)')
        plot_isomap(X[:, 0], X[:, 1], (y - Y)[:, 0], title='y(X)  -Y')

        print('*** X:', X.shape, 'Y:', Y.shape, 'y:', y.shape)

        self.assertTrue(True)

    def test2(self):
        s = 'Dark gray box model 2'
        print('-' * len(s) + '\n' + s + '\n' + '-' * len(s))

        data_frame = pd.read_csv(raw, sep=',', comment='#')
        data_frame.rename(columns=df.iloc[0])
        data_frame = data_frame.apply(pd.to_numeric, errors='coerce')
        X = np.asfarray(data_frame.loc[:, ['mDot', 'p']])
        Y = np.asfarray(data_frame.loc[:, ['A']])

        def f(x, *args, **kwargs):
            return x[0] + x[1]

        y = DarkGray(f, 'test2')(X=X, Y=Y, x=X, silent=True, neurons=[10])

        plot_isomap(X[:, 0], X[:, 1], Y[:, 0], title='Y(X)')
        plot_isomap(X[:, 0], X[:, 1], y[:, 0], title='y(X)')
        plot_isomap(X[:, 0], X[:, 1], y[:, 0] - Y[:, 0], title='y(X) - Y')

        print('*** X:', X.shape, 'Y:', Y.shape, 'y:', y.shape)

        self.assertTrue(True)

    def test3(self):
        s = 'Black box model, measured Y(X) = E(mDot, p)'
        print('-' * len(s) + '\n' + s + '\n' + '-' * len(s))

        raw.seek(0)
        data_frame = pd.read_csv(raw, sep=',', comment='#')
        data_frame.rename(columns=data_frame.iloc[0])
        data_frame = data_frame.apply(pd.to_numeric, errors='coerce')
        X = np.asfarray(data_frame.loc[:, ['mDot', 'p']])
        Y = np.asfarray(data_frame.loc[:, ['A']])

        y = Black('test3')(X=X, Y=Y, neurons=[], silent=True, x=X)

        plot_isomap(X.T[0], X.T[1], Y.T[0] * 1e3, title=r'$A_{prc}\cdot 10^3$')
        plot_isomap(X.T[0], X.T[1], y.T[0] * 1e3, title=r'$A_{blk}\cdot 10^3$')
        plot_isomap(X.T[0], X.T[1], (Y.T[0] - y.T[0]) * 1e3,
                    title=r'$(A_{prc} - A_{blk})\cdot 10^3$')
        plot_wireframe(X.T[0], X.T[1], Y.T[0] * 1e3, 
                       title=r'$A_{prc}\cdot 10^3$')
        plot_wireframe(X.T[0], X.T[1], y.T[0] * 1e3, 
                       title=r'$A_{blk}\cdot 10^3$')
        plot_wireframe(X.T[0], X.T[1], (Y.T[0] - y.T[0]) * 1e3,
                       title=r'$(A_{prc} - A_{blk})\cdot 10^3$')

        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
