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
      2018-05-08 DWW
"""

from io import StringIO
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from Black import Black
from plotArrays import plotIsoMap, plotWireframe


# Examples ####################################################################

if __name__ == '__main__':
    ALL = 0

    # anonymised data of an observation: A = mDotInd - mDot = F(mDot, p)
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

    if 1 or ALL:
        s = 'Error compensation (black box): Y(X) = A(mDot, p)'
        print('-' * len(s) + '\n' + s + '\n' + '-' * len(s))

        plotWireframes = False
        plotIsoMaps = True

        df = pd.read_csv(raw, sep=',', comment='#')
        df.rename(columns=df.iloc[0])
        df = df.apply(pd.to_numeric, errors='coerce')
        X = np.asfarray(df.loc[:, ['mDot', 'p']])
        Y = np.asfarray(df.loc[:, ['A']])
        YDiff = round(Y[:, 0].max() - Y[:, 0].min(), 5)

        plotIsoMap(X[:, 0], X[:, 1], Y[:, 0]*1e3, title=r'$A_{prc}\cdot 10^3' +
                   r'\ \ (\Delta A$: ' + str(YDiff*1e3) + 'e-3)',
                   labels=[r'$\dot m$', '$p$'])
        plotWireframe(X[:, 0], X[:, 1], Y[:, 0]*1e3,
                      title=r'$A_{prc}\cdot 10^3$',
                      labels=[r'$\dot m$', '$p$'])

        model = Black()
        dyDiffAll = []
        hidden = range(1, 20+1)

        for hid in hidden:
            print('+++ hidden:', hid, end='')
            print(' ==> autodefinition') if hid == 0 else print()

            y = model(X=X, Y=Y, x=X, goal=1e-5, trials=5, epochs=1000,
                      neural=hid, trainers='rprop', silent=True)
            # print('*** x:', model.x, 'y:', model.y, y)

            yDiff = round(y[:, 0].max() - y[:, 0].min(), 5)
            dy = Y[:, 0] - y[:, 0]
            dyDiff = round(dy.max() - dy.min(), 5)
            dyDiffAll.append(dyDiff)

            if plotIsoMaps:
                plotIsoMap(X[:, 0], X[:, 1], y[:, 0] * 1e3, title='$A_{mod}' +
                           r'\cdot 10^3\ \ (\Delta A$: ' + str(yDiff*1e3) +
                           'e-3) [' + str(hid) + ']',
                           labels=[r'$\dot m$', '$p$'])
                plotIsoMap(X[:, 0], X[:, 1], dy*1e3, title='$(A_{prc}-' +
                           r'A_{mod})\cdot 10^3\ \ (\Delta A$: ' +
                           str(dyDiff*1e3) + 'e-3)',
                           labels=[r'$\dot m$', '$p$'])
            if plotWireframes:
                plotWireframe(X[:, 0], X[:, 1], y[:, 0]*1e3,
                              title=r'$A_{mod}\cdot 10^3$')
                plotWireframe(X[:, 0], X[:, 1], dy * 1e3,
                              title=r'$(A_{prc} - A_{mod})\cdot 10^3$')

        plt.title('Compensation error vs hidden neurons')
        plt.xlabel('hidden neurons [/]')
        plt.ylabel('(max $\Delta A$ - min $\Delta A)\cdot 10^3$ [/]')
        plt.plot(hidden, np.asarray(dyDiffAll)*1e3)
        plt.grid()
        plt.show()
