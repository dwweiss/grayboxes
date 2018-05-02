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
      2018-05-01 DWW
"""

import numpy as np

from Model import Model
from Neural import Neural


class MediumGray(Model):
    """
    Medium gray box model comprising a white box and a black box submodel

    It is assumed that self.xTun, ..., self.yMod are imported from a DataFrame

    The variant of the model is a string starting with 'med' and ending with a
    dash followed by a postfix, e.g '-loc1'. The postfix diferentiates between:
        - local  calibration (contains substring '-loc') and
        - global calibration (otherwise).
    If the last character is a number, it identifies the index of the
    particular calibration algorihm within the 'local' or global' trainers

    Note:
        Degree of model blackness: $0 \le \beta_{blk} \le 1$
    """

    def __init__(self, f, identifier='MediumGray'):
        """
        Args:
            f (method or function):
                theoretical model y = f(x) for single data point

            identifier (string, optional):
                object identifier
        """
        super().__init__(identifier=identifier, f=f)

        self.variant = '-loc1'

        # submodels
        self._white = self
        self._black = Neural(f=f if self.isGlobal() else None)

        # list of keys for selecting columns of data frame
        self.xTunKeys, self.xComKeys, self.xUnqKeys = None, None, None
        self.xModKeys, self.xPrcKeys = None, None
        self.yModKeys, self.yPrcKeys, self.yComKeys = None, None, None

        # 2D arrays of inputs and outputs
        self.xTun, self.xCom, self.xUnq = None, None, None
        self.xMod, self.xPrc = None, None
        self.yMod, self.yPrc, self.yCom = None, None, None

    @property
    def silent(self):
        return self._silent

    @silent.setter
    def silent(self, value):
        self._silent = value
        self._white._silent = value
        self._black._silent = value

    def isLocal(self):
        return 'loc' in self.variant

    def isGlobal(self):
        return not self.isLocal()

    def setArrays(self, df, xModKeys, xPrcKeys, yModKeys, yPrcKeys):
        """
        - Extracts common input and common output keys
        - Extracts 2D arrays self.xTun ... self.yPrc from DataFrame

        Args:
            df (DataFrame):
                data frame with model and process input and output

            xModKeys (1D array_like of string):
                list of model input keys

            xPrcKeys (1D array_like of string):
                list of process input keys

            yModKeys (1D array_like of string):
                list of model output keys

            yPrcKeys (1D array_like of string):
                list of process output keys
        """
        self.xModKeys = list(xModKeys) if xModKeys else None
        self.xPrcKeys = list(xPrcKeys) if xPrcKeys else None
        self.yModKeys = list(yModKeys) if yModKeys else None
        self.yPrcKeys = list(yPrcKeys) if yPrcKeys else None

        if xModKeys is None or xPrcKeys is None:
            self.xComKeys = None
        else:
            self.xComKeys = list(set(xModKeys).intersection(self.xPrcKeys))
            self.xTunKeys = list(set(xModKeys).difference(self.xComKeys))
            self.xUnqKeys = list(set(xPrcKeys).difference(self.xComKeys))
        if yModKeys is None or yPrcKeys is None:
            self.yComKeys = None
        else:
            self.yComKeys = list(set(yModKeys).intersection(self.yPrcKeys))

        assert self.xComKeys, str(self.xComKeys)
        assert self.yComKeys, str(self.yComKeys)
        assert not set(self.yModKeys).isdisjoint(self.yPrcKeys), \
            str(self.yModKeys) + str(self.yPrcKeys)

        self.xTun, self.xCom, self.xUnq, self.xMod, self.xPrc, \
            self.yMod, self.yCom, self.yPrc = \
            self.frame2arrays(df, self.xTunKeys, self.xComKeys, self.xUnqKeys,
                              self.xModKeys, self.xPrcKeys,
                              self.yModKeys, self.yComKeys, self.yPrcKeys)
        if 0:
            print('xy mod:', self.xModKeys, self.yModKeys)
            print('xy prc:', self.xPrcKeys, self.yPrcKeys)
            print('x tun com unq', self.xTunKeys, self.xComKeys, self.xUnqKeys)
            print('y com:', self.yComKeys, self.yCom.shape)
            print('xy mod:', self.xModKeys, self.yModKeys)
            print('xy prc:', self.xPrcKeys, self.yPrcKeys)
            print('x tun com unq', self.xTunKeys, self.xComKeys, self.xUnqKeys)

    def trainLocal(self, X, Y, **kwargs):
        """
        Trains medium gray box with local calibration

        Args:
            X (2D array_like of float):
                traing input

            Y (2D array_like of float):
                training target

            kwargs (dict, optional):
                keyword arguments:

                ... network options

        Returns:
            (3-tuple of float):
                (L2-norm, max{|Delta_Y|}, index(max{|Delta_Y|})
        """
        assert self.isLocal()

        XPrc, XTun = [], []
        for x in X:
            xPrc = x
            xTun = [1 + np.random.normal(0, 0.1)]

            XPrc.append(xPrc)
            XTun.append(xTun)
        XPrc, XTun = np.asfarray(XPrc), np.asfarray(XTun)
        norm = self._black(X=XPrc, Y=XTun, **kwargs)

        norm = (0, 0, 0)
        return norm

    def train(self, X=None, Y=None, **kwargs):
        """
        Trains medium gray model

        Args:
            X (2D array_like of float, optional):
                training input X_prc

            Y (2D array_like of float, optional):
                training target Y_com

            kwargs (dict, optional):
                keyword arguments:

                ... network options

        Returns:
            (3-tuple of float):
                (L2-norm, max{|Delta_Y|}, index(max{|Delta_Y|})

        Example:
            Method f(self, x) or function f(x) is assigned to self.f, example:
                def f(self, x, c0=1, c1=1, c2=1, c3=1, c4=1, c5=1, c6=1, c7=1):
                    y0 = c0 * x[0]*x[0] + c1 * x[1]
                    y1 = x[1] * c3
                    return [y0, y1]

                X = [[..], [..], ..]
                Y = [[..], [..], ..]
                x = [[..], [..], ..]

                # expanded form:
                mod = MediumGray(f=f)
                mod.train(X, Y, color='-loc1')
                y = mod.predict(x)

                # same in compact form:
                y = MediumGray(f=f)(X=X, Y=Y, x=x, color='-loc1')
        """
        self.X = X if X is not None else self.xPrc
        self.Y = Y if Y is not None else self.yPrc

        opt = self.kwargsDel(kwargs, ['X', 'Y'])
        if self.isLocal():
            self.write('+++ Trains medium gray box type loc-1')
            norm = self.trainLocal(self.X, self.Y, f=self.f, **opt)
        else:
            if self.variant.endswith('1'):
                self.write('+++ Trains medium gray box type glob-1')
                norm = self._black.train(self.X, self.Y, f=self.f,
                                         trainers='genetic', **opt)
            elif self.variant.endswith('2'):
                self.write('+++ Trains medium gray box type glob-2')
                norm = self._black.train(self.X, self.Y, f=self.f,
                                         trainers='derivative', **opt)
            else:
                assert 0, str(self.variant)

        self.ready = True

        norm = (0, 0, 0)
        return norm

    def predict(self, x=None, **kwargs):
        """
        Executes medium gray box model

        Args:
            x (2D array_like of float, optional):
                process input x_prc

            kwargs (dict, optional):
                keyword arguments

        Returns:
            self.y (2D array of float):
                common output y_com
        """
        assert self._black is not None and self._black.ready

        self.x = x if x is not None else self.x
        opt = self.kwargsDel(kwargs, 'x')

        if self.isLocal():
            xTun = self._black.predict(x=x, **opt)
            xMod = np.c_[self.xCom, xTun]
            self.y = self._white.predict(x=xMod, **opt)

        elif self.isGlobal():
            self.y = self._black.predict(x=x, **opt)

        else:
            assert 0

        return self.y


# Examples ####################################################################

if __name__ == '__main__':
    ALL = 0

    from io import StringIO
    import pandas as pd
    from Black import Black

    df = pd.DataFrame({'x0': [2, 3, 4, 5],
                       'x1': [3, 4, 5, 6],
                       'x2': [4, 5, 6, 7],
                       'x3': [5, 6, 7, 8],
                       'x4': [6, 7, 8, 9],
                       'y0': [7, 8, 9, 10],
                       'y1': [8, 9, 10, 11],
                       'y2': [9, 10, 11, 12],
                       })

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
        s = 'Medium gray box model'
        print('-' * len(s) + '\n' + s + '\n' + '-' * len(s))

        model = MediumGray(f='demo')

        model.setArrays(df, xModKeys=['x0', 'x2', 'x3'], xPrcKeys=['x0', 'x4'],
                        yModKeys=['y0'], yPrcKeys=['y0'])
        print('*'*20)
        #print(foo.xPrc, foo.yPrc)
        model(X=model.xPrc, Y=model.yPrc, silent=True, neural=[])
        y = model(x=model.xPrc)
        print('*'*20)
        print('*** x:', model.x, 'y:', y)

    if 0 or ALL:
        s = 'Medium gray box model, measured Y(X) = E(mDot, p)'
        print('-' * len(s) + '\n' + s + '\n' + '-' * len(s))

        df = pd.read_csv(raw, sep=',', comment='#')
        df.rename(columns=df.iloc[0])
        df = df.apply(pd.to_numeric, errors='coerce')
        X = np.asfarray(df.loc[:, ['mDot', 'p']])
        Y = np.asfarray(df.loc[:, ['A']])

        foo = Black()
        y = foo(X=X, Y=Y, hidden=[], x=X)
        print('*** x:', foo.x, 'y:', foo.y, y)

        from plotArrays import plotIsoMap, plotWireframe
        plotIsoMap(X.T[0], X.T[1], Y.T[0] * 1e3, title=r'$A_{prc}\cdot 10^3$')
        plotIsoMap(X.T[0], X.T[1], y.T[0] * 1e3, title=r'$A_{blk}\cdot 10^3$')
        plotIsoMap(X.T[0], X.T[1], (Y.T[0] - y.T[0]) * 1e3,
                   title=r'$(A_{prc} - A_{blk})\cdot 10^3$')
        plotWireframe(X.T[0], X.T[1], Y.T[0]*1e3, title=r'$A_{prc}\cdot 10^3$')
        plotWireframe(X.T[0], X.T[1], y.T[0]*1e3, title=r'$A_{blk}\cdot 10^3$')
        plotWireframe(X.T[0], X.T[1], (Y.T[0] - y.T[0]) * 1e3,
                      title=r'$(A_{prc} - A_{blk})\cdot 10^3$')
