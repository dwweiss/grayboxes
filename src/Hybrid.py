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
      2018-03-06 DWW
"""

import inspect
import numpy as np

from Model import Model
from Empirical import Empirical
from Theoretical import Theoretical


def dob2str(x):
    """
    Converts degree of blackness of gray box model to hybrid model type

    Args:
        x (float or int):
            degree of blackness (DOB): 0.0 <= beta_blk <= 1.0

    Returns:
        (string)
            model identifier
    """
    assert isinstance(x, (int, float)), str(x)

    if x == 0:
        return 'white'
    if x == 1:
        return 'black'
    if x <= 0.25:
        return 'light'
    if x >= 0.75:
        return 'dark'
    return 'medium-a'


class Hybrid(Model):
    """
    Hybrid model comprising theoretical and empirical submodel

    It is assumed that self.xTun..self.yMod are imported from a DataFrame

    The type of the hybrid model is defined with the setter: self.hybridType.
    If hybridType == 'medium gray', outputActivation = {None | function} is the
    differentiator between the medium gray box trainers

    TODO:
        Test connection of log files of 'empirical' and 'theoretical' to log
        file of Hybrid

    Note:
        Index notation in documentation: (wht, lgt, med, drk, blk),
        Degree of blackness (DOB) 0 <= beta_blk <= 1
    """

    def __init__(self, identifier='Hybrid', model=None, f=None, trainer=None):
        """
        Args:
            identifier (string, optional):
                class identifier

            model (string, optional):
                hybridType out of self._valisHybridTypes

            f (method, optional):
                white box model f(x)

            trainer (method, optional):
                training method
        """
        super().__init__(identifier)

        # type of hybrid model (first three characters are significant)
        self._validHybridTypes = ['white', 'lightgray', 'mediumgray-a',
                                  'mediumgray-b', 'mediumgray-c',
                                  'darkgray', 'black']
        self._hybridType = self._validHybridTypes[-1]

        # submodels
        self._empirical = Empirical()
        self._theoretical = Theoretical()

        # white box model f(x)
        if f is not None:
            self.theoretical.f = f
        if trainer is not None:
            self.trainLoc = trainer.__get__(self, self.__class__)
        if model is not None:
            self.hybridType = model

        # 1D arrays of keys for selecting columns of data frame
        self.xTunKeys, self.xComKeys, self.xUnqKeys = None, None, None
        self.xModKeys, self.xPrcKeys = None, None
        self.yModKeys, self.yPrcKeys, self.yComKeys = None, None, None

        # 2D arrays of parameter and results
        self.xTun, self.xCom, self.xUnq = None, None, None
        self.xMod, self.xPrc = None, None
        self.yMod, self.yPrc, self.yCom = None, None, None

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
            self.frameToArrays(df, self.xTunKeys, self.xComKeys, self.xUnqKeys,
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

    @property
    def f(self):
        return self.theoretical._f

    @f.setter
    def f(self, method):
        assert self.theoretical is not None
        assert issubclass(type(self.theoretical), Theoretical)

        if method is None:
            self.theoretical._f = None
        else:
            if method == 'demo':
                method = self.theoretical.f_demo
                self.write('!!! set Model.f_demo to Hybrid.theoretical.f')
            firstArg = list(inspect.signature(method).parameters.keys())[0]
            if firstArg == 'self':
                method = method.__get__(self, Model)
            self.theoretical._f = method

    @property
    def hybridType(self):
        """
        Returns:
            (string):
                type of hybrid model, see: self._validHybridTypes
        """
        return self._hybridType

    @hybridType.setter
    def hybridType(self, value):
        """
        Sets hybrid type, corrects too short strings and assigns default if
        value is unknown

        Args:
            value (string):
                type of hybrid model, see self._validHybridTypes
        """
        default = self._validHybridTypes[-1]
        if value is None:
            value = default
        value = str(value).lower().replace(' ', '')
        if len(value) < 3:
            self.write("??? Hybrid type too short (min. 3 char): '", value,
                       "', continue with: '", default, "'")
            self._hybridType = default

        valids = [value[0] == x[0] for x in self._validHybridTypes]
        if not any(valids):
            self.write("??? Unknown hybrid type: '", value,
                       "', continue with: '", default, "'")
            self._hybridType = default
        else:
            self._hybridType = self._validHybridTypes[valids.index(True)]

    @property
    def empirical(self):
        """
        Returns:
            (Empirical):
                empirical model
        """
        return self._empirical

    @empirical.setter
    def empirical(self, value):
        """
        Sets empirical model and assigns Hybrid's logFile to it

        Args:
            value (Empirical):
                empirical model
        """
        assert value is None or issubclass(type(value), (Empirical)), \
            'invalid model type: ' + str(type(value))
        self._empirical = value
        if self.empirical is not None:
            self.empirical.logFile = self.logFile

    @property
    def theoretical(self):
        """
        Returns:
            (Theoretical):
                theoretical model
        """
        return self._theoretical

    @theoretical.setter
    def theoretical(self, value):
        """
        Sets theoretical model and assigns Hybrid's logFile to it

        Args:
            value (Theoretical):
                theoretical model
        """
        assert value is None or issubclass(type(value), (Theoretical)),\
            'invalid model type: ' + str(type(value))
        self._theoretical = value
        if self.theoretical is not None:
            self.theoretical.logFile = self.logFile

    def trainLocal(self, X, Y, **kwargs):
        """
        Trains medium gray box model type B

        Args:
            X (2D array_like of float):
                traing input

            Y (2D array_like of float):
                training target

            kwargs (dict, optional):
                keyword arguments:

                placeholder (string):
                    string defining data

        Returns:
            (float):
                L2-norm
        """
        assert self.hybridType[0] == 'm' and self.hybridType.endswith('-b'), \
            str(self.hybridType)
        assert 0
        return -1.

    def train(self, X=None, Y=None, **kwargs):
        """
        Trains hybrid model

        Args:
            X (2D array_like of float, optional):
                training input

            Y (2D array_like of float, optional):
                training target

            kwargs (dict, optional):
                keyword arguments:

                model (string):
                    hybridType out of self._validHybridTypes
        Returns:
            (float):
                L2-norm

        Example:
            An external function or method is assigned to Theoretical.f():
                def f(self, x, c0=1, c1=1, c2=1, c3=1, c4=1, c5=1, c6=1, c7=1):
                    y0 = c0 * x[0]*x[0] + c1 * x[1]
                    y1 = x[1] * c3
                    return [y0, y1]

                X = [[..], [..], ..]
                Y = [[..], [..], ..]
                x = [[..], [..], ..]

                foo = Theoretical(f=f)
                foo.train(X, Y, model='light gray')
                y_a = foo.predict(x)

                foo = Theoretical(f=f)
                y_b = foo(X, Y, x=x, model='light gray')
        """
        self.X = X if X is not None else self.xPrc
        self.Y = Y if Y is not None else self.yPrc

        model = kwargs.get('model', None)
        if model is not None:
            self.hybridType = model
        # if 'model' is not string but scalar, it contains degree of blackness
        if isinstance(model, (int, float)):
            model = dob2str(model)

        assert self.hybridType.startswith(('white', 'light', 'medium', 'dark',
                                           'black')), str(self.hybridType)
        assert not self.hybridType.startswith('medium') or \
            self.hybridType.endswith(('-a', '-b', '-c')), str(self.hybridType)

        if self.hybridType.startswith(('medium', 'dark', 'black')):
            if self.empirical is None:
                self.empirical = Empirical()
        else:
            self.empirical = None

        if self.hybridType.startswith(('white', 'light', 'medium', 'dark')):
            if self.theoretical is None:
                self.theoretical = Theoretical()
            f = kwargs.get('f', None)
            if f is not None:
                self.theoretical.f = f
            else:
                if self.theoretical.f is None:
                    self.theoretical.f = Model.f_demo
                    self.write('!!! Assigns f_demo to Hybrid.theoretical.f')
        else:
            self.theoretical = None

        kw = self.kwargsDel(kwargs, ['X', 'Y'])
        norm = -1.0

        if self.hybridType.startswith('white'):
            self.warning('White box model -> no training')

        elif self.hybridType.startswith('light'):
            self.write('+++ Trains light gray box')
            norm = self.theoretical.train(self.X, self.Y, **kw)

        elif self.hybridType.startswith('medium'):
            if self.hybridType.endswith('-a'):
                self.write('+++ Trains medium gray box Type A')
                self.empirical.f = self.theoretical.f
                norm = self.empirical.train(self.X, self.Y, **kw)

            elif self.hybridType.endswith('-b'):
                self.write('+++ Trains medium gray box Type B')
                norm = self.trainLocal(self.X, self.Y, **kw)

            elif self.hybridType.endswith('-c'):
                self.write('+++ Trains medium gray box Type C')
                assert 0, str(self.hybridType)

            else:
                assert 0, str(self.hybridType)

        elif self.hybridType.startswith('dark'):
            self.write('+++ Trains dark gray box')
            y = self.theoretical.predict(self.X, **kw)
            norm = self.empirical.train(np.c_[self.X, y], y - self.Y, **kw)

        elif self.hybridType.startswith('black'):
            self.write('+++ Trains black box')
            norm = self.empirical.train(self.X, self.Y, **kw)

        else:
            print('self.hybridType:', self.hybridType)
            print('self.empirical is None:', self.empirical is None)
            print('self.theoretical is None:', self.theoretical is None)
            assert 0

        return norm

    def ready(self):
        return self.empirical is not None and self.empirical.ready()

    def predict(self, x=None, **kwargs):
        """
        Executes hybrid model

        Args:
            x (2D array_like of float, optional):
                prediction input

            kwargs (dict, optional):
                keyword arguments

        Returns:
            self.y (2D array of float):
                prediction result
        """
        self.x = x if x is not None else self.x

        y = None

        assert self.empirical is None or self.empirical.ready()
        assert self.theoretical is None or self.theoretical.ready()

        kw = self.kwargsDel(kwargs, 'x')

        if self.hybridType.startswith('white'):
            self.write('Predicts white box')
            y = self.theoretical.predict(x, **kw)

        elif self.hybridType.startswith('light'):
            self.write('Predicts light gray box')
            y = self.theoretical.predict(x, **kw)

        elif self.hybridType.startswith('medium'):
            self.write('Predicts medium gray box')
            xTun = self.empirical.predict(x, **kw)
            xMod = np.c_[self.xCom, xTun]
            y = self.theoretical.predict(xMod, **kw)

        elif self.hybridType.startswith('dark'):
            self.write('Predicts dark gray box')
            y = self.theoretical.predict(x, **kw)
            dy = self.empirical.predict(np.c_[x, y], **kw)
            y -= dy

        elif self.hybridType.startswith('black'):
            self.write('Predicts black box')
            y = self.empirical.predict(x, **kw)

        else:
            assert 0

        self._y = np.atleast_2d(y)
        return self._y

    def pre(self, **kwargs):
        """
        Args:
            kwargs (dict, optional):
                keyword arguments:

                model (string):
                    hybrid type out of self._validHybridTypes
        """
        super().pre(**kwargs)

        hybridType = kwargs.get('model', None)
        if hybridType is not None:
            self.hybridType = hybridType
        assert hybridType in self._validHybridTypes, str(hybridType)


# Examples ####################################################################

if __name__ == '__main__':
    ALL = 1

    from io import StringIO
    import pandas as pd

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

    if 0 or ALL:
        s = 'Black Box Model'
        print('-' * len(s) + '\n' + s + '\n' + '-' * len(s))

        foo = Hybrid(model='black')
        foo.setArrays(df, xModKeys=['x0', 'x2', 'x3'], xPrcKeys=['x0', 'x4'],
                      yModKeys=['y0', 'y1'], yPrcKeys=['y0', 'y2'])
        foo.train(X=foo.xPrc, Y=foo.yPrc)
        y = foo.predict(x=foo.xPrc)
        print('*** x:', foo.x, 'y:', y)

    if 1 or ALL:
        s = 'Dark Gray Box Model (subtract)'
        print('-' * len(s) + '\n' + s + '\n' + '-' * len(s))

        foo = Hybrid(model='dark', f='demo')
        foo.setArrays(df, xModKeys=['x0', 'x2', 'x3'], xPrcKeys=['x0', 'x4'],
                      yModKeys=['y0'], yPrcKeys=['y0'])
        foo.train(X=foo.xPrc, Y=foo.yPrc, silent=True)
        y = foo.predict(x=foo.xPrc)
        print('*** x:', foo.x, 'y:', y)

    if 0 or ALL:
        s = 'Black Box Model, measured Y(X) = E(mDot, p)'
        print('-' * len(s) + '\n' + s + '\n' + '-' * len(s))

        df = pd.read_csv(raw, sep=',', comment='#')
        df.rename(columns=df.iloc[0])
        df = df.apply(pd.to_numeric, errors='coerce')
        X = np.asfarray(df.loc[:, ['mDot', 'p']])
        Y = np.asfarray(df.loc[:, ['A']])

        foo = Hybrid(model='black')
        foo.train(X=X, Y=Y, definition=[])
        y = foo.predict(x=X)
        print('*** x:', foo.x, 'y:', y)

        from plotArrays import plotIsoMap, plotWireframe
        plotIsoMap(X.T[0], X.T[1], Y.T[0] * 1e3, title=r'$A_{prc}\cdot 10^3$')
        plotIsoMap(X.T[0], X.T[1], y.T[0] * 1e3, title=r'$A_{blk}\cdot 10^3$')
        plotIsoMap(X.T[0], X.T[1], (Y.T[0] - y.T[0]) * 1e3,
                   title=r'$(A_{prc} - A_{blk})\cdot 10^3$')
        plotWireframe(X.T[0], X.T[1], Y.T[0]*1e3, title=r'$A_{prc}\cdot 10^3$')
        plotWireframe(X.T[0], X.T[1], y.T[0]*1e3, title=r'$A_{blk}\cdot 10^3$')
        plotWireframe(X.T[0], X.T[1], (Y.T[0] - y.T[0]) * 1e3,
                      title=r'$(A_{prc} - A_{blk})\cdot 10^3$')
