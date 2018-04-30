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
      2018-04-30 DWW
"""

import numpy as np

from Base import Base
from Model import Model
from White import White

from Model import gridInit, crossInit, randInit


class Forward(Base):
    """
    Predicts $y = \phi(x)$ for series of data points, x.shape: (nPoint, nInp)

    Examples:
        X_prc = [[... ]]  input of training
        Y_prc = [[... ]]  target of training
        x_prc = [[... ]]  input of prediction

        def func(x, c0=1, c1=1, c2=1, c3=1, c4=1, c5=1, c6=1, c7=1):
            return 2.2 * np.array(np.sin(x[0]) + (x[1] - 1)**2)

        def meth(self, x, c0=1, c1=1, c2=1, c3=1, c4=1, c5=1, c6=1, c7=1):
            return 3.3 * np.array(np.sin(x[0]) + (x[1] - 1)**2)

        # create operation on generic model phi(x)
        op = Forward(White(func))
        op = Forward(White(meth))

        # compact training and prediction with phi(x)
        _, y = op(X=X_prc, Y=Y_prc, x=x_prc)

        # separated training and prediction
        op(X=X_prc, Y=Y_prc)     # train
        _, y = op(x=x_mod)       # predict

    Note:
        - Forward.__call__() returns 2-tuple of 2D arrays of float (x and y)
    """

    def __init__(self, model, identifier='Forward'):
        """
        Args:
            model (Model_like):
                generic model object

            identifier (string, optional):
                object identifier
        """
        super().__init__(identifier)
        self.model = model

    @property
    def model(self):
        """
        Returns:
            (Model_like):
                generic model
        """
        return self._model

    @model.setter
    def model(self, value):
        """
        Sets generic model and assigns Forward's logFile

        Args:
            value (Model_like):
                generic model
        """
        self._model = value
        if self._model is not None:
            assert issubclass(type(value), Model), \
                'invalid model type: ' + str(type(value))
            self._model.logFile = self.logFile

    def _initialInput(self, ranges, n=5, shape='cross'):
        """
        Sets uniformly spaced or randomly distributed input, for instance for
        3 input with 5 axis points: _initialInput(ranges=[(2, 3), (-5, -4),
            (4, 9)], n=5, shape='cross') ==>
        [[2 -5 4] [2.25 -4.75  5.25] [2.5 -4.5 6.5] [2.75 -4.25 7.75] [3 -4 9]]

        Args:
            ranges (2D array_like of float):
                array of (min, max) pairs of float, first index is input index

            n (int or array_like of int, optional):
                number of x-variations for which initial values are generated

            shape (string, optional)            x        x---x---x     x  x  x
                pattern of input variation      |        |   |   |        x x
                'cross'                     x---x---x    x---x---x      x     x
                'grid'                          |        |   |   |     x    x
                'rand'                          x        x---x---x      x  x  x
                                             'cross'      'grid'        'rand'
        Returns:
            (2D array of float):
                uniformely spaced or random values, first index is trial index,
                second index is input index
        """
        assert ranges is not None
        ranges = np.atleast_2d(ranges)
        assert ranges.shape[1] == 2, 'ranges:' + str(ranges)
        assert all(rng[0] <= rng[1] for rng in ranges), 'ranges:' + str(ranges)
        if n is None:
            n = 1

        shape = shape.lower()
        assert shape[0] in ('c', 'g', 'r'), 'shape: ' + str(shape)

        if shape.startswith('r'):
            x = randInit(ranges, n)
        elif shape.startswith('c'):
            x = crossInit(ranges, n)
        elif shape.startswith('g'):
            x = gridInit(ranges, n)
        return x

    def pre(self, **kwargs):
        """
        - Assigns Model instance
        - Sets training data set (X, Y) or XY, and model input x
        - Trains model if (X, Y) or XY are not None
        - Generates model input x if x is None (from 'ranges' and ('cross',
           or 'grid' or 'rand'))

        Args:
            kwargs (dict, optional):
                keyword arguments:


                XY (2-tuple of 2D array_like of float):
                    input and target of training, this argument supersedes X, Y

                X (2D array_like of float):
                    input of training

                Y (2D array_like of float):
                    target of training

                x (2D array_like of float):
                    input to forward prediction or to sensitivity analysis

                ranges (2D array_like of float):
                    array of min/max pairs

                cross (int or list of int):
                    number of cross points per axis if cross is generated

                grid (int or list of int):
                    number of grid points per axis if grid is generated

                rand (int):
                    number of points if random array is generated

                n (int):
                    number of points per axis. if 'rand', n is total number

        Note:
            1) 'x' keyword overrules 'ranges' keyword
            2) if more than one arg. out of [cross, grid, rand, n] is given,
                the order in this list defines the relevant argument
        """
        super().pre(**kwargs)

        # trains model
        XY = kwargs.get('XY', None)
        if isinstance(XY, (list, tuple, np.ndarray)) and len(XY) > 1:
            X, Y = XY[0], XY[1]
        else:
            X, Y = kwargs.get('X', None), kwargs.get('Y', None)
        if not isinstance(self.model, White):
            if X is not None and Y is not None:
                self.model.train(X, Y, **self.kwargsDel(kwargs, ['X', 'Y']))

        # - assigns x to self.model.x for prediction or
        # - generates x from ranges=[(xl, xu),(xl, xu), ..] and from 1) or 2):
        #    1) grid=(nx,ny,..), grid=nx, cross=(nx,ny,..), cross=nx, or rand=n
        #    2) shape='grid', shape='cross', or shape='rand' and n=(nx,ny, ..)
        x = kwargs.get('x', None)
        if x is None:
            ranges = kwargs.get('ranges', None)
            if ranges is not None:
                n = None
                for shape in ('cross', 'grid', 'rand'):
                    if shape in kwargs:
                        n = kwargs[shape]
                        break
                if n is None:
                    print('??? ranges given without: cross/grid/rand=n')
                else:
                    x = self._initialInput(ranges=ranges, n=n, shape=shape)
        if type(self).__name__ in ('Minimum', 'Maximum', 'Inverse'):
            self.x = np.atleast_2d(x) if x is not None else None
        else:
            self.model.x = np.atleast_2d(x) if x is not None else None

    def task(self, **kwargs):
        """
        This task() method is only for Forward and Sensitivity.
        Minimum, Maximum and Inverse have another implementation of task()

        Args:
            kwargs (dict, optional):
                keyword arguments passed to super.task()

        Return:
            x, y (2-tuple of 2D arrays of float):
                input and output of model prediction
        """
        super().task(**kwargs)

        if self.model.x is None:
            self.model.y = None
        else:
            self.model.y = np.asfarray(self.model.predict(x=self.model.x,
                                       **self.kwargsDel(kwargs, 'x')))
        return self.model.x, self.model.y

    def post(self, **kwargs):
        """
        Args:
            kwargs (dict, optional):
                keyword arguments passed to super.post()
        """
        super().post(**kwargs)

        if not self.silent:
            self.plot()

    def plot(self):
        pass


# Examples ####################################################################

if __name__ == '__main__':
    ALL = 1

    import matplotlib.pyplot as plt
    from plotArrays import plotIsoMap
    from LightGray import LightGray
    from MediumGray import MediumGray
    from DarkGray import DarkGray
    from Black import Black

    # function without access to 'self' attributes
    def function(x, c0=1, c1=1, c2=1, c3=1, c4=1, c5=1, c6=1, c7=1):
        print('0')
        return 3.3 * np.array(np.sin(x[0]) + (x[1] - 1)**2)

    # method with access to 'self' attributes
    def method(self, x, c0=1, c1=1, c2=1, c3=1, c4=1, c5=1, c6=1, c7=1):
        print('1')
        return 3.3 * np.array(np.sin(x[0]) + (x[1] - 1)**2)

    if 1 or ALL:
        s = 'Forward() with demo function build-in into Model'
        print('-' * len(s) + '\n' + s + '\n' + '-' * len(s))

        x = gridInit(ranges=[(0, 1), (0, 1)], n=3)
        x, y = Forward(White(function))(x=x)
        plotIsoMap(x[:, 0], x[:, 1], y[:, 0])

    if 0 or ALL:
        s = 'Forward() with demo function build-in into Model'
        print('-' * len(s) + '\n' + s + '\n' + '-' * len(s))
        x, y = Forward(White('demo'))(ranges=[(1, 2), (3, 4)], cross=5)
        plotIsoMap(x[:, 0], x[:, 1], y[:, 0], scatter=True)

    if 0 or ALL:
        s = "Test of _initialInput()"
        print('-' * len(s) + '\n' + s + '\n' + '-' * len(s))

        op = Forward(White(function))
        for shape in ['cross', 'grid', 'rand']:
            s = "_initialInput(), shape: '" + shape + "'"
            print('-' * len(s) + '\n' + s + '\n' + '-' * len(s))

            x = op._initialInput(ranges=[(2, 4), (3, 7)], n=7, shape=shape)

            plt.title('test: _initialInput() shape: ' + shape)
            plt.scatter(x.T[0], x.T[1])
            plt.show()

        op(ranges=[[2, 2], [4, 5], [3, 3]], cross=(3, 5, 4))
        print('x:', op.model.x)
        plt.title('test: cross 3x1x8')
        plt.scatter(op.model.x.T[0], op.model.x.T[1])
        plt.show()

        op(ranges=[(2, 3), (3, 4)], grid=4)
        plt.title('test: n --> cross')
        plt.scatter(op.model.x.T[0], op.model.x.T[1])
        plt.show()

        op(ranges=[[2, 3], [3, 4]], grid=4)
        plt.title('test: grid 4x4')
        plt.scatter(op.model.x.T[0], op.model.x.T[1])
        plt.show()

        op(ranges=[[2, 3], [3, 4]], grid=(3, 5))
        plt.title('test: grid 3x5')
        plt.scatter(op.model.x.T[0], op.model.x.T[1])
        plt.show()

        op(ranges=[[2, 3], [3, 4]], rand=50)
        plt.title('test: rand')
        plt.scatter(op.model.x.T[0], op.model.x.T[1])
        plt.show()

    if 0 or ALL:
        s = "Forward, assign external function (without self-argument) to f"
        print('-' * len(s) + '\n' + s + '\n' + '-' * len(s))

        op = Forward(White(function))
        _, y = op(x=[[2, 3], [3, 4], [4, 5], [5, 6], [6, 7]])
        print('x:', op.model.x, '\ny1:', op.model.y)

    if 0 or ALL:
        s = "Forward, assign method (with 'self'-argument) to f"
        print('-' * len(s) + '\n' + s + '\n' + '-' * len(s))

        op = Forward(White(function))
        _, y = op(x=[[2, 3], [3, 4], [4, 5], [5, 6], [6, 7]])
        print('x:', op.model.x, '\ny1:', op.model.y)
