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
      2018-02-23 DWW
"""

import numpy as np

from Base import Base
from Model import randInit, crossInit, gridInit
from Empirical import Empirical
from Theoretical import Theoretical
from Hybrid import Hybrid


class Forward(Base):
    """
    Predicts with generic model for series of data points: x(iPoint, jInp)

    Examples:
        foo = Forward()

        X_prc = [[... ]]  input of training
        Y_prc = [[... ]]  target of training
        x_mod = [[... ]]  input of prediction

        def function(x, c0=1, c1=1, c2=1, c3=1, c4=1, c5=1, c6=1, c7=1):
            return 2.2 * np.array(np.sin(x[0]) + (x[1] - 1)**2)

        def method(self, x, c0=1, c1=1, c2=1, c3=1, c4=1, c5=1, c6=1, c7=1):
            return 3.3 * np.array(np.sin(x[0]) + (x[1] - 1)**2)

        # 'f' has to be assigned only one time
        y = foo(X=X_prc, Y=Y_prc, x=x_mod, f=function)   # train and predict
        y = foo(X=X_prc, Y=Y_prc, x=x_mod, f=method)     # train and predict
        y = foo(x=x_mod)      # predict only, 'f' assignment is still active

    Note:
        - Class Forward is not derived from class Model
        - The instance of the generic model has to be assigned to: self.model
    """

    def __init__(self, identifier='Forward', model=None, f=None, trainer=None):
        """
        Args:
            identifier (string, optional):
                class identifier

            model (method, optional):
                generic model

            f (method, optional):
                white box model f(x)

            trainer (method, optional):
                training method
        """
        super().__init__(identifier)

        self._model = model if model is not None else Theoretical()
        if f is not None:
            self.model.f = f
        if trainer is not None:
            self.model.trainer = trainer

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
        if isinstance(value, str):
            value = value.lower()
            Type = type(self.method)
            if value[0] in ('w', 'l'):
                if not issubclass(Type, Theoretical):
                    self.model = Theoretical()
            elif value[0] == 'b':
                if not issubclass(Type, Empirical):
                    self.model = Empirical()
            if value[0] in ('m', 'd'):
                if not issubclass(Type, Hybrid):
                    self.model = Hybrid()
                    self.model.hybridType = value
            else:
                assert 0, 'value:' + value
        else:
            assert issubclass(type(value), (Theoretical, Empirical, Hybrid)), \
                'invalid model type: ' + str(type(value))
            xx, ff, tt = None, None, None
            if self._model is not None:
                xx, ff, tt = self.model.x, self.model.f, self.model.trainer
            self._model = value
            if self._model is not None:
                if xx is not None:
                    self.model.x = xx
                if ff is not None:
                    self.model.f = ff
                if tt is not None:
                    self.model.trainer = tt
                self._model.logFile = self.logFile

    def initialInput(self, ranges, n=5, shape='cross'):
        """
        Sets uniformly spaced or randomly distributed input, for instance for
        3 input with 5 axis points: initialInput(ranges=[(2, 3), (-5, -4),
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
        1) Assigns Model instance
        2) Assigns f() method, both variants: with or without 'self' argument
        3) Sets training data set (X, Y) or XY, and model input x
        4) Trains model if (X, Y) or XY are not None
        5) Generates model input x from 'ranges' and shape ('cross', 'grid',
           'rand') if x is None

        Args:
            kwargs (dict, optional):
                keyword arguments:

                model (Model_like or string):
                    model to be assigned to attribute self.model or
                    string defining the model type ('w', 'l', 'm', 'd', 'b')

                f (method):
                    method with or without 'self' argument to be assigned to
                    attribute self.model.f

                X (2D array_like of float):
                    input of training

                Y (2D array_like of float):
                    target of training

                x (2D array_like of float):
                    input to forward prediction or to sensitivity analysis

                ranges (2D array_like of float):
                    array of min/max pairs if cross should be generated

                cross (int or list of int):
                    number of cross point per axis if cross should be generated

                grid (int or list of int):
                    number of grid point per axis if grid should be generated

                rand (int):
                    number of points if random array should be generated

                n (int):
                    number of cross point per axis if cross should be generated

        Note:
            1) 'x' keyword overrules 'ranges' keyword
            2) if more than one arg. out of [cross, grid, rand, n] is given,
                the order in this list defines the relevant argument
        """
        super().pre(**kwargs)

        # assign model to operation
        model = kwargs.get('model', None)
        if model is not None:
            self.model = model

        # assigns f() method to model
        f = kwargs.get('f', None)
        if f is not None:
            self.model.f = f

        # trains model
        X, Y = kwargs.get('X', None), kwargs.get('Y', None)
        if X is not None and Y is not None:
            self.model.train(X, Y, **self.kwargsDel(kwargs, ['X', 'Y']))
            # assert self.model.ready()

        # sets input for prediction
        x = kwargs.get('x', None)
        if x is None:
            ranges = kwargs.get('ranges', None)
            if ranges is not None:
                for shape in ['cross', 'grid', 'rand', 'n']:
                    if shape in kwargs:
                        n = kwargs[shape]
                        if shape == 'n':
                            shape = 'cross'
                        if not n:
                            n = 5
                        x = self.initialInput(ranges, n, shape)
                        break
            else:
                self.write("!!! Forward: neither 'x' nor 'ranges' is not None")
                self.write("!!! Continues with build-in: Theoretical.f()")
                x = None
        if type(self).__name__ in ('Optimum', 'Inverse'):
            self.x = np.atleast_2d(x) if x is not None else None
        else:
            self.model.x = np.atleast_2d(x) if x is not None else None

    def task(self, **kwargs):
        """
        This task() method is only for Forward and Sensitivity; Optimum has its
        own task() method implementation

        Args:
            kwargs (dict, optional):
                keyword arguments passed to super.task()

        Return:
            x, y (2D arrays of float):
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

    # function without access to 'self' attributes
    def function(x, c0=1, c1=1, c2=1, c3=1, c4=1, c5=1, c6=1, c7=1):
        print('0')
        return 3.3 * np.array(np.sin(x[0]) + (x[1] - 1)**2)

    # method with access to 'self' attributes
    def method(self, x, c0=1, c1=1, c2=1, c3=1, c4=1, c5=1, c6=1, c7=1):
        print('1')
        return 3.3 * np.array(np.sin(x[0]) + (x[1] - 1)**2)

    if 0 or ALL:
        s = "Test of initialInput()"
        print('-' * len(s) + '\n' + s + '\n' + '-' * len(s))

        foo = Forward(f=function)
        for shape in ['cross', 'grid', 'rand']:
            s = "initialInput(), shape: '" + shape + "'"
            print('-' * len(s) + '\n' + s + '\n' + '-' * len(s))

            x = foo.initialInput(ranges=[(2, 4), (3, 7)], n=7, shape=shape)

            plt.title('test: initialInput() shape: ' + shape)
            plt.scatter(x.T[0], x.T[1])
            plt.show()

        foo(ranges=[[2, 2], [4, 5], [3, 3]], cross=(3, 5, 4))
        print('x:', foo.model.x)
        plt.title('test: cross 3x1x8')
        plt.scatter(foo.model.x.T[0], foo.model.x.T[1])
        plt.show()

        foo(ranges=[[2, 3], [3, 4]], n=4)
        plt.title('test: n --> cross')
        plt.scatter(foo.model.x.T[0], foo.model.x.T[1])
        plt.show()

        foo(ranges=[[2, 3], [3, 4]], grid=4)
        plt.title('test: grid 4x4')
        plt.scatter(foo.model.x.T[0], foo.model.x.T[1])
        plt.show()

        foo(ranges=[[2, 3], [3, 4]], grid=(3, 5))
        plt.title('test: grid 3x5')
        plt.scatter(foo.model.x.T[0], foo.model.x.T[1])
        plt.show()

        foo(ranges=[[2, 3], [3, 4]], rand=50)
        plt.title('test: rand')
        plt.scatter(foo.model.x.T[0], foo.model.x.T[1])
        plt.show()

    if 0 or ALL:
        s = "Forward, assign external function (without self-argument) to f"
        print('-' * len(s) + '\n' + s + '\n' + '-' * len(s))

        foo = Forward(f=function)
        y = foo(x=[[2, 3], [3, 4], [4, 5], [5, 6], [6, 7]])
        print('x:', foo.model.x, '\ny1:', foo.model.y)

    if 0 or ALL:
        s = "Forward, assign method (with 'self'-argument) to f"
        print('-' * len(s) + '\n' + s + '\n' + '-' * len(s))

        foo = Forward(f=function)
        y = foo(x=[[2, 3], [3, 4], [4, 5], [5, 6], [6, 7]])
        print('x:', foo.model.x, '\ny1:', foo.model.y)

    if 0 or ALL:
        s = "Forward, assign new 'model'  to model"
        print('-' * len(s) + '\n' + s + '\n' + '-' * len(s))

        foo = Forward()
        model = foo.model
        model.f = method.__get__(model, Theoretical)
        y = foo(x=[[2, 3], [3, 4], [4, 5], [5, 6], [6, 7]], model=model)
        print('x:', foo.model.x, '\ny1:', foo.model.y)
