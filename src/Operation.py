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
      2018-03-13 DWW
"""

from Base import Base
from Forward import Forward
from Sensitivity import Sensitivity
from Optimum import Optimum
from Inverse import Inverse


class Operation(Base):
    """
    Convenience class proving access to all operations on models (Forward,
    Sensitivity, Optimum, Inverse).
    """

    def __init__(self, identifier='Operation'):
        super().__init__(identifier)

        self._validOperations = [Forward, Sensitivity, Optimum, Inverse]

        # initial value of self._operation is an instance of class Forward
        self._operation = self._validOperations[0]()

    def op2str(self, value):
        """
        Converts operation class to string

        Args:
            value (None, operations class or instance of it):
                instance of an operation class (see self._validOperations)

        Returns:
            (string):
                name of operation class in lower case
        """
        if issubclass(value, self._validOperations):
            s = type(value).__name__.lower()
        elif value in self._validOperations:
            s = value.__name__.lower()
        elif value is None:
            s = str(None)
        else:
            self.warning("invalid value: '", value, "' is not valid operation")
            s = None
        return s

    @property
    def operation(self):
        """
        Returns:
            (an operation class):
                Actual instance of an operation class
        """
        return self._operation

    @operation.setter
    def operation(self, value):
        """
        Sets actual instance of an operation class

        Args:
            value (None, string, class or instance of an operation class):
                new operation

        Side effects:
            if self.operation is not None, it will be deleted and set to None
        """
        if self._operation is not None:
            del self._operation
        self._operation = None
        if value is None:
            return
        if isinstance(value, str):
            for x in self._validOperations:
                if value.lower().startswith(x.__name__.lower()[:3]):
                    self._operation = x()
                    return
            assert 0, "operation.setter, value:'" + str(value) + "'"
        elif value in self._validOperations:
            self._operation = value()
        elif type(value) in self._validOperations:
            self._operation = value
        else:
            assert 0, "operation.setter, else: invalid type:'" + \
                str(type(value)) + "' value: '" + str(value) + "'"

    def pre(self, **kwargs):
        super().pre(**kwargs)

        assert self.operation is not None
        assert isinstance(self.operation, self._validOperations)
        assert self.operation.model is not None, 'model is not assigned'

        if not self.operation.model.ready():
            # trains model
            XY = kwargs.get('XY', None)
            if XY is not None:
                self.model.train(XY=XY, **self.kwargsDel(kwargs, ['XY']))
            else:
                X, Y = kwargs.get('X', None), kwargs.get('Y', None)
                if X is not None and Y is not None:
                    assert X.shape[0] == Y.shape[0], \
                        str(X.shape[0]) + ' ' + str(Y.shape[0])
                    self.model.train(X=X, Y=Y,
                                     **self.kwargsDel(kwargs, ['X', 'Y']))

    def task(self, **kwargs):
        super().task(**kwargs)

        x = kwargs.get('x', None)
        if x is None:
            return None, None
        x, y = self.operation(x=x, **self.kwargsDel(kwargs, 'x'))
        return x, y


# Examples ####################################################################

if __name__ == '__main__':
    ALL = 1

    import numpy as np
    from Theoretical import Theoretical
    from plotArrays import plot_X_Y_Yref

    # user defined method
    def f(self, x, c0=1, c1=1, c2=1, c3=1, c4=1, c5=1, c6=1, c7=1):
        x0, x1 = x[0], x[1]
        y0 = c0 * np.sin(c1 * x0) + c2 * (x1 - 1.5)**2 + c3
        return [y0]

    if 0 or ALL:
        s = 'Check operation.setter'
        print('-' * len(s) + '\n' + s + '\n' + '-' * len(s))

        foo = Operation()
        for op in ['inv', 'Inverse', Optimum(), Sensitivity, None,
                   # 1.234,
                   # 'wrong'
                   ]:
            foo.operation = op
            print('operation:', type(foo.operation).__name__,
                  type(foo.operation))
        print()

        for op in foo._validOperations:
            foo.operation = op
            print('operation:', type(foo.operation).__name__,
                  type(foo.operation))

    if 1 or ALL:
        s = 'All operation types'
        print('-' * len(s) + '\n' + s + '\n' + '-' * len(s))
        model = Theoretical(f=f)

        noise = 0.5
        n = 20
        X0 = np.linspace(-1, 5, n)
        X1 = np.linspace(0, 3, X0.size)
        X0, X1 = np.meshgrid(X0, X1)
        X = np.asfarray([X0.ravel(), X1.ravel()]).T     # X.shape = (nPoint, 2)

        Y_exa = np.array([np.array(model.f(x)) for x in X])
        Y_noise = Y_exa + (1 - 2 * np.random.rand(Y_exa.shape[0],
                                                  Y_exa.shape[1])) * noise
        plot_X_Y_Yref(X, Y_noise, Y_exa, ['X', 'Y_{nos}', 'Y_{exa}'])

        Y_fit = model(X=X, Y=Y_noise, x=X)
        plot_X_Y_Yref(X, Y_fit, Y_exa, ['X', 'Y_{fit}', 'Y_{exa}'])

        operations = Operation()
        for op in operations._validOperations:
            print('=', op.__name__, '=' * 50)
            foo = op(model=model)
            x, y = foo(X=X, Y=Y_noise, x=[(0, 0.5), (1, 3), (1, 2)], y=(0.5))
            if type(foo) in (Optimum, Inverse):
                foo.plot()
            print("+++ Operation:'" + type(foo).__name__ + "'x:", x, 'y:', y)
