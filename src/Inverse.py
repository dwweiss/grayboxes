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
      2018-02-07 DWW
"""

import numpy as np
from Model import randInit
from Optimum import Optimum


class Inverse(Optimum):
    """
    Finds the inverse x = f^{-1}(y)

    Examples:
        foo = Inverse(model='dark gray', f=f)

        X = [[... ]]  input of training
        Y = [[... ]]  target of training
        XY = (X, Y, xKeys, yKeys)  input, target and array keys of input
        xIni = [x00, x01, ... ]
        bounds = [(x0min, x0max), (x1min, x1max), ... ]
        yInv = (y0, y1, ... )

        def f(self, x): return x[0]

        x, y = foo(X=X, Y=Y, x=xIni, y=yInv)             # training & inverse
        x, y = foo(ranges=[(1,4),(0,9)], rand=5, y=yInv) # only inverse
        x, y = foo(x=xIni, bounds=bounds, y=yInv)        # only inverse
        x, y = foo(XY=(X, Y, xKeys, yKeys), y=yInv)      # only train
        x, y = foo(XY=(X, Y), y=yInv, f='demo')          # train with f_demo()
    """

    def __init__(self, identifier='Inverse', model=None, f=None, trainer=None):
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
        super().__init__(identifier, model=model, f=f, trainer=trainer)

    def objective(self, x, **kwargs):
        """
        Defines objective function for inverse problem

        Args:
            x (1D array_like of float):
                (input array of single data point)

            kwargs (dict, optional):
                keyword arguments

        Returns:
            (float):
                norm measuring difference between actual prediction and target
        """
        # x is input of prediction, x.shape: (nInp)
        yInv = self.model.predict(np.asfarray(x),
                                  **self.kwargsDel(kwargs, 'x'))

        # self.y is target, self.y.shape: (nOut,), yInv.shape: (1, nOut)
        norm = np.sqrt(np.mean(np.square(yInv[0] - self.y))) + self.penalty(x)

        self._trialHistory.append([x, yInv[0], norm])

        return norm


# Examples ####################################################################

if __name__ == '__main__':
    ALL = 0

    from Empirical import Empirical
    from Theoretical import Theoretical
    from plotArrays import plot_X_Y_Yref

    def f(self, x, c0=1, c1=1, c2=1, c3=1, c4=1, c5=1, c6=1, c7=1):
        return [np.sin(c0 * x[0]) + c1 * (x[1] - 1)**2 + c2]

    if 0 or ALL:
        s = 'Inverse, ranges+rand replaced method f()'
        print('-' * len(s) + '\n' + s + '\n' + '-' * len(s))

        foo = Inverse(f=f)
        foo(ranges=[(-1, 1), (4, 8), (3, 5)], grid=3, y=[0.5])
        foo.plot()

    if 0 or ALL:
        s = 'Inverse, replaced model'
        print('-' * len(s) + '\n' + s + '\n' + '-' * len(s))

        foo = Inverse()
        model = Theoretical(f=f)
        xIni = randInit([(-1, 1), (4, 8), (3, 5), (1, 1.1)], n=2)
        xInv, yInv = foo(x=xIni, y=[0.5], model=model)
        foo.plot()

    if 1 or ALL:
        s = 'Inverse operation on fine-tuned theoretial model'
        print('-' * len(s) + '\n' + s + '\n' + '-' * len(s))

        X0 = np.linspace(-1, 5, 20)
        X1 = np.linspace(0, 3, X0.size)
        X0, X1 = np.meshgrid(X0, X1)
        X = np.asfarray([X0.ravel(), X1.ravel()]).T
        noise = 0.9

        #################

        teo = Theoretical(f=f)
        Y_exa = np.array([np.array(teo.f(x)) for x in X])
        Y_noise = Y_exa + (1 - 2 * np.random.rand(Y_exa.shape[0],
                                                  Y_exa.shape[1])) * noise
        plot_X_Y_Yref(X, Y_noise, Y_exa, ['X', 'Y_{nos}', 'Y_{exa}'])

        #################

        Y_fit = teo(X=X, Y=Y_noise, f=f, x=X)

        plot_X_Y_Yref(X, Y_fit, Y_exa, ['X', 'Y_{fit}', 'Y_{exa}'])
        xInv, yInv = Inverse()(model=teo, y=[0.5], x=[(-10, 0), (1, 19)],
                               rand=10)
        foo.plot()

    if 0 or ALL:
        s = 'Inverse operation on empirical model'
        print('-' * len(s) + '\n' + s + '\n' + '-' * len(s))

        noise = 0.2
        n = 10
        X0 = np.linspace(-1, 5, n)
        X1 = np.linspace(0, 3, X0.size)
        X0, X1 = np.meshgrid(X0, X1)
        X = np.asfarray([X0.ravel(), X1.ravel()]).T

        Y_exa = np.array([np.array(f('dummy', x)) for x in X])
        delta = noise * (Y_exa.max() - Y_exa.min())
        Y_noise = Y_exa + np.random.normal(-delta, +delta, size=Y_exa.shape)

        plot_X_Y_Yref(X, Y_noise, Y_exa, ['X', 'Y_{nos}', 'Y_{exa}'])
        model = Empirical()
        Y_emp = model(X=X, Y=Y_noise.copy(), definition=[8], n=3, epochs=500,
                      x=X)
        plot_X_Y_Yref(X, Y_emp, Y_exa, ['X', 'Y_{emp}', 'Y_{exa}'])
        foo = Inverse()
        xInv, yInv = foo(model=model, y=[0.5], x=[(-10, 0), (1, 19)], rand=5)
        foo.plot()

    if 0 or ALL:
        s = 'Inverse operation on empirical model of tuned theoretial model'
        print('-' * len(s) + '\n' + s + '\n' + '-' * len(s))

        noise = 0.1
        n = 10

        X0 = np.linspace(-1, 5, n)
        X1 = np.linspace(0, 3, X0.size)
        X0, X1 = np.meshgrid(X0, X1)
        X = np.asfarray([X0.ravel(), X1.ravel()]).T

        teo = Theoretical(f=f)

        # synthetic training data
        Y_exa = np.array([np.array(teo.f(x)) for x in X])
        delta = noise * (Y_exa.max() - Y_exa.min())
        Y_noise = Y_exa + np.random.normal(-delta, +delta, size=(Y_exa.shape))
        plot_X_Y_Yref(X, Y_noise, Y_exa, ['X', 'Y_{nos}', 'Y_{exa}'])

        # trains and executes of theoretical model Y_fit=f(X,w)
        Y_fit = teo(X=X, Y=Y_noise, x=X)
        plot_X_Y_Yref(X, Y_fit, Y_exa, ['X', 'Y_{fit}', 'Y_{exa}'])

        # meta-model of theoretical model Y_emp=g(X,w)
        emp = Empirical()
        Y_emp = emp(X=X, Y=Y_fit, definition=[10], x=X)
        plot_X_Y_Yref(X, Y_fit, Y_emp, ['X', 'Y_{emp}', 'Y_{exa}'])

        # inverse solution with meta-model (emp.model of tuned theo.model)
        if 1:
            foo = Inverse()
            xInv, yInv = foo(y=[0.5], x=[(-10, 0)], model=emp)
            foo.plot()
            print('id:', foo.identifier, 'xInv:', xInv, 'yInv:', yInv)
