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
from Minimum import Minimum


class Inverse(Minimum):
    """
    Finds the inverse x = f^{-1}(y)

    Examples:

        X = [[... ]]  input of training
        Y = [[... ]]  target of training
        xIni = [x00, x01, ... ]
        bounds = [(x0min, x0max), (x1min, x1max), ... ]
        yInv = (y0, y1, ... )

        def f(self, x): return (x[0])
        op = Inverse(color='darkGray', f=f)

        x, y = op(XY=(X, Y, xKeys, yKeys))               # only training
        x, y = op(X=X, Y=Y, x=xIni, y=yInv)              # training & inverse
        x, y = op(ranges=[(1,4),(0,9)], rand=5, y=yInv)  # only inverse
        x, y = op(x=xIni, bounds=bounds, y=yInv)         # only inverse

        x, y = Inverse(f='demo', color='lightGray')(XY=(X, Y), x=xIni, y=yInv)
    """

    def __init__(self, model, identifier='Inverse'):
        """
        Args:
            model (Model_like):
                generic model object, supersedes all following options

            identifier (string, optional):
                class identifier
        """
        super().__init__(model=model, identifier=identifier)

    def objective(self, x, **kwargs):
        """
        Defines objective function for inverse problem

        Args:
            x (1D array_like of float):
                (input array of single data point)

            kwargs (dict, optional):
                keyword arguments for predict()

        Returns:
            (float):
                norm measuring difference between actual prediction and target
        """
        # x is input of prediction, x.shape: (nInp,)
        yInv = self.model.predict(np.asfarray(x),
                                  **self.kwargsDel(kwargs, 'x'))

        # self.y is target, self.y.shape: (nOut,), yInv.shape: (1, nOut)
        norm = np.sqrt(np.mean(np.square(yInv[0] - self.y))) + self.penalty(x)

        self._trialHistory.append([x, yInv[0], norm])

        return norm


# Examples ####################################################################

if __name__ == '__main__':
    ALL = 1

    from Model import gridInit
    from White import White
    from LightGray import LightGray
    from Black import Black
    from plotArrays import plot_X_Y_Yref

    def f(self, x, c0=1, c1=1, c2=1, c3=1, c4=1, c5=1, c6=1, c7=1):
        return [np.sin(c0 * x[0]) + c1 * (x[1] - 1)**2 + c2]

    if 0 or ALL:
        s = 'Inverse, ranges+rand replaced method f()'
        print('-' * len(s) + '\n' + s + '\n' + '-' * len(s))

        op = Inverse(White(f))
        x, y = op(ranges=[(-1, 1), (4, 8), (3, 5)], grid=3, y=[0.5])
        op.plot()

    if 0 or ALL:
        s = 'Inverse, replaced model'
        print('-' * len(s) + '\n' + s + '\n' + '-' * len(s))

        op = Inverse(White('demo1'))
        xInv, yInv = op(ranges=[(-10, 10), (-15, 15)], grid=(5, 4), y=[0.5])
        op.plot()

    if 0 or ALL:
        s = 'Inverse operation on light gray box model'
        print('-' * len(s) + '\n' + s + '\n' + '-' * len(s))

        noise = 0.9
        n = 5
        X = gridInit(ranges=[(-1, 5), (0, 3)], n=n)

        #################

        Y_exa = White(f)(x=X)
        Y_noise = Y_exa + (1 - 2 * np.random.rand(Y_exa.shape[0],
                                                  Y_exa.shape[1])) * noise
        plot_X_Y_Yref(X, Y_noise, Y_exa, ['X', 'Y_{nse}', 'Y_{exa}'])

        #################

        model = LightGray(f)
        Y_fit = model(X=X, Y=Y_noise, x=X)
        plot_X_Y_Yref(X, Y_fit, Y_exa, ['X', 'Y_{fit}', 'Y_{exa}'])

        x, y = Inverse(model)(y=[0.5], ranges=[(-10, 0), (1, 19)], grid=(3, 2))
        op.plot()

    if 1 or ALL:
        s = 'Inverse operation on empirical model'
        print('-' * len(s) + '\n' + s + '\n' + '-' * len(s))

        noise = 0.2
        n = 10
        X = gridInit(ranges=((-1, 5), (0, 3)), n=n)

        Y_exa = White(f)(x=X)
        delta = noise * (Y_exa.max() - Y_exa.min())
        Y_noise = Y_exa + np.random.normal(-delta, +delta, size=Y_exa.shape)

        plot_X_Y_Yref(X, Y_noise, Y_exa, ['X', 'Y_{nos}', 'Y_{exa}'])
        model = Black()
        Y_blk = model(X=X, Y=Y_noise.copy(), neural=[8], n=3, epochs=500, x=X)
        plot_X_Y_Yref(X, Y_blk, Y_exa, ['X', 'Y_{blk}', 'Y_{exa}'])
        xInv, yInv = Inverse(model)(y=[0.5], x=[(-10, 0), (1, 19)])
        op.plot()

    if 0 or ALL:
        s = 'Inverse operation on empirical model of tuned theoretial model'
        print('-' * len(s) + '\n' + s + '\n' + '-' * len(s))

        noise = 0.1
        n = 10
        X = gridInit(ranges=((-1, 5), (0, 3)), n=n)

        # synthetic training data
        Y_exa = White(f)(x=X)
        delta = noise * (Y_exa.max() - Y_exa.min())
        Y_noise = Y_exa + np.random.normal(-delta, +delta, size=(Y_exa.shape))
        plot_X_Y_Yref(X, Y_noise, Y_exa, ['X', 'Y_{nos}', 'Y_{exa}'])

        # trains and executes of theoretical model Y_fit=f(X,w)
        Y_fit = LightGray(f)(X=X, Y=Y_noise, x=X)
        plot_X_Y_Yref(X, Y_fit, Y_exa, ['X', 'Y_{fit}', 'Y_{exa}'])

        # meta-model of theoretical model Y_emp=g(X,w)
        meta = Black()
        Y_meta = meta(X=X, Y=Y_fit, neural=[10], x=X)
        plot_X_Y_Yref(X, Y_fit, Y_meta, ['X', 'Y_{mta}', 'Y_{exa}'])

        # inverse solution with meta-model (emp.model of tuned theo.model)
        if 1:
            op = Inverse(meta)
            xInv, yInv = op(y=[0.5], x=[(-10, 0)])
            op.plot()
            print('id:', op.identifier, 'xInv:', xInv, 'yInv:', yInv)
