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
      2018-05-07 DWW
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
        op = Inverse(DarkGray(f=f))

        x, y = op(XY=(X, Y, xKeys, yKeys))               # only training
        x, y = op(X=X, Y=Y, x=xIni, y=yInv)              # training & inverse
        x, y = op(x=rand(5, [1, 4], [0, 9]), y=yInv)     # only inverse
        x, y = op(x=xIni, bounds=bounds, y=yInv)         # only inverse

        x, y = Inverse(LightGray('demo'))(XY=(X, Y), x=xIni, y=yInv)
    """

    def __init__(self, model, identifier='Inverse'):
        """
        Args:
            model (Model_like):
                box type model object

            identifier (string, optional):
                object identifier
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
    ALL = 0

    from plotArrays import plot_X_Y_Yref
    import Model as md

    from White import White
    from LightGray import LightGray
    from Black import Black

    def f(self, x, c0=1, c1=1, c2=1, c3=1, c4=1, c5=1, c6=1, c7=1):
        return [np.sin(c0 * x[0]) + c1 * (x[1] - 1)**2 + c2]

    if 0 or ALL:
        s = 'Inverse, ranges+rand replaced method f()'
        print('-' * len(s) + '\n' + s + '\n' + '-' * len(s))

        op = Inverse(White(f))
        x, y = op(x=md.grid(3, [-1, 1], [4, 8], [3, 5]), y=[0.5])
        op.plot()

    if 0 or ALL:
        s = 'Inverse, replaced model'
        print('-' * len(s) + '\n' + s + '\n' + '-' * len(s))

        op = Inverse(White('demo1'))
        xInv, yInv = op(x=md.grid((5, 4), [-10, 10], [-15, 15]), y=[0.5])
        op.plot()

    if 1 or ALL:
        s = 'Inverse operation on light gray box model'
        print('-' * len(s) + '\n' + s + '\n' + '-' * len(s))

        noise = 0.9
        n = 5
        X = md.grid((n, n), [-1, 5], [0, 3])

        #################

        Y_exact = White(f)(x=X, silent=True)
        Y_noise = md.noise(Y_exact, absolute=noise)
        plot_X_Y_Yref(X, Y_noise, Y_exact, ['X', 'Y_{nse}', 'Y_{exa}'])

        #################

        model = LightGray(f)
        Y_fit = model(X=X, Y=Y_noise, x=X, silent=True)
        plot_X_Y_Yref(X, Y_fit, Y_exact, ['X', 'Y_{fit}', 'Y_{exa}'])

        op = Inverse(model)
        x, y = op(x=md.grid((3, 2), [-10, 0], [1, 19]), y=[0.5])
        op.plot()

    if 0 or ALL:
        s = 'Inverse operation on empirical model'
        print('-' * len(s) + '\n' + s + '\n' + '-' * len(s))

        noise = 0.2random.
        n = 10
        X = md.grid(n, [-1, 5], [0, 3])

        Y_exact = White(f)(x=X)
        Y_noise = md.noise(Y_exact, absolute=noise)

        plot_X_Y_Yref(X, Y_noise, Y_exact, ['X', 'Y_{nse}', 'Y_{exa}'])
        model = Black()
        Y_blk = model(X=X, Y=Y_noise, neural=[8], n=3, epochs=500, x=X)
        plot_X_Y_Yref(X, Y_blk, Y_exact, ['X', 'Y_{blk}', 'Y_{exa}'])
        op = Inverse(model)
        xInv, yInv = op(y=[0.5], x=[(-10, 0), (1, 19)])
        op.plot()

    if 0 or ALL:
        s = 'Inverse operation on empirical model of tuned theoretial model'
        print('-' * len(s) + '\n' + s + '\n' + '-' * len(s))

        noise = 0.1
        n = 10
        X = md.grid(n, [-1, 5], [0, 3])

        # synthetic training data
        Y_exact = White(f)(x=X)
        Y_noise = md.noise(Y_exact, absolute=noise)
        plot_X_Y_Yref(X, Y_noise, Y_exact, ['X', 'Y_{nse}', 'Y_{exa}'])

        # trains and executes theoretical model Y_fit=f(X,w)
        Y_fit = LightGray(f)(X=X, Y=Y_noise, x=X)
        plot_X_Y_Yref(X, Y_fit, Y_exact, ['X', 'Y_{fit}', 'Y_{exa}'])

        # meta-model of theoretical model Y_emp=g(X,w)
        meta = Black()
        Y_meta = meta(X=X, Y=Y_fit, neural=[10], x=X)
        plot_X_Y_Yref(X, Y_fit, Y_meta, ['X', 'Y_{met}', 'Y_{exa}'])

        # inverse solution with meta-model (emp.model of tuned theo.model)
        if 1:
            op = Inverse(meta)
            xInv, yInv = op(x=[(-10, 0)], y=[0.5])
            op.plot()
            print('id:', op.identifier, 'xInv:', xInv, 'yInv:', yInv)
