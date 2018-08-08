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
      2018-08-08 DWW
"""

import numpy as np
from grayboxes.model import Model
from grayboxes.lightgray import LightGray
from grayboxes.black import Black
from grayboxes.neural import Neural


class MediumGray(Model):
    """
    Medium gray box model comprising light gray box and black box submodels

    Training input self.X (process input) is union of common and unique input:
        X = X_com + X_unq
    """

    def __init__(self, f, identifier='MediumGray'):
        """
        Args:
            f (method or function):
                theoretical submodel f(self, x, *args, **kwargs) or
                f(x, *args, **kwargs) for single data point

                - argument 'x' to function f() corresponds to test input x_prc
                - model input is x_prc as the union of the common part of 'x'
                  and x_unq
                - in f() the subset x_unq of is unused

            identifier (str, optional):
                object identifier
        """
        super().__init__(identifier=identifier, f=f)

        self._local = None
        self._lightGray = LightGray(f=f)
        self._black = Black()

    @property
    def silent(self):
        return self._silent

    @silent.setter
    def silent(self, value):
        self._silent = value
        self._lightGray._silent = value
        if self._black is not None:
            self._black._silent = value

    def train(self, X, Y, **kwargs):
        """
        Trains model, stores X and X as self.X and self.Y, and stores result of
        best training trial as self.best

        Args:
            X (2D or 1D array_like of float):
                training input X_prc, shape: (nPoint, nInp) or shape: (nPoint,)

            Y (2D or 1D array_like of float):
                training target Y_com, shape: (nPoint, nOut) or shape:(nPoint,)

            kwargs (dict, optional):
                keyword arguments:

                bounds (2-tuple of float or 2-tuple of 1D array_like of float):
                    list of pairs (xMin, xMax) limiting tuning parameters

                local (int or None):
                    size of subset sizes if local training type of medium gray
                        box model
                    if 'local' is None, FalsE or 0, a single global network is
                        trained without local tuning and data collection

                methods (str or list of str):
                    optimizer method of
                        - scipy.optimizer.minimize or
                        - genetic algorithm
                    see LightGray.validMethods
                    default: 'BFGS'

                shuffle (bool):
                    if 'local' is geater 1 and 'shuffled' is True, then x- and
                    y-datasets are shuffled before split to local datasets
                    default: True

                tun0 (2D or 1D array_like of float):
                    sequence of initial guess of the tuning parameter set,
                    If missing, then initial values will be all 1.0
                    tun0.shape[1] is the number of tuning parameters if 2D arr.
                    see LightGray.train()

                ... network options, see class Neural

        Returns:
            see Model.train()

        Example:
            Method f(self, x) or function f(x) is assigned to self.f, example:

                def f(self, x, *args, **kwargs):
                    tun = args if len(args) >= 3 else np.ones(3)

                    y0 = tun[0] * x[0]*x[0] + tun[1] * x[1]
                    y1 = x[1] * tun[2]
                    return [y0, y1]

                # training data
                X = [[..], [..], ..]
                Y = [[..], [..], ..]

                # test data
                x = [[..], [..], ..]

                # expanded form:
                model = MediumGray(f=f)
                model.train(X, Y, methods='ga', neurons=[])
                y = model.predict(x)

                # compact form:
                y = MediumGray(f)(X=X, Y=Y, x=x, methods='ga', neurons=[])
        """
        self.X = X if X is not None else self.X
        self.Y = Y if Y is not None else self.Y

        opt = self.kwargsDel(kwargs, ['X', 'Y', 'local'])
        self.silent = kwargs.get('silent', self.silent)
        self._local = kwargs.get('local', None)
        neurons = kwargs.get('neurons', [])

        if self._local:
            self.write('+++ Medium gray (local tun: ' + str(self._local) + ')')

            shuffle = kwargs.get('shuffle', self._local > 1)
            methods = kwargs.get('methods', ['bfgs', 'rprop'])
            nPoint = self.X.shape[0]
            nSub = nPoint // np.clip(self._local, 1, nPoint)

            xyRnd2d = np.c_[self.X, self.Y]
            if shuffle:
                np.random.shuffle(xyRnd2d)
            xyAll3d = np.array_split(xyRnd2d, nSub)          # list of 2d array

            xTunAll2d = []
            nInp = self.X.shape[1]
            for xy in xyAll3d:
                XY = np.hsplit(xy, [nInp])
                X, Y = XY[0], XY[1]
                self._lightGray.Y = None
                res = self._lightGray.train(X=X, Y=Y, **opt)
                xTun1d = res['weights']
                if xTun1d is not None:
                    for i in range(xy.shape[0]):
                        xTunAll2d.append(xTun1d)

            if len(xyAll3d) > 1:
                self.write('            (generalization)')
                res = self._black.train(X=xyRnd2d[:, :nInp], Y=xTunAll2d,
                                        neurons=neurons, methods=methods)
                self.weights = None
            else:
                self.weights = xTunAll2d[0]        # local==X.shape[0]: const w

            # TODO remove next line after test
            self.__weightsForPresentation = xTunAll2d   # only for presentation

        else:
            self.write('+++ Medium gray (global training)')

            methods = kwargs.get('methods', ['genetic', 'derivative'])

            if self._black is not None:
                del self._black
            self._black = Neural(f=self.f)
            self._black.train(self.X, self.Y, neurons=neurons, methods=methods)

        self.ready = True
        self.best = self.error(self.X, self.Y, **opt)
        return self.best

    def predict(self, x, **kwargs):
        """
        Executes Model, stores input x as self.x and output as self.y

        Args:
            x (2D or 1D array_like of float):
                prediction input, shape: (nPoint, nInp) or shape: (nInp,)

            kwargs (dict, optional):
                keyword arguments

        Returns:
            (2D array of float):
                prediction output, shape: (nPoint, nOut)
        """
        assert self._black is not None and self._black.ready
        opt = self.kwargsDel(kwargs, 'x')
        self.x = x

        if self._local is not None:
            if self.weights is None:
                yAll = []
                for xPrc in self.x:
                    if xPrc[0] is not None:
                        xTun = self._black.predict(x=xPrc, **opt)[0]
                        yAll.append(Model.predict(self, xPrc, *xTun, **opt)[0])
                self.y = yAll
            else:
                # local==X.shape[0]: const w
                self.y = Model.predict(self, self.x, *self.weights, **opt)
        else:
            self.y = self._black.predict(x=x, **opt)

        return self.y


# Examples ####################################################################

if __name__ == '__main__':
    ALL = 0

    from grayboxes.plotarrays import plot_X_Y_Yref
    from grayboxes.model import Model, grid, noise, rand
    from grayboxes.white import White
    import matplotlib.pyplot as plt

    nTun = 3

    def f(self, x, *args, **kwargs):
        """
        Theoretical submodel for single data point

        Aargs:
            x (1D array_like of float):
                common input

            args (argument list, optional):
                tuning parameters as positional arguments

            kwargs (dict, optional):
                keyword arguments {str: float/int/str}
        """
        if x is None:
            return np.ones(nTun)
        tun = args if len(args) >= nTun else np.ones(nTun)

        y0 = tun[0] + tun[1] * np.sin(tun[2] * x[0]) + tun[1] * (x[1] - 1.5)**2
        return [y0]

    s = 'Creates exact output y_exa(X) and adds noise. Target is Y(X)'
    print('-' * len(s) + '\n' + s + '\n' + '-' * len(s))

    noise_abs = 0.1
    noise_rel = 5e-2
    X = grid(10, [-1., 2.], [0., 3.])
    y_exa = White(f)(x=X, silent=True)
    Y = noise(y_exa, absolute=noise_abs, relative=noise_rel)
    if 0:
        plot_X_Y_Yref(X, Y, y_exa, ['X', 'Y_{nse}', 'y_{exa}'])

    methods = [
                # 'all',
                # 'L-BFGS-B',
                'BFGS',
                # 'Powell',
                # 'Nelder-Mead',
                # 'differential_evolution',
                # 'basinhopping',
                # 'ga',
                ]

    if 1 or ALL:
        s = 'Tunes model, compare: y(X) vs y_exa(X)'
        print('-' * len(s) + '\n' + s + '\n' + '-' * len(s))

        # train with n1 random initial tuning parameter help, each of size n2
        local, n1, n2 = 10, 1, 3
        mgr, lgr = MediumGray(f), LightGray(f)
        mgr.silent, lgr.silent = True, True
        tun0 = rand(n1, *(n2 * [[0., 2.]]))

        L2 = np.inf
        yMgr, tunMgr = None, None
        for local in range(1, 3):
            for neurons in range(2, 4):
                y = mgr(X=X, Y=Y, x=X, methods=methods, tun0=tun0, nItMax=5000,
                        bounds=nTun*[(-1., 3.)], neurons=[neurons], trials=3,
                        local=local)
                print('L2(neurons:', str(neurons)+'): ', mgr.best['L2'],
                      end='')
                if L2 > mgr.best['L2']:
                    L2 = mgr.best['L2']
                    print('  *** better', end='')
                    yMgr, tunMgr = y, mgr.weights
                print()
        assert yMgr is not None

        yLgr = lgr(X=X, Y=Y, x=X, methods=methods, nItMax=5000, tun0=tun0)
        print('lgr.w:', lgr.weights)

        if mgr.weights is None:
            xTun = mgr._black.predict(x=X)
            for i in range(xTun.shape[1]):
                plt.plot(xTun[:, i], ls='-',
                         label='$x^{loc}_{tun,'+str(i)+'}$')
        for i in range(len(lgr.weights)):
            plt.axhline(lgr.weights[i], ls='--',
                        label='$x^{lgr}_{tun,'+str(i)+'}$')
        # plt.ylim(max(0, 1.05*min(lgr.weights)),
        #          min(2, 0.95*max(lgr.weights)))
        plt.legend(bbox_to_anchor=(1.1, 1.05))
        plt.show()

        plot_X_Y_Yref(X, yLgr, y_exa, ['X', 'y_{lgr}', 'y_{exa}'])
        plot_X_Y_Yref(X, yMgr, y_exa, ['X', 'y_{mgr}', 'y_{exa}'])
