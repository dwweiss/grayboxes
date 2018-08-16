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
      2018-08-16 DWW
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
