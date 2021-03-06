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
      2020-02-03 DWW
"""

import numpy as np
from typing import Any, Dict, List

try:
    from grayboxes.black import Black
    from grayboxes.boxmodel import BoxModel
    from grayboxes.datatype import Float2D, Function
    from grayboxes.lightgray import LightGray
    from grayboxes.neuraltf import Neural
except:
    from black import Black
    from boxmodel import BoxModel
    from datatype import Float2D, Function
    from lightgray import LightGray
    from neuraltf import Neural


class MediumGray(BoxModel):
    """
    Medium gray box model comprising light gray and black box submodels

    Training input self.X (process input) is union of common and unique
    input:
        X = X_com + X_unq
    """

    def __init__(self, f: Function, identifier: str = 'MediumGray') -> None:
        """
        Args:
            f:
                theoretical submodel f(self, x, *c, **kwargs) or
                f(x, *c, **kwargs) for single data point

                - argument 'x' to function f() corresponds to test input
                  x_prc
                - model input is x_prc, (x_prc = x_com + x_unq)
                - the subset x_unq of x_prc is unused in f()

            identifier:
                Unique object identifier
        """
        super().__init__(identifier=identifier, f=f)

        self._local_size = None
        self._light_gray = LightGray(f=f)
        self._black = Black()

    @property
    def silent(self) -> bool:
        return self._silent

    @silent.setter
    def silent(self, value: bool) -> None:
        self._silent = value
        self._light_gray._silent = value
        if self._black is not None:
            self._black._silent = value

    def train(self, X: Float2D, Y: Float2D, **kwargs: Any) -> Dict[str, Any]:
        """
        Trains model, stores X and X as self.X and self.Y, and stores 
        result of best training trial as self.metrics

        Args:
            X:
                training input X_prc, shape: (n_point, n_inp)
                shape (n_point,) is tolerated

            Y:
                training target Y_com, shape: (n_point, n_out)
                shape (n_point,) is tolerated

        Kwargs:
            bounds (2-tuple of float or 2-tuple of 1D array of float):
                list of pairs (x_min, x_max) limiting tuning parameters

            local (int or None):
                size of subset sizes if local training type of medium gray
                    box model
                if 'local' is None, False or 0, a single global network is
                    trained without local tuning and data collection

            trainer (str or list of str):
                optimizer method of
                    - scipy.optimizer.minimize or
                    - genetic algorithm
                see LightGray.valid_trainers
                default: 'BFGS'

            shuffle (bool):
                if 'local' is geater 1 and 'shuffle' is True, then
                x- and y-datasets are shuffled before split to local
                datasets
                default: True

            c_ini (2D or 1D array of float):
                sequence of initial guess of tuning parameter set.
                If missing, then initial values will be all 1.0
                c_ini.shape[1] is the number of tuning parameters
                    if c_ini is an 2D array
                see LightGray.train()

            ... network options, see class Neural

        Returns:
            metrics of best training, see definition in BoxModel.train()

        Example:
            Method f(self, x) or function f(x) is assigned to self.f:

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
                model.train(X, Y, trainer='ga', neurons=[])
                y = model.predict(x)

                # compact form:
                y = MediumGray(f)(X=X, Y=Y, x=x, trainer='ga', neurons=[])
        """
        self.set_XY(X, Y)
        print('mdm 150 X Y', self.X.shape, self.Y.shape)

        kwargs_ = self.kwargs_del(kwargs, ['X', 'Y', 'local'])
        kwargs_['correct_xy_shape'] = False
        self.silent = kwargs.get('silent', self.silent)
        self._local_size = kwargs.get('local', None)
        neurons = kwargs.get('neurons', [])

        if self._local_size:
            self.write('+++ Medium gray (local size: ' +
                       str(self._local_size) + ')')

            shuffle = kwargs.get('shuffle', self._local_size > 1)
            trainer = kwargs.get('trainer', ['bfgs', 'rprop'])
            n_point = self.X.shape[0]
            n_sub = n_point // np.clip(self._local_size, 1, n_point)

            xy_rnd2d = np.c_[self.X, self.Y]
            if shuffle:
                np.random.shuffle(xy_rnd2d)
            xy_all3d = np.array_split(xy_rnd2d, n_sub)   # 2D arr. list

            x_tun_all2d: List[np.ndarray] = []
            n_inp = self.X.shape[1]
            for xy in xy_all3d:
                XY = np.hsplit(xy, [n_inp])
                X, Y = XY[0], XY[1]
                self._light_gray.Y = None
                
                print('mgr176 X Y', X.shape, Y.shape)
                
                res = self._light_gray.train(X=X, Y=Y, **kwargs_)
                
                print('mgr179 X Y', self._light_gray.X.shape, 
                      self._light_gray.Y.shape)
                
                x_tun1d = self._light_gray.weights
                if x_tun1d is not None:
                    for i in range(xy.shape[0]):
                        x_tun_all2d.append(x_tun1d)

            if len(xy_all3d) > 1:
                self.write('            (generalization)')
                res = self._black.train(X=xy_rnd2d[:, :n_inp], Y=x_tun_all2d,
                                        neurons=neurons, trainer=trainer)
                self.weights = None
            else:
                # constant weights if local == X.shape[0], (1 group)
                self.weights = x_tun_all2d[0]

            # TODO remove next line after final release of this module
            self.__weightsForPresentation = x_tun_all2d

        else:
            self.write('+++ Medium gray (global training)')

            trainer = kwargs.get('trainer', ['genetic', 'derivative'])

            if self._black is not None:
                del self._black
            self._black = Neural(f=self.f)
            self._black.train(self.X, self.Y, neurons=neurons, 
                              trainer=trainer)

        self.ready = True
        self.metrics = self.evaluate(self.X, self.Y, **kwargs_)
        self.metrics['ready'] = self.ready
        
        return self.metrics

    def predict(self, x: Float2D, *args, **kwargs: Any) -> Float2D:
        """
        Executes box model, stores input x as self.x and output as self.y

        Args:
            x:
                prediction input, shape: (n_point, n_inp) 
                shape: (n_inp,) is tolerated

            args:
                positional arguments to be passed to theoretical
                submodel f()

        Kwargs:
            Keyword arguments to be passed to BoxModel.predict() and to
            self._black.predict()

        Returns:
            prediction output, shape: (n_point, n_out)
        """
        assert self._black is not None
        assert self._black.ready
        
        kwargs_ = self.kwargs_del(kwargs, 'x')
        
        self.x = x                            # setter ensuring 2D array
        
        if not self.ready or self._n_inp == -1:
            self.y = None
            return self.y
        
        assert self._n_inp == self.x.shape[1], \
            str((self._n_inp, self.x.shape))

        if self._local_size is not None:
            if self.weights is None:
                y_all = []
                for x_prc in self.x:
                    if x_prc[0] is not None:
                        x_tun = self._black.predict(x=x_prc, **kwargs_)[0]
                        y_all.append(BoxModel.predict(self, x_prc, *x_tun,
                                                      **kwargs_)[0])
                self.y = y_all
            else:
                # local==X.shape[0]: const w
                self.y = BoxModel.predict(self, self.x, *self.weights, 
                                          **kwargs_)
        else:
            self.y = self._black.predict(x=x, **kwargs_)
                                              # setter ensuring 2D array
        return self.y
