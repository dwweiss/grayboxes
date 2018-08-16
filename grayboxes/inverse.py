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
from grayboxes.minimum import Minimum


class Inverse(Minimum):
    """
    Finds the inverse x = f^{-1}(y)

    Examples:
        X = [[... ]]  input of training
        Y = [[... ]]  target of training
        xIni = [x00, x01, ... ]
        bounds = [(x0min, x0max), (x1min, x1max), ... ]
        yInv = (y0, y1, ... )

        def f(self, x):
            c0 = args if len(args) > 0 else 1
            return [c0 * x[0]]
        op = Inverse(DarkGray(f=f))

        x, y = op(XY=(X, Y, xKeys, yKeys))               # only training
        x, y = op(X=X, Y=Y, x=xIni, y=yInv)              # training & inverse
        x, y = op(x=rand(5, [1, 4], [0, 9]), y=yInv)     # only inverse
        x, y = op(x=xIni, bounds=bounds, y=yInv)         # only inverse

        x, y = Inverse(LightGray(f))(XY=(X, Y), c0=1, x=xIni, y=yInv)
    """

    def objective(self, x, **kwargs):
        """
        Defines objective function for inverse problem

        Args:
            x (2D or 1D array_like of float):
                input of multiple or single data points,
                shape: (nPoint, nInp) or shape: (nInp)

            kwargs (dict, optional):
                keyword arguments for predict()

        Returns:
            (float):
                L2-norm as measure of difference between prediction and target
        """
        # x is input of prediction, x.shape: (nInp,)
        yInv = self.model.predict(np.asfarray(x),
                                  **self.kwargsDel(kwargs, 'x'))

        # self.y is target, self.y.shape: (nOut,), yInv.shape: (1, nOut)
        L2 = np.sqrt(np.mean((yInv[0] - self.y)**2))

        self._trialHistory.append([x, yInv[0], L2])

        return L2
