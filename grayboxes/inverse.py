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
      2019-12-09 DWW
"""

import numpy as np
from typing import Any, Union

from grayboxes.datatype import Float1D, Float2D
from grayboxes.minimum import Minimum


class Inverse(Minimum):
    """
    Finds the inverse x = phi^{-1}(y)

    Examples:
        X = [[... ]]  input of training
        Y = [[... ]]  target of training
        x_keys, y_keys = ('a', 'b'), ('y', 'z')
        x_ini = [x00, x01, ... ]
        bounds = [(x0min, x0max), (x1min, x1max), ... ]
        y_inv = (y0, y1, ... )

        def f(self, x: Iterable[float], *c: float) -> List[float]:
            c = args if len(weights) > 0 else np.ones(1)
            return [c[0] * x[0]]
        op = Inverse(DarkGray(f=f))

        x, y = op(XY=(X, Y, x_keys, y_keys))             # only training
        x, y = op(X=X, Y=Y, x=x_ini, y=y_inv)       # training & inverse
        x, y = op(x=rand(5, (1, 4), (0, 9)), y=y_inv)     # only inverse
        x, y = op(x=x_ini, bounds=bounds, y=y_inv)        # only inverse

        x, y = Inverse(LightGray(f))(XY=(X, Y), c0=1, x=x_ini, y=y_inv)
    """

    def objective(self, x: Union[Float2D, Float1D], **kwargs: Any) -> float:
        """
        Defines objective function for inverse problem

        Args:
            x:
                input of multiple or single data points,
                shape: (1, n_inp) or (n_inp,)

        Kwargs:
            Keyword arguments to be passed to self.model.predict()

        Note: 
            The target is given as self.y with self.y.shape: (n_out,)

        Returns:
            L2-norm as measure of difference between prediction & target
            or
            np.inf if self.model.prediction() returns None
        """
        # x is prediction input, x.shape:(n_inp,), y_opt.shape:(1, n_out)
        y_opt = self.model.predict(np.asfarray(x),
                                   **self.kwargs_del(kwargs, 'x'))
        
        if y_opt is None:
            return np.inf

        # y is target, self.y.shape: (n_out,), y_opt[0].shape: (n_out,)
        # Note: This L2-norm is only computed for a single data point 
        L2_norm = np.sqrt(np.mean((y_opt[0] - self.y)**2))

        self._single_hist.append((x, y_opt[0], L2_norm))

        return L2_norm
