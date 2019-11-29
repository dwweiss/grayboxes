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
      2019-11-21 DWW
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Any, Tuple

from grayboxes.base import Float1D
from grayboxes.boxmodel import BoxModel
from grayboxes.forward import Forward
from grayboxes.plot import plot_bar_arrays


class Sensitivity(Forward):
    """
    Sensitivity operation on box type model for a given input range

    Examples:
        op = Sensitivity(LightGray('demo')                         x
        X = [[... ]]  input of training                            |
        Y = [[... ]]  target of training                       x--ref--x
        ranges = ([x0min, x0max], [x1min, x1max], ... )            |
        xCross = cross(3, *ranges)                                 x

        dy_dx = op(X=X, Y=Y, x=xCross)        # training and sensitivity
        dy_dx = op(x=xCross)        # sensitivity only, x contains cross
        dy_dx = op(x=cross(5, *ranges)    # gen. x from ranges and cross
    """

    def __init__(self, model: BoxModel, identifier: str='Sensitivity') -> None:
        """
        Args:
            model:
                Box type model

            identifier:
                unique object identifier
        """
        super().__init__(model=model, identifier=identifier)

        self.axis_indices = None  # point indices for which x[j] is equal
        self.dy_dx = None
        self.indices_with_equal_Xj = None

    def task(self, **kwargs: Any) -> Tuple[Float1D, Float1D]:
        """
        Analyzes sensitivity

        Kwargs:
            X (2D array of float, optional):
                training input, shape: (n_point, n_inp) 
                default: self.model.X

            Y (2D array of float, optional):
                training target, shape: (n_point, n_out)
                default: self.model.Y

            x (2D array of float):
                cross-type input points, see BoxModel.cross()
                default: self.model.x

        Returns:
            2-tuple:
                x:
                    reference point in the center of the cross
                dy/dx:
                    gradient of y with respect to x in reference point
        """
        # trains (if X and Y not None) and predicts self.y = f(self.x)
        super().task(**self.kwargs_del(kwargs, 'x'))

        if self.model.x is None:
            return None, None

        # ref point (x, y)_ref is stored as: (self.model.x[0], self.model.y[0])
        x, y = self.model.x, self.model.y
        x_ref = x[0]
        n_point, n_inp, n_out = x.shape[0], x.shape[1], y.shape[1]

        self.indices_with_equal_Xj = [[] for _ in range(n_inp)]

        # i is point index, j is input index and k is output index
        for i in range(1, n_point):
            for j in range(n_inp):
                if np.isclose(x[0, j], x[i, j]):
                    self.indices_with_equal_Xj[j].append(i)

        self.dy_dx = np.full((n_inp, n_out), np.inf)
        j_center = n_inp // 2
        for k in range(n_out):
            for j in range(n_inp):
                xx, yy = [], []
                for i in range(n_point):
                    if i not in self.indices_with_equal_Xj[j]:
                        xx.append(x[i, j])
                        yy.append(y[i, k])
                    yy = [a for _, a in sorted(zip(xx, yy))]
                    xx = sorted(xx)
                dx = (xx[j_center+1] - xx[j_center-1]) * 0.5
                grad = np.gradient(yy, dx)
                self.dy_dx[j, k] = grad[j_center]
        s = np.array2string(self.dy_dx).replace(' ', '').replace('\n',
                                                                 '\n' + ' '*22)
        self.write('    grad: ' + str(s[1:-1]))

        return x_ref, self.dy_dx

    def plot(self) -> None:
        """
        Plots gradient df(x)/dx
        """
        if self.silent:
            return

        n_point, n_inp = self.model.x.shape
        n_out = self.model.y.shape[1]

        for k in range(n_out):
            for j in range(n_inp):
                plt.title('y' + str(k) + ' (x' + str(j) + ')')
                plt.xlabel('x' + str(j))
                plt.ylabel('y' + str(k))
                xx, yy = [], []
                for i in range(n_point):
                    if i not in self.indices_with_equal_Xj[j]:
                        xx.append(self.model.x[i, j])
                        yy.append(self.model.y[i, k])
                    yy = [a for _, a in sorted(zip(xx, yy))]
                    xx = sorted(xx)
                plt.plot(xx, yy, '-o')
                plt.grid()
                plt.legend(bbox_to_anchor=(1.1, 1.03), loc='upper left')
                plt.show()

        for k in range(n_out):
            plt.title('y' + str(k) + ' (x0..' + str(n_inp-1) + ')')
            for j in range(n_inp):
                plt.xlabel('x0..' + str(n_inp-1))
                plt.ylabel('y' + str(k))
                xx, yy = [], []
                for i in range(n_point):
                    if i not in self.indices_with_equal_Xj[j]:
                        xx.append(self.model.x[i, j])
                        yy.append(self.model.y[i, k])
                    yy = [a for _, a in sorted(zip(xx, yy))]
                    xx = sorted(xx)
                plt.plot(xx, yy, '-o', label='y'+str(k)+'(x'+str(j)+')')
            plt.grid()
            plt.legend(bbox_to_anchor=(1.1, 1.04), loc='upper left')
            plt.show()

        if n_out > 1:
            for j in range(n_inp):
                plt.title('y0..' + str(n_out-1) + ' (x' + str(j) + ')')
                for k in range(n_out):
                    plt.xlabel('x' + str(j))
                    plt.ylabel('y0..' + str(n_out-1))
                    xx, yy = [], []
                    for i in range(n_point):
                        if i not in self.indices_with_equal_Xj[j]:
                            xx.append(self.model.x[i, j])
                            yy.append(self.model.y[i, k])
                        yy = [a for _, a in sorted(zip(xx, yy))]
                        xx = sorted(xx)
                    plt.plot(xx, yy, '-o',
                             label='y'+str(k)+' (x'+str(j)+')')
                plt.grid()
                plt.legend(bbox_to_anchor=(1.1, 1.04), loc='upper left')
                plt.show()

        plot_bar_arrays(yarrays=self.dy_dx.T, legend_position=(1.1, 1.03),
                        title=r'Gradient $d y_k \ / \ d x_j$', grid=True,
                        figsize=(6, 4))
