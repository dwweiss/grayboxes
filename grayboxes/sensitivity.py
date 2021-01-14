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
      2020-11-26 DWW
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Any, List, Optional, Tuple, Union

from grayboxes.boxmodel import BoxModel
from grayboxes.datatype import Float1D, Float2D
from grayboxes.forward import Forward
from grayboxes.plot import plot_bar_arrays


class Sensitivity(Forward):
    """
    Sensitivity operation on box type model in given input range

    Example:
        from grayboxes.array import rand, cross
        op = Sensitivity(LightGray('demo')                         x
        ranges = ([x0min, x0max], [x1min, x1max], ... )            |
        X = rand(100, *ranges)       # input of training       x--ref--x    
        Y = f(X)                     # target of training          |
        x_cross = cross(3, *ranges)  # 3 points per axis           x

        x, dy_dx = op(X=X, Y=Y, x=x_cross)    # training and sensitivity
        x, dy_dx = op(x=x_cross)    # sensitivity only, x contains cross
        x, dy_dx = op(x=cross(5, *ranges) # gen. x from ranges and cross
    """

    def __init__(self, model: BoxModel, 
                 identifier: str = 'Sensitivity') -> None:
        """
        Args:
            model:
                Box type model

            identifier:
                Unique object identifier
        """
        super().__init__(model=model, identifier=identifier)

        self.axis_indices: Optional[List[int]] = None  
                                 # point indices for which x[j] is equal
        self.x_ref: Optional[np.array] = None
        self.dy_dx: Optional[np.array] = None
        self.indices_with_equal_Xj: Optional[List[List[int]]] = None

    def task(self, **kwargs: Any) -> Union[Tuple[None, None],
                                           Tuple[Float1D, Float2D]]:
        """
        Analyzes sensitivity dy/dx

        Kwargs:
            X (2D array of float):
                training input, shape: (n_point, n_inp) 
                default: self.model.X

            Y (2D array of float):
                training target, shape: (n_point, n_out)
                default: self.model.Y

            x (2D array of float):
                cross-type input points, see grayboxes.array.cross()
                default: self.model.x

        Returns:
            2-tuple:
                x:
                    reference point in the center of the cross
                dy/dx:
                    gradient of y with respect to x in reference point
            OR
            2-typle (None, None) if self.model.x is None
        """
        # 1. training if X and Y are not None and/or 
        # 2. prediction of self.model.y if self.model.x is not None:
        super().task(**self.kwargs_del(kwargs, 'x'))
        if self.model.x is None:
            return None, None    
        
        # (x, y)_ref is stored as: (self.model.x[0], self.model.y[0])
        x, y = self.model.x, self.model.y

        if y.ndim == 3:
            print('!!! y.ndim: 3 -> reduce dim from:', np.shape(y), end='')
            if y.shape[1] == 1:
                y = y[:, 0, :]
            print(' to:', np.shape(y))
        
        self.x_ref = x[0]
        
        n_point, n_inp, n_out = x.shape[0], x.shape[1], y.shape[1]

        self.indices_with_equal_Xj = [[] for empty_item in range(n_inp)]

        # i is point index, j is input index
        for i in range(1, n_point):
            for j in range(n_inp):
                if np.isclose(x[0, j], x[i, j]):
                    self.indices_with_equal_Xj[j].append(i)

        self.dy_dx = np.zeros((n_inp, n_out))
        for k in range(n_out):
            for j in range(n_inp):
                x_unq, y_unq = [], []
                for i in range(n_point):
                    if i not in self.indices_with_equal_Xj[j]:
                        x_unq.append(x[i, j])
                        y_unq.append(y[i, k])
                    y_unq = [_y for _x, _y in sorted(zip(x_unq, y_unq))]
                    x_unq = sorted(x_unq)
                n_unq = len(x_unq)

                if n_unq == 1: 
                    grad = 0
                elif n_unq == 2:
                    j_cen = 1
                    dx = x_unq[1] - x_unq[0]
                    dy = y_unq[1] - y_unq[0]
                    grad = dy / dx
                else:
                    j_cen = len(x_unq) // 2
                    dx = (x_unq[j_cen+1] - x_unq[j_cen-1]) * 0.5                    
                    grad = np.gradient(y_unq, dx)
                    grad = grad[j_cen]
                self.dy_dx[j, k] = grad

        s = np.array2string(self.dy_dx).replace('  ', ' ').replace('\n',
                                                                 '\n' + ' '*22)
        self.write('    grad: ' + str(s[1:-1]))

        return self.x_ref, self.dy_dx

    def plot(self, **kwargs) -> None:
        """
        Plots gradient df(x)/dx

        Kwargs:
            None            
        """
        if self.silent:
            return

        if self.model.x is None:
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

        plot_bar_arrays(x=self.x_ref,
                        yarrays=self.dy_dx.T,
                        title=r'Gradient d$y_k$ / d$x_j$',
                        labels=['$x_{ref}$'] + ['$y^\prime_' + str(j) + '$' \
                                for j in range(self.dy_dx.shape[1])],
                        figsize=(10, 8),
                        )

        hat_dy_dx = np.zeros(shape=self.dy_dx.shape)
        for i in range(self.dy_dx.shape[0]):
            hat_dy_dx[i] = self.dy_dx[i] / self.dy_dx[0]
        plot_bar_arrays(x=self.x_ref,
                        yarrays=hat_dy_dx.T,
                        title=r'Normalized d$\hat y_k$ / d$x_j$',
                        labels=['$x_{ref}$'] + ['$\hat y^\prime_' + str(j) 
                                + '$' for j in range(self.dy_dx.shape[1])],
                        figsize=(10, 8),
                        )

        hat_dy_dx_minus_one = np.zeros(shape=self.dy_dx.shape)
        for i in range(self.dy_dx.shape[0]):
            hat_dy_dx_minus_one[i] = self.dy_dx[i] / self.dy_dx[0]
            hat_dy_dx_minus_one[i] -= 1.        
        plot_bar_arrays(x=self.x_ref,
                        yarrays=hat_dy_dx_minus_one.T,
                        title=r'Normalized d$\hat y_k$ / d$x_j$ - 1',
                        labels=['$x_{ref}$'] + ['$\hat y^\prime_' + str(j) 
                                + '$' for j in range(self.dy_dx.shape[1])],
                        figsize=(10, 8),
                        )

