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
      2018-05-20 DWW
"""

import numpy as np
import matplotlib.pyplot as plt

from Forward import Forward
from plotArrays import plotBarArrays


class Sensitivity(Forward):
    """
    Sensitivity operation on box type model for a given input range

    Examples:
        op = Sensitivity(LightGray('demo')                        x
        X = [[... ]]  input of training                           |
        Y = [[... ]]  target of training                      x--ref--x
        ranges = ([x0min, x0max], [x1min, x1max], ... )           |
        xCross = cross(3, *ranges)                                x

        dy_dx = op(X=X, Y=Y, x=xCross)               # training and sensitivity
        dy_dx = op(x=xCross)        # sensitivity only, x contains cross points
        dy_dx = op(x=cross(5, *ranges)       # generate x from ranges and cross
    """

    def __init__(self, model, identifier='Sensitivity'):
        """
        Args:
            model (Model_like):
                box type model

            identifier (str, optional):
                object identifier
        """
        super().__init__(model=model, identifier=identifier)

        self.axisIndices = None         # point indices for which x[j] is equal
        self.dy_dx = None

    def task(self, **kwargs):
        """
        Analyzes sensistivity

        Args:
            kwargs (dict, optional):
                keyword arguments

                X (2D or 1D array_like of float, optional):
                    training input, shape: (nPoint, nInp) or shape: (nPoint)
                    default: self._X

                Y (2D or 1D array_like of float, optional):
                    training target, shape: (nPoint, nOut) or shape: (nPoint)
                    default: self._Y

                x (2D array_like of float):
                    cross-type input points, see Model.cross()
                    default: self._x

        Returns:
            x (1D array of float):
                reference point in the center of the cross

            dy/dx (1D array of float):
                gradient of y with respect to x in reference point
        """
        # trains (if X and Y not None) and predicts self.y = f(self.x)
        super().task(**self.kwargsDel(kwargs, 'x'))

        if self.model.x is None:
            return None, None

        # ref point (x, y)_ref is stored as: (self.model.x[0], self.model.y[0])
        x, y = self.model.x, self.model.y
        xRef = x[0]
        nPoint, nInp, nOut = x.shape[0], x.shape[1], y.shape[1]

        self.indicesWithEqualXj = [[] for _ in range(nInp)]

        # i is point index, j is input index and k is output index
        for i in range(1, nPoint):
            for j in range(nInp):
                if np.isclose(x[0, j], x[i, j]):
                    self.indicesWithEqualXj[j].append(i)

        self.dy_dx = np.full((nInp, nOut), np.inf)
        jCenter = nInp // 2
        for k in range(nOut):
            for j in range(nInp):
                xx, yy = [], []
                for i in range(nPoint):
                    if i not in self.indicesWithEqualXj[j]:
                        xx.append(x[i, j])
                        yy.append(y[i, k])
                    yy = [a for _, a in sorted(zip(xx, yy))]
                    xx = sorted(xx)
                dx = (xx[jCenter+1] - xx[jCenter-1]) * 0.5
                grad = np.gradient(yy, dx)
                self.dy_dx[j, k] = grad[jCenter]
        s = np.array2string(self.dy_dx).replace(' ', '').replace('\n',
                                                                 '\n' + ' '*10)
        self.write('    grad: ', s[1:-1])

        return xRef, self.dy_dx

    def plot(self):
        """
        Plots gradient df(x)/dx
        """
        if self.silent:
            return

        nPoint, nInp = self.model.x.shape
        nOut = self.model.y.shape[1]

        for k in range(nOut):
            for j in range(nInp):
                plt.title('y' + str(k) + ' (x' + str(j) + ')')
                plt.xlabel('x' + str(j))
                plt.ylabel('y' + str(k))
                xx, yy = [], []
                for i in range(nPoint):
                    if i not in self.indicesWithEqualXj[j]:
                        xx.append(self.model.x[i, j])
                        yy.append(self.model.y[i, k])
                    yy = [a for _, a in sorted(zip(xx, yy))]
                    xx = sorted(xx)
                plt.plot(xx, yy, '-o')
                plt.grid()
                plt.legend(bbox_to_anchor=(1.1, 1.03), loc='upper left')
                plt.show()

        for k in range(nOut):
            plt.title('y' + str(k) + ' (x0..' + str(nInp-1) + ')')
            for j in range(nInp):
                plt.xlabel('x0..' + str(nInp-1))
                plt.ylabel('y' + str(k))
                xx, yy = [], []
                for i in range(nPoint):
                    if i not in self.indicesWithEqualXj[j]:
                        xx.append(self.model.x[i, j])
                        yy.append(self.model.y[i, k])
                    yy = [a for _, a in sorted(zip(xx, yy))]
                    xx = sorted(xx)
                plt.plot(xx, yy, '-o', label='y'+str(k)+'(x'+str(j)+')')
            plt.grid()
            plt.legend(bbox_to_anchor=(1.1, 1.04), loc='upper left')
            plt.show()

        if nOut > 1:
            for j in range(nInp):
                plt.title('y0..' + str(nOut-1) + ' (x' + str(j) + ')')
                for k in range(nOut):
                    plt.xlabel('x' + str(j))
                    plt.ylabel('y0..' + str(nOut-1))
                    xx, yy = [], []
                    for i in range(nPoint):
                        if i not in self.indicesWithEqualXj[j]:
                            xx.append(self.model.x[i, j])
                            yy.append(self.model.y[i, k])
                        yy = [a for _, a in sorted(zip(xx, yy))]
                        xx = sorted(xx)
                    plt.plot(xx, yy, '-o',
                             label='y'+str(k)+' (x'+str(j)+')')
                plt.grid()
                plt.legend(bbox_to_anchor=(1.1, 1.04), loc='upper left')
                plt.show()

        plotBarArrays(yArrays=self.dy_dx.T, legendPosition=(1.1, 1.03),
                      title=r'Gradient $d y_k \ / \ d x_j$', grid=True,
                      figsize=(6, 4))


# Examples ####################################################################

if __name__ == '__main__':
    ALL = 1

    from White import White
    import Model as md

    def f(self, x, *args, **kwargs):
        return np.sin(x[0]) + (x[1] - 1)**2

    if 0 or ALL:
        s = 'Sensitivity with method f(self, x)'
        print('-' * len(s) + '\n' + s + '\n' + '-' * len(s))

        xRef, dy_dx = Sensitivity(White(f))(x=md.cross(3, [2, 3], [3, 4]))
        if dy_dx.shape[0] == 1 or dy_dx.shape[1] == 1:
            dy_dx = dy_dx.tolist()
        print('dy_dx:', dy_dx)
        print('x_ref:', xRef)

    if 1 or ALL:
        s = 'Sensitivity with demo function'
        print('-' * len(s) + '\n' + s + '\n' + '-' * len(s))

        op = Sensitivity(White('demo'))
        xRef, dy_dx = op(x=md.cross(3, [2, 3], [3, 4], [4, 5]))
        if dy_dx.shape[0] == 1 or dy_dx.shape[1] == 1:
            dy_dx = dy_dx.tolist()
        print('dy_dx:', dy_dx)
        print('x_ref:', xRef)
