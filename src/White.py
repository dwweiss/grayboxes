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
      2018-05-01 DWW
"""

from Model import Model


class White(Model):
    """
    White box model y=f(x)
    """

    def __init__(self, f, identifier='White'):
        """
        Args:
            f (method or function):
                theoretical model y = f(x) for single data point

            identifier (string, optional):
                object identifier
        """
        super().__init__(f=f, identifier=identifier)

    def train(self, X, Y, **kwargs):
        """
        Sets 'ready' to True without performing training

        Args:
            X (2D array_like of float):
                training input, not used

            Y (2D array_like of float):
                training target, not used

            kwargs (dict, optional):
                keyword arguments, not used

        Returns:
            None
        """
        self.ready = True
        return None


if __name__ == '__main__':
    ALL = 1

    import numpy as np
    from plotArrays import plotIsoMap
    import Model
    from Forward import Forward

    def fUser(self, x, c0=1, c1=1, c2=1, c3=1, c4=1, c5=1, c6=1, c7=1):
        x0, x1 = x[0], x[1]
        y0 = c0 + c1 * np.sin(c2 * x0) + c3 * (x1 - 1.5)**2
        return [y0]

    x = Model.gridInit(ranges=[(-1, 8), (0, 3)], n=[8, 8])

    if 0 or ALL:
        s = 'White box (expanded)'
        print('-' * len(s) + '\n' + s + '\n' + '-' * len(s))

        wht = White(fUser)
        y = wht(x=x)

        plotIsoMap(x[:, 0], x[:, 1], y[:, 0])

    if 0 or ALL:
        s = 'White box (compact 1)'
        print('-' * len(s) + '\n' + s + '\n' + '-' * len(s))

        y = White(fUser)(x=x)

        plotIsoMap(x[:, 0], x[:, 1], y[:, 0])

    if 0 or ALL:
        s = 'Forward operator on White box model'
        print('-' * len(s) + '\n' + s + '\n' + '-' * len(s))

        model = White(fUser)
        x, y = Forward(model)(ranges=[(-1, 8), (0, 3)], grid=[8, 8])

        plotIsoMap(x[:, 0], x[:, 1], y[:, 0])

    if 0 or ALL:
        s = 'Forward operator on demo White box'
        print('-' * len(s) + '\n' + s + '\n' + '-' * len(s))

        x, y = Forward(White('demo'))(ranges=[(-1, 8), (0, 3)], cross=9)

        plotIsoMap(x[:, 0], x[:, 1], y[:, 0])
