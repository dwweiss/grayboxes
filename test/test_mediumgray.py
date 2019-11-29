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
      2019-06-04 DWW
"""

import __init__
__init__.init_path()

import unittest
import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Any, List, Optional, Sequence

from grayboxes.mediumgray import MediumGray
from grayboxes.array import grid, noise, rand
from grayboxes.lightgray import LightGray
from grayboxes.white import White
from grayboxes.plot import plot_x_y_y_ref


n_tun = 3

def f(self, x: Optional[Sequence[float]], *args: float, **kwargs: Any) \
        -> List[float]:
    """
    Theoretical submodel for single data point

    Aargs:
        x:
            common input

        args:
            tuning parameters as positional arguments

        kwargs:
            keyword arguments {str: float/int/str}
    """
    if x is None:
        return np.ones(n_tun)
    
    tun = args if len(args) >= n_tun else np.ones(n_tun)

    y0 = tun[0] + tun[1] * np.sin(tun[2] * x[0]) + tun[1] * (x[1] - 1.5)**2
    return [y0]


trainer = [
           # 'all',
           # 'L-BFGS-B',
           'BFGS',
           # 'Powell',
           # 'Nelder-Mead',
           # 'differential_evolution',
           # 'basinhopping',
           # 'ga',
           ]


class TestUM(unittest.TestCase):
    def setUp(self):
        print('///', os.path.basename(__file__))

        s = 'Creates exact output y_exa(X) and adds noise. Target is Y(X)'
        print('-' * len(s) + '\n' + s + '\n' + '-' * len(s))
        
        noise_abs = 0.1
        noise_rel = 5e-2
        self.X = grid(10, [-1., 2.], [0., 3.])
        self.y_exa = White(f)(x=self.X, silent=True)
        self.Y = noise(self.y_exa, absolute=noise_abs, relative=noise_rel)
        if 0:
            plot_x_y_y_ref(self.X, self.Y, self.y_exa, ['X', 'Y_{nse}', 
                                                        'y_{exa}'])

        pass

    def tearDown(self):
        pass

    def test1(self):
        s = 'Tunes model, compare: y(X) vs y_exa(X)'
        print('-' * len(s) + '\n' + s + '\n' + '-' * len(s))

        # train with n1 random initial tuning parameter help, each of size n2
        local, n1, n2 = 10, 1, 3
        mgr, lgr = MediumGray(f), LightGray(f)
        mgr.silent, lgr.silent = True, True
        tun0 = rand(n1, *(n2 * [[0., 2.]]))

        l2 = np.inf
        y_mgr, tun_mgr = None, None
        for local in range(1, 3):
            for neurons in range(2, 4):
                y = mgr(X=self.X, Y=self.Y, x=self.X, trainer=trainer, 
                        tun0=tun0, nItMax=5000,
                        bounds=n_tun*[(-1., 3.)], neurons=[neurons], trials=3,
                        local=local)
                print('l2(neurons:', str(neurons)+'): ', mgr.metrics['L2'],
                      end='')
                if l2 > mgr.metrics['L2']:
                    l2 = mgr.metrics['L2']
                    print('  *** better', end='')
                    y_mgr, tun_mgr = y, mgr.weights
                print()

        self.assertFalse(y_mgr is None)

        y_lgr = lgr(X=self.X, Y=self.Y, x=self.X, trainer=trainer, 
                    nItMax=5000, tun0=tun0)
        print('lgr.w:', lgr.weights)

        if mgr.weights is None:
            x_tun = mgr._black.predict(x=self.X)
            for i in range(x_tun.shape[1]):
                plt.plot(x_tun[:, i], ls='-',
                         label='$x^{loc}_{tun,'+str(i)+'}$')
        for i in range(len(lgr.weights)):
            plt.axhline(lgr.weights[i], ls='--',
                        label='$x^{lgr}_{tun,'+str(i)+'}$')
        # plt.ylim(max(0, 1.05*min(lgr.weights)),
        #          min(2, 0.95*max(lgr.weights)))
        plt.legend(bbox_to_anchor=(1.1, 1.05))
        plt.show()

        plot_x_y_y_ref(self.X, y_lgr, self.y_exa, ['X', 'y_{lgr}', 'y_{exa}'])
        plot_x_y_y_ref(self.X, y_mgr, self.y_exa, ['X', 'y_{mgr}', 'y_{exa}'])

        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
