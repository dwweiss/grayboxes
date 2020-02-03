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

import initialize
initialize.set_path()

import numpy as np
from typing import Any, List, Optional, Iterable
import unittest

from grayboxes.boxmodel import BoxModel
from grayboxes.white import White 
from grayboxes.lightgray import LightGray 
from grayboxes.mediumgray import MediumGray 
from grayboxes.darkgray import DarkGray 
from grayboxes.black import Black 

from grayboxes.forward import Forward
from grayboxes.sensitivity import Sensitivity 
from grayboxes.minimum import Minimum 
from grayboxes.maximum import Maximum 
from grayboxes.inverse import Inverse 

from grayboxes.array import grid, noise
from grayboxes.plot import plot_isomap


# function without access to 'self' attributes
def func(x: Optional[Iterable[float]], *c: float, 
         **kwargs: Any) -> List[float]:
    """
    x: [1, 1] y: 0.8414709848078965

    """
    c0, c1 = 1., 1.
    if x is None:
        return c0, c1
    if len(c) == 2:
        c0, c1 = c
    return [c0 * np.array(np.sin(c1 * x[0]) + (x[1] - 1)**2)]


# method with access to 'self' attributes
def method(self, x: Optional[Iterable[float]], *c: float,
           **kwargs: Any) -> List[float]:
    return func(x, *c, **kwargs)


class TestUM(unittest.TestCase):
    def setUp(self):
        pass
    

    def tearDown(self):
        pass


    def test0(self):
        s = 'Return result of func(x)'
        print('-' * len(s) + '\n' + s + '\n' + '-' * len(s))
        
        X = grid(10, (-2, 2), (-1, 3))
        model = White('demo')
        Y = model(x=X, silent=True)
        
        y = model(X=X, Y=Y, x=X[0], silent=True)
        print('y(X[0]):', str(y))

        y = model(X=X, Y=Y, x=X, silent=True)
        print('y(X):', y[:5], y[-5:])
        
        X0, X1 = model.X.T[0], model.X.T[1]
        Y0, y0 = model.Y, model.y
        dY = (model.y - model.Y).T[0]
        s = model.identifier
        
        plot_isomap(X0, X1, Y0, title=s + ': Y',  labels=['x0', 'x1', 'Y'])
        plot_isomap(X0, X1, y0, title=s + ': y',  labels=['x0', 'x1', 'y'])
        plot_isomap(X0, X1, dY, title=s + ': dY', labels=['x0', 'x1', 'dY'])        
        
        self.assertTrue(np.isclose(1., 1.))


    def test1(self):
        s = 'Return value of all operations if x is not passed, no training'
        print('-' * len(s) + '\n' + s + '\n' + '-' * len(s))

        operation_types = (Forward, Sensitivity, Minimum, Maximum, Inverse)
        model_types = (White, LightGray, MediumGray, DarkGray, Black, BoxModel)
        for operation_type in operation_types:
            print('*** operation:', operation_type.__name__) 
            for model_type in model_types:
                operation = operation_type(model_type(func))
                print('    model:', model_type.__name__, end='') 
                xy = operation(silent=True, neurons=[10, 10])
                print(', xy:', xy) 
                
        self.assertTrue(True)


    def test2(self):
        s = 'Return value of all models if x is not passed, no training'
        print('-' * len(s) + '\n' + s + '\n' + '-' * len(s))

        model_types = (White, LightGray, MediumGray, DarkGray, Black, BoxModel)
        for model_type in model_types:
            print('    model:', model_type.__name__, end='')
            model= model_type(func)
            x = None
            y = model(x=x, silent=True)
            print(', x:', x, 'y:', y) 
                
        self.assertTrue(True)


    def test3(self):
        s = 'Return value of all models if x is passed, no training'
        print('-' * len(s) + '\n' + s + '\n' + '-' * len(s))

        model_types = (White, LightGray, MediumGray, DarkGray, Black, BoxModel)
        for model_type in model_types:
            print('    model:', model_type.__name__, end='')
            model= model_type(func)
            x = [2, 3]
            y = model(x=x, silent=True)
            print(', x:', x, 'y:', y) 
                
        self.assertTrue(True)


    def test4(self):
        s = 'Return value of all models if x is passed, with training'
        print('-' * len(s) + '\n' + s + '\n' + '-' * len(s))

        model_types = (
                       White, 
                       LightGray, 
                       Black, 
                       DarkGray, 
                       BoxModel
                       )

        for model_type in model_types:
            print('    model:', model_type.__name__, end='')
            model = model_type('demo')
            X = grid(20, [-2, 2], [-1, 3])
            Y_tru = White('demo')(x=X, silent=True)
            Y = noise(Y_tru, relative=10e-2)
            x = [[1, 2]]
            y = model(X=X, Y=Y, x=x, 
                      trainer='adam', 
                      expected=1e-3, 
                      trials=3, 
                      epochs=250, 
                      neurons=[6, 6])
            print('metrics:', model.metrics)
            print(', shapes:', (np.asarray(x).shape, np.asarray(y).shape), 
                  'x:', x, 'y:', y, '\n')

        self.assertTrue(True)


    def test5(self):
        s = 'Return value of all models if x is None, with training'
        print('-' * len(s) + '\n' + s + '\n' + '-' * len(s))

        model_types = (
                       White, 
                       LightGray, 
                       Black, 
                       DarkGray, 
                       BoxModel
                       )
        
        X = grid(64, (-2, 2), (-1, 3))
        Y_tru = White('demo')(x=X, silent=True)
        Y = noise(Y_tru, relative=5e-2)
        X0, X1 = X.T[0], X.T[1]
        plot_isomap(X0, X1, Y, title='Init. White: Y', 
                    labels=['x0', 'x1', 'Y'])

        for model_type in model_types:            
            print('    model:', model_type.__name__, end='')
            f = 'demo' if model_type.__name__ != 'Black' else None

            model = model_type(f)
            neurons = []
            if model_type.__name__ == 'LightGray':
                trainer = ['lm', 'leastsq']
            elif model_type.__name__ == 'Black':
                trainer = 'rprop'
                neurons = 'auto'
            else:
                trainer = 'auto'
                neurons = [8, 6]
            y = model(X=X, Y=Y, x=X, 
                      expected=0.5e-3,
                      neurons='auto',
                      plot=1,
                      show=0, 
                      silent=0, 
                      tolerated=10e-3,
                      trainer='adam', 
                      trials=5, 
                      )
            
            if y is not None:
                print('+++ metrics:', model.metrics)
                dY = (model.y - model.Y).T[0]
                s = model.identifier + ': '
                
                plot_isomap(X0, X1, Y, title=s + 'Y', labels=['x0', 'x1', 'Y'])
                plot_isomap(X0, X1, y, title=s + 'y', labels=['x0', 'x1', 'y'])
                plot_isomap(X0, X1, dY, title=s + 'dY', 
                            labels=['x0', 'x1', 'dY'])        
                
        self.assertTrue(True)


    def _test6(self):
        s = 'Return value of all models if x is passed, with training'
        print('-' * len(s) + '\n' + s + '\n' + '-' * len(s))

        model_types = (White, 
                       LightGray, 
                       Black, 
                       DarkGray, 
#                       BoxModel
                       )
        
        X = grid(10, (-2, 2), (-1, 3))
        Y_tru = White('demo')(x=X, silent=True)
        Y = noise(Y_tru, relative=10e-2)
        X0, X1 = X.T[0], X.T[1]
        plot_isomap(X0, X1, Y, title='Init. White: Y', 
                    labels=['x0', 'x1', 'Y'])

        for model_type in model_types:            
            print('    model:', model_type.__name__, end='')

            f = 'demo' if model_type.__name__ != 'Black' else None
            model = model_type(f)
            
            neurons = []
            if model_type.__name__ == 'LightGray':
                trainer = ['lm', 'leastsq']
            elif model_type.__name__ == 'Black':
                trainer = 'rprop'
                neurons = [8, 6, 4]
            else:
                trainer = 'auto'
            
            metrics = model(X=X, Y=Y, x=None, silent=1, trainer=trainer, 
                            show=0, goal=1e-5, trials=3, neurons=neurons)
            
            print('metrics:', metrics)
                
        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
