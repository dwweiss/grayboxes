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
      2019-07-17 DWW
"""

import __init__
__init__.init_path()

import unittest
import os
from collections import OrderedDict
from typing import Any, Dict, Optional, Union
import numpy as np
import matplotlib.pyplot as plt

from grayboxes.array import noise, rand, xy_rand_split
from grayboxes.base import Base
from grayboxes.boxmodel import BoxModel
from grayboxes.black import Black
from grayboxes.darkgray import DarkGray
from grayboxes.forward import Forward
from grayboxes.inverse import Inverse
from grayboxes.white import White


def f(x):
    x = np.atleast_1d(x)
    y = (1 - np.cos(x * np.pi)) / 2
#    if x[0] < 0.:
#        y = 0.
#    if x[0] > 1.:
#        y = 1.
    return y


def plot_f():
    x = np.linspace(-1, 2, 100)
    plt. plot(x, f(x), label='f(x)')
    plt.legend()
    plt.show()


class Foo(Base):
    """
    This example compares two methods for inverse problem solution:
        1. inverse model: x=phi^-1(y)
        2. minimization: x = arg min ||phi(x) - Y)||_2
    """
    def __init__(self, identifier='foo'):
        super().__init__(identifier=identifier)

        self.operation: Optional[Union[Forward, Inverse]] = None

        self.X_exa: Optional[np.ndarray] = None  # exact solution
        self.Y_exa: Optional[np.ndarray] = None  # exact solution
        self.X_trn: Optional[np.ndarray] = None  # training data
        self.Y_trn: Optional[np.ndarray] = None  # training data
        self.X_tst: Optional[np.ndarray] = None  # test data
        self.Y_tst: Optional[np.ndarray] = None  # test data
        self.x_tst: Optional[np.ndarray] = None  # prediction of input
        self.y_tst: Optional[np.ndarray] = None  # prediction of output

        self.metrics_trn: Optional[Dict[str, Any]] = None  # training
        self.metrics_tst: Optional[Dict[str, Any]] = None  # test
        
        self.data: Dict[str, Any] = OrderedDict()

    def generate_artificial_data(self, **kwargs):
        
        # gets parameter from key word arguments
        n_point = kwargs.get('n_point', 50)
        noise_rel = kwargs.get('noise_rel', 5e-2)
        noise_abs = kwargs.get('noise_abs', 0.05)
        train_test_ratio = kwargs.get('train_test_ratio', (0.6, 0.4))
        x_min, x_max = kwargs.get('x_min', 0.1), kwargs.get('x_max', 0.9)

        # generates artificial data '$y = 1 - \cos \( x \pi \) / 2$'
        self.X_exa = np.atleast_2d(np.linspace(x_min, x_max, n_point)).T
        self.Y_exa = f(self.X_exa)

        # adds noise to Y
        X_nse = self.X_exa
        Y_nse = noise(self.Y_exa, relative=noise_rel, absolute=noise_abs)

        # splits randomly into training data and test data
        X_spl, Y_spl = xy_rand_split(X_nse, Y_nse, train_test_ratio)
        self.X_trn, self.Y_trn = X_spl[0], Y_spl[0]
        self.X_tst, self.Y_tst = X_spl[1], Y_spl[1]
                
        # ensures max(train) >= max(test) and min(train) <= min(test) 
        self.X_tst = np.clip(self.X_tst, 0.01, 0.99)
        self.Y_tst = np.clip(self.Y_tst, 0.01, 0.99)
        
        # plots exact solution, training and test data 
        if not False and not self.silent:
            plt.plot(self.X_exa, self.Y_exa, '.', c='0.', label='exact')
            plt.plot(X_nse, Y_nse, '--', c='0.', label='noise')
            plt.plot(self.X_trn, self.Y_trn, 'v', c='g', label='train')
            plt.plot(self.X_tst, self.Y_tst, 'x', c='b', label='test')
            plt.xlabel('$X$')
            plt.ylabel('$Y$')
            plt.legend()
            plt.grid()
            plt.show()

#        print(self.X_trn.min(), self.X_trn.max(),
#              self.Y_trn.min(), self.Y_trn.max())
#        print(self.X_tst.min(), self.X_tst.max(),
#              self.Y_tst.min(), self.Y_tst.max())
#
#        assert self.X_trn.min() < self.X_tst.min()
#        assert self.X_trn.max() > self.X_tst.max()
#        assert self.Y_trn.min() < self.Y_tst.min()
#        assert self.Y_trn.max() > self.Y_tst.max()

    def forward_operator(self, inverse_model, **kwargs):

        # trains inverse model x=phi^{-1}(y)
        self.metrics_trn = inverse_model.train(self.Y_trn, self.X_trn,**kwargs)

        # optional: predicts y for x of exact and training data
        self.y_trn, self.x_trn = self.Y_trn, inverse_model.predict(self.Y_trn)
        self.y_exa, self.x_exa = self.Y_exa, inverse_model.predict(self.Y_exa)

        # optional: computes error norm with test data
        self.metrics_tst = inverse_model.evaluate(self.Y_tst, self.X_tst)
        
        # defines forward operator employing inverse model x=phi^{-1}(y)
        self.operation = Forward(inverse_model)
        
        # forward simulation for all Y_tst (lower case 'x' -> execution)
        self.y_tst, self.x_tst = self.operation(x=self.Y_tst)

    def inverse_operator(self, forward_model, **kwargs):        
        if not isinstance(forward_model, White):
            
            # trains forward model y=phi(x)
            self.metrics_trn = forward_model.train(X=self.X_trn, Y=self.Y_trn, 
                                                   **kwargs)
        else:
            self.metrics_trn = BoxModel.init_metrics()

        # optional: predicts y(x) for exact and training data
        self.x_exa, self.y_exa = self.X_exa, forward_model.predict(self.X_exa)
        self.x_trn, self.y_trn = self.X_trn, forward_model.predict(self.X_trn)

        # optional: computes error norm with test data
        self.metrics_tst = forward_model.evaluate(self.X_tst, self.Y_tst)
        
        # defines inverse operation employing forward model y=phi(x)
        self.operation = Inverse(forward_model)

        # inverse search for every point in (X_tst, Y_tst)
        x_tst, y_tst = [], []
        for x_ini, y_trg in zip(self.X_tst.copy(), self.Y_tst.copy()):
            x_ini = rand(3, [0.1, 0.9])
            x_opt, y_opt = self.operation(x=x_ini, y=y_trg, **kwargs)
            x_tst.append(x_opt[0])
            y_tst.append(y_opt)
        self.x_tst, self.y_tst = np.asfarray(x_tst), np.asfarray(y_tst)
                
    def pre(self, **kwargs):
        """
        The pre-process provides training data (X_trn, Y_trn),
        test data (X_tst, Y_tst) and exact solution (X_exa, Y_exa)
        """
        super().pre(**kwargs)

        # skips data split after first call of pre() for having same 
        # train/test data for all trials 
        if all([x is not None for x in [self.X_trn, self.Y_trn, 
                                        self.X_tst, self.Y_tst]]):
            return 
        
        self.generate_artificial_data(**kwargs)

    def task(self, **kwargs):
        super().task(**kwargs)

        variant = kwargs.get('variant', 1)
        self.write('*** variant: ' + str(variant))

        if variant == 1:
            self.forward_operator(inverse_model=Black(), **kwargs)
        elif variant == 2:
            print('*** DarkGray ' + '*'*30)
            self.inverse_operator(forward_model=DarkGray(f), **kwargs)
        else:
            assert 0

    def post(self, **kwargs):
        super().post(**kwargs)

        variant = kwargs.get('variant', 1)
        optimizer = kwargs.get('optimizer', '')
        nse_abs = kwargs.get('noise_abs', '')
        nse_rel = kwargs.get('noise_rel', '')

        self.write('+++ self.metrics_tst: ' + str(self.metrics_tst))

        if False:
            plt.cla()
            plt.plot(self.x_exa, self.y_exa, '--', label='pred exa')
            plt.plot(self.x_trn, self.y_trn, '.', label='pred trn')
            plt.plot(self.x_tst, self.y_tst, '.', label='pred tst')
            plt.plot(self.X_trn, self.Y_trn, 'v', 'g', label='train data')
            plt.plot(self.X_tst, self.Y_tst, 'x', c='b', label='test data')
            plt.legend(bbox_to_anchor=(1.1, 0), loc='lower left')
            plt.show()               

        # Y over X diagram
        plt.cla()
        s = 'V' + str(variant)
        if not isinstance(self.operation.model, White):
            s += ', ' + str(kwargs.get('neurons', None)) + \
                ', $L_{2}^{trn} ' + str(round(self.metrics_trn['L2'], 4)) + \
                ', L_{2}^{tst} ' + str(round(self.metrics_tst['L2'], 4)) + '$'
        if variant == 2:
            s += ', ' + optimizer            
        plt.title(s)
        plt.xlim(min(0.0, self.X_tst.min(), self.x_tst.min()) - 0.1,
                 max(1.0, self.X_tst.max(), self.x_tst.max()) + 0.1)
        plt.yscale('linear')

        if variant == 1:
            y = np.atleast_2d(np.linspace(0, 1, 200)).T
            x = self.operation.model.predict(x=y)
            plt.plot(x, y, '.', c='0.', label=r'$\varphi(y)$')
        else:
            plt.plot(self.x_tst, self.Y_tst, '.', c='0.', 
                     label=r'$inv \varphi^{}(x)$')
            
        if False:
            plt.plot(self.X_exa, self.Y_exa, '--', c='0.', label='exact')
        plt.scatter(self.X_trn, self.Y_trn, marker='v', c='g',
                    label='train data')
        plt.scatter(self.X_tst, self.Y_tst, marker='x', c='b',
                    label='test data')

        plt.scatter(self.x_tst, self.y_tst, marker='+', color='r',
                    label=r'$x = \varphi( y_{tst} )$')
        if not isinstance(self.operation.model, White):
            if 'iAbs' in self.metrics_trn:
                i_abs_trn = self.metrics_trn['iAbs']
                plt.scatter([self.X_trn[i_abs_trn]], [self.Y_trn[i_abs_trn]],
                            marker='o', color='g', s=66, label='max abs train')
            if 'iAbs' in self.metrics_tst:
                i_abs_tst = self.metrics_tst['iAbs']
                plt.scatter([self.x_tst[i_abs_tst]], [self.y_tst[i_abs_tst]],
                            marker='o', color='b', s=66, label='max abs test')

#        plt.scatter(self.x_tst - self.X_tst, self.Y_tst, marker='+', 
#                    color='r', label=r'$x = \varphi( Y_{tst} )$')

        plt.legend(bbox_to_anchor=(1.1, 0), loc='lower left')
        plt.xlabel('$x$')
        plt.ylabel('$y$')
        plt.grid()
        nrn = str(kwargs.get('neurons', None))
        x_max = str(kwargs.get('x_max', None))
        f = 'var' + str(variant) + '_nrn' + \
            nrn.replace('[', '').replace(']', '').replace(', ', '.')
        f += '_xmax' + str(x_max) 
        f += '_nse' + str(nse_abs) + '_' + str(nse_rel)
        plt.savefig(os.path.join(self.path, f + '.png'))
        plt.show()

        # X over Y diagram
        plt.title('$(x_{tst} - X_{tst})$ versus $Y_{tst}$')
        plt.xlabel('$Y_{tst}$')
        plt.ylabel(r'$\Delta X$')
        plt.grid()
        y = self.Y_tst if variant == 1 else self.y_tst
        plt.scatter(y, self.x_tst, marker='>', 
                    color='y', label=r'$\varphi(Y_{tst})$')
        plt.scatter(y, self.X_tst, marker='v', color='g', label=r'$X_{tst}$')
        plt.scatter(y, self.x_tst - self.X_tst, marker='+', 
                    color='r', label=r'$\varphi(Y_{tst}) - X_{tst}$')
        plt.plot([self.Y_tst.min(), self.Y_tst.max()], [0, 0])
        plt.legend(bbox_to_anchor=(1.1, 0), loc='lower left')
        plt.show()


class TestUM(unittest.TestCase):
    def setUp(self):
        print('///', os.path.basename(__file__))

        plot_f()
        
        # settings of data generation and training
        self.opt = {
                    'epochs': 500,
                    'goal': 1e-5, 
                    'noise_abs': 0.02, 
                    'noise_rel': 5e-2,
                    'n_point': 64,
                    'silent': True,
                    'trainer': 'rprop',
                    'train_test_ratio': (0.8, 0.2),                    
                    'transf': 'tansig',
                    'outputf': 'lin',
                    'plot': True,
                    'trials': 5, 
                    'x_min': 0.0,
                    'x_max': 1.2,
                    }

    def tearDown(self):
        pass

    def test1(self):
        foo = Foo()
        foo.generate_artificial_data(**self.opt)
        nrn = ([1], [2], [3], [4], [5], [6], [7],)
        nrn =([2],)
        optimizer = 'BFGS'
        for variant in (1, 2, ):
            for neurons in nrn:
                for optimizer in ('ga'): # ('BFGS', 'ga'):
                    dx = 2e-2 * (foo.X_trn.max() - foo.X_trn.min())   
                    dx = 0
                    foo(variant=variant, neurons=neurons,
                        optimizer=optimizer, bounds=((foo.X_trn.min() + dx, 
                                                      foo.X_trn.max() - dx),),                        
                        **self.opt)


if __name__ == '__main__':
    unittest.main()
