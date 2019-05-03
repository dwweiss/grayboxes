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
      2019-03-20 DWW

  Acknowledgement:
      Modestga is a contribution by Krzyzstof Arendt, SDU, Denmark
"""

import sys
import numpy as np
import scipy.optimize
from typing import Any, Callable, Dict, Sequence

from grayboxes.boxmodel import BoxModel
try:
    import modestga
except ImportError:
    print('!!! Package modestga not imported')


class LightGray(BoxModel):
    """
    Light gray box model y=f(x_com, x_tun)

    Extends functionality of class BoxModel by a train() method which 
    tunes theoretical submodel f(x) with set of tuning parameters x_tun

    Notes:
        - tun0 (int, 1D or 2D array_like of float) is principally a 
          mandatory argument. If tun0 is missing, self.f() can return 
          this value when called as self.f(x=None)

        - The number of outputs (y.shape[1]) is limited to 1, see 
          self._n_max_out

    Examples:
        def func(x, *args, **kwargs):
            if x is None:
                return np.ones(4)
            tun = args if len(args) == 4 else np.ones(4)
            return [tun[0] + tun[1] * (tun[2] * np.sin(x[0]) +
                    tun[3] * (x[1] - 1)**2)]

        def meth(self, x, *args, **kwargs):
            if x is None:
                return np.ones(4)
            tun = args if len(args) == 4 else np.ones(4)
            return [tun[0] + tun[1] * (tun[2] * np.sin(x[0]) +
                    tun[3] * (x[1] - 1)**2)]

        ### compact form:
        y = LightGray(func)(X=X, Y=Y, x=x, tun0=4, trainer='lm')

        ### expanded form:
        # assign theoretical submodel as function or method
        model = LightGray(func)  or
        model = LightGray(meth)

        # (X, Y): training data
        X = [(1,2), (2,3), (4,5), (6,7), (7,8)]
        Y = [(1,), (2,), (3,), (4,), (5,)]

        # x: test data
        x = [(1, 4), (6, 6)]

        # before training, result of theor. submodel f(x) is returned
        y = model(x=x)                    # predict with white box model

        # train light gray with (X, Y), tun0 has 9 rand init. tun param
        model(X=X, Y=Y, tun0=rand(9, [[-10, 10]] * 4))           # train

        # after model is trained, it keeps its weights for further pred
        y = model(x=x)               # predict with light gray box model

        # alternatively: combined train and prediction, single initial 
        # tuning parameter set tun0
        y = model(X=X, Y=Y, tun0=4, x=x)             # train and predict
    """

    def __init__(self, f: Callable, identifier: str='LightGray') -> None:
        """
        Args:
            f:
                Theoretical submodel f(self, x, *args, **kwargs) or
                f(x, *args, **kwargs) for single data point
                x is common parameter set and args is tuning param. set

            identifier:
                Unique object identifier
        """
        super().__init__(identifier=identifier, f=f)

        self._n_max_out = 1  # max n_out is '1' due to implementation

        # populate list of valid trainers
        self.scipy_minimizers = ['BFGS',
                                'L-BFGS-B',      # BFGS with less memory
                                'Nelder-Mead',   # gradient-free simplex
                                'Powell',       # gradient-free shooting
                                'CG',
                                 # 'Newton-CG',      # requires Jacobian
                                'TNC',
                                 # 'COBYLA',  # failed in grayBoxes test
                                'SLSQP',
                                 # 'dogleg',         # requires Jacobian
                                 # 'trust-ncg',      # requires Jacobian
                                 'basinhopping',   # global (brute) opt.
                                 'differential_evolution',  # global opt
                                 ]
        self.scipy_root_finders = ['lm',
                                   # 'hybr', 'broyden1', 'broyden2',
                                   # 'anderson', 'linearmixing',
                                   # 'diagbroyden', 'excitingmixing',
                                   # 'krylov', 'df-sane'
                                   ]
        self.scipy_equ_minimizers = ['least_squares',
                                     # Levenberg-Marquardt
                                     'leastsq',
                                     ]
        self.genetic_minimizers = ['genetic', 'ga'] if 'modestga' \
                                                       in sys.modules else []

        self.valid_trainers = self.scipy_minimizers + \
                              self.scipy_root_finders + \
                              self.scipy_equ_minimizers + \
                              self.genetic_minimizers

    # function wrapper for scipy minimize
    def _mean_square_errror(self, weights: Sequence[float], **kwargs: Any) \
            -> np.ndarray:
        y = BoxModel.predict(self, self.X, *weights,
                             **self.kwargs_del(kwargs, 'x'))
        return np.mean((y - self.Y)**2)

    # function wrapper for scipy least_square and leastsq
    def _difference(self, weights: Sequence[float], **kwargs: Any) \
            -> np.ndarray:
        return (BoxModel.predict(self, self.X, *weights,
                                 **self.kwargs_del(kwargs, 'x')) -
                self.Y).ravel()

    def _minimize_least_squares(self, trainer: str, tun0: np.ndarray,
                                **kwargs: Any) -> Dict[str, Any]:
        """
        Minimizes least squares: sum(self.f(self.X)-self.Y)^2) / X.size
            for a SINGLE initial tuning parameter set
        Updates self.ready and self.weights according to success of 
            optimizer

        Args:
            trainer:
                optimizing method for minimizing objective function
                [recommended: 'BFGS' or 'L-BFGS-B' if ill-conditioned.
                 'Nelder-Mead' or 'Powell' if noisy data]

            tun0:
                initial guess of tuning parameter set

        Kwargs:
            bounds (2-tuple of float or 2-tuple of sequence of float):
                list of pairs (x_min, x_max) limiting x

            ... specific optimizer options

        Returns:
            (dictionary):
                results, see BoxModel.train()

        """
        results = self.init_metrics('trainer', trainer)
        self.weights = None             # required by BoxModel.predict()
        self.ready = True               # required by BoxModel.predict()

        if trainer in self.scipy_minimizers:
            if trainer.startswith('bas'):
                n_it_max = kwargs.get('n_it_max', 100)

                res = scipy.optimize.basinhopping(
                    func=self._mean_square_errror, x0=tun0, niter=n_it_max,
                    T=1.0,
                    stepsize=0.5, minimizer_kwargs=None,
                    take_step=None, accept_test=None, callback=None,
                    interval=50, disp=False, niter_success=None)
                if 'success' in res.message[0]:
                    results['weights'] = np.atleast_1d(res.x)
                    results['iterations'] = res.nit
                    results['evaluations'] = res.nfev
                else:
                    self.write(4 * ' ' + res.message)

            elif trainer.startswith('dif'):
                n_it_max = kwargs.get('n_it_max', None)

                res = scipy.optimize.differential_evolution(
                    func=self._mean_square_errror, 
                    bounds=[[-10, 10]] * tun0.size,
                    strategy='best1bin', maxiter=n_it_max, popsize=15,
                    tol=0.01, mutation=(0.5, 1), recombination=0.7,
                    seed=None, disp=False, polish=True,
                    init='latinhypercube')
                if res.success:
                    results['weights'] = np.atleast_1d(res.x)
                    results['iterations'] = res.nit
                    results['evaluations'] = res.nfev
                else:
                    self.write(4 * ' ' + '!!! ' + res.message)
            else:
                valid_keys = ['n_it_max', 'adaptive', 'goal']
                kw = {}
                if any(k in kwargs for k in valid_keys):
                    kw['options'] = {}
                    if trainer in ('SLSQP', 'Nelder-Mead', 'L-BFGS-B'):
                        kw['options']['maxiter'] = kwargs.get('n_it_max', 100)
                    else:
                        kw['options']['maxiter'] = kwargs.get('n_it_max', None)
                    if trainer == 'Nelder-Mead':
                        kw['options']['xatol'] = kwargs.get('goal', 1e-4)
                try:
                    res = scipy.optimize.minimize(fun=self._mean_square_errror,
                                                  x0=tun0, method=trainer, 
                                                  **kw)
                    if res.success:
                        results['weights'] = np.atleast_1d(res.x)
                        results['iterations'] = res.nit \
                            if trainer != 'COBYLA' else -1
                        results['evaluations'] = res.nfev
                    else:
                        self.write(4 * ' ' + '!!! ' + res.message)
                except scipy.optimize.OptimizeWarning:
                    results['weights'] = None
                    self.write(4 * ' ' + '!!! ' + res.message)

        elif trainer in self.scipy_root_finders:
            n_it_max = kwargs.get('n_it_max', 0)

            if trainer.startswith('lm'):
                res = scipy.optimize.root(
                    fun=self._difference, x0=tun0, args=(), method='lm',
                    jac=None, tol=None, callback=None,
                    options={  # 'func': None,mesg:_root_leastsq() got 
                             #           multiple values for argument 'func'
                             'col_deriv': 0, 'xtol': 1.49012e-08,
                             'ftol': 1.49012e-8, 'gtol': 0., 
                             'maxiter': n_it_max, 'eps': 0.0, 'factor': 100, 
                             'diag': None})
                if res.success:
                    results['weights'] = np.atleast_1d(res.x)
                    results['iterations'] = -1
                    results['evaluations'] = res.nfev
                else:
                    self.write(4 * ' ' + '!!! ' + res.message)
            else:
                print("\n??? trainer:'" + str(trainer) + "' not implemented")

        elif trainer in self.scipy_equ_minimizers:
            if trainer.startswith('leastsq'):
                x, cov_x, infodict, mesg, ier = scipy.optimize.leastsq(
                    self._difference, tun0, full_output=True)
                if ier in [1, 2, 3, 4]:
                    results['weights'] = np.atleast_1d(x)
                    results['iterations'] = -1
                    results['evaluations'] = infodict['nfev']
                else:
                    self.write(4 * ' ' + '!!! ' + mesg)

            elif trainer == 'least_squares':
                res = scipy.optimize.least_squares(self._difference, tun0)
                if res.success:
                    results['weights'] = np.atleast_1d(res.x)
                    results['iterations'] = -1
                    results['evaluations'] = res.nfev
                else:
                    self.write(4 * ' ' + '!!! ' + res.message)

        elif trainer in self.genetic_minimizers:
            if 'modestga' in sys.modules and trainer in ('genetic', 'ga'):
                valid_keys = ['tol', 'options', 'bounds']
                # see scipy's minimize
                kw = {k: kwargs[k] for k in valid_keys if k in kwargs}
                if 'bounds' not in kw:
                    kw['bounds'] = [[0, 2]]*np.atleast_2d(tun0).shape[1]
                    self.write(4 * ' ' + '!!! bounds is missing ==> ' + 
                               str(kw['bounds']))
                res = modestga.minimize(fun=self._mean_square_errror, x0=tun0,
                                        # TODO method=trainer,
                                        **kw)
                if True:  # TODO replace 'if True' with 'if res.success'
                    results['weights'] = np.atleast_1d(res.x)
                    results['iterations'] = -1            # TODO res.nit
                    results['evaluations'] = -1          # TODO res.nfev
                else:
                    self.write(4 * ' ' + '!!! ' + res.message)
        else:
            assert 0, '??? LightGray, invalid trainer: ' + str(trainer)

        self.weights = results['weights']
        self.ready = self.weights is not None

        return results

    def train(self, X: np.ndarray, Y: np.ndarray, **kwargs: Any) \
            -> Dict[str, Any]:
        """
        Trains model, stores X and Y as self.X and self.Y, and stores 
        result of best training trial as self.metrics
        The tuning parameter set is stored as self._weights

        Args:
            X (2D array of float):
                training input, shape: (n_point, n_inp)

            Y (2D array of float):
                training target, shape: (n_point, n_out)

        Kwargs:
            tun0 (2D or 1D array of float):
                sequence of initial guess of the tuning parameter sets,
                If missing, then initial values will be all 1
                tun0.shape[1] is the number of tuning parameters
                [IS PASSED IN KWARGS to be compatible to parallel.py]

            trainer (Union[str, Sequence[str]]):
                optimizer method of
                - scipy.optimizer.minimize or
                - genetic algorithm
                see: self.valid_trainers
                default: 'BFGS'

            bounds (2-tuple of float or 2-tuple of iterable of float):
                list of pairs (x_min, x_max) limiting x

        Returns:
            (dictionary):
                results, see BoxModel.train()

        Note:
            If argument 'tun0' is not given, self.f(None) must return an
            iterable of float providing the initial tuning parameter set
        """
        
        correct_xy_shape = kwargs.get('correct_xy_shape', True)
        self.set_XY(X=X, Y=Y, correct_xy_shape=correct_xy_shape)

        # get series of initial tun par sets from 'tun0' or self.f(None)
        tun0seq = self.kwargs_get(kwargs, ('tun0', 'c0', 'C0'), None)
        if tun0seq is None:
            tun0seq = self.f(None)
            print(4 * ' ' + '!!! tun0 is None, from f(x=None) ==> tun0:', 
                  tun0seq)
        assert not isinstance(tun0seq, int), str(tun0seq)
        tun0seq = np.atleast_2d(tun0seq)         # shape: (nTrial, nTun)

        trainer = self.kwargs_get(kwargs, 'trainer')
        if trainer is None:
            trainer = self.valid_trainers[0]
        trainer = np.atleast_1d(trainer)
        if trainer[0].lower() == 'all':
            trainer = self.valid_trainers
        if any([tr not in self.valid_trainers for tr in trainer]):
            trainer = self.valid_trainers[0]
            self.write("!!! correct trainer: '" + trainer + "' ==> " + trainer)
        trainer = np.atleast_1d(trainer)

        # set detailed print (only if not silent)
        self.silent = kwargs.get('silent', self.silent)
        print_details = kwargs.get('detailed', False) and not self.silent

        # loop over all trainer
        self.metrics = self.init_metrics()
        message = ''
        for _trainer in trainer:
            self.write(4 * ' ' + _trainer)

            # tries all initial tuning par sets if not global method
            if _trainer in ('basinhopping', 'differential_evolution', 
                            'genetic'):
                tun0seq = [tun0seq[0]]

            for iTrial, tun0 in enumerate(tun0seq):
                if print_details:
                    message = (4+4) * ' ' + 'tun0: ' + str(np.round(tun0, 2))

                results = self._minimize_least_squares(_trainer, tun0, \
                    **self.kwargs_del(kwargs, ('trainer', 'tun0')))

                if results['weights'] is not None:
                    self.weights = results['weights']  
                                                # for BoxModel.predict()
                    err = self.evaluate(X=X, Y=Y, silent=True)
                    self.weights = None         # back to None for train
                    if self.metrics['L2'] > err['L2']:
                        self.metrics.update(results)
                        self.metrics.update(err)
                        self.metrics['iTrial'] = iTrial
                        if print_details:
                            message += ' +++'
                    if print_details:
                        message += ' L2: ' + str(round(err['L2'], 6))
                else:
                    if print_details:
                        message += ' ---'
                if print_details:
                    self.write(message)

        self.weights = self.metrics['weights']
        self.ready = self.weights is not None

        self.write('+++ ' + "Best trainer: '" + self.metrics['trainer'] + "'")
        message = (4+4) * ' ' 
        for key in ['L2', 'abs']:
            if key in self.metrics:
                message += key + ': ' + str(float(str(round(self.metrics[key], 
                                                            4)))) + ', '
        self.write(message)
        message = (4+4) * ' ' + 'w: '
        if self.weights is not None:
            message += str(np.round(self.weights, 4))
        else:
            message = str(None)
            self.write(message)
        message = (4+4) * ' '
        for key in ['iTrial', 'iterations', 'evaluations']:
            if key in self.metrics:
                message += key + ': ' + str(self.metrics[key]) + ', '
        self.write(message)

        return self.metrics

    def predict(self, x: np.ndarray, *args: float, **kwargs) -> np.ndarray:
        """
        Executes box model,stores input x as self.x and output as self.y

        Args:
            x (2D or 1D array of float):
                prediction input, shape: (n_point, n_inp) or (n_inp,)

        Args:
            Tuning parameter set if self._weights is None

        Kwargs:
            Keyword arguments

        Returns:
            (2D array of float):
                prediction output, shape: (n_point, n_out)
        """
        args = self.weights if self.weights is not None else args
        return BoxModel.predict(self, x, *args, **self.kwargs_del(kwargs, 'x'))
