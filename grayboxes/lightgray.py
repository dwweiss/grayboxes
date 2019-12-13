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
      2019-12-12 DWW

  Acknowledgement:
      Modestga is a contribution by Krzyzstof Arendt, SDU, Denmark
"""

import sys
import numpy as np
import scipy.optimize
from typing import Any, Dict, List, Sequence, Union

from grayboxes.base import Float1D, Float2D, Function
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
        - c_ini (int, 1D or 2D array_like of float) is principally a
          mandatory argument. If c_ini is missing, self.f() can return 
          this value when called as self.f(x=None)

        - The number of outputs (y.shape[1]) is limited to 1, see 
          self._n_max_out

    Examples:
        def func(x, *c, **kwargs):
            c0, c1, c2, c3 = 1, 1, 1, 1
            if x is None:
                return c0, c1, c2, c3
            if len(c) == 4:
                c0, c1, c2, c3 = c
            return [c0 + c1 * (c2 * np.sin(x[0]) + c3 * (x[1] - 1)**2)]

        def meth(self, x, *c, **kwargs):
            c0, c1, c2, c3 = 1, 1, 1, 1
            if x is None:
                return c0, c1, c2, c3
            if len(c) == 4:
                c0, c1, c2, c3 = c
            return [c0 + c1 * (c2 * np.sin(x[0]) + c3 * (x[1] - 1)**2)]

        ### compact form:
        y = LightGray(func)(X=X, Y=Y, x=x, trainer='lm')

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

        # train light gray with (X, Y), c_ini has 9 random init. weigths
        model(X=X, Y=Y, c_ini=rand(9, [(-10, 10)] * 4))          # train

        # after model is trained, it keeps its weights for further pred
        y = model(x=x)                          # predict with box model

        # alternatively: combined train and prediction with single  
        # initial tuning parameter set c_ini
        y = model(X=X, Y=Y, c_ini=4, x=x)            # train and predict
    """

    def __init__(self, f: Function, identifier: str = 'LightGray') -> None:
        """
        Args:
            f:
                Theoretical submodel f(self, x, *c, **kwargs) or
                f(x, *c, **kwargs) for single data point
                x is common parameter set and c is tuning parameter set

            identifier:
                Unique object identifier
        """
        super().__init__(identifier=identifier, f=f)

        self._n_max_out: int = 1  # max n_out is '1' due to implementation

        # populate list of valid trainers
        self.scipy_minimizers: List[str] = [
                'BFGS',                                  # standard BFGS
                'L-BFGS-B',                      # BFGS with less memory
                'Nelder-Mead',                   # gradient-free simplex
                'Powell',                       # gradient-free shooting
                'CG',
                # 'Newton-CG',                       # requires Jacobian
                'TNC',
                # 'COBYLA',                   # failed in grayboxes test
                'SLSQP',
                # 'dogleg',                          # requires Jacobian
                # 'trust-ncg',                       # requires Jacobian
                'basinhopping',            # global (brute) optimization
                'differential_evolution',          # global optimization
                ]
        self.scipy_root_finders: List[str] = [
                'lm',                              # Levenberg-Marquardt
                # 'hybr', 'broyden1', 'broyden2',
                # 'anderson', 'linearmixing',
                # 'diagbroyden', 'excitingmixing',
                # 'krylov', 'df-sane'
                ]
        self.scipy_equ_minimizers: List[str] = [
                'leastsq',                # good for n_inp == n_out == 1
                'least_squares',
                # Levenberg-Marquardt
                ]
        self.genetic_minimizers: List[str] = \
            ['genetic', 'ga'] if 'modestga' in sys.modules else []

        self.scipy_curve_fitters: List[str] = ['curve_fit']

        self.valid_trainers: List[str] = self.scipy_minimizers + \
                                         self.scipy_root_finders + \
                                         self.scipy_equ_minimizers + \
                                         self.genetic_minimizers + \
                                         self.scipy_curve_fitters

    # function wrapper for scipy minimize
    def _mean_square_errror(self, c: Sequence[float], 
                            **kwargs: Any) -> Float2D:
        y: Float2D = BoxModel.predict(self, self.X, *c,
                                      **self.kwargs_del(kwargs, 'x'))
        dy = y - self.Y
        np.clip(dy, None, 1e20)   # max value: 1e308, sqrt(1e308) :1e154
        
        return np.mean(dy**2)

    # function wrapper for scipy least_square and leastsq
    def _difference(self, c: Sequence[float], **kwargs: Any) -> Float2D:
        y = BoxModel.predict(self, self.X, *c, **self.kwargs_del(kwargs, 'x'))
        
        return (y - self.Y).ravel()

    def _minimize_least_squares(self, trainer: str, c_ini: Sequence[float],
                                **kwargs: Any) -> Dict[str, Any]:
        """
        Minimizes least squares: sum(self.f(self.X)-self.Y)^2) / X.size
            for a SINGLE initial tuning parameter set
        Updates self.ready and self.weights if optimizer succedes

        Args:
            trainer:
                optimizing method for minimizing objective function
                [recommended: 'BFGS' or 'L-BFGS-B' if ill-conditioned.
                 'Nelder-Mead' or 'Powell' if noisy data]
                
                if trainer is 'all', all trainers will be tried 
                if trainer is 'auto', an optimal trainer will be chosen
                    base on heuristics

            c_ini:
                initial guess of tuning parameter set

        Kwargs:
            bounds (2-tuple of float or 2-tuple of sequence of float):
                list of pairs (x_min, x_max) limiting x

            ... specific optimizer options

        Returns:
            results, see BoxModel.train()
        """
        results = self.init_metrics('trainer', trainer)
        
        # self.weights and self.reday are used in BoxModel.predict()
        self.weights = None
        self.ready = True

        if trainer in self.scipy_minimizers:
            if trainer.startswith('bas'):
                n_it_max = kwargs.get('n_it_max', 100)

                res = scipy.optimize.basinhopping(
                    func=self._mean_square_errror, x0=c_ini, niter=n_it_max,
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
                    bounds=[(-10, 10)] * c_ini.size,
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
                kw: Dict[str, Any] = {}
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
                                                  x0=c_ini, method=trainer, 
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
                    fun=self._difference, x0=c_ini, args=(), method='lm',
                    jac=None, tol=None, callback=None,
                    options={# 'func': None, mesg:_root_leastsq() got 
                             #       multiple values for argument 'func'
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
                    self._difference, c_ini, full_output=True)
                if ier in [1, 2, 3, 4]:
                    results['weights'] = np.atleast_1d(x)
                    results['iterations'] = -1
                    results['evaluations'] = infodict['nfev']
                else:
                    self.write(4 * ' ' + '!!! ' + mesg)

            elif trainer == 'least_squares':
                res = scipy.optimize.least_squares(self._difference, c_ini)
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
                    kw['bounds'] = [[0, 2]]*np.atleast_2d(c_ini).shape[1]
                    self.write(4 * ' ' + '!!! bounds is missing ==> ' + 
                               str(kw['bounds']))
                res = modestga.minimize(fun=self._mean_square_errror, x0=c_ini,
                                        # TODO method=trainer,
                                        **kw)
                if True:  # TODO replace 'if True' with 'if res.success'
                    results['weights'] = np.atleast_1d(res.x)
                    results['iterations'] = -1            # TODO res.nit
                    results['evaluations'] = -1          # TODO res.nfev
                else:
                    self.write(4 * ' ' + '!!! ' + res.message)

        elif trainer in self.scipy_curve_fitters:
            if trainer == 'curve_fit':
                param, param_cov = scipy.optimize.curve_fit(self.f, 
                    self.X.ravel(), self.Y.ravel(), 
                    p0=c_ini)
                results['weights'] = param
        else:
            assert 0, '??? LightGray, invalid trainer: ' + str(trainer)

        if 'weights' in results:
            self.weights = results['weights']
            self.ready = self.weights is not None
        else:
            results['weights'] = None
            self.weights = None
            self.ready = False
        
        return results

    def train(self, X: Float2D, Y: Float2D, **kwargs: Any) -> Dict[str, Any]:
        """
        Trains model, stores X and Y as self.X and self.Y, and stores 
        result of best training trial as self.metrics
        The optimal tuning parameter set is stored as self.weights

        Args:
            X:
                training input, shape: (n_point, n_inp)

            Y:
                training target, shape: (n_point, n_out)

        Kwargs:
            bounds (2-tuple of float or 2-tuple of iterable of float):
                list of pairs (x_min, x_max) limiting x

            correct_xy_shape (bool):
                if False, shape of X and Y arrays will not be corrected
                if an error is assumed 
            
            trainer (str or sequence of str):
                optimizer method of
                - scipy.optimizer.minimize or
                - genetic algorithm
                see: self.valid_trainers
                default: 'BFGS'

            c_ini (2D or 1D array of float):
                sequence of initial guess of tuning parameter set 
                per trial 
                If c_ini is None, the initialvalues are returned from
                f(x=None)

                c_ini.shape: (number of trials,number of tuning params)
                [IS PASSED IN KWARGS to be compatible to parallel.py]

        Returns:
            metrics of training, see BoxModel.train()

        Note:
            If argument 'c_ini' is not given, self.f(None) must return an
            iterable of float providing the initial tuning parameter set
        """
        correct_xy_shape = kwargs.get('correct_xy_shape', True)
        correct_xy_shape = False
        self.set_XY(X=X, Y=Y, correct_xy_shape=correct_xy_shape)

        # get series of initial tun par sets from 'c_ini' or self.f(None)
        c_ini_trials = self.kwargs_get(kwargs, ('c_ini', 'tun0', 'c0'), None)
        if c_ini_trials is None:
            c_ini_trials = self.f(None)
            print(4 * ' ' + '!!! c_ini is None ==> from f(x=None):', 
                  c_ini_trials)
        assert not isinstance(c_ini_trials, int), '??? invalid c_ini ' + \
            str(c_ini_trials)
        c_ini_trials = np.atleast_2d(c_ini_trials)     # shape: (n_trial, n_tun)

        # replace 'all' and 'auto' in trainer, checksvalidity of trainer 
        trainer = self.kwargs_get(kwargs, 'trainer')
        trainer = np.atleast_1d(trainer)
        if trainer[0] is None:
            trainer = ['auto']
        if trainer[0].lower() == 'all':
            trainer = self.valid_trainers
        if any([trn not in self.valid_trainers + ['auto'] for trn in trainer]):
            self.write("    !!! trainer: '" + str(trainer) + "' ==> 'auto'")
            trainer = ['auto']
        if trainer[0].lower() == 'auto':
            if self.X.shape[1] == 1 and self.Y.shape[1] == 1:
                # consider 'lm' or 'curve_fit' if n_inp == n_out == 1
                trainer = ['leastsq']
            else:
                trainer = ['BFGS']
            self.write("    !!! trainer: 'auto' ==> '" + trainer[0] + "'")
            
        assert all([trn in self.valid_trainers for trn in trainer]), \
            str(trainer)
            
        # sets detailed print (only if not silent)
        if 'silent' in kwargs:
            self.silent = kwargs.get('silent', False)
        print_details = kwargs.get('detailed', False) and not self.silent

        # loop over all trainers
        self.metrics = self.init_metrics()
        self.metrics['trainer'] = trainer[0]
        self.metrics['weights'] = None
        message = ''
        self.write('+++ Loop over trainers')
                
        for trainer_ in trainer:
            self.write(4 * ' ' + trainer_)

            # tries all initial tuning par sets if not global method
            if trainer_ in ('basinhopping', 'differential_evolution', 
                            'genetic'):
                c_ini_trials = [c_ini_trials[0]]

            for i_trial, c_ini in enumerate(c_ini_trials):
                if print_details:
                    message = (4+0) * ' ' + 'c_ini: ' + str(np.round(c_ini, 2))

                results = self._minimize_least_squares(trainer_, c_ini, \
                    **self.kwargs_del(kwargs, ('trainer', 'c_ini', 'tun0')))

                if results['weights'] is not None:
                    self.weights = results['weights']
                                                # for BoxModel.predict()
                    metrics = self.evaluate(X=X, Y=Y, silent=True)
                    self.weights = None         # back to None for train
                    if self.metrics['L2'] > metrics['L2']:
                        self.metrics['trainer'] = trainer_
                        self.metrics.update(results)
                        self.metrics.update(metrics)
                        self.metrics['i_trial'] = i_trial
                        if print_details:
                            message += ' +++'
                    if print_details:
                        message += ' L2: ' + str(round(metrics['L2'], 6))
                else:
                    if print_details:
                        message += ' ---'
                if print_details:
                    self.write(message)

        self.weights = self.metrics['weights']
        self.ready = self.weights is not None

        self.write('+++ ' + "Best trainer: '" + self.metrics['trainer'] + "'")
        message = (4+0) * ' ' 
        for key in ['L2', 'abs']:
            if key in self.metrics:
                message += key + ': ' + str(float(str(round(self.metrics[key], 
                                                            4)))) + ', '
        self.write(message)
        message = (4+0) * ' ' + 'w: '
        if self.weights is not None:
            message += str(np.round(self.weights, 4))
        else:
            message += str(None)
            self.write(message)
        message = (4) * ' '
        for key in ['i_trial', 'iterations', 'evaluations']:
            if key in self.metrics:
                message += key + ': ' + str(self.metrics[key]) + ', '
        self.write(message)

        # the 'weights' keys is only locally used in self.train()
        del self.metrics['weights']

        return self.metrics

    def predict(self, x: Union[Float1D, Float2D], *c: float,
                **kwargs) -> Float2D:
        """
        Executes box model,stores input x as self.x and output as self.y

        Args:
            x (2D or 1D array of float):
                prediction input, shape: (n_point, n_inp)
                shape (n_inp,) is tolerated

            c:
                tuning parameters as positional arguments if 
                self.weights is None

        Kwargs:
            Keyword arguments

        Returns:
            prediction output, shape: (n_point, n_out)
        """
        c = self.weights if self.weights is not None else c
        return BoxModel.predict(self, x, *c, 
                                **self.kwargs_del(kwargs, ('x', 'c')))
