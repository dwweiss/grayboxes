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
      2018-08-29 DWW

  Acknowledgement:
      Modestga is a contribution by Krzyzstof Arendt, SDU, Denmark
"""

import sys
import numpy as np
import scipy.optimize
from grayboxes.boxmodel import BoxModel
try:
    import modestga
except ImportError:
    print('??? Package modestga not imported')


class LightGray(BoxModel):
    """
    Light gray box model y=f(x_com, x_tun)

    Extends functionality of class BoxModel by a train() method which 
    tunes theoretical submodel f(x) with a set of tuning parameters x_tun

    Notes:
        - tun0 (int, 1D or 2D array_like of float) is principally a 
          mandatory argument. If tun0 is missing, self.f() can return 
          this value when called as self.f(x=None)

        - The number of outputs (y.shape[1]) is limited to 1, see 
          self._nMaxOut

    Examples:
        def function(x, *args, **kwargs):
            if x is None:
                return np.ones(4)
            tun = args if len(args) == 4 else np.ones(4)
            return [tun[0] + tun[1] * (tun[2] * np.sin(x[0]) +
                    tun[3] * (x[1] - 1)**2)]

        def method(self, x, *args, **kwargs):
            if x is None:
                return np.ones(4)
            tun = args if len(args) == 4 else np.ones(4)
            return [tun[0] + tun[1] * (tun[2] * np.sin(x[0]) +
                    tun[3] * (x[1] - 1)**2)]

        ### compact form:
        y = LightGray(function)(X=X, Y=Y, x=x, tun0=4, methods='lm')

        ### expanded form:
        # assign theoretical submodel as function or method
        model = LightGray(function)  or
        model = LightGray(method)

        # (X, Y): training data
        X = [(1,2), (2,3), (4,5), (6,7), (7,8)]
        Y = [(1,), (2,), (3,), (4,), (5,)]    # or with: [1, 2, 3, 4, 5]

        # x: test data
        x = [(1, 4), (6, 6)]

        # before training, result of theor. submodel f(x) is returned
        y = model(x=x)                    # predict with white box model

        # train light gray with (X, Y), tun0 has 9 rand init. tun param
        model(X=X, Y=Y, tun0=rand(9, [[-10, 10]] * 4))           # train

        # after model is trained, it keeps its weights for further pred
        y = model(x=x)               # predict with light gray box model

        # alternatively: combined train and prediction, single initial 
        # tunining parameter set tun0
        y = model(X=X, Y=Y, tun0=4, x=x)             # train and predict
    """

    def __init__(self, f, identifier :str='LightGray') -> None:
        """
        Args:
            f (method or function):
                theoretical submodel f(self, x, *args, **kwargs) or
                f(x, *args, **kwargs) for single data point
                x is common parameter set and args is tuning param. set

            identifier (str, optional):
                object identifier
        """
        super().__init__(identifier=identifier, f=f)

        self._nMaxOut = 1        # max nOut is '1' due to implementation limits

        # populate 'validmethods' list
        self.scipyMinimizers = ['BFGS',
                                'L-BFGS-B',      # BFGS with less memory
                                'Nelder-Mead',   # gradient-free simplex
                                'Powell',       # gradient-free shooting
                                'CG',
                                # 'Newton-CG',       # requires Jacobian
                                'TNC',
                                # 'COBYLA',   # failed in grayBoxes test
                                'SLSQP',
                                # 'dogleg',          # requires Jacobian
                                # 'trust-ncg',       # requires Jacobian
                                'basinhopping',    # global (brute) opt.
                                'differential_evolution',  # global opt.
                                ]
        self.scipyRootFinders = ['lm',
                                 # 'hybr', 'broyden1', 'broyden2', 
                                 # 'anderson', 'linearmixing', 
                                 # 'diagbroyden', 'excitingmixing', 
                                 # 'krylov', 'df-sane'
                                 ]
        self.scipyEquationMinimizers = ['least_squares',  
                                                   # Levenberg-Marquardt
                                        'leastsq',
                                        ]
        self.geneticMinimizers = ['genetic', 'ga'] if 'modestga' \
            in sys.modules else []

        self.validMethods = self.scipyMinimizers + self.scipyRootFinders + \
            self.scipyEquationMinimizers + self.geneticMinimizers

    # function wrapper for scipy minimize
    def meanSquareErrror(self, weights, **kwargs):
        y = BoxModel.predict(self, self.X, *weights,
                          **self.kwargsDel(kwargs, 'x'))
        return np.mean((y - self.Y)**2)

    # function wrapper for scipy least_square and leastsq
    def difference(self, weights, **kwargs):
        return (BoxModel.predict(self, self.X, *weights,
                                 **self.kwargsDel(kwargs, 'x')) -
                self.Y).ravel()

    def minimizeLeastSquares(self, method, tun0, **kwargs):
        """
        Minimizes least squares: sum(self.f(self.X)-self.Y)^2) / X.size
            for a SINGLE initial tuning parameter set
        Updates self.ready and self.weights according to success of 
            optimizer

        Args:
            method (str):
                optimizing method for minimizing objective function
                [recommendation: 'BFGS' or 'L-BFGS-B' if ill-conditioned.
                 'Nelder-Mead' or 'Powell' if noisy data]

            tun0 (Iterable[float]:
                initial guess of tuning parameter set

            kwargs (Dict[str, Any], optional):
                keyword arguments:

                bounds (2-tuple of float or 2-tuple of iterable of float):
                    list of pairs (xMin, xMax) limiting x

                ... specific optimizer options

        Returns:
            (Dict[str, Union[float, int, str]]):
                results, see BoxModel.train()

        """
        results = self.initResults('method', method)
        self.weights = None             # required by BoxModel.predict()
        self.ready = True               # required by BoxModel.predict()

        if method in self.scipyMinimizers:
            if method.startswith('bas'):
                nItMax = kwargs.get('nItMax', 100)

                res = scipy.optimize.basinhopping(
                    func=self.meanSquareErrror, x0=tun0, niter=nItMax,
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

            elif method.startswith('dif'):
                nItMax = kwargs.get('nItMax', None)

                res = scipy.optimize.differential_evolution(
                    func=self.meanSquareErrror, bounds=[[-10, 10]]*tun0.size,
                    strategy='best1bin', maxiter=nItMax, popsize=15,
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
                validKeys = ['nItMax', 'adaptive', 'goal']
                kw = {}
                if any(k in kwargs for k in validKeys):
                    kw['options'] = {}
                    if method in ('SLSQP', 'Nelder-Mead', 'L-BFGS-B'):
                        kw['options']['maxiter'] = kwargs.get('nItMax', 100)
                    else:
                        kw['options']['maxiter'] = kwargs.get('nItMax', None)
                    if method == 'Nelder-Mead':
                        kw['options']['xatol'] = kwargs.get('goal', 1e-4)
                try:
                    res = scipy.optimize.minimize(
                        fun=self.meanSquareErrror, x0=tun0, method=method,
                        **kw)
                    if res.success:
                        results['weights'] = np.atleast_1d(res.x)
                        results['iterations'] = res.nit \
                            if method != 'COBYLA' else -1
                        results['evaluations'] = res.nfev
                    else:
                        self.write(4 * ' ' + '!!! ' + res.message)
                except scipy.optimize.OptimizeWarning:
                    results['weights'] = None
                    self.write(4 * ' ' + '!!! ' + res.message)

        elif method in self.scipyRootFinders:
            nItMax = kwargs.get('nItMax', 0)

            if method.startswith('lm'):
                res = scipy.optimize.root(
                    fun=self.difference, x0=tun0, args=(), method='lm',
                    jac=None, tol=None, callback=None,
                    options={  # 'func': None,mesg:_root_leastsq() got multiple
                             #                       values for argument 'func'
                             'col_deriv': 0, 'xtol': 1.49012e-08,
                             'ftol': 1.49012e-8, 'gtol': 0., 'maxiter': nItMax,
                             'eps': 0.0, 'factor': 100, 'diag': None})
                if res.success:
                    results['weights'] = np.atleast_1d(res.x)
                    results['iterations'] = -1
                    results['evaluations'] = res.nfev
                else:
                    self.write(4 * ' ' + '!!! ' + res.message)
            else:
                print("\n??? method:'" + str(method) + "' not implemented")

        elif method in self.scipyEquationMinimizers:
            if method.startswith('leastsq'):
                x, cov_x, infodict, mesg, ier = scipy.optimize.leastsq(
                    self.difference, tun0, full_output=True)
                if ier in [1, 2, 3, 4]:
                    results['weights'] = np.atleast_1d(x)
                    results['iterations'] = -1
                    results['evaluations'] = infodict['nfev']
                else:
                    self.write(4 * ' ' + '!!! ' + mesg)

            elif method == 'least_squares':
                res = scipy.optimize.least_squares(self.difference, tun0)
                if res.success:
                    results['weights'] = np.atleast_1d(res.x)
                    results['iterations'] = -1
                    results['evaluations'] = res.nfev
                else:
                    self.write(4 * ' ' + '!!! ' + res.message)

        elif method in self.geneticMinimizers:
            if 'modestga' in sys.modules and method in ('genetic', 'ga'):
                validKeys = ['tol', 'options', 'bounds']
                # see scipy's minimize
                kw = {k: kwargs[k] for k in validKeys if k in kwargs}
                if 'bounds' not in kw:
                    kw['bounds'] = [[0, 2]]*np.atleast_2d(tun0).shape[1]
                    self.write(4 * ' ' + '!!! bounds is missing ==> ' + 
                               str(kw['bounds']))
                res = modestga.minimize(fun=self.meanSquareErrror, x0=tun0,
                                        # TODO method=method,
                                        **kw)
                if True:  # TODO replace 'if true' with 'if res.success'
                    results['weights'] = np.atleast_1d(res.x)
                    results['iterations'] = -1            # TODO res.nit
                    results['evaluations'] = -1          # TODO res.nfev
                else:
                    self.write(4 * ' ' + '!!! ' + res.message)
        else:
            assert 0, '??? LightGray, invalid method: ' + str(method)

        self.weights = results['weights']
        self.ready = self.weights is not None

        return results

    def train(self, X, Y, **kwargs):
        """
        Trains model, stores X and Y as self.X and self.Y, and stores 
        result of best training trial as self.best.
        The tuning parameter set is stored as self._weights

        Args:
            X (2D or 1D array_like of float):
                training input, shape: (nPoint, nInp) or (nPoint,)

            Y (2D or 1D array_like of float):
                training target, shape: (nPoint, nOut) or (nPoint,)

            tun0 (2D or 1D array_like of float, optional):
                sequence of initial guess of the tuning parameter sets,
                If missing, then initial values will be all 1
                tun0.shape[1] is the number of tuning parameters
                [IS PASSED IN KWARGS to be compatible to parallel.py]

            kwargs (Any, optional):
                keyword arguments:

                methods (Union[str, Iterable[str]]):
                    optimizer method of
                    - scipy.optimizer.minimize or
                    - genetic algorithm
                    see: self.validMethods
                    default: 'BFGS'

                bounds (2-tuple of float or 2-tuple of iterable of float):
                    list of pairs (xMin, xMax) limiting x

        Returns:
            (Dict[str, Union[float, int, str]]):
                results, see BoxModel.train()

        Note:
            If argument 'tun0' is not given, self.f(None) must return an
            iterable of float providing the initial tuning parameter set
        """
        self.X = X if X is not None and Y is not None else self.X
        self.Y = Y if X is not None and Y is not None else self.Y

        # get series of initial tun par sets from 'tun0' or self.f(None)
        tun0seq = self.kwargsGet(kwargs, ('tun0', 'c0', 'C0'), None)
        if tun0seq is None:
            tun0seq = self.f(None)
            print(4 * ' ' + '!!! tun0 is None, from f(x=None) ==> tun0:', 
                  tun0seq)
        assert not isinstance(tun0seq, int), str(tun0seq)
        tun0seq = np.atleast_2d(tun0seq)         # shape: (nTrial, nTun)

        # get methods from kwargs
        methods = self.kwargsGet(kwargs, ('methods', 'method'))
        if methods is None:
            methods = self.validMethods[0]
        methods = np.atleast_1d(methods)
        if methods[0].lower() == 'all':
            methods = self.validMethods
        if any([tr not in self.validMethods for tr in methods]):
            methods = self.validMethods[0]
            self.write("!!! correct method: '" + methods + "' ==> ", methods)
        methods = np.atleast_1d(methods)

        # set detailed print (only if not silent)
        self.silent = kwargs.get('silent', self.silent)
        printDetails = kwargs.get('detailed', False) and not self.silent

        # loop over all methods
        self.best = self.initResults()
        for method in methods:
            self.write(4 * ' ' + method)

            # tries all initial tuning par sets if not global method
            if method in ('basinhopping', 'differential_evolution', 'genetic'):
                tun0seq = [tun0seq[0]]

            for iTrial, tun0 in enumerate(tun0seq):
                if printDetails:
                    message = (4+4) * ' ' + 'tun0: ' + str(np.round(tun0, 2))

                results = self.minimizeLeastSquares(
                    method, tun0, **self.kwargsDel(kwargs, ('method', 'tun0')))

                if results['weights'] is not None:
                    self.weights = results['weights']  
                                                # for BoxModel.predict()
                    err = self.error(X=X, Y=Y, silent=True)
                    self.weights = None         # back to None for train
                    if self.best['L2'] > err['L2']:
                        self.best.update(results)
                        self.best.update(err)
                        self.best['iTrial'] = iTrial
                        if printDetails:
                            message += ' +++'
                    if printDetails:
                        message += ' L2: ' + str(round(err['L2'], 6))
                else:
                    if printDetails:
                        message += ' ---'
                if printDetails:
                    self.write(message)

        self.weights = self.best['weights']
        self.ready = self.weights is not None

        self.write('+++ ' + "Best method: '" + self.best['method'] + "'")
        message = (4+4) * ' ' 
        for key in ['L2', 'abs']:
            if key in self.best:
                message += key + ': ' + str(float(str(round(self.best[key], 
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
            if key in self.best:
                message += key + ': ' + str(self.best[key]) + ', '
        self.write(message)

        return self.best

    def predict(self, x, *args, **kwargs):
        """
        Executes box model,stores input x as self.x and output as self.y

        Args:
            x (2D or 1D array_like of float):
                prediction input, shape: (nPoint, nInp) or (nInp,)

            args(float, optional):
                tuning parameter set if self._weights is None

            kwargs (Any], optional):
                keyword arguments

        Returns:
            (2D array of float):
                prediction output, shape: (nPoint, nOut)
        """
        args = self.weights if self.weights is not None else args
        return BoxModel.predict(self, x, *args, **self.kwargsDel(kwargs, 'x'))
