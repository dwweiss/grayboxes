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
      2018-05-23 DWW
"""

import sys
import numpy as np
import scipy.optimize
from Model import Model
#############################################
try:
    import krza_ga
except ImportError:
    print('??? module krza_ga not imported')
HAS_KRZA_GA = 'krza_ga' in sys.modules
#############################################


class LightGray(Model):
    """
    Light gray box model y=f(x, C)

    Extends the functionality of class Model by a train() method which fits
    the theoretical submodel f(x) with constant fit parameters 'C'

    Notes:
        - C0 (int, 1D or 2D array_like of float) is principally a
          mandatory argument. Alternatively, self.f(x=None) must return
          a C0-like value if 'C0' is not given as an argument
        - The number of outputs y.shape[1] is limited to 1, see self._nMaxOut

    Examples:
        def function(x, *args):
            c0, c1, c2, c3 = args if len(args) > 0 else np.ones(4)
            return [c0 + c1 * (c2 * np.sin(x[0]) + c3 * (x[1] - 1)**2)]

        def function2(x, *args):
            if x is None:
                return [1, 1, 1, 1]                    # initial fit parameters
            c0, c1, c2, c3 = args if len(args) > 0 else np.ones(4)
            return [c0 + c1 * (c2 * np.sin(x[0]) + c3 * (x[1] - 1)**2)]

        def method(self, x, *args):
            c0, c1, c2, c3 = args if len(args) > 0 else np.ones(4)
            return [c0 + c1 * (c2 * np.sin(x[0]) + c3 * (x[1] - 1)**2)]

        ### compact form:
        y = LightGray(function)(X=X, Y=Y, x=x, C0=4, trainers='lm')

        ### expanded form:
        # assign theoretical submodel as function or method ('self'-attribute)
        model = LightGray(function)  or
        model = LightGray(method)

        # (X, Y): training data
        X = [(1,2), (2,3), (4,5), (6,7), (7,8)]
        Y = [(1,), (2,), (3,), (4,), (5,)]     # alternatively: [1, 2, 3, 4, 5]

        # x: test data
        x = [(1, 4), (6, 6)]

        # before training, result of theoretical submodel f(x) is returned
        y = model(x=x)                           # predict with white box model

        # train light gray with data (X, Y), C0 has 9 random initial fit params
        model(X=X, Y=Y, C0=rand(9, [[-10, 10]] * 4))                    # train

        # after model is trained, it keeps its weights for further preddictions
        y = model(x=x)                      # predict with light gray box model

        # alternatively: combined train and pred, single initial fit par set C0
        y = model(X=X, Y=Y, C0=4, x=x)                      # train and predict
    """

    def __init__(self, f, identifier='LightGray'):
        """
        Args:
            f (method or function):
                theoretical submodel f(self, x) or f(x) for single data point

            identifier (string, optional):
                object identifier
        """
        super().__init__(identifier=identifier, f=f)

        self._nMaxOut = 1        # max nOut is '1' due to implementation limits
        self._nMaxWeights = 8           # number of arguments of f() except 'x'

    def train(self, X, Y, **kwargs):
        """
        Trains model, stores X and Y as self.X and self.Y, and stores result
        of best training trial as self.best.
        Fitted coefficients are stored as self._weights

        Args:
            X (2D or 1D array_like of float):
                training input, shape: (nPoint, nInp) or shape: (nPoint,)

            Y (2D or 1D array_like of float):
                training target, shape: (nPoint, nOut) or shape: (nPoint,)

            C0 (2D or 1D array_like of float, optional):
                sequence of initial guess of the tuning parameter sets,
                If missing, then initial values will be all 1
                C0.shape[1] is the number of fit parameters
                [IS PASSED IN KWARGS to be compatible to parallel.py]

            kwargs (dict, optional):
                keyword arguments:

                bounds (2-tuple of float or 2-tuple of 1D array_like of float):
                    list of pairs (xMin, xMax) limiting x

                trainers (string or 1D array_like of string):
                    optimizer method of
                    - scipy.optimizer.curve_fit or
                    - scipy.optimizer.minimize or
                    - genetic algorithm
                    see: self.validYrainers
                    default: 'trf'

        Returns:
            see Model.train()

        Note:
            If argument 'C0' is not given, self.f(None) must return the number
            of tuning parameters or an array of initial tuning parameter sets
        """
        self.X = X if X is not None and Y is not None else self.X
        self.Y = Y if X is not None and Y is not None else self.Y

        # get series of initial fit par sets from 'C0', 'CInit' or self.f(None)
        C0 = self.kwargsGet(kwargs, ('C0', 'CInit'))
        if C0 is None:
            C0 = self.f(None)
        if isinstance(C0, int):
            C0 = np.ones(C0)
        C0 = np.atleast_2d(C0)                    # shape: (nTrial, nTun)

        # get trainers from kwargs or 'validTrainers' list
        scipyCurveFitTrainers = ['trf', 'lm', 'dogbox']
        scipyMinimizeOptimizers = ['Nelder-Mead',
                                   'Powell',
                                   'CG',
                                   'BFGS',
                                   # 'Newton-CG',           # requires Jacobian
                                   'L-BFGS-B',
                                   'TNC',
                                   'COBYLA',
                                   'SLSQP',
                                   # 'dogleg',              # requires Jacobian
                                   # 'trust-ncg',           # requires Jacobian
                                   'basinhopping',             # GLOBAL optimum
                                   'differential_evolution',   # GLOBAL optimum
                                   ]
#############################################
        if HAS_KRZA_GA:
            geneticOptimizers = ['krza_ga']           # Krzystof's GA optimizer
        else:
            geneticOptimizers = []
#############################################

        self.validTrainers = scipyCurveFitTrainers + scipyMinimizeOptimizers +\
            geneticOptimizers
        trainers = self.kwargsGet(kwargs, ('trainers', 'trainer', 'train'))
        if trainers is None:
            trainers = self.validTrainers[0]
        trainers = np.atleast_1d(trainers)
        if trainers[0].lower() == 'all':
            trainers = self.validTrainers
        if any([tr not in self.validTrainers for tr in trainers]):
            trainers = self.validTrainers[0]
            self.write('??? unknown trainer found, correct to:', trainers)

        bounds = kwargs.get('bounds', (-np.inf, np.inf))
        printDetails = kwargs.get('detailed', False)

        # function wrapper for scipy curve_fit
        def f_curve_fit(xT, *args):
            # xT.shape:(nInp, nPoint), xT.T.shape:(nPoint,nInp)
            return Model.predict(self, xT.T, *args,
                                 **self.kwargsDel(kwargs, 'x')).ravel()

        # function wrapper for scipy minimize etc
        def objective(weights):
            y = Model.predict(self, self.X, *weights,
                              **self.kwargsDel(kwargs, 'x'))
            return np.sqrt(np.mean((y - self.Y)**2))

        # loop over all trainers
        self.write('    fit (', None)
        self.best = self.initBest()
        self.ready = True                         # required by Model.predict()
        for trainer in trainers:
            self.write(trainer, ', ' if trainer != trainers[-1] else '', None)

            # loop over all initial fit parameter sets
            for iTrial, c0 in enumerate(C0):
                if printDetails:
                    if iTrial == 0:
                        self.write()
                    self.write('        C0: ', str(np.round(c0, 2)), None)

                self.weights = None
                self.ready = True                 # required by Model.predict()

                if trainer in scipyCurveFitTrainers:
                    _bounds = (-np.inf, +np.inf) if trainer == 'lm' else bounds
                    try:
                        self.weights, cov = scipy.optimize.curve_fit(
                            f=f_curve_fit, xdata=self.X.T,     # (nInp, nPoint)
                            ydata=self.Y.ravel(),                   # (nPoint,)
                            method=trainer, p0=c0,                    # (nTun,)
                            sigma=None, absolute_sigma=False, bounds=_bounds)
                    except RuntimeError:
                        print('\n??? scipy curve_fit: maxiter exceeded')

                elif trainer in scipyMinimizeOptimizers:
                    if trainer.startswith('bas'):
                        res = scipy.optimize.basinhopping(
                            func=objective,
                            x0=c0,
                            niter=100, T=1.0, stepsize=0.5,
                            minimizer_kwargs=None, take_step=None,
                            accept_test=None, callback=None, interval=50,
                            disp=False, niter_success=None)
                        if 'success' in res.message[0]:
                            self.weights = res.x
                        else:
                            self.weights = None

                    elif trainer.startswith('dif'):
                        res = scipy.optimize.differential_evolution(
                            func=objective,
                            bounds=[[-10, 10]]*c0.size,
                            strategy='best1bin', maxiter=None,
                            popsize=15, tol=0.01, mutation=(0.5, 1),
                            recombination=0.7, seed=None, disp=False,
                            polish=True, init='latinhypercube')
                        if res.success:
                            self.weights = np.atleast_1d(res.x)
                        else:
                            self.weights = None

                    else:
                        validKeys = ['maxiter', 'tol', 'options']
                        kw = {k: kwargs[k] for k in validKeys if k in kwargs}
                        res = scipy.optimize.minimize(fun=objective, x0=c0,
                                                      method=trainer, **kw)
                        if res.success:
                            self.weights = np.atleast_1d(res.x)
                        else:
                            self.weights = None

                elif trainer in geneticOptimizers:
                    if HAS_KRZA_GA and trainer == 'ga_ka':
                        #############################################
                        validKeys = ['tol', 'options']  # see scipy's minimize
                        kw = {k: kwargs[k] for k in validKeys if k in kwargs}
                        res = krza_ga.minimize(fun=objective, x0=c0,
                                               method=trainer, **kw)
                        #############################################
                        if res.success:
                            self.weights = np.atleast_1d(res.x)
                        else:
                            self.weights = None
                else:
                    assert 0, str(trainer)

                if self.weights is not None:
                    actual = self.error(X=X, Y=Y, silent=True)
                    if self.best['L2'] > actual['L2']:
                        self.best = actual
                        self.best['weights'] = self.weights
                        self.best['trainer'] = trainer
                        self.best['epochs'] = -1
                        self.best['iTrial'] = iTrial
                        if printDetails:
                            self.write(' +++', None)
                    if printDetails:
                        self.write(' L2: ', round(actual['L2'], 14))
                else:
                    self.write(' ----------')

        self.weights = self.best['weights']
        self.ready = self.weights is not None

        self.write('), w: ', None)
        self.write(str(np.round(self.weights, 4)))
        self.write('    best trainer: ', "'", self.best['trainer'], "'",
                   ', L2: ', float(str(round(self.best['L2'], 4))),
                   ', abs: ', float(str(round(self.best['abs'], 4))),
                   ', iTrial: ', self.best['iTrial'])

        return self.best

    def predict(self, x, *args, **kwargs):
        """
        Executes Model, stores input x as self.x and output as self.y

        Args:
            x (2D or 1D array_like of float):
                prediction input, shape: (nPoint, nInp) or shape: (nInp)

            args(list arguments, optional):
                constant fit parameters if self._weights is None

            kwargs (dict, optional):
                keyword arguments

        Returns:
            (2D array of float):
                prediction output, shape: (nPoint, nOut)
        """
        args = self._weights if self._weights is not None else args
        return Model.predict(self, x, *args, **self.kwargsDel(kwargs, 'x'))


# Examples ####################################################################

if __name__ == '__main__':
    ALL = 0

    from plotArrays import plot_X_Y_Yref
    import Model as md
    from White import White

    def f(self, x, *args, **kwargs):
        """
        Theoretical submodel for single data point

        Aargs:
            x (1D array_like of float):
                input

            args (argument list):
                fit parameters as positional arguments

            kwargs (dict, optional):
                keyword arguments {str: float or int or str}
        """
        p = args if len(args) > 0 else np.ones(4)
        y0 = p[0] + p[1] * np.sin(p[2] * x[0]) + p[3] * (x[1] - 1.5)**2
        return [y0]

    s = 'Creates exact output y_exa(X), add noise, target is Y(X)'
    print('-' * len(s) + '\n' + s + '\n' + '-' * len(s))

    noise_abs = 0.25
    noise_rel = 10e-2
    X = md.grid(8, [-1, 8], [0, 3])
    y_exa = White(f)(x=X, silent=True)
    Y = md.noise(y_exa, absolute=noise_abs, relative=noise_rel)
    plot_X_Y_Yref(X, Y, y_exa, ['X', 'Y_{nse}', 'y_{exa}'])

    if 0 or ALL:
        s = 'Fits model, compare: y(X) vs y_exa(X)'
        print('-' * len(s) + '\n' + s + '\n' + '-' * len(s))

        # train with 9 random initial tuning parameter sets
        C0 = md.rand(16, [0, 2], [0, 2], [0, 2], [0, 2])
        model = LightGray(f)

#############################################
        if HAS_KRZA_GA:
            trainer = 'krza_ga'
        else:
            trainer = ['differential_evolution', 'lm']
        y = model(X=X, Y=Y, C0=C0, x=X, trainer=trainer, detailed=True)
#############################################

        plot_X_Y_Yref(X, y, y_exa, ['X', 'y', 'y_{exa}'])
        if 0:
            print('best:', model.best)
            df = model.xy2frame()
            print('=== df:\n', df)

    if 1 or ALL:
        def f2(self, x, *args, **kwargs):
            if x is None:
                return 4
            p = args if len(args) > 0 else np.ones(4)
            y0 = p[0] + p[1] * np.sin(p[2] * x[0]) + p[3] * (x[1] - 1.5)**2
            return [y0]

        # train with single initial tuning parameter set, nTun from f2(None)
        y = LightGray(f2)(X=X, Y=Y, x=X, silent=not True, trainer='all')
