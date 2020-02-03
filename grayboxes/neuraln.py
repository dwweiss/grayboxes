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

  Acknowledgements:
      Neurolab is a contribution by E. Zuev (pypi.python.org/pypi/neurolab)
"""

__all__ = ['Neural']

from collections import OrderedDict
import numpy as np
import matplotlib.pyplot as plt
from typing import Any, Callable, Dict, List, Optional
try:
    import neurolab as nl
    _has_neurolab = True
except ImportError:
    print('??? Package neurolab not imported')
    _has_neurolab = False
    
try:
    from grayboxes.datatypes import Float2D, Function
except:
    try:
        from datatypes import Float2D, Function
    except:
        print('??? module datatypes not imported')
        print('    ==> copy file datatypes.py to this directory')
        print('    continue with unauthorized definition of Float2D, Function')
        Float2D = Optional[np.ndarray]
        Function = Optional[Callable[..., List[float]]]
        
# TODO .
#try:
#    from grayboxes.metrics import init_metrics
#except:
#    try:
#        from metrics import init_metrics
#    except:
#        print('??? module metrics not imported')
#        print('    ==> copy file metrics.py to this directory')
try:
    from grayboxes.neural0 import Neural0
except:
    try:
        from neural0 import Neural0
    except:
        print('??? module neural0 not imported')
        print('    ==> copy file neural0.py to this directory')
        

class Neural(Neural0):
    """
    - Wraps neural network implementations from
        - Neurolab: trains exclusively with backpropagation
        - NeuralGenetic: trains exclusively with genetic algorithm

    References:
        - Recommended training algorithms:
              'rprop': resilient backpropagation (NO REGULARIZATION)
                       wikipedia: 'Rprop'
              'bfgs':  Broyden–Fletcher–Goldfarb–Shanno algorithm,
                       see: scipy.optimize.fmin_bfgs() and wikipedia: 
                           'Broyden-Fletcher-Goldfarb-Shanno_algorithm'
        - http://neupy.com/docs/tutorials.html#tutorials
    """

    def __init__(self, f: Function = None) -> None:
        """
        Args:
            f:
                theor. submodel as method f(self, x) or function f(x)

        Note: if f is not None, genetic training or training with
            derivative dE/dy and dE/dw is employed
        """
        super().__init__(f)

        self._final_errors = []    # error (SSE, MSE) of best trial of 
                                   #   each training method
        self._final_L2_norms = []  # L2-norm of best trial of each train
        self._best_epochs = []     # epochs of best trial of each method

    def train(self, X: Float2D = None, Y: Float2D = None, 
              **kwargs: Any) -> Dict[str, Any]:
        """
        Trains model, stores X and Y as self.X and self.Y, and stores 
        metrics of best training trial as self.metrics

        Args:
            X:
                training input, shape: (n_point, n_inp)
                shape: (n_point,) is tolerated
                default: self.X

            Y:
                training target, shape: (n_point, n_out)
                shape: (n_point,) is tolerated
                default: self.Y

        Kwargs:
            see Neural0.train()
            
        Returns:
            metrics of best training trial:
                see Neural0.train()
        """
        default_activation = 'sigmoid'
        default_activation_out = 'sigmoid'
        
        self._scale_margin = 0.1

        if X is not None and Y is not None:
            self.set_XY(X, Y)
                                
        assert self._X is not None and self._Y is not None, \
            str(self.X) + ' ' + str(self.Y)

        transf = kwargs.get('activation', default_activation)
        if isinstance(transf, str):
            if transf.lower() in ('tansig', 'tanh', ):
                transf = nl.trans.TanSig
            elif transf.lower() in ('lin', 'linear', 'purelin', ):
                transf = nl.trans.PureLin
            elif transf.lower() in ('sigmoid', 'logsig', ):
                transf = nl.trans.LogSig
            else:
                transf = nl.trans.LogSig
        assert transf in (nl.trans.TanSig, nl.trans.LogSig, nl.trans.PureLin),\
            str(transf)

        epochs = kwargs.get('epochs', 1000)

        errorf = kwargs.get('errorf', nl.error.MSE())
        if isinstance(errorf, str):
            if errorf.lower() == 'mse':
                errorf = nl.error.MSE()
            elif errorf.lower() == 'sse':
                errorf = nl.error.SSE()
            elif errorf.lower() == 'sae':
                errorf = nl.error.SAE()
            elif errorf.lower() == 'mae':
                errorf = nl.error.MAE()
            else:
                errorf = nl.error.MSE()

        mse_expected = kwargs.get('expected', None)
        if mse_expected is None:
            mse_expected = kwargs.get('goal', 0.5e-3)
        mse_tolerated = kwargs.get('tolerated', 5e-3)

        neurons = kwargs.get('neurons', None)
        outputf = kwargs.get('output', default_activation_out)
        if isinstance(outputf, str):
            if outputf.lower() in ('tansig', 'tanh', ):
                outputf = nl.trans.TanSig
            elif outputf.lower() in ('lin', 'linear', 'purelin', ):
                outputf = nl.trans.PureLin
            elif outputf.lower() in ('logsig', 'sigmoid', ):
                outputf = nl.trans.LogSig
            else:
                outputf = nl.trans.TanSig
        assert outputf in (nl.trans.TanSig, nl.trans.LogSig, nl.trans.PureLin)

        plot = kwargs.get('plot', 1)
        rr = kwargs.get('regularization', None)
        if rr is None:
            rr = kwargs.get('rr', 1.)
        show = kwargs.get('show', 0)
        self.silent  = kwargs.get('silent', self.silent)
        if show:
            self.silent = False
        shuffle = kwargs.get('shuffle', True)
        smart_trials = kwargs.get('smart_trials', True)
        trainers = kwargs.get('trainer', ['bfgs', 'rprop'])
        trials = kwargs.get('trials', 3)

        if self.silent:
            show = 0
            plot = 0

        if errorf is None:
            errorf = nl.error.MSE()
        if show is None:
            show = epochs // 10
        if self.silent:
            plot = False

        if neurons is None:
            neurons = [1]
        if isinstance(neurons, (int, float)):
            neurons = list([int(neurons)])

        if not neurons or len(neurons) == 0 or not all(neurons):
            neurons = [10, 10]
        if not isinstance(neurons, list):
            neurons = list(neurons)

        assert all(nrn > 0 for nrn in np.array(neurons).reshape(-1))

        size = neurons.copy()
        size.append(self._Y.shape[1])
        assert size[-1] == self._Y.shape[1]

        trainer_pool = {
            'genetic':    nl.train.train_bfgs,      # TODO .
            'derivative': nl.train.train_bfgs,      # TODO .
            'bfgs':       nl.train.train_bfgs,
            'cg':         nl.train.train_cg,
            'gd':         nl.train.train_gd,
            'gda':        nl.train.train_gda,
            'gdm':        nl.train.train_gdm,
            'gdx':        nl.train.train_gdx,
            'rprop':      nl.train.train_rprop
            }

        default_trainers = ['rprop', 'bfgs']
        assert all([tr in trainer_pool for tr in default_trainers])
        
        if self.f is not None:
            # alternative training if theoretical submodel 'f'
            trainers = [x for x in trainers if x in ('genetic', 'derivative')]
            if not trainers:
                trainers = ['genetic']
        else:
            if not trainers:
                trainers = 'auto'
            if isinstance(trainers, str):
                if trainers.lower() == 'all':
                    trainers = ['cg', 'gd', 'gdx', 'gdm', 'gda', 'rprop', 
                                'bfgs', 'genetic']
                elif trainers.lower() == 'auto':
                    trainers = default_trainers
                else:
                    trainers = [trainers]
            trainers = list(OrderedDict.fromkeys(trainers))   # redundancy
        trainers = [trn.lower() for trn in trainers if trn in trainer_pool]
        if not trainers:
            trainers = default_trainers 

        if not self.silent:
            print('+++ trainers:', trainers)

        # shuffle argument of fit() is applied after split
        X_scaled = self._scale(self.X, self._X_stats, self._min_max_scale)
        Y_scaled = self._scale(self.Y, self._Y_stats, self._min_max_scale)
        if shuffle:
            X_scaled, Y_scaled = self._shuffle(X_scaled, Y_scaled)
        X_trn, Y_trn = X_scaled, Y_scaled

        self._ready = True # predict() returns None if self._ready is False

        self._net = None
        sequence_error = np.inf
        best_trainer = trainers[0]
        self._final_errors, self._final_L2norms, self._best_epochs = [], [], []

        for trainer in trainers:
            trainf = trainer_pool[trainer]
            trainer_err = np.inf
            trainer_epochs = None
            trainer_l2norm = np.inf

            minmax = [[_['min'], _['max']] for _ in self._X_stats]
                
            assert len(minmax) == len(self._X_stats), str((minmax, 
                                                           self._X_stats))
            net = nl.net.newff(minmax, size)
            net.transf = transf
            net.trainf = trainf
            net.errorf = errorf
            net.f = self.f
            
            if self.f is not None:
                net.outputf = nl.trans.PureLin
            else:
                net.outputf = outputf

            for j_trial in range(trials):
                if trainer in ('genetic', ):
                    net.init()
                    trial_errors = net.train(X_trn, Y_trn, f=self.f,
                        epochs=epochs, goal=mse_expected, rr=rr, show=show)
                elif trainer == 'rprop':
                    net.init()
                    trial_errors = net.train(X_trn, Y_trn, epochs=epochs,
                        show=show, goal=mse_expected)
                else:
                    for i_repeat in range(3):
                        del net
                        net = nl.net.newff(minmax, size)
                        net.init()
                        trial_errors = net.train(X_trn, Y_trn, 
                            epochs=epochs, show=show, goal=mse_expected, rr=rr)
                        if i_repeat > 0:
                            print('!!! Neural, L697, i_repeat:', i_repeat,
                                  'trainer:', trainer,'trial:', j_trial)
                        if len(trial_errors) >= 1:
                            break
                if len(trial_errors) < 1:
                    trial_errors.append(np.inf)
                
                if sequence_error > trial_errors[-1]:
                    sequence_error = trial_errors[-1]
                    del self._net
                    self._net = net.copy()
                if (trainer_err < mse_expected and \
                        trainer_epochs > len(trial_errors)) or \
                        (trainer_err >= mse_expected and \
                        trainer_err > trial_errors[-1]):
                    trainer_err = trial_errors[-1]
                    trainer_epochs = len(trial_errors)
                    trainer_l2norm = np.sqrt(np.mean(np.square(
                        self._predict_scaled(X_scaled) - Y_scaled)))
                if plot:
                    plt.plot(range(len(trial_errors)), trial_errors,
                             label='trial: ' + str(j_trial))
                if smart_trials:
                    if trial_errors[-1] < mse_expected:
                        break

            self._final_errors.append(trainer_err)
            self._final_L2norms.append(trainer_l2norm)
            self._best_epochs.append(trainer_epochs)
            i_best = trainers.index(best_trainer)
            if trainer_err < self._final_errors[i_best]:
                best_trainer = trainer

            if plot:
                plt.title("'" + trainer + "' mse:" +
                          str(round(trainer_err*1e3, 2)) + 'e-3 L2:' +
                          str(round(trainer_l2norm, 3)) +
                          ' [' + str(trainer_epochs) + ']')
                plt.xlabel('epochs')
                plt.ylabel('error')
                plt.yscale('log', nonposy='clip')
                plt.legend(bbox_to_anchor=(1.1, 1), loc='upper left')
                plt.grid()
                plt.show()
            if not self.silent:
                print('    ' + trainer + ':' + str(round(trainer_err, 5)) +
                      '[' + str(trainer_epochs) + '], ')

        if self._net is None:
            if not self.silent:
                print('??? All training trials failed')
            self._ready = False
            return self.metrics

        if plot:
            self._plot_train_vs_pred()

        i_best = trainers.index(best_trainer)
        if not self.silent:
            if len(trainers) > 1:
                print("    best trainer: '" + trainers[i_best] +
                      "' out of: [" + ' '.join(trainers) +
                      '], error:', round(self._final_errors[i_best], 5))
                if len(self._final_errors) > 1:
                    print("    (trainer:err): [", end='')
                    s = ''
                    for trn, err in zip(trainers, self._final_errors):
                        s += trn + ':' + str(round(err, 5)) + ' '
                    print(s[:-2] + ']')

        Y_prd_scaled = self._predict_scaled(X_trn)
        Y_prd = self._descale(Y_prd_scaled, self._Y_stats, self._min_max_scale)
       
        dy = Y_prd - self.Y
        
        i_abs_max = np.abs(dy).argmax()
        self._metrics = {'trainer': trainers[i_best],
                         'L2': np.sqrt(np.mean(np.square(dy))),
                         'abs': self.Y.ravel()[i_abs_max],
                         'i_abs': i_abs_max,
                         'epochs': self._best_epochs[i_best],
                         'ready': self.ready}

        return self.metrics

    def _predict_scaled(self, x_scaled: Float2D, **kwargs) -> Float2D:
        """
        Executes the network with scaled input
        
        Args:
            x_scaled:
                scaled input, shape: (n_point, self.X.shape[1])

        Kwargs:
            None

        Returns:
            scaled output, shape: (n_point, self.Y.shape[1])
        """
        return self._net.sim(x_scaled)
