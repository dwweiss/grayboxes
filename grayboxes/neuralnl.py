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
      2021-02-24 DWW

  Acknowledgements:
      Neurolab is a contribution by E. Zuev (pypi.python.org/pypi/neurolab)
"""

__all__ = ['Neural']

import numpy as np
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple
try:
    import neurolab as nl
    _has_neurolab = True
except ImportError:
    print('??? Package neurolab not imported')
    _has_neurolab = False
    
try:
    from grayboxes.datatype import Float2D, Function
except:
    try:
        from datatype import Float2D, Function
    except:
        print('??? module datatype not imported')
        print('    ==> copy file datatype.py to this directory')
        print('    continue with unauthorized definition of Float2D, Function')
        
        Float2D = Optional[np.ndarray]
        Function = Optional[Callable[..., List[float]]]
try:
    from grayboxes.bruteforce import BruteForce
except:
    try:
        from bruteforce import BruteForce
    except:
        print('??? module bruteforce not imported')
        print('    ==> copy file bruteforce.py to this directory')
        

class Neural(BruteForce):
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
        self._backend = 'neurolab'
    
    def _get_trainers(self) -> Tuple[Dict[str, Any], List[str]]:
        """
        See super()._get_trainers()
        """
        trainer_pool = {
            'bfgs':       nl.train.train_bfgs,
            'cg':         nl.train.train_cg,
            'gd':         nl.train.train_gd,
            'gda':        nl.train.train_gda,
            'gdm':        nl.train.train_gdm,
            'gdx':        nl.train.train_gdx,
            'rprop':      nl.train.train_rprop
        }
        default_trainer = ['rprop', 'bfgs']
        
        return trainer_pool, default_trainer

    def _set_trainer(self, trainer: str, 
                     trainer_pool: Dict[str, Any],
                     net_metric_keys: Iterable[str],
                     **kwargs: Any) -> None:                                           
        """
        See super()._set_trainer()
        """
        self._net.trainf = trainer_pool[trainer]    

    def _create_net(self, n_inp: int, 
                    hiddens: Iterable[int], 
                    n_out: int, 
                    activation: str, 
                    output_activation: str,
                    X_stats: Iterable[Dict[str, float]]) -> Any:
        """
        See super()._create_net()
        """        
        minmax = [[x['min'], x['max']] for x in X_stats]
        assert len(minmax) == len(self._X_stats), str((minmax, 
                                                       self._X_stats))
        size = np.append(np.atleast_1d(hiddens), [n_out])
        net = nl.net.newff(minmax, size)

        net.errorf = nl.error.MSE()
        
        if isinstance(activation, str):
            if activation.lower() in ('tansig', 'tanh', ):
                activation = nl.trans.TanSig
            elif activation.lower() in ('lin', 'linear', 'purelin', ):
                activation = nl.trans.PureLin
            elif activation.lower() in ('sigmoid', 'logsig', ):
                activation = nl.trans.LogSig
            else:
                activation = nl.trans.LogSig
        assert activation in (nl.trans.TanSig, nl.trans.LogSig, 
                              nl.trans.PureLin), str(activation)
        net.trainf = activation
        
        if isinstance(output_activation, str):
            if output_activation.lower() in ('tansig', 'tanh', ):
                output_activation = nl.trans.TanSig
            elif output_activation.lower() in ('lin', 'linear', 'purelin', ):
                output_activation = nl.trans.PureLin
            elif output_activation.lower() in ('sigmoid', 'logsig', ):
                output_activation = nl.trans.LogSig
            else:
                output_activation = nl.trans.LogSig
        assert output_activation in (nl.trans.TanSig, nl.trans.LogSig, 
                              nl.trans.PureLin), str(output_activation)
        net.outputf = nl.trans.PureLin
        
        return net
      
    def _train_scaled(self, X: Float2D = None, Y: Float2D = None, 
                      **kwargs: Any) -> Dict[str, Any]:
        """
        see super().train()
        """
        epochs = kwargs.get('epochs', 300)
        mse_expected = kwargs.get('mse_expected', 1e-3)
        rr = kwargs.get('regularization', None)
        if rr is None:
            rr = kwargs.get('rr', 1.)
        show = kwargs.get('show', 0)
        if show is None:
            show = epochs // 10
        trainer = kwargs.get('trainer', 'rprop')
        validation_data = kwargs.get('validation_data', None)
        
        if trainer == 'rprop':
            hist = self._net.train(X, Y, epochs=epochs, show=show, 
                                   goal=mse_expected)
        else:
            for i in range(5+1):
                self._net.init()
                hist = self._net.train(X, Y, epochs=epochs, show=show, 
                                       goal=mse_expected, rr=rr)
                if len(hist):
                    break
                                
        mse_history = hist
        
        if validation_data is None:
            mse_val_last = np.nan
        else:
            x_val_scaled = self._scale(validation_data[0], self._X_stats, 
                                       self._min_max_scale)
            y_val = self._predict_scaled(x_val_scaled)
            mse_val_last = np.mean((y_val - validation_data[1])**2)
            
#            y_trn = self._predict_scaled(X)
#            mse_trn_last = np.mean((y_trn - Y)**2)
                       
#            print('mse_trn[-1]', mse_history[-1])
#            print('mse_trn_last', mse_trn_last)
#            print('mse_val_last', mse_val_last)
        val_mse_history = [np.nan] * (len(mse_history)-1) + [mse_val_last] 
        
        return {'mse': mse_history, 'val_mse': val_mse_history}

    def _predict_scaled(self, x_scaled: Float2D, **kwargs) -> Float2D:
        """
        See super()._predict_scaled()
        """
        return self._net.sim(x_scaled)
