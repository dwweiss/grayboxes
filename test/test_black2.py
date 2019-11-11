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
      2019-10-10 DWW
"""

import __init__
__init__.init_path()

import unittest
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from grayboxes.black import Black
from grayboxes.neural import Neural


class TestUM(unittest.TestCase):
    def setUp(self):
        print('///', os.path.basename(__file__))
        self.save_figures = True

    def tearDown(self):
        pass

    def test3(self):
        """
        Demonstrates the poor extrapolation capability of a multi-layer
        perceptron

        - Generates training and test daat by adding noise on 
          known white box model 
            sin(x) with noise
            
        -
        """
        
        path = Black().path

        print('*' * 30, 'path:', path)

        file = 'sin_x_-3..3.5pi'
        n_point = 20                 
        max_neurons_in_layer = 3     # <==== 20
        n_hidden_layers = 2          # <==== 5
        MAX_HIDDEN_LAYERS = 6        # MUST NOT be modified
        assert MAX_HIDDEN_LAYERS >= n_hidden_layers
        max_noise = 0.0

        # 1. compute training target (X, Y) and test data (x, y_ref)
        def f(x, *args):
            c0, c1 = args if len(args) > 0 else 1, 0
            return np.sin(x) * c0 + c1
        X = np.linspace(-2*np.pi, +2*np.pi, n_point)   # X of train data
        if 0:
            X = np.flipud(X)                        # reverse order of X
        x = np.linspace(-3*np.pi, 3*np.pi, 100)
        dx = (X.max() - X.min()) / n_point
        x = x + 0.5 * dx                          # shift x of test data

        Y = f(X)
        if max_noise > 0.0:
            Y += np.random.normal(-max_noise, +max_noise, X.size)
        X, Y = X.reshape(X.size, 1), Y.reshape(Y.size, 1)
        x = x.reshape(x.size, 1)
        y_ref = f(x)

        # 2. define 'collect' as DataFrame for result collection
        columns = ['n' + str(i+1) for i in range(MAX_HIDDEN_LAYERS)]
        columns.extend(['L2Train', 'absTrain', 'iAbsTrain',
                        'L2Test', 'absTest', 'iAbsTest',
                        'mse', 'trainer', 'epochs'])
        collect = pd.DataFrame(columns=columns)
        definition_max = [max_neurons_in_layer for _ in range(n_hidden_layers)]
        definition_max = definition_max + [0] * (MAX_HIDDEN_LAYERS -
                                                 n_hidden_layers)
        print('definition_max:', definition_max)

        # 3. definition of all network configurations 
        definitions = []
        for n1 in range(1, 1 + definition_max[0]):
            for n2 in range(0, 1 + min(n1, definition_max[1])):
                for n3 in range(0, 1 + min(n2, definition_max[2])):
                    for n4 in range(0, 1 + min(n3, definition_max[3])):
                        for n5 in range(0, 1 + min(n4, definition_max[4])):
                            for n6 in range(0, 1 + min(n5, definition_max[5])):
                                definitions.append([n1, n2, n3, n4, n5, n6])
        definitions = [np.trim_zeros(a) for a in definitions]
        print('+++ definitions:', definitions)

        # 4. loop over all network configurations
        l2_tst_best, i_def_best = 1.0, 0
        for i_def, definition in enumerate(definitions):
            definition_copy = definition.copy()
            print('+++ hidden layers:', definition)

            # training
            blk = Black()
            metrics_trn = blk(X=X, Y=Y, neurons=definition, trials=5, 
                              epochs=500, show=500, algorithms='bfgs', 
                              goal=1e-5)

            # prediction
            y = blk.predict(x=x)

            metrics_tst = blk.evaluate(x, y_ref, silent=False)
            if l2_tst_best > metrics_tst['L2']:
                l2_tst_best = metrics_tst['L2']
                i_def_best = i_def
            row = definition_copy.copy()
            row = row + [0]*(MAX_HIDDEN_LAYERS - len(row))
            row.extend([metrics_trn['L2'], metrics_trn['abs'], 
                        metrics_trn['iAbs'], metrics_tst['L2'], 
                        metrics_tst['abs'], metrics_tst['iAbs'],
                        0., 0., 0.  # mse training method epochs
                        ])
            # print('row:', row, len(row), 'columns:', collect.keys)
            collect.loc[collect.shape[0]] = row

            if isinstance(blk._empirical, Neural):
                print('+++ neural network definition:', definition)
            plt.title('$' + str(definition_copy) + '\ \ L_2(tr/te):\ ' +
                      str(round(metrics_trn['L2'], 5)) + r', ' +
                      str(round(metrics_tst['L2'], 4)) +
                      '$')
            plt.xlim(x.min() - 0.25, x.max() + 0.25)
            plt.ylim(-2, 2)
            plt.grid()
            plt.scatter(X, Y, marker='>', c='g', label='training data')
            plt.plot(x, y, linestyle='-', label='prediction')
            plt.plot(x, y_ref, linestyle=':', label='analytical')
            plt.scatter([X[metrics_trn['iAbs']]], [Y[metrics_trn['iAbs']]], 
                        marker='o', color='c', s=60, label='max err train')
            plt.scatter([x[metrics_tst['iAbs']]], [y[metrics_tst['iAbs']]], 
                        marker='o', color='r', s=60, label='max err test')
            plt.legend(bbox_to_anchor=(1.15, 0), loc='lower left')
            if self.save_figures:
                f = file
                for s in definition_copy:
                    f += '_' + str(s)
                plt.savefig(os.path.join(path, f + '.png'))
                
            plt.show()

            print('+++ optimum: definition: ', definitions[i_def_best],
                  ' index: [', i_def_best, '], L2: ',
                  round(l2_tst_best, 5), sep='')
            if i_def % 10 == 0 or i_def == len(definitions) - 1:
                print('+++ collect:\n', collect)
                collect.to_csv(os.path.join(path, file + '.collect.csv'))
                collect.to_pickle(os.path.join(path, file + '.collect.pkl'))
                collect.plot(y='L2Test', use_index=True)

        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
