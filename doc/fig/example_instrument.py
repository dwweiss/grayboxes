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
      2020-01-29 DWW
"""
import initialize
initialize.set_path()

from io import StringIO
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from grayboxes.black import Black
from grayboxes.plot import plot_isomap, plot_wireframe


# anonymized data of an observation: A = m_dot_obs - m_dot = F(m_dot, p)
# E in [%], m_dot and m_dot_obs normalized with min/max of both arrays
raw = StringIO("""m_dot,p,E,A,m_dot_obs
    0.003393,  0.000,    NaN,  0.000154,  0.003547
    0.597247,  0.054, -0.785, -0.004662,  0.592586
    0.858215,  0.054, -0.334, -0.002855,  0.855360
    0.901367,  0.262, -0.621, -0.005576,  0.895790
    0.893147,  0.516, -0.857, -0.007625,  0.885522 ## outlier (regular)
    0.884928,  0.771, -0.879, -0.007749,  0.877179
    0.849995,  0.931, -0.865, -0.007323,  0.842672
    0.003393,  0.000,    NaN, -0.003391,  0.000002
    0.862324,  0.054, -0.687, -0.005901,  0.856423
    0.525327,  0.250, -0.962, -0.005021,  0.520306 ## outlier (extra)
    1.000000,  0.260, -0.616, -0.006139,  0.993861
    0.003393,  0.056,    NaN,  0.000616,  0.004009
    0.765746,  0.056, -0.249, -0.001898,  0.763848
    0.003393,  0.261,    NaN, -0.000411,  0.002982
    0.843831,  0.261, -0.471, -0.003958,  0.839872 ## outlier for 2D
    0.003393,  0.000,    NaN, -0.003156,  0.000236
    0.003393,  0.000,    NaN, -0.003386,  0.000006
    0.003393,  0.100,    NaN, -0.002885,  0.000508
    0.003393,  0.100,    NaN, -0.003319,  0.000074
    0.003393,  0.250,    NaN, -0.003393,  0.000000
    0.003393,  0.270,    NaN, -0.002817,  0.000575
    0.003393,  0.260,    NaN, -0.002860,  0.000532
    0.003393,  0.260,    NaN, -0.002922,  0.000471
    0.003393,  0.500,    NaN, -0.002774,  0.000619
    0.003393,  0.770,    NaN, -0.002710,  0.000682
    0.003393,  1.000,    NaN, -0.002770,  0.000623
    0.003393,  1.000,    NaN, -0.002688,  0.000705
    0.003393,  1.000,    NaN, -0.002686,  0.000707
""")


if __name__ == '__main__':
    s = 'Error compensation with black box model: Y(X) = A(m_dot, p)'
    print('-' * len(s) + '\n' + s + '\n' + '-' * len(s))

    plotWireframes = False
    plot_isomaps = True

    df = pd.read_csv(raw, sep=',', comment='#')
    df.rename(columns=df.iloc[0])
    df = df.apply(pd.to_numeric, errors='coerce')
    X = np.asfarray(df.loc[:, ['m_dot', 'p']])
    Y = np.asfarray(df.loc[:, ['A']])
    y_diff = round(Y[:, 0].max() - Y[:, 0].min(), 5)

    plot_isomap(X[:, 0], X[:, 1], Y[:, 0]*1e3, title=r'$A_{obs}\cdot 10^3'+
                r'\ \ (\Delta A$: ' + str(y_diff*1e3) + 'e-3)',
                labels=[r'$\dot m$', '$p$'])
    plot_wireframe(X[:, 0], X[:, 1], Y[:, 0]*1e3,
                   title=r'$A_{obs}\cdot 10^3$',
                   labels=[r'$\dot m$', '$p$'])

    phi = Black()
    dy_diff_all = [], []
    
    backend = 'keras'
    backend = 'neurolab'
    neurons = [[2*j]*i for j in range(1, 2+1) for i in range(1, 2+1)]
#    for neurons in hidden_neurons:
    for i in range(1 if backend == 'keras' else len(neurons)):
#        str_neurons = ':' + str(neurons).replace(', ', ':')
#        str_neurons = str_neurons[1:max(-3, len(str_neurons))]
#        print('+++ hidden:', neurons)

        y = phi(X=X, Y=Y, x=X,
                activation='sigmoid',
                epochs=250,
                expected=0.5e-3,
                backend=backend,
                neurons=neurons if backend == 'keras' else neurons[i],
                output='sigmoid',
                patience=25,
                plot=1,
                silent=0, 
                tolerated=20e-3,
                trainer=('adam', 'sgd', ),
                trials=3,
                validation_split=0.20,
                )
        
        print('mtc', phi.metrics)

        if backend != 'keras':
            str_neurons = str(neurons[i]) + ', '
        else:
            str_neurons = ''
        if phi.ready:  
            dy_diff_all[0].append(str_neurons)
            dy_diff_all[1].append((Y - y).ptp())

            if plot_isomaps:
                plot_isomap(X[:, 0], X[:, 1], Y[:, 0] * 1e3, 
                            title=r'$A_{obs} \cdot 10^3$, ' + 
                            str_neurons +
                            ', span: ' + str(round(Y.ptp() * 1e3, 3)) + 'e-3',
                            labels=[r'$\dot m$', '$p$'],
                            fontsize=10)
                plot_isomap(X[:, 0], X[:, 1], y[:, 0] * 1e3, 
                            title=r'$A_{prd} \cdot 10^3$, ' + 
                            str_neurons + 
                            ', span: ' + str(round(y.ptp() * 1e3, 3)) + 'e-3', 
                            labels=[r'$\dot m$', '$p$'],
                            fontsize=10)
                plot_isomap(X[:, 0], X[:, 1], (y - Y)[:, 0] * 1e3,
                            title=r'$(A_{prd}-A_{obs})\cdot 10^3$, '+ 
                            str_neurons + 
                            ', span: '+str(round((Y - y).ptp()*1e3, 3))+'e-3',
                            labels=[r'$\dot m$', '$p$'],
                            fontsize=10)
            if plotWireframes:
                plot_wireframe(X[:, 0], X[:, 1], Y[:, 0] * 1e3,
                               title=r'$A_{obs}\cdot 10^3$')
                plot_wireframe(X[:, 0], X[:, 1], y[:, 0] * 1e3,
                               title=r'$A_{prd}\cdot 10^3$')
                plot_wireframe(X[:, 0], X[:, 1], (y - Y)[:, 0] * 1e3,
                               title=r'$(A_{prd} - A_{obs})\cdot 10^3$')

    if len(dy_diff_all[0]) > 1:
        plt.rcParams.update({'font.size': 10})
        plt.title('Compensation error vs hidden neurons')
        plt.xlabel('hidden neurons [/]')
        plt.ylabel(r'span $\Delta A \cdot 10^3$ [/]')
        plt.plot([np.sum(nrn) for nrn in dy_diff_all[0]], 
                 np.asarray(dy_diff_all[1]) * 1e3, 'o')
        plt.grid()
        plt.show()
