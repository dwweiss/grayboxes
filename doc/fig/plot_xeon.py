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
      2018-03-16 DWW
"""

import pandas as pd
from io import StringIO
import matplotlib.pyplot as plt
import numpy as np


def plotXeon5690_twoForLoops():
    """
    Plots performance of MPI on two Xeon X5690 3.5 GHz
    
    Task:
        Repeat 500 times: 
        
        def f(x, **kwargs):
            for i in range(10*1000):
                sum = 0
                for i in range(1000):
                  sum += 0.001  
            return [x[0] * 2, x[1]**2]
            
        The example file is stored as parallel.py
            
    Call:
        mpiexec -n 12 python3 parallel.py
            
    Hardware:    
        Intel(R) Xeon(R) CPU X5690  @ 3.47GHz
    $ lscpu
        Architecture:          x86_64
        CPU op-mode(s):        32-bit, 64-bit
        Byte Order:            Little Endian
        CPU(s):                24
        On-line CPU(s) list:   0-23
        Thread(s) per core:    2
        Core(s) per socket:    6
        Socket(s):             2
        NUMA node(s):          2
        Vendor ID:             GenuineIntel
        CPU family:            6
        Model:                 44
        Model name:            Intel(R) Xeon(R) CPU X5690 @ 3.47GHz
        Stepping:              2
    """    
    
    # results: wall clock time and user time measured versus process number
    csvSeparator = ','
    s = StringIO("""processes,real,user
        1,  257,  257 
        2,  137,  273 
        3,   92,  274
        4,   70,  281
        5,   57,  282  
        6,   48,  279
        7,   44,  286
        8,   37,  277
        9,   32,  278
        10,  30,  281
        11,  29,  286
        12,  25,  282
        13,  27,  315
        14,  30,  347
        15,  29,  336
        16,  28,  320
        17,  30,  422
        18,  26,  411
        19,  28,  451
        20,  26,  467
        21,  26,  488
        22,  25,  449
        23,  24,  523
        24,  25,  543
    """)
    df = pd.read_csv(s, sep=csvSeparator, comment='#')
    df.rename(columns=df.iloc[0])
    df = df.apply(pd.to_numeric, errors='coerce')
    print(df)

    plt.clf()
    plt.title('MPI on 2 Xeon X5690 (12 physical, 24 logical cores)')
    plt.plot(df['processes'], df['real'])
    plt.xlabel('Number of processes')
    plt.ylabel('Wall clock time [s]')
    plt.xticks(np.arange(df.shape[0]+1, step=2))
    plt.grid()
    plt.show()

    plt.title('User time vs. processes (12 physical cores)')
    plt.plot(df['processes'], df['user'])
    plt.xlabel('Number of processes')
    plt.ylabel('User time [s]')
    plt.xticks(np.arange(df.shape[0]+1, step=2))
    plt.grid()
    plt.show()


def plotXeon5690_ft07_navier_stokes_channel():
    """
    Plots performance of MPI on two Xeon X5690 3.5 GHz
    
    Task:
        Execute ft07_navier_stokes_channel.py from FEniCS tutorial [FEN04] with
        nx = 100 (space steps) and num_steps = 50 (time steps)

    Call:
        mpiexec -n 12 python3 ft07_navier_stokes_channel.py
        
    Hardware:    
        Intel(R) Xeon(R) CPU X5690  @ 3.47GHz
    $ lscpu
        Architecture:          x86_64
        CPU op-mode(s):        32-bit, 64-bit
        Byte Order:            Little Endian
        CPU(s):                24
        On-line CPU(s) list:   0-23
        Thread(s) per core:    2
        Core(s) per socket:    6
        Socket(s):             2
        NUMA node(s):          2
        Vendor ID:             GenuineIntel
        CPU family:            6
        Model:                 44
        Model name:            Intel(R) Xeon(R) CPU X5690 @ 3.47GHz
        Stepping:              2
    """    
    
    # results: wall clock time and user time measured versus process number
    csvSeparator = ','
    s = StringIO("""processes,real
        1, 210
        2, 153
        3, 121
        4, 114
        5, 110
        6, 122
        7, 97
        8, 99
        9, 99
        10, 102
        11, 108
        12, 98
        13, 125
        14, 144
        15, 144
        16, 136
        17, 136
        18, 126
        19, 145
        20, 178
        21, 156
        22, 160
        23, 173
        24, 167

    """)
    df = pd.read_csv(s, sep=csvSeparator, comment='#')
    df.rename(columns=df.iloc[0])
    df = df.apply(pd.to_numeric, errors='coerce')
    print(df)

    plt.clf()
    plt.title('MPI on 2 Xeon X5690 (12 physical, 24 logical cores)')
    plt.plot(df['processes'], df['real'])
    plt.xlabel('Number of processes')
    plt.ylabel('Wall clock time [s]')
    plt.xticks(np.arange(df.shape[0]+1, step=2))
    plt.grid()
    plt.show()


# Examples ###################################################################

if __name__ == '__main__':
    plotXeon5690_twoForLoops()
    plotXeon5690_ft07_navier_stokes_channel()