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
      2018-06-18 DWW
"""

import numpy as np
import matplotlib.pyplot as plt


def partlyRegular(nCells, *ranges, spacing='lin', dtype=np.float32,
                  silent=True):
    """
    Coordinates of vertices and cell centers for partly regular spaced grid
    in 1D, 2D and 3D space.
    The spacing can be linear, logarithmic and double-logarithmic

    - The boundary cells are void ghost cells with zero volume
    - The number of vertices and cell centers per axis equals (cell number + 1)


         -+-------+-------+-------+-------+-+
        |-|   -   |   -   |   -   |   -   |-| <-- ghost cells
         -+-------+-------+-------+-------+-+
        | |       |       |       |       | |
        |-|   o   |   o   |   o   |   o   |-|
        | |       |       |       |       | |
         -+-------+-------+-------+-------+-+                     vertices: '+'
        | |       |       |       |       | |         regular cell centers: 'o'
        |-|   o   |   o   |   o   |   o   |-|           ghost cell centers: '-'
        | |       |       |       |       | |
         -+-------+-------+-------+-------+-+
        | |       |       |       |       | |
        |-|   o   |   o   |   o   |   o   |-|
        | |       |       |       |       | |
         -+-------+-------+-------+-------+-+
        |-|   -   |   -   |   -   |   -   |-| <-- ghost cells
         - ------- ------- ------- ------- -
         ^                                 ^
         |                                 |
       ghost cells                       ghost cells


    Args:
        nCells (int or 1D array_like of int):
            number of cells per axis for which coordinates are generated.
            If nCells is single and negative, arrays will be transformed to 2D
                and transposed:
                    e.g. partlyRegular(nCells, [0, 1])
                    nCells: -5 ==> X1d:[[0], [.25], [.5], [.75], [1]]
                    nCells: +5 ==> X1d:[[0, .25, .5, .75, 1]]
            Currently the length of 'nCells' is limited to 3

        ranges (variable length argument list of pairs of float):
            list of (min, max) pairs of vertex node ranges

        spacing (str or 1D array_like of str, optional):
            spacing of axes: ('lin', 'log+', 'log-', 'log+-', 'log-+')

            with 'log-', grid densities are higher at the lower range limit.
            log+- and log-+ require the zero point between lower and upper
            limits of this range, e.g.
                ranges=(-3, +5), spacing='log+-')
            default: 'lin'

        dtype (np.float32 or np.float64, optional):
            floating point data type for coordinates, length, areas and volumes
            default: np.float32

        silent (bool, optional):
            test plot if True
            default: True

    Returns:
        (4-tuple of arrays of float with dimension: n+1):
            - list of vertex coordinates X
            - list center coordinates x
            - list of step sizes dx
            - list of cell volumes V
    """
    ranges, spacing = list(ranges), list(np.atleast_1d(spacing))
    N = list(np.atleast_1d(nCells))
    n = N + [N[-1]] * (len(ranges) - len(N))  # fill n-array up to: len(ranges)
    assert len(n) == len(ranges), str(n)+' '+str(nCells)+' '+str(ranges)
    ranges = np.asfarray(ranges)
    spacing = spacing + [spacing[-1]] * (len(ranges) - len(spacing))

    # vertex 'X' and center 'x' nodes as 1D arrays
    X1d, x1d, dx1d = [], [], []
    for rng, _n, _spc in zip(ranges, n, spacing):
        rngLo = min(rng[0], rng[1])
        rngUp = max(rng[0], rng[1])
        abs_n = abs(_n)

        def L(n, start=0, stop=1, base=10, reverse=False):
            """
            Normalized logarithmic spaced array (grid density decreases with
            index if reverse is False)
            """
            x = (np.logspace(start, stop, n, base=base, dtype=dtype) -
                 base**start) / (base**stop - base**start)
            if reverse:
                x = x[::-1]
            return x

        if _spc.lower() == 'lin':
            X_j = np.linspace(rngLo, rngUp, abs(_n), dtype=dtype)
        elif _spc.lower() == 'log-':
            X_j = rngLo + (rngUp-rngLo) * L(abs_n)
        elif _spc.lower() == 'log+':
            X_j = rngUp - (rngUp-rngLo) * (L(abs_n, reverse=True))
        elif _spc.lower() in ('log+-', 'log-+'):
            assert rngLo < 0 and rngUp > 0, str((rngLo, rngUp))

            nLo = max(1, round(abs_n * rngLo / (-rngUp + rngLo)))
            nUp = abs_n - nLo
            if _spc.lower() == 'log-+':
                lo = rngLo + (0-rngLo) * L(nLo)
                up = 0 + (rngUp-0) * (1 - L(nUp, reverse=True)[1:])
            else:
                lo = rngLo + (0-rngLo) * (1-L(nLo, reverse=True))
                up = 0 + (rngUp-0) * (L(nUp))
            X_j = np.append(lo, up)
        else:
            assert 0, str(_spc)

        X_j = np.append(X_j, X_j[-1])
        X1d.append(X_j)

        x_j = np.append(X_j[0], 0.5 * (X_j[:-1] + X_j[1:]))
        x1d.append(x_j)

        dx_j = np.append(np.array([0], dtype=dtype), (X_j[1:] - X_j[:-1]))
        dx1d.append(dx_j)

    if ranges.shape[0] == 1:
        X, x, dx = np.atleast_2d(X1d[0]), np.atleast_2d(x1d[0]), \
            np.atleast_2d(dx1d[0])
        if _n < 0:
            X, x, dx = X.T, x.T, dx.T
        V = dx

    elif ranges.shape[0] == 2:
        n0, n1 = x1d[0].size, x1d[1].size
        X = np.asarray([np.asarray([X1d[0]] * n1).T, [X1d[1]] * n0])
        x = np.asarray([np.asarray([x1d[0]] * n1).T, [x1d[1]] * n0])
        dx = np.asarray([np.asarray([dx1d[0]] * n1).T, [dx1d[1]] * n0])
        V = dx[0] * dx[1]

    elif ranges.shape[0] == 3:
        n0, n1, n2 = x1d[0].size, x1d[1].size, x1d[2].size
        X = np.asarray([np.asarray([[X1d[0]] * n1] * n2).T,
                       np.asarray([np.asarray([X1d[1]] * n0).T] * n2).T,
                       [[X1d[2]] * n1] * n0])
        x = np.asarray([np.asarray([[x1d[0]] * n1] * n2).T,
                       np.asarray([np.asarray([x1d[1]] * n0).T] * n2).T,
                       [[x1d[2]] * n1] * n0])
        dx = np.asarray([np.asarray([[dx1d[0]] * n1] * n2).T,
                        np.asarray([np.asarray([dx1d[1]] * n0).T] * n2).T,
                        [[dx1d[2]] * n1] * n0])
        V = np.asarray(dx[0] * dx[1] * dx[2])

    else:
        assert 0, 'ranges.shape[0] <= 3 ' + str(ranges.shape)

    assert all([_.dtype == dtype for _ in [x, dx, X, V]]), \
        str([_.dtype for _ in [x, dx, X, V]])

    if not silent:
        plt.title("1D coordinates: center 'x' vs neighbor vertices 'X'")
        plt.xlabel('index')
        plt.ylabel('x, X')
        colors = ('r', 'g', 'b')
        for i in range(np.asarray(x1d).shape[0]):
            plt.plot(x1d[i], colors[i]+'.-', label='x'+str(i))
            plt.plot(X1d[i], colors[i]+'v', label='X'+str(i))
            plt.plot(range(1, len(X1d[i])), X1d[i][:-1], colors[i]+'^',
                     label='X'+str(i)+'_prv')

        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.show()
        plt.title("Cell volume, shape: " + str(V.shape))
        plt.xlabel('index')
        plt.ylabel('V')
        plt.plot(V.ravel(), 'x')
        plt.show()

        if X.shape[0] < X.shape[1] and X.shape[0] == 1:
            plt.title(r"1D Center ($\bullet$) and vertices (+)")
            plt.xlabel('index')
            plt.ylabel('x1, X1')
            plt.plot(X[0].ravel(), '+', x[0].ravel(), '.')
            plt.show()

        if X.shape[0] < X.shape[1] and X.shape[0] > 1:
            plt.title(r"Center ($\bullet$) and vertices (+)")
            plt.xlabel('x0, X0')
            plt.ylabel('x1, X1')
            plt.plot(X[0].ravel(), X[1].ravel(), '+',
                     x[0].ravel(), x[1].ravel(), '.')
            plt.show()

        if X.shape[0] < X.shape[1] and X.shape[0] > 2:
            plt.title(r"Center ($\bullet$) and vertices (+)")
            plt.xlabel('x0, X0')
            plt.ylabel('x2, X2')
            plt.plot(X[0].ravel(), X[2].ravel(), '+',
                     x[0].ravel(), x[2].ravel(), '.')
            plt.legend()
            plt.show()

            plt.title(r"Center ($\bullet$) and vertices (+)")
            plt.xlabel('x1, X1')
            plt.ylabel('x2, X2')
            plt.plot(X[1].ravel(), X[2].ravel(), '+',
                     x[1].ravel(), x[2].ravel(), '.')
            plt.legend()
            plt.show()

    return X, x, dx, V


# Examples ####################################################################

if __name__ == '__main__':
    Ls = [
            [(-2, 3), (0, 3), (0, 3)],
            [(-2, 3), (0, 3)],
            [(-2, 3)]
            ]
    ns = [
            (14, 10, 8),
            (10, 10),
            (10)
            ]
    for L, n in zip(Ls, ns):
        X, x, dx, vol = partlyRegular(n, *L, spacing=('log+-', 'log-', 'lin'),
                                      silent=False)

        print('### X x dx vol:', X.shape, x.shape, dx.shape, vol.shape, )
