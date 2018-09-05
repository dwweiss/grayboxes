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
      2018-09-05 DWW
"""

__all__ = ['plot1', 'plotCurves', 'plotSurface', 'plotWireframe',
           'plotIsoMap', 'plotIsolines', 'plotVector', 'plotTrajectory',
           'plotBarArrays', 'plotBars']

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import mlab, cm
from mpl_toolkits.mplot3d import Axes3D          # for "projection='3d'"
from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA
from typing import Iterable, List, Optional, Tuple, Union


def isIrregularMesh(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> bool:
    """
    Checks if all three arrays are 1D and of the same length

    Args:
        x:
            array of coordinates

        y:
            array of coordinates

        z:
            array of dependent variable

    Returns:
        True if all arrays are 1D and of same length

    """
    return x.ndim == 1 and y.ndim == 1 and z.ndim == 1 and \
        len(x) == len(y) and len(x) == len(z)


def isRegularMesh(x: np.ndarray, y: np.ndarray, z: np.ndarray,
                  dim: int=2) -> bool:
    """
    Checks if all three arrays are of dimension 'dim' and of same length

    Args:
        x:
            array of coordinate

        y:
            array of coordinate

        z:
            array of dependent variable

        dim:
            dimension of arrays

    Returns:
        True if all arrays are of dimension 'dim' and of same length
    """
    return all([arr.ndim == dim for arr in [x, y, z]]) and \
        all([arr.shape[i] == x.shape[i]
             for arr in [y, z] for i in range(dim)])


def toRegularMesh(x: np.ndarray, y: np.ndarray, z: np.ndarray,
                  nx: Optional[int]=50, ny: Optional[int]=None) \
        -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Maps irregular arrays to regular 2D arrays for x, y and z

    Args:
        x (1D or 2D array of float):
            array of coordinate

        y (1D or 2D array of float):
            array of coordinate

        z (1D or 2D array of float):
            irregular 1D or 2D-array of dependent variable,
             size: len(x)*len(y)

        nx:
            number of nodes along x-axis

        ny:
            number of nodes along y-axis

    Returns:
        x, y, z (3-tuple of array of float): coordinates and
            independent variable as regular 2D-arrays of same
            shape
    """
    x = np.asfarray(x)
    y = np.asfarray(y)
    z = np.asfarray(z)

    assert x.ndim == y.ndim and x.ndim < 3 and y.ndim < 3 and z.ndim < 3, \
        'incompatible arrays'

    if x.ndim == 1:
        if z.ndim == 2:
            assert x.size * y.size == z.size, 'incompatible arrays'
            X, Y = np.meshgrid(x, y)
            return X, Y, z
        else:
            assert x.size == y.size and x.size == z.size, \
              'incompatible arr x..z.shape:' + str((x.shape, y.shape, z.shape))
            if nx is None:
                nx = 32
            if ny is None:
                ny = nx
            X = np.linspace(x.min(), x.max(), nx)
            Y = np.linspace(y.min(), y.max(), ny)
            X, Y = np.meshgrid(X, Y)
            Z = mlab.griddata(x, y, z, X, Y, interp='linear')
            return X, Y, Z
    else:
        assert x.size == y.size and x.size == z.size, 'incompatible arrays'
        return x, y, z


def clip_xyz(x: np.ndarray,
             y: np.ndarray,
             z: np.ndarray,
             z2: Optional[np.ndarray]=None,
             xrange: Optional[Tuple[float, float]]=None,
             yrange: Optional[Tuple[float, float]]=None,
             zrange: Optional[Tuple[float, float]]=None) \
        -> Union[Tuple[np.ndarray, np.ndarray, np.ndarray],
                 Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """
    - Clips IRREGULAR arrays for x, y, z (1D arrays of same length)
    - Assigns array of size 2 to ranges if size is not 2
    - Clips x, y, z, z2 arrays according to x-, y- and z-ranges if lower and
      upper bound of a range is not equal

    Args:
        x (1D or 2D array of float):
            array of coordinates

        y (1D or 2D array of float):
            array of coordinates

        z (1D or 2D array of float):
            array of values over (x, y)

        z2 (1D or 2D array of float):
            second z-data as 1D-array

        xrange:
            ranges in x-direction

        yrange:
            ranges in y-direction

        zrange:
            ranges in z-direction

    Returns:
        (3-tuple of arrays of float):
            if z2 is None
        or
        (4-tuple of arrays of float):
            if z2 is not None
    """
    x = np.asfarray(x)
    y = np.asfarray(y)
    z = np.asfarray(z)

    if not isIrregularMesh(x, y, z):
        return x, y, z

    if not xrange:
        xrange = [None, None]
    if not yrange:
        yrange = [None, None]
    if not zrange:
        zrange = [None, None]
    if xrange[0] is None:
        xrange[0] = min(x)
    if xrange[1] is None:
        xrange[1] = max(x)
    if yrange[0] is None:
        yrange[0] = min(y)
    if yrange[1] is None:
        yrange[1] = max(y)
    if zrange[0] is None:
        zrange[0] = min(z)
    if zrange[1] is None:
        zrange[1] = max(z)
    hasXRange = xrange[0] != xrange[1]
    hasYRange = yrange[0] != yrange[1]
    hasZRange = zrange[0] != zrange[1]

    indices = []
    for i in range(len(z)):
        if not hasXRange or (xrange[0] <= x[i] <= xrange[1]):
            if not hasYRange or (yrange[0] <= y[i] <= yrange[1]):
                if not hasZRange or (zrange[0] <= z[i] <= zrange[1]):
                    indices.append(i)
    if z2 is None:
        return x[indices], y[indices], z[indices]
    else:
        return x[indices], y[indices], z[indices], z2[indices]


def plt_pre(xLabel: str='x', yLabel: str='y', title: str='',
            xLog: bool=False, yLog: bool=False, grid: bool=True,
            figsize: Optional[Tuple[float, float]]=None,
            fontsize: Optional[int]=None) \
        -> plt.figure:
    """
    Begin multiple plots
    """
    if figsize is None:
        figsize = (6, 3.5)
    if not fontsize:
        fontsize = round(0.4 * (figsize[0] * 3)) * 2
    plt.rcParams.update({'font.size': fontsize})
    plt.rcParams['legend.fontsize'] = fontsize

    fig = plt.figure(figsize=figsize)
    if xLabel:
        plt.xlabel(xLabel)
    if yLabel:
        plt.ylabel(yLabel)
    if xLog:
        plt.xscale('log', nonposy='clip')
    # else:
    #    plt.xscale('linear')
    if yLog:
        plt.yscale('log', nonposy='clip')
    # else:
    #    plt.yscale('linear')
    if title:
        plt.title(title, y=1.01)
    if grid:
        plt.grid()
    return fig


def plt_post(file: str='', legendPosition:
        Optional[Tuple[float, float]]=None) -> None:
    """
    Ends multiple plots
    """

    if legendPosition is None:
        legendPosition = (1.1, 1.05)
        plt.legend(bbox_to_anchor=legendPosition, loc='upper left')
    if file:
        if not file.endswith('.png'):
            file += '.png'
        f = file
        for c in "[](){}$?#!%&^*=+,': \\":
            if len(f) > 1:
                f1 = f[1]
                f = f.replace(c, '_')
                if f1 == ':':
                    f = f[:1] + ':' + f[2:]
            else:
                f = f.replace(c, '_')
        plt.savefig(f)
    plt.show()


def plot1(x: np.ndarray,
          y: np.ndarray,
          title: str = '',
          labels: Tuple[str, str]=('x', 'y'),
          xLog: bool=False, yLog: bool=False,
          grid: bool=True,
          figsize: Optional[Tuple[int, int]]=None,
          fontsize: Optional[int]=None,
          legendPosition: Optional[Tuple[float, float]]=None,
          file: str='') \
        -> None:
    plt_pre(xLabel=labels[0], yLabel=labels[1], title=title,
            xLog=xLog, yLog=yLog, grid=grid,
            figsize=figsize, fontsize=fontsize)
    plt.plot(x, y)
    plt_post(file, legendPosition=legendPosition)


def plotCurves(x,
               y1, y2=[], y3=[], y4=[],
               labels: Optional[List[str]]=None,  # axis labels
               title: str='',                     # title of plot
               styles: List[str]=['', '', ''],    # curve styles ('-:')
               marker: str='',                    # plot markers ('<>*+')
               linestyle: str='-',                # line style ['-','--',':'']
               units: Optional[List[str]]=None,   # axis units
               offsetAxis2: int=90,               # space to 1st right-hand axis
               offsetAxis3: int=180,              # space to 2sn right-hand axis
               xrange: Optional[Tuple[float, float]]=(0., 0.),
               y1range: Optional[Tuple[float, float]]=(0., 0.),
               y2range: Optional[Tuple[float, float]]=(0., 0.),
               y3range: Optional[Tuple[float, float]]=(0., 0.),
               y4range: Optional[Tuple[float, float]]=(0., 0.),
               xLog: bool=False,
               yLog: bool=False,
               grid: bool=False,
               figsize: Optional[Tuple[float, float]]=(6, 3.5),
               fontsize: int=14,
               legendPosition: Optional[Tuple[float, float]]=None,
                                        # dimensionless leg pos in (1,1)-space
               file: str='',                 # file name of image (no save if empty)
               ):
    plt.figure(figsize=figsize)
    plt.rcParams.update({'font.size': fontsize})
    plt.rcParams['legend.fontsize'] = fontsize

    if labels is None:
        labels = ['x', 'y1', 'y2', 'y3', 'y4']
    if units is None:
        units = ['[/]', '[/]', '[/]', '[/]', '[/]']
    if len(labels) > len(units):
        for i in range(len(units), len(labels)):
            units.append('?')
    if marker is None or marker not in ',ov^<>12348sp*hH+xDd|_':
        marker = ''
    if linestyle is None or linestyle not in ['', None, '-', '--', '-.', ':']:
        linestyle = ''

    par1 = host_subplot(111, axes_class=AA.Axes)
    if xrange[0] != xrange[1]:
        par1.set_xlim(xrange[0], xrange[1])
    if y1range[0] != y1range[1]:
        par1.ylim(y1range[0], y1range[1])
    if len(y2):
        par2 = par1.twinx()
        if y2range[0] != y2range[1]:
            par2.set_ylim(y2range[0], y2range[1])
    if len(y3):
        par3 = par1.twinx()
        if y3range[0] != y3range[1]:
            par3.set_ylim(y3range[0], y3range[1])
    if len(y4):
        par4 = par1.twinx()
        if y4range[0] != y4range[1]:
            par4.set_ylim(y4range[0], y4range[1])

    plt.subplots_adjust(right=0.75)
    if len(y2):
        new_fixed_axis = par2.get_grid_helper().new_fixed_axis
    if len(y3):
        par3.axis['right'] = new_fixed_axis(loc='right',
                                            axes=par3, offset=(offsetAxis2, 0))
        par3.axis['right'].toggle(all=True)
    if len(y4):
        par4.axis['right'] = new_fixed_axis(loc='right',
                                            axes=par4, offset=(offsetAxis3, 0))
        par4.axis['right'].toggle(all=True)

    for i in range(len(units)):
        if units[i] is None or units[i] == '':
            units[i] = '/'
        if not units[i].startswith('['):
            units[i] = '[' + units[i] + ']'
    par1.set_xlabel(labels[0] + ' ' + units[0])
    par1.set_ylabel(labels[1] + ' ' + units[1])
    if len(y2) and len(labels) > 2 and labels[2] != '[ ]':
        par2.set_ylabel(labels[2] + ' ' + units[2])
    if len(y3) and len(labels) > 3 and labels[3] != '[ ]':
        par3.set_ylabel(labels[3] + ' ' + units[3])
    if len(y4) and len(labels) > 4 and labels[4] != '[ ]':
        par4.set_ylabel(labels[4] + ' ' + units[4])
    p1, = par1.plot(x, y1, label=labels[1], marker=marker, linestyle=linestyle)
    if len(y2):
        label = ''
        if len(labels) > 2 and labels[2] != '[ ]':
            label = labels[2]
        p2, = par2.plot(x, y2, label=label, marker=marker, linestyle=linestyle)
    if len(y3):
        label = ''
        if len(labels) > 3 and labels[3] != '[ ]':
            label = labels[3]
        p3, = par3.plot(x, y3, label=label, marker=marker, linestyle=linestyle)
    if len(y4):
        label = ''
        if len(labels) > 4 and labels[4] != '[ ]':
            label = labels[4]
        p4, = par4.plot(x, y4, label=label, marker=marker, linestyle=linestyle)
    if figsize[0] <= 8:
            par1.legend(bbox_to_anchor=(1.5, 1), loc='upper left')
    else:
        if legendPosition is None:
            par1.legend()
        elif legendPosition[1] < 0.1:
            par1.legend(bbox_to_anchor=legendPosition, loc='lower right')
        elif legendPosition[1] > 0.9:
            par1.legend(bbox_to_anchor=legendPosition, loc='upper right')
        else:
            par1.legend(bbox_to_anchor=legendPosition, loc='center right')

    par1.axis['left'].label.set_color(p1.get_color())
    if len(y2):
        par2.axis['right'].label.set_color(p2.get_color())
    if len(y3):
        par3.axis['right'].label.set_color(p3.get_color())
    if len(y4):
        par4.axis['right'].label.set_color(p4.get_color())

    if legendPosition is None:
        xLegend = 1.05
        for yy in [y2, y3, y4]:
            if len(yy):
                xLegend += 0.30
        legendPosition = (xLegend, 1)
    plt.title(title)
    plt.grid(grid)
    plt_post(file=file, legendPosition=legendPosition)


def plotSurface(x: np.ndarray, y: np.ndarray, z: np.ndarray,
                labels: Tuple[str, str, str]=('x', 'y', 'z'),  # axis labels
                units: List[str]=['[/]', '[/]', '[/]'],  # axis units
                title: str='',
                xrange: Tuple[float, float]=(0., 0.),  # axis range
                yrange: Tuple[float, float]=(0., 0.),  # axis range
                zrange: Tuple[float, float]=(0., 0.),  # axis range
                xLog: bool=False,
                yLog: bool=False,
                grid: bool=False,
                figsize: Optional[Tuple[float, float]]=(8, 6),
                fontsize: int=10,
                legendPosition: Optional[Tuple[float, float]]=None,
                file='',
                ):
    """
    Plots u(x,y) data as colored 3D surface
    """
    if len(x.shape) == 1 and len(y.shape) == 1:
        if len(z.shape) == 2:
            x, y = np.meshgrid(x, y)
        else:
            x, y, z = toRegularMesh(x, y, z)

    fig = plt_pre(xLabel=labels[0], yLabel=labels[1], title=title,
                  xLog=False, yLog=False, grid=False, figsize=figsize,
                  fontsize=fontsize)

    for i in range(len(units)):
        if not units[i].startswith('['):
            units[i] = '(' + units[i] + ')'
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    if len(labels) > 0 and labels[0] != '[ ]':
        ax.set_xlabel(labels[0] + ' ' + units[0])
    if len(labels) > 1 and labels[1] != '[ ]':
        ax.set_ylabel(labels[1] + ' ' + units[1])
    if len(labels) > 2 and labels[2] != '[ ]':
        ax.set_zlabel(labels[2] + ' ' + units[2])
    surf = ax.plot_surface(x, y, z, rstride=1, cstride=1,
                           cmap=cm.coolwarm, linewidth=0, antialiased=False)
    # ax.set_zlim3d(-0.01 * Z.min(), 1.01)
    fig.colorbar(surf, shrink=0.5, aspect=10)

    plt_post(file=file, legendPosition=legendPosition)


def plotWireframe(x : np.ndarray, y : np.ndarray, z : np.ndarray,
                  labels: List[str]=['x', 'y', 'z'],
                  units: List[str]=['[/]', '[/]', '[/]'],
                  title: str='',
                  xrange: Optional[Tuple[float, float]]=(0., 0.),
                  yrange: Optional[Tuple[float, float]]=(0., 0.),
                  zrange: Optional[Tuple[float, float]]=(0., 0.),
                  xLog: bool=False,
                  yLog: bool=False,
                  grid: bool=False,
                  figsize: Tuple[float, float]=(7., 6.),
                  fontsize: int=14,
                  legendPosition: Optional[Tuple[float, float]]=None,
                  file: str='',
                  ):
    """
    Plots one z(x,y) array as 3D wiremesh surface
    """
    x2 = np.asarray(x)
    y2 = np.asarray(y)
    z2 = np.asarray(z)

    if all([arr.ndim == 1 for arr in [x, y, z]]):
        x2, y2, z2 = toRegularMesh(x, y, z)
    if zrange[0] != zrange[1]:
        x2, y2, z2 = clip_xyz(x2, y2, z2, zrange=zrange)
    """
    if len(x.shape) == 1 and len(y.shape) == 1:
        if len(z.shape) == 2:
            x, y = np.meshgrid(x, y)
        else:
            x, y, z = toRegularMesh(x, y, z)
    """
    fig = plt.figure(figsize=figsize)  # plt.figaspect(1.0) )
    for i in range(len(units)):
        if not units[i].startswith('['):
            units[i] = '[' + units[i] + ']'
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    if len(labels) > 0 and labels[0] != '[ ]':
        ax.set_xlabel(labels[0] + ' ' + units[0])
    if len(labels) > 1 and labels[1] != '[ ]':
        ax.set_ylabel(labels[1] + ' ' + units[1])
    if len(labels) > 2 and labels[2] != '[ ]':
        ax.set_zlabel(labels[2] + ' ' + units[2])

    ax.plot_wireframe(x2, y2, z2, rstride=2, cstride=2)
    if xrange[0] != xrange[1]:
        ax.set_xlim3d(xrange[0], xrange[1])
    if yrange[0] != yrange[1]:
        ax.set_ylim3d(yrange[0], yrange[1])

    plt_post(file=file, legendPosition=legendPosition)


def plotIsoMap(x, y, z,
               labels=['x', 'y', 'z'],       # axis labels
               units=['[/]', '[/]', '[/]'],  # axis units
               title='',                     # title of plot
               xrange=[0., 0.],
               yrange=[0., 0.],
               zrange=[0., 0.],
               xLog=False,        # logarithmic scaling of axis
               yLog=False,        # logarithmic scaling of axis
               levels=100,
               scatter=False,     # indicate irregular data with marker
               triangulation=False,
               figsize=None,      # figure size in inch
               fontsize=None,
               legendPosition='',
               cmap=None,    # color map
               file='',           # name of image file
               ):
    """
    Plots one z(x,y) array as 2D isomap
    """
    for i in range(len(units)):
        if not units[i].startswith('['):
            units[i] = '[' + units[i] + ']'
    if cmap is None:
        cmap = 'jet'   # 'rainbow'
    plt.set_cmap(cmap)

    if not title and len(labels) > 2:
        title = labels[2] + ' ' + units[2]
    fig = plt_pre(xLabel=labels[0] + ' ' + units[0],
                  yLabel=labels[1] + ' ' + units[1], xLog=xLog, yLog=yLog,
                  title=title, figsize=figsize, fontsize=fontsize)

    assert len(x) == len(y) and len(x) == len(z), 'size of x, y & z unequal' +\
        ' (' + str(len(x)) + ', ' + str(len(y)) + ', ' + str(len(z)) + ')'
    assert len(x) >= 3, 'size of x, y and z less 4'
    if zrange[0] != zrange[1]:
        zMin = min(z)
        zMax = max(z)
        x2, y2, z2 = clip_xyz(x, y, z,
                              zrange=[None, zMin + 1. * (zMax - zMin)])
        assert len(x2) == len(y2) and len(x2) == len(z2), 'not len x2==y2==z2'
        assert len(x2) >= 3, 'size of x2, y2 and z2 less 4'
    else:
        x2, y2, z2 = x.copy(), y.copy(), z.copy()
    if any(arr.ndim != 1 for arr in [x, y, z]):
        x2 = x2.ravel()
        y2 = y2.ravel()
        z2 = z2.ravel()
    # plt.tripcolor(x, y, z)
    plt.tricontourf(x2, y2, z2, levels)  # plot interpolation
    plt.colorbar()
    if scatter:
        plt.scatter(x, y, marker='o')
    # if xrange[0] != xrange[1]:
    #    plt.xlim(xrange[0], xrange[1])
    # if yrange[0] != yrange[1]:
    #    plt.ylim(yrange[0], yrange[1])
    plt_post(file='', legendPosition=None)


def plotIsolines(x, y, z,
                 labels=['x', 'y', 'z'],
                 units=['[/]', '[/]', '[/]'],
                 title='',
                 xrange=[0., 0.],
                 yrange=[0., 0.],
                 zrange=[0., 0.],
                 xLog=False,
                 yLog=False,
                 levels=None,  # 1D array_like of isolines
                 grid=False,
                 figsize=None,
                 fontsize=None,
                 legendPosition='',
                 file=''):
    """
    Plots u(x,y) data as 2D isolines
    """
    if xrange[0] is None:
        xrange[0] = min(x)
    if xrange[1] is None:
        xrange[1] = max(x)
    if yrange[0] is None:
        yrange[0] = min(y)
    if yrange[1] is None:
        yrange[1] = max(y)
    if zrange[0] is None:
        zrange[0] = min(z)
    if zrange[1] is None:
        zrange[1] = max(z)

    for i in range(len(units)):
        if not units[i].startswith('['):
            units[i] = '[' + units[i] + ']'
    fig = plt_pre(xLabel=labels[0], yLabel=labels[1], title='', 
                  xLog=xLog, yLog=yLog,
                  figsize=figsize, grid=grid, fontsize=fontsize, )

    if all([arr.ndim == 1 for arr in [x, y, z]]):
        x, y, z = toRegularMesh(x, y, z)
    if zrange[0] != zrange[1]:
        x, y, z = clip_xyz(x, y, z, zrange=zrange)

    if levels is not None:
        levels = np.atleast_1d(levels)
        CS = plt.contour(x, y, z, levels)
    else:
        CS = plt.contour(x, y, z)
    plt.clabel(CS, inline=1, fontsize=10)

    # ax = fig.add_subplot(1, 1, 1, projection='3d')
    if len(labels) > 0 and labels[0] != '[ ]':
        plt.xlabel(labels[0] + ' ' + units[0])
    if len(labels) > 1 and labels[1] != '[ ]':
        plt.ylabel(labels[1] + ' ' + units[1])
    if labels[2]:
        # title() setting must be done here, does not work with plt_pre()
        plt.title(labels[2] + ' ' + units[2])
    plt.show()


def plotVector(x, y,
               vx, vy,
               labels=['x', 'y', 'z', '', '', ''],
               units=['[/]', '[/]', '[/]'],
               title='',
               xrange=[0, 0], yrange=[0, 0],
               xLog=False, yLog=False,
               grid=False,
               figsize=None,
               fontsize=None,
               legendPosition='',
               file=''):
    for i in range(len(units)):
        if not units[i].startswith('['):
            units[i] = '[' + units[i] + ']'
    if len(labels) > 0 and labels[0] != '[ ]':
        plt.xlabel(labels[0] + ' ' + units[0])
    if len(labels) > 1 and labels[1] != '[ ]':
        plt.ylabel(labels[1] + ' ' + units[1])

    if figsize is None:
        figsize = (8, 6)
#    fig = plt.figure(figsize=figsize)
    if labels[0]:
        plt.xlabel(labels[0])
    if labels[1]:
        plt.ylabel(labels[1])
    if xLog:
        plt.xscale('log', nonposy='clip')
    # else:
    #    plt.xscale('linear')
    if yLog:
        plt.yscale('log', nonposy='clip')
    # else:
    #    plt.yscale('linear')
    if title:
        plt.title(title, y=1.01)
    if grid:
        plt.grid()

    if not fontsize:
        fontsize = round(0.5 * (figsize[0] * 3)) * 1.33
    plt.rcParams.update({'font.size': fontsize})
    plt.rcParams['legend.fontsize'] = fontsize

    plt.quiver(x, y, vx, vy, angles='xy', scale_units='xy')
    plt_post(file=file, legendPosition=legendPosition)


def plotTrajectory(x, y, z,                   # trajectory to be plotted
                   x2=[], y2=[], z2=[],       # second trajectory to be plotted
                   x3=[], y3=[], z3=[],       # third trajectory to be plotted
                   labels=['x', 'y', 'z', '', '', ''],
                   units=['[/]', '[/]', '[/]'],
                   title='',
                   xrange=[0., 0.],
                   yrange=[0., 0.],
                   zrange=[0., 0.],
                   yLog=False,
                   grid=False,
                   figsize=(10, 8),
                   fontsize=None,
                   legendPosition='',
                   file='',
                   startPoint=False,
                   ):
    """
    Plots up to three (x,y,z)  trajectories in 3D space
    """
    if xrange[0] is None:
        xrange[0] = min(x)
    if xrange[1] is None:
        xrange[1] = max(x)
    if yrange[0] is None:
        yrange[0] = min(y)
    if yrange[1] is None:
        yrange[1] = max(y)
    if zrange[0] is None:
        zrange[0] = min(z)
    if zrange[1] is None:
        zrange[1] = max(z)

    mpl.rcParams['legend.fontsize'] = 10
    fig = plt.figure(figsize=figsize)
    ax = fig.gca(projection='3d')
    if len(labels) > 0 and labels[0] != '[ ]':
        ax.set_xlabel(labels[0] + ' ' + units[0])
    if len(labels) > 1 and labels[1] != '[ ]':
        ax.set_ylabel(labels[1] + ' ' + units[1])
    if len(labels) > 2 and labels[2] != '[ ]':
        ax.set_zlabel(labels[2] + ' ' + units[2])

    if len(labels) > 3 and labels[3] != '':
        ax.plot(x, y, z, label=labels[3])
    else:
        ax.plot(x, y, z)
    if startPoint:
        if len(x) > 0:
            ax.scatter([x[0]], [y[0]], [z[0]], label='start')
        if len(x2) > 0:
            ax.scatter([x2[0]], [y2[0]], [z2[0]], label='start 2')
        if len(x3) > 0:
            ax.scatter([x3[0]], [y3[0]], [z3[0]], label='start 3')

    if len(labels) > 4 and labels[4] != '':
        ax.plot(x2, y2, z2, label=labels[4])
    else:
        ax.plot(x2, y2, z2)
    if len(labels) > 5 and labels[5] != '':
        ax.plot(x3, y3, z3, label=labels[5])
    else:
        ax.plot(x3, y3, z3)
    ax.legend()

    plt.show()


def plotBarArrays(x: Optional[np.ndarray]=None,
                  yArrays: Optional[np.ndarray]=None,
                  labels: Optional[List[str]]=None,
                  units: Optional[List[str]]=None,
                  title: str='',
                  yrange: Optional[Tuple[float, float]]=None,
                  grid: bool=False,
                  figsize: Optional[Tuple[float, float]]=None,
                  fontsize: int=14,
                  legendPosition: Optional[str]=None,
                  showYLabel: bool=True,
                  width: float=0.15,
                  file: str=''):
    """
    Plots bars without errorbars if yArrays is 2D array of float
    """
    assert yArrays is not None

    yArrays = np.atleast_2d(yArrays)
    ny = yArrays.shape[0]
    y1 = yArrays[0] if ny > 0 else []
    y2 = yArrays[1] if ny > 1 else []
    y3 = yArrays[2] if ny > 2 else []
    y4 = yArrays[3] if ny > 3 else []
    y5 = yArrays[4] if ny > 4 else []
    y6 = yArrays[5] if ny > 5 else []

    plotBars(x=x,
             y1=y1, y1Error=[],
             y2=y2, y2Error=[],
             y3=y3, y3Error=[],
             y4=y4, y4Error=[],
             y5=y5, y5Error=[],
             y6=y6, y6Error=[],
             labels=labels, units=units, title=title, yrange=yrange,
             grid=grid, figsize=figsize, fontsize=fontsize,
             legendPosition=legendPosition, showYLabel=showYLabel,
             width=width, file=file)


def plotBars(x=[],
             y1=[], y1Error=[],
             y2=[], y2Error=[],
             y3=[], y3Error=[],
             y4=[], y4Error=[],
             y5=[], y5Error=[],
             y6=[], y6Error=[],
             labels=[],
             units=[],
             title='',
             yrange=[],
             grid=False,
             figsize=None,
             fontsize=14,
             legendPosition=None,
             showYLabel=True,
             width=0.15,
             file=''):

    assert x is None or len(x) == 0 or len(x) == len(y1)
    x = [] if x is None else x
    yrange = [] if yrange is None else yrange
    if not figsize:
        figsize = (5, 3.5)
    plt.rcParams.update({'font.size': fontsize})
    plt.rcParams['legend.fontsize'] = fontsize
    ind = np.arange(len(y1))     # the x locations for the group

    fig, ax = plt.subplots(figsize=figsize)
    if len(yrange):
        ymin = yrange[0]
        ymax = yrange[1]
    else:
        ymin = min(y1)
        ymax = max(y1)
        if len(y2):
            ymin = min(ymin, min(y2))
            ymax = max(ymax, max(y2))
        if len(y3):
            ymin = min(ymin, min(y3))
            ymax = max(ymax, max(y3))
        if len(y4):
            ymin = min(ymin, min(y4))
            ymax = max(ymax, max(y4))
        if len(y5):
            ymin = min(ymin, min(y5))
            ymax = max(ymax, max(y5))
        if len(y6):
            ymin = min(ymin, min(y6))
            ymax = max(ymax, max(y6))
    ax.set_ylim([ymin, ymax * 1.05])

    if not labels or not units:
        ny = 1
        if len(y2):
            ny = 2
        if len(y3):
            ny = 3
        if len(y4):
            ny = 4
        if len(y5):
            ny = 5
        if len(y6):
            ny = 6
        if not labels:
            labels = ['x'] + ['y'+str(i) for i in range(ny)]      # axis labels
        if not units:
            units = ['[/]']*(ny + 1)                               # axis units

    if len(y1Error):
        rects1 = ax.bar(ind + 0*width, y1, width, color='b', yerr=y1Error)
    else:
        rects1 = ax.bar(ind + 0*width, y1, width, color='b')
    if len(y2):
        if len(y2Error):
            rects2 = ax.bar(ind + 1*width, y2, width, color='r', yerr=y2Error)
        else:
            rects2 = ax.bar(ind + 1*width, y2, width, color='r')
    else:
        rects2 = None
    if len(y3):
        if len(y3Error):
            rects3 = ax.bar(ind + 2*width, y3, width, color='y', yerr=y3Error)
        else:
            rects3 = ax.bar(ind + 2*width, y3, width, color='y')
    else:
        rects3 = None
    if len(y4):
        if len(y4Error):
            rects4 = ax.bar(ind + 3*width, y4, width, color='g', yerr=y3Error)
        else:
            rects4 = ax.bar(ind + 3*width, y4, width, color='g')
    else:
        rects4 = None
    if len(y5):
        if len(y5Error):
            rects5 = ax.bar(ind + 4*width, y5, width, color='b', yerr=y5Error)
        else:
            rects5 = ax.bar(ind + 4*width, y5, width, color='b')
    else:
        rects5 = None
    if len(y6):
        if len(y6Error):
            rects6 = ax.bar(ind + 5*width, y6, width, color='c', yerr=y6Error)
        else:
            rects6 = ax.bar(ind + 5*width, y6, width, color='c')
    else:
        rects6 = None

    plt.xlabel(labels[0])
    if showYLabel:
        la = ''
        for i in range(1, len(labels) - 1):
            la += labels[i] + ', '
        la += labels[-1]
        ax.set_ylabel(la)

    if title:
        ax.set_title(title)

    ax.set_xticks(ind + width)
    tics = []
    for i in range(len(y1)):
        if len(x):
            tics.append(str(x[i]))           # tics.append(labels[0] + str(i))
        else:
            tics.append(str(i))              # tics.append(labels[0] + str(i))
    ax.set_xticklabels(tics)

    r = [rects1[0]]
    la = [labels[1]]
    if len(y2):
        r.append(rects2[0])
        la.append(labels[2])
    if len(y3):
        r.append(rects3[0])
        la.append(labels[3])
    if len(y4):
        r.append(rects4[0])
        la.append(labels[4])
    if len(y5):
        r.append(rects5[0])
        la.append(labels[5])
    if len(y6):
        r.append(rects6[0])
        la.append(labels[6])
    ax.legend(r, la)
    if legendPosition:
        ax.legend(r, la, bbox_to_anchor=legendPosition, loc='upper left')
    else:
        if figsize[0] <= 8:
            ax.legend(r, la, bbox_to_anchor=(1, 1), loc='upper right')
        else:
            ax.legend(r, la)

    if 0:
        for rects in [rects1, rects2, rects3, rects4, rects5, rects6]:
            if rects:
                for r in rects:
                    height = r.get_height()
                    ax.text(r.get_x() + 1.0*r.get_width(), 1.0*height+0.02,
                            '%d' % int(height), ha='center', va='bottom')
    if grid:
        plt.grid()
    plt_post(file='', legendPosition=None)


def plot_X_Y_Yref(X: np.ndarray, Y: np.ndarray, Y_ref: np.ndarray,
                  labels:  List[str]=['X', 'Y', 'Y_{ref}']) -> None:
    """
    Plots Y(X). Y_ref(X) and the difference Y-Y_ref(X) as isoMap and surface
    """

    # plots Y
    plotIsoMap(X[:, 0], X[:, 1], Y[:, 0], title='$'+labels[1]+'$',
               labels=['$'+labels[0]+'_0$',
                       '$'+labels[0]+'_1$', '$'+labels[1]+'$'])
    plotSurface(X[:, 0], X[:, 1], Y[:, 0], title='$'+labels[1]+'$',
                labels=['$'+labels[0]+'_0$',
                        '$'+labels[0]+'_1$', '$'+labels[1]+'$'])

    # plots Y_ref
    plotIsoMap(X[:, 0], X[:, 1], Y_ref[:, 0], title='$'+labels[2]+'$',
               labels=['$'+labels[0]+'_0$',
                       '$'+labels[0]+'_1$', '$'+labels[2]+'$'])
    plotSurface(X[:, 0], X[:, 1], Y_ref[:, 0], title='$'+labels[2]+'$',
                labels=['$'+labels[0]+'_0$',
                        '$'+labels[0]+'_1$', '$'+labels[2]+'$'])

    # plots Y - Y_ref
    plotIsoMap(X[:, 0], X[:, 1], (Y - Y_ref)[:, 0],
               title='$'+labels[1]+' - '+labels[2]+'$',
               labels=['$'+labels[0]+'_0$', '$'+labels[0]+'_1$',
                       '$'+labels[1]+' - '+labels[2]+'$'])
    plotSurface(X[:, 0], X[:, 1], (Y - Y_ref)[:, 0],
                title='$'+labels[1]+' - '+labels[2]+'$',
                labels=['$'+labels[0]+'_0$', '$'+labels[0]+'_1$',
                        '$'+labels[1]+' - '+labels[2]+'$'])
