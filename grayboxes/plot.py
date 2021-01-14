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
      2020-11-25 DWW
"""

__all__ = ['plot1', 'plot_curves', 'plot_surface', 'plot_wireframe',
           'plot_isomap', 'plot_isolines', 'plot_vectors', 'plot_trajectory',
           'plot_2bars', 'plot_bar_arrays', 'plot_bars']

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D          # for "projection='3d'"
from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA
from scipy.interpolate import griddata 
from typing import Iterable, List, Optional,Tuple, Union


def is_irregular_mesh(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> bool:
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


def is_regular_mesh(x: np.ndarray, y: np.ndarray, z: np.ndarray,
                    dim: int = 2) -> bool:
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


def to_regular_mesh(x: np.ndarray, 
                    y: np.ndarray, 
                    z: np.ndarray,
                    nx: Optional[int] = 50, 
                    ny: Optional[int] = None) \
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
            size: len(x)*len(y)==len(z) or len(x)==len(y)==len(z)

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
            
            Z = griddata((x, y), z, (X, Y))
            
            return X, Y, Z
    else:
        assert x.size == y.size and x.size == z.size, 'incompatible arrays'
        return x, y, z


def clip_xyz(x: np.ndarray,
             y: np.ndarray,
             z: np.ndarray,
             z2: Optional[np.ndarray] = None,
             xrange: Optional[Tuple[Optional[float], Optional[float]]] = None,
             yrange: Optional[Tuple[Optional[float], Optional[float]]] = None,
             zrange: Optional[Tuple[Optional[float], Optional[float]]] = None)\
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
            optional second z-data as 1D-array

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

    if not is_irregular_mesh(x, y, z):
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
    has_x_range = xrange[0] != xrange[1]
    has_y_range = yrange[0] != yrange[1]
    has_z_range = zrange[0] != zrange[1]

    indices = []
    for i in range(len(z)):
        if not has_x_range or (xrange[0] <= x[i] <= xrange[1]):
            if not has_y_range or (yrange[0] <= y[i] <= yrange[1]):
                if not has_z_range or (zrange[0] <= z[i] <= zrange[1]):
                    indices.append(i)
    if z2 is None:
        return x[indices], y[indices], z[indices]
    else:
        return x[indices], y[indices], z[indices], z2[indices]


def _plt_pre(xlabel: str = 'x',
             ylabel: str = 'y',
             title: str ='',
             xlog: bool = False,
             ylog: bool = False,
             grid: bool = True,
             figsize: Optional[Tuple[float, float]] = None,
             fontsize: Optional[int] = None) \
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
    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)
    if xlog:
        plt.xscale('log', nonposy='clip')
    # else:
    #    plt.xscale('linear')
    if ylog:
        plt.yscale('log', nonposy='clip')
    # else:
    #    plt.yscale('linear')
    if title:
        plt.title(title, y=1.01)
    if grid:
        plt.grid()
    return fig


def _plt_post(file: str='',
             legend_position: Optional[Tuple[float, float]] = None) -> None:
    """
    Ends multiple plots
    """

    if legend_position is None:
        legend_position = (1.1, 1.05)
    plt.legend(bbox_to_anchor=legend_position, loc='upper left')
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
          labels: Optional[Tuple[str, str]]=None,
          xlog: bool = False,
          ylog: bool = False,
          grid: bool = True,
          figsize: Optional[Tuple[float, float]] = None,
          fontsize: Optional[int] = None,
          legend_position: Optional[Tuple[float, float]] = None,
          file: str = '') \
        -> None:
    if not labels:
        labels = ('x', 'y')
    _plt_pre(xlabel=labels[0], ylabel=labels[1], title=title,
            xlog=xlog, ylog=ylog, grid=grid,
            figsize=figsize, fontsize=fontsize)
    plt.plot(x, y)
    _plt_post(file, legend_position=legend_position)


def plot_curves(x: np.ndarray,
                y1: np.ndarray,
                y2: Optional[np.ndarray] = None,
                y3: Optional[np.ndarray] = None,
                y4: Optional[np.ndarray] = None,
                labels: Optional[List[str]] = None,        # axis labels
                title: str = '',                         # title of plot
                styles: Optional[Tuple[str, str, str]] = None, 
                                                     # curve style('-:')
                marker: str = '',                # plot markers ('<>*+')
                linestyle: str = '-',       # line style ['-','--',':'']
                units: Optional[Tuple[str, str, str]] = None,  
                                                            # axis units
                offset_axis2: int = 90,    # space to 1st right-hd. axis
                offset_axis3: int = 180,   # space to 2nd right-hd. axis
                xrange: Optional[Tuple[float, float]] = None,
                y1range: Optional[Tuple[float, float]] = None,
                y2range: Optional[Tuple[float, float]] = None,
                y3range: Optional[Tuple[float, float]] = None,
                y4range: Optional[Tuple[float, float]] = None,
                xlog: bool = False,
                ylog: bool = False,
                grid: bool = False,
                figsize: Optional[Tuple[float, float]] = (6, 3.5),
                fontsize: Optional[int] = 14,
                legend_position: Optional[Tuple[float, float]] = None,
                       # dimensionless legend position in (1., 1.)-space
                file: str = '',  # file name of image (no save if empty)
                ) -> None:
    x, y1 = np.asfarray(x), np.asfarray(y1)
    
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
    if not fontsize:
        fontsize = 14
    if not styles:
        styles = ('', '', '')
    if not xrange:
        xrange = (0., 0.)
    if not y1range:
        y1range = (0., 0.)
    if not y2range:
        y2range = (0., 0.)
    if not y3range:
        y3range = (0., 0.)
    if not y4range:
        y4range = (0., 0.)

    par1 = host_subplot(111, axes_class=AA.Axes)
    if xrange[0] != xrange[1]:
        par1.set_xlim(xrange[0], xrange[1])
    if y1range[0] != y1range[1]:
        par1.ylim(y1range[0], y1range[1])
    if y2 is not None and len(y2):
        par2 = par1.twinx()
        if y2range[0] != y2range[1]:
            par2.set_ylim(y2range[0], y2range[1])
    if y3 is not None and len(y3):
        par3 = par1.twinx()
        if y3range[0] != y3range[1]:
            par3.set_ylim(y3range[0], y3range[1])
    if y4 is not None and len(y4):
        par4 = par1.twinx()
        if y4range[0] != y4range[1]:
            par4.set_ylim(y4range[0], y4range[1])

    plt.subplots_adjust(right=0.75)
    if y2 is not None and len(y2):
        new_fixed_axis = par2.get_grid_helper().new_fixed_axis
    if y3 is not None and len(y3):
        par3.axis['right'] = new_fixed_axis(loc='right',
                                            axes=par3, offset=(offset_axis2, 0))
        par3.axis['right'].toggle(all=True)
    if y4 is not None and len(y4):
        par4.axis['right'] = new_fixed_axis(loc='right',
                                            axes=par4, offset=(offset_axis3, 0))
        par4.axis['right'].toggle(all=True)

    for i in range(len(units)):
        if units[i] is None or units[i] == '':
            units[i] = '/'
        if not units[i].startswith('['):
            units[i] = '[' + units[i] + ']'
    par1.set_xlabel(labels[0] + ' ' + units[0])
    par1.set_ylabel(labels[1] + ' ' + units[1])
    if y2 is not None and len(y2) and len(labels) > 2 and labels[2] != '[ ]':
        par2.set_ylabel(labels[2] + ' ' + units[2])
    if y3 is not None and len(y3) and len(labels) > 3 and labels[3] != '[ ]':
        par3.set_ylabel(labels[3] + ' ' + units[3])
    if y4 is not None and len(y4) and len(labels) > 4 and labels[4] != '[ ]':
        par4.set_ylabel(labels[4] + ' ' + units[4])
    p1, = par1.plot(x, y1, label=labels[1], marker=marker, linestyle=linestyle)
    if y2 is not None and len(y2):
        label = ''
        if len(labels) > 2 and labels[2] != '[ ]':
            label = labels[2]
        p2, = par2.plot(x, y2, label=label, marker=marker, linestyle=linestyle)
    if y3 is not None and len(y3):
        label = ''
        if len(labels) > 3 and labels[3] != '[ ]':
            label = labels[3]
        p3, = par3.plot(x, y3, label=label, marker=marker, linestyle=linestyle)
    if y4 is not None and len(y4):
        label = ''
        if len(labels) > 4 and labels[4] != '[ ]':
            label = labels[4]
        p4, = par4.plot(x, y4, label=label, marker=marker, linestyle=linestyle)
    if figsize[0] <= 8:
            par1.legend(bbox_to_anchor=(1.5, 1), loc='upper left')
    else:
        if legend_position is None:
            par1.legend()
        elif legend_position[1] < 0.1:
            par1.legend(bbox_to_anchor=legend_position, loc='lower right')
        elif legend_position[1] > 0.9:
            par1.legend(bbox_to_anchor=legend_position, loc='upper right')
        else:
            par1.legend(bbox_to_anchor=legend_position, loc='center right')

    par1.axis['left'].label.set_color(p1.get_color())
    if y2 is not None and len(y2):
        par2.axis['right'].label.set_color(p2.get_color())
    if y3 is not None and len(y3):
        par3.axis['right'].label.set_color(p3.get_color())
    if y4 is not None and len(y4):
        par4.axis['right'].label.set_color(p4.get_color())

    if legend_position is None:
        x_legend = 1.05
        for yy in [y2, y3, y4]:
            if yy is not None and len(yy):
                x_legend += 0.30
        legend_position = (x_legend, 1)
    plt.title(title)
    plt.grid(grid)
    _plt_post(file=file, legend_position=legend_position)


def plot_surface(x: np.ndarray,
                 y: np.ndarray,
                 z: np.ndarray,
                 labels: Optional[Tuple[str, str, str]] = None,  # axis labels
                 units: Optional[Tuple[str, str, str]] = None,  # ax.unit
                 title: str = '',
                 xrange: Optional[Tuple[float, float]] = None,  # axis range
                 yrange: Optional[Tuple[float, float]] = None,  # axis range
                 zrange: Optional[Tuple[float, float]] = None,  # axis range
                 xlog: bool = False,
                 ylog: bool = False,
                 grid: bool = False,
                 figsize: Optional[Tuple[float, float]] = None,
                 fontsize: Optional[int] = None,
                 legend_position: Optional[Tuple[float, float]] = None,
                 file: str = '',
                 ) -> None:
    """
    Plots u(x,y) data as colored 3D surface
    """
    x, y, z = np.asfarray(x), np.asfarray(y), np.asfarray(z) 

    if len(x.shape) == 1 and len(y.shape) == 1:
        if len(z.shape) == 2:
            x, y = np.meshgrid(x, y)
        else:
            x, y, z = to_regular_mesh(x, y, z)

    if not labels:
        labels = ('x', 'y', 'z')
    if not units:
        units = ('[/]', '[/]', '[/]')
    units = list(units)
    if not figsize:
        figsize = (8, 6)
    if not fontsize:
        fontsize = 10
    if xrange is None:
        xrange = (0., 0.)
    if yrange is None:
        yrange = (0., 0.)
    if zrange is None:
        zrange = (0., 0.)

    fig = _plt_pre(xlabel=labels[0], ylabel=labels[1], title=title,
                   xlog=False, ylog=False, grid=False, figsize=figsize,
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


def plot_wireframe(x: np.ndarray,
                   y: np.ndarray,
                   z: np.ndarray,
                   labels: Optional[Tuple[str, str, str]] = None,
                   units: Optional[Tuple[str, str, str]] = None,
                   title: str = '',
                   xrange: Optional[Tuple[float, float]] = None,
                   yrange: Optional[Tuple[float, float]] = None,
                   zrange: Optional[Tuple[float, float]] = None,
                   xlog: bool = False,
                   ylog: bool = False,
                   grid: bool = False,
                   figsize: Optional[Tuple[float, float]] = None,
                   fontsize: Optional[int] = None,
                   legend_position: Optional[Tuple[float, float]] = None,
                   file: str = '',
                   ) -> None:
    """
    Plots one z(x,y) array as 3D wiremesh surface
    """
    x, y, z = np.asfarray(x), np.asfarray(y), np.asfarray(z)
    x2, y2, z2 = np.asfarray(x), np.asfarray(y), np.asfarray(z)

    if not xrange:
        xrange = (0., 0.)
    if not yrange:
        yrange = (0., 0.)
    if not zrange:
        zrange = (0., 0.)
    if not figsize:
        figsize = (7, 6)
    if not fontsize:
        fontsize = 14
    if not labels:
        labels = ('x', 'y', 'z')
    if not units:
        units = ('[/]', '[/]', '[/]')
    units = list(units)

    if all([arr.ndim == 1 for arr in [x, y, z]]):
        x2, y2, z2 = to_regular_mesh(x, y, z)
    if zrange[0] != zrange[1]:
        x2, y2, z2 = clip_xyz(x2, y2, z2, zrange=zrange)
    """
    if len(x.shape) == 1 and len(y.shape) == 1:
        if len(z.shape) == 2:
            x, y = np.meshgrid(x, y)
        else:
            x, y, z = to_regular_mesh(x, y, z)
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


def plot_isomap(x: np.ndarray,
                y: np.ndarray,
                z: np.ndarray,
                labels: Optional[Iterable[str]] = None,  # axis labels
                units: Optional[Iterable[str]] = None,  # axis units
                title: str = '',  # title of plot
                xrange: Optional[Tuple[float, float]] = None,
                yrange: Optional[Tuple[float, float]] = None,
                zrange: Optional[Tuple[float, float]] = None,
                xlog: bool = False,  # logarithmic scaling of axis
                ylog: bool = False,  # logarithmic scaling of axis
                levels: int = 100,
                scatter: bool = False,  # indicate irregular data with marker
                triangulation: bool = False,
                figsize: Optional[Tuple[float, float]] = None,
                fontsize: Optional[int] = None,
                legend_position: Optional[Tuple[float, float]] = None,
                cmap: Optional[str] = None,  # color map
                grid: bool = True,
                file: str = '',  # name of image file
                ) -> None:
    """
    Plots one z(x,y) array as 2D isomap
    """
    x, y, z = np.asfarray(x), np.asfarray(y), np.asfarray(z) 
    if labels is None:
        labels = ['x', 'y', 'z']
    if units is None:
        units = ['[/]', '[/]', '[/]']
    
    if xrange is None:
        xrange = (0., 0.)
    if yrange is not None:
        yrange = (0., 0.)
    if zrange is None:
        zrange = (0., 0.)
    for i in range(len(units)):
        if not units[i].startswith('['):
            units[i] = '[' + units[i] + ']'

    if not title and len(labels) > 2:
        title = labels[2] + ' ' + units[2]

    if figsize is None:
        figsize = (6, 3.5)
    if not fontsize:
        fontsize = round(0.4 * (figsize[0] * 3)) * 2
    plt.rcParams.update({'font.size': fontsize})
    plt.rcParams['legend.fontsize'] = fontsize

    fig = plt.figure(figsize=figsize)
    if labels[0]:
        plt.xlabel(labels[0] + ' ' + units[0])
    if labels[1]:
        plt.ylabel(labels[1] + ' ' + units[1])
    if xlog:
        plt.xscale('log', nonposy='clip')
    # else:
    #    plt.xscale('linear')
    if ylog:
        plt.yscale('log', nonposy='clip')
    # else:
    #    plt.yscale('linear')
    if title:
        plt.title(title, y=1.01)
    if grid:
        plt.grid()

    if cmap is None:
        cmap = 'jet'   # 'rainbow'
    plt.set_cmap(cmap)

    assert len(x) == len(y) and len(x) == len(z), 'size of x, y & z unequal' +\
        ' (' + str(len(x)) + ', ' + str(len(y)) + ', ' + str(len(z)) + ')'
    assert len(x) >= 3, 'size of x, y and z less 4'
    if zrange[0] != zrange[1]:
        z_min = min(z)
        z_max = max(z)
        x2, y2, z2 = clip_xyz(x, y, z,
                              zrange=[None, z_min + 1. * (z_max - z_min)])
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


def plot_isolines(x: np.ndarray, 
                  y: np.ndarray, 
                  z: np.ndarray,
                  labels: Optional[Tuple[str, str, str]] = None,
                  units: Optional[Tuple[str, str, str]] = None,
                  title: str = '',
                  xrange: Optional[Tuple[Optional[float], 
                                         Optional[float]]] = None,
                  yrange: Optional[Tuple[Optional[float], 
                                         Optional[float]]] = None,
                  zrange: Optional[Tuple[Optional[float], 
                                         Optional[float]]] = None,
                  figsize: Optional[Tuple[float, float]] = None,
                  file: str = '',
                  fontsize: Optional[int] = None,
                  grid: bool = False,
                  levels: Optional[Iterable[float]] = None, 
                  xlog: bool = False,
                  ylog: bool = False,
                  ) -> None:
    """
    Plots u(x,y) data as 2D isolines
    """
    if labels is None:
        labels = ['x', 'y', 'z']
    if units is None:
        units = '[/]', '[/]', '[/]'
    if xrange is None:
        xrange = [None, None]
    if yrange is None:
        yrange = [None, None]
    if zrange is None:
        zrange = [None, None]
    
    x, y, z = np.asfarray(x), np.asfarray(y), np.asfarray(z) 
    
    if xrange[0] is None:
        xrange[0] = x.min()
    if xrange[1] is None:
        xrange[1] = x.max()
    if yrange[0] is None:
        yrange[0] = y.min()
    if yrange[1] is None:
        yrange[1] = y.max()
    if zrange[0] is None:
        zrange[0] = z.min()
    if zrange[1] is None:
        zrange[1] = z.max()

    for i in range(len(units)):
        if not units[i].startswith('['):
            units[i] = '[' + units[i] + ']'
    fig = _plt_pre(xlabel=labels[0], ylabel=labels[1], title='',
                   xlog=xlog, ylog=ylog,
                   figsize=figsize, grid=grid, fontsize=fontsize, )

    if all([arr.ndim == 1 for arr in [x, y, z]]):
        x, y, z = to_regular_mesh(x, y, z)
    if zrange[0] != zrange[1]:
        x, y, z = clip_xyz(x, y, z, zrange=zrange)

    if levels is not None:
        levels = np.atleast_1d(levels)
        cs = plt.contour(x, y, z, levels)
    else:
        cs = plt.contour(x, y, z)
    plt.clabel(cs, inline=1, fontsize=10)

    # ax = fig.add_subplot(1, 1, 1, projection='3d')
    if len(labels) > 0 and labels[0] != '[ ]':
        plt.xlabel(labels[0] + ' ' + units[0])
    if len(labels) > 1 and labels[1] != '[ ]':
        plt.ylabel(labels[1] + ' ' + units[1])
    if labels[2]:
        # title() setting must be done here, does not work with plt_pre()
        plt.title(labels[2] + ' ' + units[2])
    plt.show()


def plot_vectors(x: np.ndarray, 
                 y: np.ndarray,
                 vx: np.ndarray, 
                 vy: np.ndarray,
                 labels: Optional[Tuple[str, str, str]] = None,
                 units: Optional[Tuple[str, str, str]] = None,
                 title: str = '',
                 xrange: Tuple[float, float] = (0., 0.), 
                 yrange: Tuple[float, float] = (0., 0.), 
                 xlog: bool = False, 
                 ylog: bool = False,
                 grid: bool = False,
                 figsize: Optional[Tuple[float, float]] = None,
                 fontsize: Optional[int] = None,
                 legend_position: Optional[Tuple[float, float]] = None,
                 file: str = '',
                 ) -> None:
    x, y = np.asfarray(x), np.asfarray(y)
    vx, vy = np.asfarray(vx), np.asfarray(vy) 

    if labels is None:
        labels = ['x', 'y', 'z', '', '', '']
    if units is None:
        units = ['[/]', '[/]', '[/]']

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
    if xlog:
        plt.xscale('log', nonposy='clip')
    # else:
    #    plt.xscale('linear')
    if ylog:
        plt.yscale('log', nonposy='clip')
    # else:
    #    plt.yscale('linear')
    if title:
        plt.title(title, y=1.01)
    if grid:
        plt.grid()

    if fontsize is None:
        fontsize = round(0.5 * (figsize[0] * 3)) * 1.33
    plt.rcParams.update({'font.size': fontsize})
    plt.rcParams['legend.fontsize'] = fontsize

    plt.quiver(x, y, vx, vy, angles='xy', scale_units='xy')

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


def plot_trajectory(x, y, z,  # trajectory to be plotted
                    x2: Optional[np.ndarray] = None, 
                    y2: Optional[np.ndarray] = None, 
                    z2: Optional[np.ndarray] = None,  
                                       # second trajectory to be plotted
                    x3: Optional[np.ndarray] = None, 
                    y3: Optional[np.ndarray] = None, 
                    z3: Optional[np.ndarray] = None,  
                                        # third trajectory to be plotted
                    labels: Optional[Tuple[str, str, str, 
                                           str, str, str]] = None,
                    units: Optional[Tuple[str, str, str]] = None,
                    title: Optional[str] = '',
                    xrange: Optional[Tuple[float, float]] = (0., 0.),
                    yrange: Optional[Tuple[float, float]] = (0., 0.),
                    zrange: Optional[Tuple[float, float]] = (0., 0.),
                    ylog: bool = False,
                    grid: bool = False,
                    figsize: Optional[Tuple[float, float]] = (10, 8),
                    fontsize: Optional[int] = None,
                    legend_position: Optional[Tuple[float, float]] = None,
                    file: str = '',
                    start_point: bool = False,
                    ) -> None:
    """
    Plots up to three (x,y,z)  trajectories in 3D space
    """
    if x2 is None:
        x2 = []
    if y2 is None:
        y2 = []
    if z2 is None:
        z2 = []
    if x3 is None:
        x3 = []
    if y3 is None:
        y3 = []
    if z3 is None:
        z3 = []
    if labels is None:
        labels = ['x', 'y', 'z', '', '', '']
    if units is None:
        units = ['[/]', '[/]', '[/]']

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
    if start_point:
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


def plot_2bars(
        x_labels: Optional[Iterable[str]] = None,
        y1: Optional[Iterable[float]] = None,
        y2: Optional[Iterable[float]] = None,
        y_labels: Tuple[str, str] = ('y1', 'y2'),
        title: str = 'y1(x), y2(x)',
        y_log: bool = False,
        y_lim: Tuple[Optional[float], Optional[float]] = (None, None),
        bar_width: float = 0.3,
        ) -> None:
    """
    Plots two 1D float arrays y1 and y2 vs 1D str array with x-labels
    
    Example:
        plot_2bars(x_labels=['x1', 'x2', 'x3', 'x4', 'x5'],
                   y1=[2, 3, 1, 9, 2],
                   y2=[7, 9, 2, 4, 5],
                   y_labels=['y1', 'y2'],
                   title='y(x)',
                   y_log=True)
    """ 
    if y1 is None:
        if y2 is None:
            print('??? y1 and y2 are None')
            return
        else: 
            y1, y2 = y2, y1

    if x_labels is None:
        x_labels = [str(i) for i in range(len(y1))]

    assert len(x_labels) == len(y1), str((x_labels, y1))
    if y2 is not None:
        assert len(x_labels) == len(y2), str((x_labels, y2))
    assert len(y_labels) == 2, str(y_labels) 
    
    x = np.arange(len(x_labels))
    fig, axs = plt.subplots()
    axs.bar(x - bar_width/2, y1, bar_width, label=y_labels[0])
    if y2 is not None:
        axs.bar(x + bar_width/2, y2, bar_width, label=y_labels[1])
    
    axs.set_title(str(title))
    axs.set_xticks(x)
    axs.set_xticklabels(x_labels)
    axs.set_ylabel(str(y_labels[0]) + ', ' + str(y_labels[1]))
    axs.legend()
        
    if y_log:
        plt.yscale('log')
    plt.ylim(y_lim)
    plt.grid()
    fig.tight_layout()
    plt.xticks(rotation=45)
    legend_position = (1.1, 1.05)
    plt.legend(bbox_to_anchor=legend_position, loc='upper left')

    plt.show()


def plot_bar_arrays(x: Optional[np.ndarray] = None,
                    yarrays: Optional[np.ndarray] = None,
                    labels: Optional[Tuple[str, str, str]] = None,
                    units: Optional[Tuple[str, str, str]] = None,
                    title: str = '',
                    yrange: Optional[Tuple[float, float]] = None,
                    grid: bool = False,
                    figsize: Optional[Tuple[float, float]] = None,
                    fontsize: int = 14,
                    legend_position: Optional[Tuple[float, float]] = None,
                    show_ylabel: bool = True,
                    width: float = 0.15,
                    file: str = '',
                    ) -> None:
    """
    Plots bars without errorbars if y_arrays is 2D array of float
    """
    assert yarrays is not None

    yarrays = np.atleast_2d(yarrays)
    ny = yarrays.shape[0]
    y1 = yarrays[0] if ny > 0 else None
    y2 = yarrays[1] if ny > 1 else None
    y3 = yarrays[2] if ny > 2 else None
    y4 = yarrays[3] if ny > 3 else None
    y5 = yarrays[4] if ny > 4 else None
    y6 = yarrays[5] if ny > 5 else None

    plot_bars(x=x,
              y1=y1, y1error=None,
              y2=y2, y2error=None,
              y3=y3, y3error=None,
              y4=y4, y4error=None,
              y5=y5, y5error=None,
              y6=y6, y6error=None,
              yrange=yrange,
              labels=labels, 
              figsize=figsize, 
              file=file,
              fontsize=fontsize,
              grid=grid, 
              legend_position=legend_position, 
              show_ylabel=show_ylabel,
              title=title, 
              width=width, 
              units=units, 
              )


def plot_bars(x: Optional[np.ndarray] = None,
              y1: Optional[np.ndarray] = None, 
              y1error: Optional[np.ndarray] = None,
              y2: Optional[np.ndarray] = None, 
              y2error: Optional[np.ndarray] = None,
              y3: Optional[np.ndarray] = None, 
              y3error: Optional[np.ndarray] = None,
              y4: Optional[np.ndarray] = None, 
              y4error: Optional[np.ndarray] = None,
              y5: Optional[np.ndarray] = None, 
              y5error: Optional[np.ndarray] = None,
              y6: Optional[np.ndarray] = None, 
              y6error: Optional[np.ndarray] = None,
              labels: Optional[Iterable[str]] = None,
              units: Optional[Tuple[str, str, str]] = None,
              title: str = '',
              yrange: Optional[Tuple[float, float]] = None,
              grid: bool = False,
              figsize: Optional[Tuple[float, float]] = None,
              fontsize: Optional[int] = 14,
              legend_position: Optional[Tuple[float, float]] = None,
              show_ylabel: bool = True,
              width: float = 0.15,
              file: str = ''
              ) -> None:

    if y1 is None:
        y1 = []
    if y1error is None:
        y1error = []
    if y2 is None:
        y2 = []
    if y2error is None:
        y2error = []
    if y3 is None:
        y3 = []
    if y3error is None:
        y3error = []
    if y4 is None:
        y4 = []
    if y4error is None:
        y4error = []
    if y5 is None:
        y5 = []
    if y5error is None:
        y5error = []
    if y6 is None:
        y6 = []
    if y6error is None:
        y6error = []
        
    if labels is None:
        labels = ['x', 'y', '', '', '', '']
    if units is None:
        units = ['[/]','[/]', '']

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
            labels = ['x'] + ['y'+str(i) for i in range(ny)]  # axis labels
        if not units:
            units = ['[/]']*(ny + 1)  # axis units

    if len(y1error):
        rects1 = ax.bar(ind + 0 * width, y1, width, color='b', yerr=y1error)
    else:
        rects1 = ax.bar(ind + 0*width, y1, width, color='b')
    if len(y2):
        if len(y2error):
            rects2 = ax.bar(ind + 1*width, y2, width, color='r', yerr=y2error)
        else:
            rects2 = ax.bar(ind + 1*width, y2, width, color='r')
    else:
        rects2 = None
    if len(y3):
        if len(y3error):
            rects3 = ax.bar(ind + 2*width, y3, width, color='y', yerr=y3error)
        else:
            rects3 = ax.bar(ind + 2*width, y3, width, color='y')
    else:
        rects3 = None
    if len(y4):
        if len(y4error):
            rects4 = ax.bar(ind + 3*width, y4, width, color='g', yerr=y3error)
        else:
            rects4 = ax.bar(ind + 3*width, y4, width, color='g')
    else:
        rects4 = None
    if len(y5):
        if len(y5error):
            rects5 = ax.bar(ind + 4*width, y5, width, color='b', yerr=y5error)
        else:
            rects5 = ax.bar(ind + 4*width, y5, width, color='b')
    else:
        rects5 = None
    if len(y6):
        if len(y6error):
            rects6 = ax.bar(ind + 5*width, y6, width, color='c', yerr=y6error)
        else:
            rects6 = ax.bar(ind + 5*width, y6, width, color='c')
    else:
        rects6 = None

    plt.xlabel(labels[0])
    if show_ylabel:
        la = ''
        for i in range(1, len(labels)):
            if len(labels[i]):
                la += labels[i] + ', '
        if la.endswith(', '):
            la = la[:-2]
        ax.set_ylabel(la)

    if title:
        ax.set_title(title)

    ax.set_xticks(ind + width)
    tics = []
    for i in range(len(y1)):
        if len(x):
            tics.append(str(x[i]))     # tics.append(labels[0] + str(i))
        else:
            tics.append('x' + str(i))  # tics.append(labels[0] + str(i))
    ax.set_xticklabels(tics)
    plt.xticks(rotation=45)    

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
    if legend_position:
        ax.legend(r, la, bbox_to_anchor=legend_position, loc='upper left')
    else:
        if figsize[0] <= 8:
            ax.legend(r, la, bbox_to_anchor=(1, 1), loc='upper right')
        else:
            ax.legend(r, la)

    if False:
        for rects in [rects1, rects2, rects3, rects4, rects5, rects6]:
            if rects:
                for r in rects:
                    height = r.get_height()
                    ax.text(r.get_x() + 1.0*r.get_width(), 1.0*height+0.02,
                            '%d' % int(height), ha='center', va='bottom')
    if grid:
        plt.grid()
        
    fig.tight_layout()
    
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


def plot_x_y_y_ref(x: np.ndarray,   
                   y: np.ndarray,
                   y_ref: np.ndarray,
                   labels: Optional[Tuple[str, str, str]] = None, 
                   presentation: str = 'all'
                   ) -> None:
    """
    Plots y(x), y_ref(x) and the difference y-y_ref(x) as 
        - isomap and surface if x, y and y_ref are 2D arrays
        - or as curves otherwise
        
    Args:
        x (array of float):
            arguments, shape: (n_point, n_inp)

        y (array of float):
            dependent variable y(x), shape: (n_point, n_out)

        y_ref (array of float):
            reference y_ref(x), shape: (n_point, n_out)
        
        labels:
            list of x, y and z labels
            
        presentation:
            placeholder
    """
    if labels is None:
        labels = ('x', 'y', 'y^{ref}', )
    assert len(labels) >= 3
        
    
    if any(arr is None for arr in (x, y, y_ref)):
        print('!!! one of the plot_x_y_y_ref() args is None')
        return
    
    x, y, y_ref = np.asfarray(x), np.asfarray(y), np.asfarray(y_ref)

    if any(len(arr.shape) != 2 for arr in (x, y, y_ref, )):
        plot_curves(x.flatten(), y.flatten(), y_ref.flatten(), labels=labels)
    else:
        # plots y
        plot_isomap(x[:, 0], x[:, 1], y[:, 0], title='$' + labels[1] + '$',
                    labels=['$' + labels[0] + '_0$',
                            '$' + labels[0] + '_1$', '$' + labels[1] + '$'])
        plot_surface(x[:, 0], x[:, 1], y[:, 0], title='$' + labels[1] + '$',
                     labels=['$' + labels[0] + '_0$',
                             '$' + labels[0] + '_1$', '$' + labels[1] + '$'])    
        # plots y_ref
        plot_isomap(x[:, 0], x[:, 1], y_ref[:, 0], title='$' + labels[2] + '$',
                    labels=['$' + labels[0] + '_0$',
                            '$' + labels[0] + '_1$', '$' + labels[2] + '$'])
        plot_surface(x[:, 0], x[:, 1], y_ref[:, 0], title='$' + labels[2] +'$',
                     labels=['$' + labels[0] + '_0$',
                             '$' + labels[0] + '_1$', '$' + labels[2] + '$'])    
        # plots y - y_ref
        plot_isomap(x[:, 0], x[:, 1], (y - y_ref)[:, 0],
                    title='$' + labels[1] + ' - ' + labels[2] + '$',
                    labels=['$' + labels[0] + '_0$', '$' + labels[0] + '_1$',
                            '$' + labels[1] + ' - ' + labels[2] + '$'])
        plot_surface(x[:, 0], x[:, 1], (y - y_ref)[:, 0],
                     title='$' + labels[1] + ' - ' + labels[2] + '$',
                     labels=['$' + labels[0]+ '_0$', '$' + labels[0] + '_1$',
                             '$' + labels[1] + ' - ' + labels[2] + '$'])
