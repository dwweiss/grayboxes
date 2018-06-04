"""
  Copyright (c) 2016- by Dietmar Wilhelm Weiss, Denmark

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
      2018-06-03 DWW
"""

import os
import sys

"""
Extends sys.path with subdirectories and directory where this file is located

- Search starts from directory where this file is located
- Paths containing strings in 'blackList' are excluded
"""

# if False, search starts from '/x'
fromInitFileLocation = True

# subdirectories containing a member of blackList wont be added to sys.path
blackList = ['/archive', '/doc', '/temp', '/tmp', '/workbench', '/.', '/_']

# get start directory
if fromInitFileLocation:
    dir = os.getcwd()
else:
    dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

# add start directory to sys.path
if dir not in sys.path:
    sys.path.insert(0, dir)

# add all subdirs to sys.path excluding those which contain 'blackList' strings
subDirs = [dirs[0] for dirs in os.walk(dir)]
for subDir in list(reversed(subDirs)):
    if subDir not in sys.path:
        if not any([b in subDir for b in blackList]):
            sys.path.insert(1, subDir)

# remove parts of sys.path
if 0:
    for sub in list(reversed(subDir)):
        print('sub:', sub, 'any:', not any([b in sub for b in blackList]))
        if any([string in sub for string in blackList]):
            sys.path.remove(sub)

# print sys.path
if 1:
    for x in sys.path:
        print(x)
