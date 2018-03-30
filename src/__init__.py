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
      2018-02-02 DWW
"""

import os
import sys


# if True, this init file is placed in a pmLib sub-folder
pmLibSubFolder = False

# sub-folders containing a string from blackList are not added to sys.path
blackList = ['archive', '__pycache__', '/', '-', 'workbench']

# get pmLib folder
if not pmLibSubFolder:
    folder = os.getcwd()
else:
    folder = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

# add this folder to sys.path
if folder not in sys.path:
    sys.path.insert(0, folder)

# add all sub-folders to sys.path excluding blackList
subFolder = [x[0] for x in os.walk(folder)]
for sub in list(reversed(subFolder)):
    # print('sub:', sub, 'any:', not any([x in sub for x in blackList]))
    if sub not in sys.path and not any([x in sub for x in blackList]):
        sys.path.insert(1, sub)

# remove parts of sys.path
if 0:
    for sub in list(reversed(subFolder)):
        print('sub:', sub, 'any:', not any([x in sub for x in blackList]))
        if any([x in sub for x in blackList]):
            sys.path.remove(sub)

# print sys.path
if 1:
    for x in sys.path:
        print(x)
