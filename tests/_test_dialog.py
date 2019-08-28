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
      2018-08-28 DWW
"""

import unittest
import os
import sys
sys.path.append(os.path.abspath('..'))
from grayboxes.dialog import (dialog_yes_no, dialog_load_filenames, 
                              dialog_save_filename, dialog_directory)
    
class TestUM(unittest.TestCase):
    """
    Tests involve user interaction; The name of this file starts with 
    letter '_' in order to avoid execution by test_All.py
    
    
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test1(self):
        yes_no = dialog_yes_no(title='Question to be answered',
                               icon='question')
        print('yes_no:', yes_no)
        self.assertTrue(True)

    def test2(self):
        List = dialog_load_filenames(default_ext='data', initial_dir=None,
                                     min_files=2)
        print('List:', List)
        self.assertTrue(True)

    def test3(self):
        filename = dialog_save_filename(default_ext='data', initial_dir=None,
                                        confirm_overwrite=True)
        print('fn:', filename)
        foo = open(filename, 'w')
        result = foo is not None
        foo.write('foo')
        foo.close()
        self.assertTrue(result)

    def test4(self):
        Dir = dialog_directory(initial_dir=None)
        print('Dir:', Dir)
        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
