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
      2019-12-10 DWW
"""

import initialize
initialize.set_path()

import numpy as np
import sys
from time import time
from typing import Any, List, Optional
import unittest

from grayboxes.base import Base


class Foo(Base):
    def __init__(self, identifier: str='Foo',
                 argv: Optional[List[str]]=None) -> None:
        super().__init__(identifier=identifier, argv=argv)
        self.x = 3.0

    def pre(self, **kwargs: Any) -> bool:
        super().pre()
        
        return True

    def task(self, **kwargs: Any) -> float:
        super().task()

        self.x *= 2.0
        b = 7.0
        for i in range(int(1e6)):
            b = b / 3.0
        self.y = b
        
        return 0.0

    def post(self, **kwargs: Any) -> bool:
        super().post()
        self.write('    x(' + self.identifier + '): ' + str(self.x))
        self.write('    y(' + self.identifier + '): ' + str(self.y))
        
        return True


class TestUM(unittest.TestCase):
    def setUp(self):
        pass


    def tearDown(self):
        pass


    def test1(self):
        # creates instance

        foo = Foo('root1', sys.argv)
        foo.gui = False

        # assigns path to files

        # creates objects
        f1 = Foo('follower 1')
        f11 = Foo('follower 1->1')
        f12 = Foo('follower 1->2')
        f13 = Foo('follower 1->3')
        f2 = Foo('follower 2')
        f21 = Foo('follower 2->1 and cooperator 1--2')
        f22 = Foo('follower 2->2')

        # connects objects                                   foo
        foo.set_follower([f1, f2])           # .            /     \
        f1.set_follower([f11, f12, f13])     # .         f1 ......   f2
        f2.set_follower([f21, f22])          # .       / |  \     : /  \
        f1.set_cooperator(f21)               # .    f11 f12 f13   f21 f22

        # links between two objects
        f13.x = 6.789
        print('f13.x:', f13.x)
        f1.link = foo['follower 1->3']
        f1.link.x = 4.56
        print('f1.link.id f1_b_link.id:', f1.link.identifier)
        print('f13.id:', f13.identifier)
        print('f13.x:', f13.x)

        f1_b_link = foo.get_follower('follower 1->3')
        print('f1.link.id:', f1.link.identifier)
        print('f1_b.link.id:', f1_b_link.identifier)
        assert f1.link.identifier == f1_b_link.identifier
        print('-'*20)

        foo()

        # prints content of root and its followers
        print('\nPrint(foo): ' + str(foo))
        del foo

        print('-' * 40)
        self.assertTrue(True)


    def test2(self):
        foo = Foo('root2')
        foo()
        foo.write(str('my write id:' + foo.identifier))
        
        print('-' * 40)
        self.assertTrue(True)


    def test3(self):
        """
        Compares access time to node via get_follower (or []) and get_link
        
        Result on Intel i7-8850H @2.60GHz:

            time (get_follower): total: 4131489.992 per access: 413.148 [us]
            time (get_link):     total:    3994.226 per access:   0.399 [us]
            
        Note:
            Tree is populated 16 followers having 16 sub-followers each
            Access time for last tree node is less than a half millisecond
        """
        n_a, n_b = 16, 16  # ==> total of 1 + 16 * 16 = 257 nodes
        n_test_loop = 10*1000
        
        b = Base('root')
        for i in range(n_a):
            a_key = 'a' + str(i)
            b_key = 'b' + str(i)
            b.set_follower([Base(a_key), Base(b_key)])
            
            for j in range(n_b):
                aa_key = a_key + '_' + str(j)
                bb_key = b_key + '_' + str(j)
                b[a_key].set_follower(Base(aa_key))
                b[b_key].set_follower(Base(bb_key))
                
        b2 = Base('root2')
        test_key = 'a' + str(n_a - 1) + '_' + str(n_b - 1)
        b2.set_link(b[test_key])
        
        start = time()
        for i in range(n_test_loop):
            node = b[test_key]
        dt = np.round((time() - start) * 1e6, 3)
        print('time (get_follower), total: ', dt, 
              'per access:', dt / n_test_loop, '[us]')

        start = time()
        for i in range(n_test_loop):
            node2 = b2.get_link(test_key)
        dt = np.round((time() - start) * 1e6, 3)
        print('time (get_link), total: ', dt, 
              'per access:', dt / n_test_loop, '[us]')
        
        assert node == node2
        
        print('-' * 40)
        self.assertTrue(True)


    def test4(self):
        # searches for specific follower in tree
        foo = Foo('root3')
        identifier = 'follower 11'
        p = foo.get_follower_downwards(identifier=identifier)
        if p is None:
            print('identifier not found, p:', p)
        else:
            print('identifier found:', p.identifier == identifier)
            print('downward search, p.identifier:', p.identifier)
        
        print('-' * 40)
        self.assertTrue(True)


    def test5(self):
        # destructs tree
        foo = Foo('root4')
        foo()
        print('*** destruct')
        print('foo 4:', foo)
        foo.destruct()
        
        print('-' * 40)
        self.assertTrue(True)


    def test6(self):
        # sends warning and termination of program
        foo = Foo('root5')
        print('foo 5:', foo)
        foo.gui = not True  # TODO toggle foo.gui if TKinter interface
        foo.warn('my warning1')
        with self.assertRaises(SystemExit) as cm:
            foo.terminate('warning to GUI')

        print('-' * 40)
        self.assertEqual(cm.exception.code, None)        


    def test7(self):
        # sends warning
        foo = Foo('root6')

        foo.gui = False
        foo()
        print('foo 6:', foo)
        foo.warn('my warning1')
        
        print('-' * 40)
        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
