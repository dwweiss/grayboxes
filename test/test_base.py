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
      2019-11-22 DWW
"""

import __init__
__init__.init_path()

import unittest
import sys
from typing import Any, List, Optional

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

        self.assertTrue(True)


    def test2(self):
        foo = Foo('root2')
        foo()
        foo.write(str('my write id:' + foo.identifier))
        
        self.assertTrue(True)


    def test3(self):
        # searches for specific follower in tree
        foo = Foo('root3')
        identifier = 'follower 11'
        p = foo.get_follower_downwards(identifier=identifier)
        if p is None:
            print('identifier not found, p:', p)
        else:
            print('identifier found:', p.identifier == identifier)
            print('downward search, p.identifier:', p.identifier)
        
        self.assertTrue(True)


    def test4(self):
        # destructs tree
        foo = Foo('root4')
        foo()
        print('*** destruct')
        print('foo 4:', foo)
        foo.destruct()
        
        self.assertTrue(True)


    def test5(self):
        # sends warning and termination of program
        foo = Foo('root5')
        print('foo 5:', foo)
        foo.gui = not True  # TODO toggle foo.gui if TKinter interface
        foo.warn('my warning1')
        with self.assertRaises(SystemExit) as cm:
            foo.terminate('warning to GUI')

        self.assertEqual(cm.exception.code, None)        


    def test6(self):
        # sends warning
        foo = Foo('root6')
        foo.gui = False
        foo()
        print('foo 6:', foo)
        foo.warn('my warning1')
        
        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
