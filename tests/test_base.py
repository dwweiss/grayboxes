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
      2018-08-15 DWW
"""

import unittest	
import sys
import os

sys.path.append(os.path.abspath('..'))
from grayboxes.base import Base


class TestUM(unittest.TestCase):
 
    class Foo(Base):
        def __init__(self, identifier=''):
            super().__init__(identifier=identifier)
            self.x = 3.0
    
        def task(self, **kwargs):
            super().task()
    
            self.x *= 2.0
            b = 7.0
            for i in range(int(1e6)):
                b = b / 3.0
            self.y = b
            return 0.0

        def post(self, **kwargs):
            super().post()
            self.write('    x: ', self.x)
            self.write('    y: ', self.y)
            

    def setUp(self):
        pass
 
    def tearDown(self):
        pass
 
    def test1(self):
        # creates instance
        foo = self.Foo('root')
        foo.gui = False

        # assigns path to files
        foo.path = 'c:/Temp'

        # creates objects
        f1 = self.Foo('follower 1')
        f11 = self.Foo('follower 1->1')
        f12 = self.Foo('follower 1->2')
        f13 = self.Foo('follower 1->3')
        f2 = self.Foo('follower 2')
        f21 = self.Foo('follower 2->1 and cooperator 1--2')
        f22 = self.Foo('follower 2->2')

        # connects objects                                    foo
        foo.setFollower([f1, f2])           # .             /     \
        f1.setFollower([f11, f12, f13])     # .          f1 ......   f2
        f2.setFollower([f21, f22])          # .        / |  \     : /  \
        f1.setCooperator(f21)               # .     f11 f12 f13   f21 f22

        # links between two objects
        f13.x = 6.789
        print('f13.x:', f13.x)
        f1.link = foo['follower 1->3']
        f1.link.x = 4.56
        print('f1.link.id f1_b_link.id:', f1.link.identifier)
        print('f13.id:', f13.identifier)
        print('f13.x:', f13.x)

        f1_b_link = foo.getFollower('follower 1->3')
        print('f1.link.id:', f1.link.identifier)
        print('f1_b.link.id:', f1_b_link.identifier)
        assert f1.link.identifier == f1_b_link.identifier
        print('-'*20)

        # assigns a private logfile to follower f11
        f11.logFile = 'f11log'
        f11.write('abc')
        foo()

        # prints content of root and its followers
        print('\nPrint(foo): ' + str(foo))
        self.assertTrue(True)
 
    def test2(self): 
        # assigns a new log file common to root and its followers
        foo = self.Foo('root')
        foo.logFile = 'abc'
        foo()
        self.assertTrue(True)

    def test3(self): 
        # searches for specific follower in tree
        foo = self.Foo('root')
        identifier = 'follower 11'
        p = foo.getFollowerDownwards(identifier=identifier)
        if p is None:
            print('identifier not found, p:', p)
        else:
            print('identifier found:', p.identifier == identifier)
            print('downward search, p.identifier:', p.identifier)
        self.assertTrue(True)

    def test4(self): 
        # destructs tree
        foo = self.Foo('root')
        print('*** destruct')
        print('foo 1:', foo)
        foo.destruct()
        self.assertTrue(True)

    def test5(self): 
        # sends warning and termination of program
        foo = self.Foo('root')
        print('foo 2:', foo)
        foo.warning('my warning1')
        foo.terminate('my error message')
        self.assertRaise()


if __name__ == '__main__':
    unittest.main()
