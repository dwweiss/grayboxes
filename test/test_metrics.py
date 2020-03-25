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
      2020-01-13 DWW
"""

import initialize
initialize.set_path()

import unittest

from grayboxes.metrics import init_metrics, Metrics, update_errors


class TestUM(unittest.TestCase):

    def setUp(self):
        pass


    def tearDown(self):
        pass


    def test1(self):
        m = Metrics({'aaa': 999})
        print(m)
    
        for plot in (False, True, ):    
            x = m.update_errors([1,2,3], [3,4,5], [2.1, 4.2, 5], plot=plot)
        
        print(m)
        print(x)
        print("m['aaa']", m['aaa'])    
        

        m = init_metrics()
        print(m)
        m['abs'] = 1235
        m['ABS'] = 123567
        print(m['abs'])
        print(m)
        x = update_errors(m, X=[1,2,3], Y=[2,4,6], y=[2.1, 4.3, 5.7], 
                          silent=False, plot=1)
        print('*'*20, '\nx', x)
        del m['abs']
        print(m)

        self.assertTrue(True)


    def test2(self):
        m = init_metrics({'aaa': 3})
        
        for silent in (True, False):
            for plot in (True, False):
                x = update_errors(m, X=[1,2,3], Y=[2,4,6], y=[2.1, 4.3, 5.7], 
                                  silent=silent, plot=plot)
                print(x)

        self.assertTrue(True)

        m = Metrics({'aaa': 999})
        print(m)
    
        for plot in (False, True, ):    
            x = m.update_errors([1,2,3], [3,4,5], [2.1, 4.2, 5], plot=plot)
        
        print(m)
        print(x)
        print("m['aaa']", m['aaa'])    

        self.assertTrue(True)

                
    def test3(self):
        m = Metrics({'value1': 3.0})
        x = m['value1']
        print('x', x)
        print('m', m)
        m['value1'] = 4.56

        self.assertTrue('value1' in m)
        self.assertTrue(m['value1'] == 4.56)

        del m['value1']

        self.assertFalse('value1' in m)
        print('m', m)

        self.assertTrue(True)


    def test4(self):
        m = Metrics({'value1': 3.0})
        
        d = m.to_dict()
        print('m', m)
        print('d', d)

        self.assertTrue(True)


    def test5(self):
        histories = [Metrics({'a': 4}), Metrics({'a': 5}), Metrics({'a': 11})]
        
        m = Metrics()
        m.plot_histories('a', histories)
        
        self.assertTrue(True)


    def test6(self):
        histories = [dict({'a': [4, 5, 6]}), 
                     dict({'a': [7, 11, 2]}), 
                     dict({'a': [6, 7, 8]})]
        
        m = Metrics()
        m.plot_histories('a', histories)

        m.plot_histories('i_abs', histories)
        
        self.assertTrue(True)


    def test7(self):
        histories = [Metrics({'b': [4, 5, 6]}), 
                     Metrics({'bb': [7, 11, 2]}), 
                     Metrics({'b': [6, 7, 8]})]
        
        m = Metrics()
        m.plot_histories('b', histories)

        m.plot_histories('i_abs', histories)

        m.plot_histories('?', histories)
        
        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
