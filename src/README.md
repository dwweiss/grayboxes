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
      2018-04-01 DWW
"""

"""        
********************************************************************
                       CLASS DEPENDENCIES 
********************************************************************


                           Operation                                
                               |                                            
                               |                                               
                +--------------+--------------+                         
                |              |              |                       
                v              v              v                       
      Forward Simulation   Optimisation   Inverse of Forward    
                |
                v
    Sensitivity Analysis 

 

    Models: (Theoretical, Empirical, Model)
    --------------------------------------------------------------------------

              Base
                ^
                |
              Loop
                ^
                |
              Move   <==aggregation=== xyz
                ^
                |
              Model
              ^   ^
              |   |
    -----------   ---------
    Theoretical   Empirical   <==aggregation=== Neural
    -----------   ---------


    Models: (Hybrid)
    --------------------------------------------------------------------------

              Model
                ^
                |
             ======                      -----------  ---------
             Hybrid   <==aggregation===  Theoretical, Empirical
             ======                      -----------  ---------

    Note:
        On Model level, (X, Y) are input and target of the training. 
        model.x and model.y are input and output of the prediction.



    Operations: (Forward, Sensitivity, Optimum, Inverse)
    --------------------------------------------------------------------------

              Base
                ^
                |                        -----------  ---------  ======
             Forward  <==aggregation===  Theoretical, Empirical, Hybrid
              ^   ^                      -----------  ---------  ======
              |   |
    Sensitivity   Optimum
                     ^
                     |
                  Inverse



    Convenience class
    --------------------------------------------------------------------------

              Base
                ^
                |
           ===========
          | Operation | <==aggregation== Forward, Sensitivity, Optimum, Inverse
           ===========
           
    Note:
        On operations level, 'x' is the initial 2D input and 'ranges' is a list
        of min/max pairs constraining 'x'.
        Inverse.y is the target of the inverse search and Inverse.x is the 
        initial value. Optimum.y is the final optimum and Optimum.x is the 
        input delivering this optimum. 
"""
