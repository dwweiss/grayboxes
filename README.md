# grayboxes

_grayboxes_ contributes to the creation and evaluation of [white](https://github.com/dwweiss/grayboxes/wiki/2.-White-box-model) box, [gray](https://github.com/dwweiss/grayboxes/wiki/3.-Gray-box-model) box and [black](https://github.com/dwweiss/grayboxes/wiki/4.-Black-box-model) box models of physical and chemical transport phenomena. Gray box models 

- are hybrids of data-driven (black) and theory-driven (white) submodels

- can have adjustable [degrees of blackness](https://github.com/dwweiss/grayboxes/wiki/5.-Model-evaluation#52-degree-of-model-blackness) (the blacker, the more data driven) 

- are compatible to the [operations](https://github.com/dwweiss/grayboxes/wiki/6.-Operations-on-model) of the _grayboxes_ library:


     - Minimization / maximization     
     - Inverse problem solution
     - Sensitivity analysis 

<!--
     - [Minimization](https://github.com/dwweiss/grayboxes/tree/master/grayboxes/minimum.py)/[maximization](https://github.com/dwweiss/grayboxes/tree/master/grayboxes/maximum.py)
     - [Inverse](https://github.com/dwweiss/grayboxes/tree/master/grayboxes/inverse.py) problem solution
     - [Sensitivity](https://github.com/dwweiss/grayboxes/tree/master/grayboxes/sensitivity.py) analysis 
-->

<!-- [[Link to grayboxes Wiki]](https://github.com/dwweiss/grayboxes/wiki) -->


<p align="center"><img src="https://github.com/dwweiss/grayboxes/blob/master/doc/fig/boxTypeWithTheoretical.png"></p>


The theoretical submodels can be of different size and complexity. Package [_coloredlids_](https://github.com/dwweiss/coloredlids/wiki) supports the implementation of distributed white box models. Models based on [_coloredlids_](https://github.com/dwweiss/coloredlids/wiki) are compatible to the model [operations](https://github.com/dwweiss/grayboxes/wiki/6.-Operations-on-model) of the _grayboxes_ library.

<br>

### Content

    grayboxes
        Training of gray box models, sensitivity analysis
        Optimization and inverse problem solution with white box, gray box and black box models

    tests
        Selected test cases

    doc
        Figures and manuals used in wiki
        

### Installation

    git clone https://github.com/dwweiss/grayboxes.git
    python3 setup.py install --user `

### Dependencies

- Modules _lightgray_ and _minimum_ are dependent on package _modestga_ [[MGA18]](https://github.com/dwweiss/grayboxes/wiki/References#mga18)
- Module _neural_ is dependent on package _neurolab_ [[NLB15]](https://github.com/dwweiss/grayboxes/wiki/References#nlb15)

These dependencies are automatically satisfied if _grayboxes_ is installed with the commands in section _Installation_  
