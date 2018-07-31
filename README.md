# grayboxes

_grayboxes_ contributes to the creation and evaluation of white box, gray box and black box models of physical and chemical transport phenomena. Gray box models 

- are hybrids of data-driven (black) and theory-driven (white) submodels of the process investigated

- can have adjustable degrees of blackness (the blacker, the more data driven) 

- are compatible to the operations of the _grayboxes_ library:

     - Minimization/maximization
     - Inverse problem solution
     - Sensitivity analysis 

[[Link to grayboxes Wiki]](https://github.com/dwweiss/grayboxes/wiki)

<br>
<p align="center"><img src="https://github.com/dwweiss/grayboxes/blob/master/doc/fig/boxTypeWithTheoretical.png"></p>


The theoretical submodels can be of different size and complexity. Package [_coloredlids_](https://github.com/dwweiss/coloredlids/wiki) supports the implementation of distributed white box models. 

<br><p align="center"><img src="https://github.com/dwweiss/coloredlids/blob/master/doc/fig/colored_boxes_top.png"></p>

Models created with [_coloredlids_](https://github.com/dwweiss/coloredlids/wiki) are compatible to the above listed operations of the _grayboxes_ library.

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
