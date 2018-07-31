# grayboxes

_grayboxes_ contributes to the creation and evaluation of white box, gray box and black box models of physical and chemical transport phenomena. 

- The models are usually based on theoretical (white box) submodels of the process investigated. 
- Besides model construction, _grayboxes_ provides tools for typical operations on model such as:
     - Minimization
     - Inverse problem solution
     - Sensitivity analysis 

Such submodels can be of different size and complexity. Package [coloredlids](https://github.com/dwweiss/coloredlids/wiki) supports the implementation of distributed white box models compatible to the model operations of the _grayboxes_ library.



[[Link to grayboxes Wiki]](https://github.com/dwweiss/grayboxes/wiki)

<br>

### Content

    grayboxes
        Training of graybox models, sensitivity analysis,  
        Minimization, inverse problem solution with white box, gray box and black box models

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
