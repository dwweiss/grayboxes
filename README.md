# grayboxes

_grayboxes_ contributes to the creation and evaluation of white box, gray box and black box models of heat transfer, fluid flow and structural mechanics. 


[[Link to grayboxes Wiki]](https://github.com/dwweiss/grayboxes/wiki/1.-Introduction)

<br>

### Content

    src
        Training, sensitivity analysis and prediction with white box, gray box and black box models

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

These depdendencies are automatically satisfied if installation with _setup.py_ was successful. 
