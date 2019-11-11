# grayboxes

_grayboxes_ contributes to the creation and evaluation of [white](https://github.com/dwweiss/grayboxes/wiki/2.-White-box-model) box, [gray](https://github.com/dwweiss/grayboxes/wiki/3.-Gray-box-model) box and [black](https://github.com/dwweiss/grayboxes/wiki/4.-Black-box-model) box models of physical and chemical transport phenomena. Gray box models 

- are hybrids of theory-driven and data-driven submodels

- can have adjustable [degrees of blackness](https://github.com/dwweiss/grayboxes/wiki/5.-Model-evaluation#52-degree-of-model-blackness) (the blacker, the more data-driven) 

- are compatible to all [operations](https://github.com/dwweiss/grayboxes/wiki/6.-Operations-on-model#61-operations) of the _grayboxes_ library:

     - Minimization / maximization     
     - Inverse problem solution
     - Sensitivity analysis 

_grayboxes_  is the base of the extension package [_coloredlids_](https://github.com/dwweiss/coloredlids/wiki) for  implementation of distributed theoretical submodels. _coloredlids_  based models are compatible to the model [operations](https://github.com/dwweiss/grayboxes/wiki/6.-Operations-on-model) of the _grayboxes_ package, see figure below.


<br>
<p align="center"><img src="https://github.com/dwweiss/grayboxes/blob/master/doc/fig/operationsOnBoxTypeModels_mediumGray_observation.png">
</p>

[[Link to grayboxes Wiki]](https://github.com/dwweiss/grayboxes/wiki)


### Content

    grayboxes
        Training of gray box models, sensitivity analysis
        Optimization and inverse problem solution with white box, gray box and black box models

    test
        Module tests

    doc
        Figures and manuals used in wiki
        
### Installation

    git clone https://github.com/dwweiss/grayboxes.git
    python3 setup.py install --user `
    
[[Link]](https://github.com/dwweiss/grayboxes/blob/master/doc/installation/windowsinstallation.md#installation-proposal) 
 to alternative package installation procedure on windows
### Dependencies

- Modules _lightgray_ and _minimum_ are dependent on package _modestga_ [[MGA18]](https://github.com/dwweiss/grayboxes/wiki/References#mga18)
- Module _neural_ is dependent on package _neurolab_ [[NLB15]](https://github.com/dwweiss/grayboxes/wiki/References#nlb15)

These dependencies are satisfied if _grayboxes_ is installed with the commands in section [Installation](#installation).
