# grayBoxes

Data-driven modelling is gaining interest due to the availibility of machine learning tools such as TensorFlow, Keras etc. The success of data-driven tools is bound to the availibility of large data sets. In contrast, multi-purpose packages employing finite element or finite volume methods are not dependent on data. They are often limitated in reproducing the behaviour of the real process. This is especially true if the process to be modelled has chaotic elements as for instance in welding.

A remedy is the combination of theoretical with empirical models. Such hybrid models lower the expenses of theoretical model development and can be calibrated to given experimental sets, however it is still difficult to estimate the split in the share of the empirical submodel relative to the theoretical one. The availability of such an estimate is essential for evaluation of the model reliability and its further development.

_grayBoxes_ contributes to the creation and evaluation of hybrid models. A Python framework for implementation of empirical, theoretical and hybrid models is provided. The generic model access and the possibility to distribute models over subdomains supports long-distance collaboration.

[[Link to grayBoxes Wiki]](https://github.com/dwweiss/grayBoxes/wiki)

<br>

### Content of project 

    grayBoxes
        Training, sensitivity analysis and prediction with white box, gray box and black box models

    tests
        Selected test cases

    doc
        Figures and manuals used in wiki
        

### Installation

    git clone https://github.com/dwweiss/grayBoxes.git
    sudo python3 setup.py install `

### Dependencies

- Modules _LightGray.py_ and _Minimum.py_ are dependent on package _modestga_ [[MGA18]](https://github.com/dwweiss/grayBoxes/wiki/References#mga18)
- Module _Neural.py_ is dependent on package _neurolab_ [[NLB15]](https://github.com/dwweiss/grayBoxes/wiki/References#nlb15)
