# grayBoxes

Data-driven modelling is gaining interest due to the availibility of machine learning tools such as TensorFlow, Keras etc. The success of data-driven tools is bound to the availibility of large data sets. In contrast, multi-purpose packages employing finite element or finite volume methods are not dependent on data. They are often limitated in reproducing the behaviour of the real process. This is especially true if the process to be modelled has chaotic elements as for instance in welding.

A remedy is the combination of theoretical with empirical models. Such hybrid models lower the expenses of theoretical model development and can be calibrated to given experimental sets, however it is still difficult to estimate the split in the share of the empirical submodel relative to the theoretical one. The availability of such an estimate is essential for evaluation of the model reliability and its further development.

_grayBoxes_ contributes to the creation and evaluation of hybrid models. A Python framework for implementation of empirical, theoretical and hybrid models is provided. The generic model access and the possibility to distribute models over subdomains supports long-distance collaboration.



## Table of Contents 

    src 
        Training, sensitivity analysis and execution of white box, gray box and black box models

    src/tests
        Selected test cases

    doc/fig
        Figures and manuals used in wiki
        

## Installation

    $ wget https://github.com/dwweiss/grayBoxes.git
or

    $ git clone https://github.com/dwweiss/grayBoxes.git


## Dependency

Neural.py is dependent on `neurolab`, install with `sudo pip3 install neurolab` or:

    1) Download neurolab.0.x.y.tar.gz from: https://pypi.python.org/pypi/neurolab
    2) Change to download directory                            
    3) python -m pip install ./neurolab.0.x.y.tar.gz
