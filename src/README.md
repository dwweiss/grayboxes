### Abstract

Data-driven modelling is gaining interest due to the availibility of machine learning tools such as TensorFlow, Keras etc. The success of data-driven tools is bound to the availibility of large data sets. In contrast, multi-purpose packages employing finite element or finite volume methods are not dependent on data. They are often limitated in reproducing the behaviour of the real process. This is especially true if the process to be modelled has chaotic elements as for instance in welding.

A remedy is the combination of theoretical (white box) with empirical (black box) models. Such gray box models lower the expenses of theoretical model development and can be calibrated to observed data, however it is still difficult to estimate the split in the share of the empirical submodel relative to the theoretical one. The availability of such an estimate is essential for evaluation of the model reliability and its further development.

_grayboxes_ contributes to the creation and evaluation of combined models. A Python framework for implementation of white box, gray box and black box models is provided. The generic model access and the possibility to distribute models over subdomains supports long-distance collaboration.
