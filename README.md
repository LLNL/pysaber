# PySABER

## Documentation
Documentation for PySABER is available at the link [https://pysaber.readthedocs.io/](https://pysaber.readthedocs.io/). 

## Introduction
PySABER is a python package for characterizing the X-ray source and detector blur in cone-beam X-ray imaging systems. SABER is an abbreviation for systems approach to blur estimation and reduction. Note that even parallel beam X-rays in synchrotrons are in fact cone beams albeit with a large source to object distance. X-ray images, also called radiographs, are simultaneously blurred by both the X-ray source spot blur and detector blur. This package uses a numerical optimization algorithm to disentangle and estimate both forms of blur simultaneously.The point spread function (PSF) of X-ray source blur is modeled using an exponential density function with two parameters. The first parameter is the full width half maximum (FWHM) of the PSF along the x-axis (row-wise) and second is the FWHM along the y-axis (column-axis). The PSF of detector blur is modeled as the sum of two exponential density functions, each with its own FWHM parameter, that is mixed together by a mixture (or weighting) parameter. All these parameters are then estimated using numerical optimization from normalized radiographs of a sharp edge such as a thick Tungsten plate rollbar. It is recommended to acquire radiographs of the sharp edge at two different mutually perpendicular orientations and also repeat this process at two different values of the ratio of source to object distance (SOD) and object to detector distance (ODD). Once the parameters of both source and detector blurs are estimated, this package is also useful to reduce blur in radiographs using deblurring algorithms. Currently, Wiener filtering and regularized least squares deconvolution are two deblurring algorithms that are supported for deblurring. Both these techniques use the estimated blur parameters to deblur radiographs. The paper listed in the below reference section contains more information on the theory behind this package package. If you find this package useful, please cite the paper referenced below in your publications.


## References
K. Aditya Mohan, Robert M. Panas, and Jefferson A. Cuadra. "SABER: A Systems Approach to Blur Estimation and Reduction in X-ray Imaging." arXiv preprint arXiv:1905.03935 (2019) [pdf](https://arxiv.org/pdf/1905.03935.pdf)


## License
This project is licensed under the MIT License. LLNL-CODE-766837.


## Installation and Usage
Please refer to the documentation at the link [https://pysaber.readthedocs.io/](https://pysaber.readthedocs.io/).
 

## Contributors
* [K. Aditya Mohan](https://github.com/adityamnk)
* Robert M. Panas
* Jefferson A. Cuadra


## Recent Changes
* May 2020: Moved repository from [https://github.com/sabersw/pysaber](https://github.com/sabersw/pysaber) to [https://github.com/LLNL/pysaber](https://github.com/LLNL/pysaber).
