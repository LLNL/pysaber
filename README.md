# PySABER
PySABER is a python package for characterizing the X-ray source and detector blur in cone-beam X-ray imaging systems. SABER is an abbreviation for systems approach to blur estimation and reduction. Note that even parallel beam X-rays in synchrotrons are in fact cone beams albeit with a large source to object distance. X-ray images, also called radiographs, are simultaneously blurred by both the X-ray source spot blur and detector blur. This package uses a numerical optimization algorithm to disentangle and estimate both forms of blur simultaneously.The point spread function (PSF) of X-ray source blur is modeled using an exponential density function with two parameters. The first parameter is the full width half maximum (FWHM) of the PSF along the x-axis (row-wise) and second is the FWHM along the y-axis (column-axis). The PSF of detector blur is modeled as the sum of two exponential density functions, each with its own FWHM parameter, that is mixed together by a mixture (or weighting) parameter. All these parameters are then estimated using numerical optimization from normalized radiographs of a sharp edge such as a thick Tungsten plate rollbar. It is recommended to acquire radiographs of the sharp edge at two different mutually perpendicular orientations and also repeat this process at two different values of the ratio of source to object distance (SOD) and object to detector distance (ODD). Once the parameters of both source and detector blurs are estimated, this package is also useful to reduce blur in radiographs using deblurring algorithms. Currently, Wiener filtering and regularized least squares deconvolution are two deblurring algorithms that are supported for deblurring. Both these techniques use the estimated blur parameters to deblur radiographs. The paper listed in the below reference section contains more information on the theory behind this package package. If you find this package useful, please cite the paper referenced below in your publications.


## References
K. Aditya Mohan, Robert M. Panas, and Jefferson A. Cuadra. "SABER: A Systems Approach to Blur Estimation and Reduction in X-ray Imaging." arXiv preprint arXiv:1905.03935 (2019) [pdf](https://arxiv.org/pdf/1905.03935.pdf)


## License
This project is licensed under the MIT License. LLNL-CODE-766837.


## Dependencies
```
python 3.*
numpy
pyyaml
scipy
scikit-image
scikit-learn
matplotlib
```
The python packages in the above dependency list will be installed automatically if you follow the procedure outlined in the installation section below. 

## Installation
`pysaber` is installed using the python package manager pip.
To install `pysaber`, run the following command in a terminal, 
```bash
    pip install pysaber
```
Alternatively, to install using the source code in this repository, first download this repository using the download link in the top right corner on this webpage. Or, you can also git clone this repository directly from github. In a terminal, change the current directory to the outermost folder of this downloaded repository that contains this README and run the following command -
```bash
    pip install .
```
It is recommended to install `pysaber` within a python virtual environment.


## Usage
This python package has two useful functionalities. First, it can extract the PSFs of X-ray source and detector blurs from calibration data consisting of radiographs of a sharp edge. Second, it uses the estimated PSFs to deblur radiograph of any arbitrary sample acquired at any source to object distance (SOD) and source to detector distance (SDD). It is recommended to read the paper in the above reference section before using this python package.
### Estimate Blur PSFs
* The steps involved in estimating blur PSFs are outlined below. The example python script [fit_blur_model.py](https://github.com/sabersw/pysaber-demo/blob/master/fit_blur_model.py) demonstrates estimation of parameters of blur PSFs from radiographs of a Tungsten sharp edge rollbar.
    * Acquire radiographs of a straight sharp edge such as a Tungsten edge rollbar. Radiographs must be acquired at two different perpendicular orientations and at two different values of SOD/ODD.
    * Normalize each radiograph. For each radiograph, acquire a bright field image (measurements with X-rays but no sample) and a dark field image (measurements without X-rays). Then, compute the normalized radiograph by dividing the difference between the radiograph and the dark field image with the difference between the bright field and the dark field image.
    * Using the normalized radiographs, estimate parameters of X-ray source blur and detector blur using the function `pysaber.get_blur_params`.
* Next, ensure that the estimated parameters are indeed a good fit for the measured data. This is done by comparing line profiles across the sharp edge between the measured radiograph and the prediction from the blur model output. The output of the blur model given parameters of source and detector blurs is given by the function `pysaber.get_trans_fit`. Carefully zoom into the region where the edge lies and verify if the predicted blur matches with the blur in the measured radiograph. The python scripts [plot_horz_fit.py](https://github.com/sabersw/pysaber-demo/blob/master/plot_horz_fit.py) and [plot_vert_fit.py](https://github.com/sabersw/pysaber-demo/blob/master/plot_vert_fit.py) show examples of such comparisons. If the fit is not tight, consider reducing the value of the input argument `convg_thresh` of the function `pysaber.get_blur_params` to obtain a better fit.
* Lastly, save or visualize the PSF of source and detector blurs. The PSF of source blur is given by the function `pysaber.get_source_psf` and PSF of detector blur is given by `pysaber.get_detector_psf`. The example python scripts [plot_source_psf.py](https://github.com/sabersw/pysaber-demo/blob/master/plot_source_psf.py) and [plot_detector_psf.py](https://github.com/sabersw/pysaber-demo/blob/master/plot_detector_psf.py) display source and detector PSFs as images.
### Deblur Radiographs
Once the parameters of source and detector PSFs are estimated, radiographs of any arbitrary sample acquired at any source to object distance (SOD) and source to detector distance (SDD) can be deblurred using various techniques. To deblur a radiograph using Wiener filtering, the function `pysaber.wiener_deblur` is used. To deblur using regularized least squares deconvolution (RLSD), use the function `pysaber.least_squares_deblur`. The python scripts [deblur_wiener.py](https://github.com/sabersw/pysaber-demo/blob/master/deblur_wiener.py) and [deblur_rlsd.py](https://github.com/sabersw/pysaber-demo/blob/master/deblur_rlsd.py) are examples that demonstrate radiograph deblur. 


## Functions
This section describes the various functions available in this package along with the corresponding input arguments and return values. The information in this section can also be obtained using the python help function. For example, to get help in using the function `pysaber.get_blur_params`, run the following lines in python -
```python
from pysaber import get_blur_params #Import the function get_blur_params
help(get_blur_params) #To learn more about pysaber.get_blur_params
```

#### pysaber.get_blur_params
```
get_blur_params(norm_rads, sod, sdd, pix_wid, convg_thresh=1e-06, bdary_mask_perc=5, pad_factor=[3, 3], mask=None, edge_type=None)
    Estimate parameters of point spread functions (PSF) that model X-ray source blur and detector blur from normalized radiographs of a straight sharp edge or two mutually perpendicular sharp edges. 
    
    This function is used to estimate parameters of the PSFs that model X-ray source blur and detector blur. It takes as input the normalized radiographs at multiple source to object distances (SOD) and source to detector distances (SDD). If each radiograph has a single straight edge, then the measurement must be repeated for two different, preferably perpendicular, orientations of the edge. Currently, only a single straight edge in a radiograph is verified to work. The case of perpendicular edges in a radiograph will be supported soon. For more details, please refer to the document listed in the references.  
    Parameters:
        norm_rads (list): Python list of normalized radiographs, each of type numpy.ndarray, at various SODs and SDDs.
        sod (list): List of source to object distances (SOD), each of type float, at which each normalized radiograph in the list norm_rads was acquired.  
        sdd (list): List of source to detector distances (SDD), each of type float, at which each normalized radiograph in the list norm_rads was acquired.
        pix_wid (float): Effective width of each detector pixel. Note that this is the effective pixel size given by dividing the physical width of each detector pixel by the zoom factor of the optical lens.
        convg_thresh (float): Convergence threshold for the minimizer in the last final step when estimating source, detector, and transmission parameters. The convergence threshold for the minimizer during the first two initialization steps is ten times this value. The iterations stop when the ratio of the reduction in the error function (cost value) and the magnitude of the error function is lower than convg_thresh. This is the parameter ftol that is specified in the options parameter of scipy.optimize.minimize. The optimizer used is L-BFGS-B.
        bdary_mask_perc (float): Percentage of image region in the normalized radiographs that must be excluded from blur estimation. Pixels are excluded (or masked) beginning from the outermost periphery of the image and working inwards until the specified percentage of pixels is reached.
        pad_factor (list): Pad factor is a list of two integers that determine the amount of padding that must be applied to the radiographs to reduce aliasing during convolution. The number of rows/columns after padding is equal to pad_factor[0]/pad_factor[1] times the number of rows/columns in each normalized radiograph before padding. For example, if the first element in pad_factor is 2, then the radiograph is padded to twice its size along the first dimension.  
        mask (numpy.ndarray): Boolean mask of the same shape as the radiograph that is used to exclude pixels from blur estimation. An example use case is if some pixels in the radiographs are bad, then those pixels can be excluded from blur estimation by setting the corresponding entries in mask to false and true otherwise.
        edge_type (str): Used to indicate whether there is a single straight edge or two mutually perpendicular edges in each radiograph. If edge_type is perpendicular, then each radiograph is assumed to have two mutually perpendicular edges and a single straight edge otherwise. Currently, choosing perpendicular as edge_type is not recommended since it isn't a verified functionality and may lead to unstable behavior.  
    
    Returns:
        dict: Estimated parameters of X-ray source PSF that is returned as a python dictionary. It consists of several key-value pairs. The value for key source_FWHM_x_axis is the full width half maximum (FWHM) of the source PSF along the x-axis (i.e., second array dimension). The value for key source_FWHM_y_axis is the FWHM of source PSF along the y-axis (i.e., first array dimension). All FWHMs are for the source PSF in the plane of the X-ray source (and not the plane of the detector). The value for key cutoff_FWHM_multiplier decides the non-zero spatial extent of the exponential PSF. The PSF is clipped to zero beginning at a distance, as measured from the PSF's origin, of cutoff_FWHM_multiplier times half the maximum FWHM (maximum of source_FWHM_x_axis and source_FWHM_y_axis).
        dict: Estimated parameters of detector PSF that is returned as a python dictionary. It consists of several key-value pairs. The value for key detector_FWHM_1 is the FWHM of the first exponential in the mixture density model for detector blur. The first exponential is the most dominant part of detector blur. The value for key detector_FWHM_2 is the FWHM of the second exponential in the mixture density model. This exponential has the largest FWHM and models the long running tails of the detector blur's PSF. The value for key detector_weight_1 is between 0 and 1 and is an approximate measure of the amount of contribution of the first exponential to the detector blur. The values for keys cutoff_FWHM_1_multiplier and cutoff_FWHM_2_multiplier decide the non-zero spatial extent of the detector PSF. The PSF is clipped to zero beginning at a distance, as measured from the PSF's origin, of the maximum of cutoff_FWHM_1_multiplier*detector_FWHM_1/2 and cutoff_FWHM_2_multiplier*detector_FWHM_2/2.
        list: Estimated parameters of the transmission function for each input radiograph. This return value is a list of lists, where each inner nested list consists of two parameters of type float. These float values give the low and high values respectively of the transmission function. The number of nested lists in the returned list equals the number of input radiographs. Note that the transmission function is the normalized radiograph image that would have resulted in the absence of blur and noise.
```

#### pysaber.get_trans_fit
```
get_trans_fit(norm_rad, sod, sdd, pix_wid, src_params, det_params, trans_params, pad_factor=[3, 3], edge_type=None)
    For a measured radiograph consisting of a straight sharp edge or two mutually perpendicular edges, get the ideal transmission function and a prediction from the blur model for the normalized radiograph in the presence of X-ray source and detector blurs. 
    
    Parameters:
        norm_rad (numpy.ndarray): Normalized radiograph
        sod (float): Source to object distance (SOD) for the radiograph norm_rad.
        sdd (float): Source to detector distance (SDD) for the radiograph norm_rad.
        pix_wid (float): Effective width of each detector pixel. Note that this is the effective pixel size given by dividing the physical width of each detector pixel by the zoom factor of the optical lens.
        src_params (dict): Estimated parameters of X-ray source PSF. It should consist of several key-value pairs. The value for key source_FWHM_x_axis is the full width half maximum (FWHM) of the source PSF along the x-axis (i.e., second array dimension). The value for key source_FWHM_y_axis is the FWHM of source PSF along the y-axis (i.e., first array dimension). All FWHMs are for the source PSF in the plane of the X-ray source (and not the detector plane). The value for key cutoff_FWHM_multiplier decides the non-zero spatial extent of the source blur PSF. The PSF is clipped to zero beginning at a distance, as measured from the PSF's origin, of cutoff_FWHM_multiplier times half the maximum FWHM (maximum of source_FWHM_x_axis and source_FWHM_y_axis).
        det_params (dict): Estimated parameters of detector blur PSF. It should consist of several key-value pairs. The value for key detector_FWHM_1 is the FWHM of the first exponential in the mixture density model for detector blur. The first exponential is the most dominant part of detector blur. The value for key detector_FWHM_2 is the FWHM of the second exponential in the mixture density model. This exponential has the largest FWHM and models the long running tails of the detector blur's point spread function (PSF). The value for key detector_weight_1 is between 0 and 1 and is an approximate measure of the amount of contribution of the first exponential to the detector blur. The values for keys cutoff_FWHM_1_multiplier and cutoff_FWHM_2_multiplier decide the non-zero spatial extent of the detector PSF. The PSF is clipped to zero beginning at a distance, as measured from the PSF's origin, of the maximum of cutoff_FWHM_1_multiplier*detector_FWHM_1/2 and cutoff_FWHM_2_multiplier*detector_FWHM_2/2.
        trans_params (list): A two element integer list of transmission function parameters. The first element is the low value and the second element is the high value of the transmission function. Note that the transmission func
        pad_factor (list): Pad factor is a list of two integers that determine the amount of padding that must be applied to the radiographs to reduce aliasing during convolution. The number of rows/columns after padding is equal to pad_factor[0]/pad_factor[1] times the number of rows/columns in each normalized radiograph before padding. For example, if the first element in pad_factor is 2, then the radiograph is padded to twice its size along the first dimension.  
        edge_type (str): Used to indicate whether there is a single straight edge or two mutually perpendicular edges in each radiograph. If edge_type is perpendicular, then each radiograph is assumed to have two mutually perpendicular edges and a single straight edge otherwise. Currently, choosing perpendicular as edge_type is not recommended since it isn't a verified functionality and may lead to unstable behavior.
```

#### pysaber.get_source_psf
```
get_source_psf(pix_wid, src_params, sod=1, sdd=2)
    Get point spread function (PSF) of X-ray source blur.
    
    Parameters:
        pix_wid (float): Effective width of each detector pixel. Note that this is the effective pixel size given by dividing the physical width of each detector pixel by the zoom factor of the optical lens.
        src_params (dict): Estimated parameters of X-ray source PSF. It should consist of several key-value pairs. The value for key source_FWHM_x_axis is the full width half maximum (FWHM) of the source PSF along the x-axis (i.e., second array dimension). The value for key source_FWHM_y_axis is the FWHM of source PSF along the y-axis (i.e., first array dimension). All FWHMs are for the source PSF in the plane of the X-ray source (and not the detector plane). The value for key cutoff_FWHM_multiplier decides the non-zero spatial extent of the source blur PSF. The PSF is clipped to zero beginning at a distance, as measured from the PSF's origin, of cutoff_FWHM_multiplier times half the maximum FWHM (maximum of source_FWHM_x_axis and source_FWHM_y_axis).
        sod (float): Source to object distance (SOD) is necessary to compute PSF of source blur in the plane of the detector. Use the default value to compute source PSF at the plane of the X-ray source. Or, if sdd = 2*sod, then source blur PSF at the detector is same as the source blur PSF at the plane of the X-ray source. 
        sdd (float): Source to detector distance (SDD) is necessary to compute PSF of source blur in the plane of the detector. Use the default value to compute source PSF at the plane of the X-ray source. Or, if sdd = 2*sod, then source blur PSF at the detector is same as the source blur PSF at the plane of the X-ray source.
    
    Returns:
        numpy.ndarray: 2D array of source blur PSF
```

#### pysaber.get_detector_psf
```
get_detector_psf(pix_wid, det_params)
    Get point spread function (PSF) of detector blur
    
    Parameters:
        pix_wid (float): Effective width of each detector pixel. Note that this is the effective pixel size given by dividing the physical width of each detector pixel by the zoom factor of the optical lens.
        det_params (dict): Estimated parameters of detector blur PSF. It should consist of several key-value pairs. The value for key detector_FWHM_1 is the FWHM of the first exponential in the mixture density model for detector blur. The first exponential is the most dominant part of detector blur. The value for key detector_FWHM_2 is the FWHM of the second exponential in the mixture density model. This exponential has the largest FWHM and models the long running tails of the detector blur's point spread function (PSF). The value for key detector_weight_1 is between 0 and 1 and is an approximate measure of the amount of contribution of the first exponential to the detector blur. The values for keys cutoff_FWHM_1_multiplier and cutoff_FWHM_2_multiplier decide the non-zero spatial extent of the detector PSF. The PSF is clipped to zero beginning at a distance, as measured from the PSF's origin, of the maximum of cutoff_FWHM_1_multiplier*detector_FWHM_1/2 and cutoff_FWHM_2_multiplier*detector_FWHM_2/2.
    
    Returns:
        numpy.ndarray: 2D array of detector blur PSF
```

#### pysaber.get_effective_psf
```
get_effective_psf(pix_wid, src_params, det_params, sod=1, sdd=2)
    Get point spread function (PSF) of the combined effect of X-ray source and detector blur.
    
    Parameters:
        pix_wid (float): Effective width of each detector pixel. Note that this is the effective pixel size given by dividing the physical width of each detector pixel by the zoom factor of the optical lens.
        src_params (dict): Estimated parameters of X-ray source PSF. It should consist of several key-value pairs. The value for key source_FWHM_x_axis is the full width half maximum (FWHM) of the source PSF along the x-axis (i.e., second array dimension). The value for key source_FWHM_y_axis is the FWHM of source PSF along the y-axis (i.e., first array dimension). All FWHMs are for the source PSF in the plane of the X-ray source (and not the detector plane). The value for key cutoff_FWHM_multiplier decides the non-zero spatial extent of the source blur PSF. The PSF is clipped to zero beginning at a distance, as measured from the PSF's origin, of cutoff_FWHM_multiplier times half the maximum FWHM (maximum of source_FWHM_x_axis and source_FWHM_y_axis).
        det_params (dict): Estimated parameters of detector blur PSF. It should consist of several key-value pairs. The value for key detector_FWHM_1 is the FWHM of the first exponential in the mixture density model for detector blur. The first exponential is the most dominant part of detector blur. The value for key detector_FWHM_2 is the FWHM of the second exponential in the mixture density model. This exponential has the largest FWHM and models the long running tails of the detector blur's point spread function (PSF). The value for key detector_weight_1 is between 0 and 1 and is an approximate measure of the amount of contribution of the first exponential to the detector blur. The values for keys cutoff_FWHM_1_multiplier and cutoff_FWHM_2_multiplier decide the non-zero spatial extent of the detector PSF. The PSF is clipped to zero beginning at a distance, as measured from the PSF's origin, of the maximum of cutoff_FWHM_1_multiplier*detector_FWHM_1/2 and cutoff_FWHM_2_multiplier*detector_FWHM_2/2.
        sod (float): Source to object distance (SOD) is necessary to compute PSF of source blur in the plane of the detector. Use the default value to compute source PSF at the plane of the X-ray source. Or, if sdd = 2*sod, then source blur PSF at the detector is same as the source blur PSF at the plane of the X-ray source. 
        sdd (float): Source to detector distance (SDD) is necessary to compute PSF of source blur in the plane of the detector. Use the default value to compute source PSF at the plane of the X-ray source. Or, if sdd = 2*sod, then source blur PSF at the detector is same as the source blur PSF at the plane of the X-ray source.
    
    Returns:
        numpy.ndarray: 2D array of effective blur PSF
```

#### pysaber.wiener_deblur
```
wiener_deblur(norm_rad, sod, sdd, pix_wid, src_params, det_params, reg_param)
    Function to reduce blur (deblur) in a radiograph using Wiener filtering.     
    
    Parameters:
        norm_rad (numpy.ndarray): Normalized radiograph to deblur
        sod (float): Source to object distance (SOD) of the radiograph
        sdd (float): Source to detector distance (SDD) of the radiograph
        pix_wid (float): Effective width of each detector pixel. Note that this is the effective pixel size given by dividing the physical width of each detector pixel by the zoom factor of the optical lens.
        src_params (dict): Estimated parameters of X-ray source PSF. It should consist of several key-value pairs. The value for key source_FWHM_x_axis is the full width half maximum (FWHM) of the source PSF along the x-axis (i.e., second array dimension). The value for key source_FWHM_y_axis is the FWHM of source PSF along the y-axis (i.e., first array dimension). All FWHMs are for the source PSF in the plane of the X-ray source (and not the detector plane). The value for key cutoff_FWHM_multiplier decides the non-zero spatial extent of the source blur PSF. The PSF is clipped to zero beginning at a distance, as measured from the PSF's origin, of cutoff_FWHM_multiplier times half the maximum FWHM (maximum of source_FWHM_x_axis and source_FWHM_y_axis).
        det_params (dict): Estimated parameters of detector blur PSF. It should consist of several key-value pairs. The value for key detector_FWHM_1 is the FWHM of the first exponential in the mixture density model for detector blur. The first exponential is the most dominant part of detector blur. The value for key detector_FWHM_2 is the FWHM of the second exponential in the mixture density model. This exponential has the largest FWHM and models the long running tails of the detector blur's point spread function (PSF). The value for key detector_weight_1 is between 0 and 1 and is an approximate measure of the amount of contribution of the first exponential to the detector blur. The values for keys cutoff_FWHM_1_multiplier and cutoff_FWHM_2_multiplier decide the non-zero spatial extent of the detector PSF. The PSF is clipped to zero beginning at a distance, as measured from the PSF's origin, of the maximum of cutoff_FWHM_1_multiplier*detector_FWHM_1/2 and cutoff_FWHM_2_multiplier*detector_FWHM_2/2.
        reg_param (float): Regularization parameter for Wiener filtering deblurring algorithm. Noise and ringing artifacts in the deblurred radiograph decreases with increasing values for reg_param and vice versa. But, note that increasing reg_param can also result in excessive blurring due to over-regulization. It is recommended to empirically choose this parameter by increasing or decreasing it by a factor, greater than one (such as 2 or 10), until the desired image quality is achieved.
    
    Returns:
        numpy.ndarray: Deblurred radiograph using a Wiener filter.
```

#### pysaber.least_squares_deblur
```
least_squares_deblur(norm_rad, sod, sdd, pix_wid, src_params, det_params, reg_param, init_rad=None, weights=None, convg_thresh=0.01)
    Function to reduce blur (deblur) in radiographs using a regularized least squares iterative algorithm.     
    
    Parameters:
        norm_rad (numpy.ndarray): Normalized radiograph to deblur
        sod (float): Source to object distance (SOD) of the radiograph
        sdd (float): Source to detector distance (SDD) of the radiograph
        pix_wid (float): Effective width of each detector pixel. Note that this is the effective pixel size given by dividing the physical width of each detector pixel by the zoom factor of the optical lens.
        src_params (dict): Estimated parameters of X-ray source PSF. It should consist of several key-value pairs. The value for key source_FWHM_x_axis is the full width half maximum (FWHM) of the source PSF along the x-axis (i.e., second array dimension). The value for key source_FWHM_y_axis is the FWHM of source PSF along the y-axis (i.e., first array dimension). All FWHMs are for the source PSF in the plane of the X-ray source (and not the detector plane). The value for key cutoff_FWHM_multiplier decides the non-zero spatial extent of the source blur PSF. The PSF is clipped to zero beginning at a distance, as measured from the PSF's origin, of cutoff_FWHM_multiplier times half the maximum FWHM (maximum of source_FWHM_x_axis and source_FWHM_y_axis).
        det_params (dict): Estimated parameters of detector blur PSF. It should consist of several key-value pairs. The value for key detector_FWHM_1 is the FWHM of the first exponential in the mixture density model for detector blur. The first exponential is the most dominant part of detector blur. The value for key detector_FWHM_2 is the FWHM of the second exponential in the mixture density model. This exponential has the largest FWHM and models the long running tails of the detector blur's point spread function (PSF). The value for key detector_weight_1 is between 0 and 1 and is an approximate measure of the amount of contribution of the first exponential to the detector blur. The values for keys cutoff_FWHM_1_multiplier and cutoff_FWHM_2_multiplier decide the non-zero spatial extent of the detector PSF. The PSF is clipped to zero beginning at a distance, as measured from the PSF's origin, of the maximum of cutoff_FWHM_1_multiplier*detector_FWHM_1/2 and cutoff_FWHM_2_multiplier*detector_FWHM_2/2.
        reg_param (float): Regularization parameter for the least squares deblurring algorithm. Noise and ringing artifacts in the deblurred radiograph decreases with increasing values for reg_param and vice versa. But, note that increasing reg_param can also result in excessive blurring due to over-regulization. It is recommended to empirically choose this parameter by increasing or decreasing it by a factor, greater than one (such as 2 or 10), until the desired image quality is achieved.
        init_rad (numpy.ndarray): Initial estimate for the deblurred radiograph. If set to None, then the blurred radiograph is used as an initial estimate. If not None, then init_rad must be a numpy.ndarray of the same shape as norm_rad. 
        weights (numpy.ndarray): Used to increase or decrease reliance on a particular pixel of norm_rad in the forward model cost function. If set to None, every pixel is assigned the same weight of 1.
        convg_thresh (float): Convergence threshold for the minimizer used to deblur the input radiograph. The iterations stop when the ratio of the reduction in the error function (cost value) and the magnitude of the error function is lower than convg_thresh. This is the parameter ftol that is specified in the options parameter of scipy.optimize.minimize. The optimizer used is L-BFGS-B.
    
    Returns:
        numpy.ndarray: Deblurred radiograph using regularized least squares algorithm.
```

## Authors
* [K. Aditya Mohan](https://github.com/adityamnk)


## Acknowledgements
The following people contributed to the theoretical formulation and experimental validation of SABER -
* Robert M. Panas
* Jefferson A. Cuadra


## Bug Reports & Feedback
This software is under development and may have bugs. If you run into any problems, please raise a issue on github. There is a lot of scope to improve the performance and functionality of this python package. Furthermore, since this package solves a non-convex optimization problem, there is a remote possibility that the final solution may be a local optima that does not properly fit the data. If there is sufficient interest, I will invest time to significantly reduce the run time, improve convergence and usability, and add additional features and functionalities.


## Contributing
TBA
