Deblur Radiographs
==================

* Once the parameters of source and detector PSFs are estimated, radiographs of any arbitrary sample acquired at any source to object distance (SOD) and source to detector distance (SDD) can be deblurred using various techniques.

* To deblur a radiograph using Wiener filter, the function :meth:`pysaber.wiener_deblur` is used. To deblur using regularized least squares deconvolution (RLSD), use the function :meth:`pysaber.least_squares_deblur`.

* Deblurring increases sharpness and resolution. However, it also introduces ringing artifacts and increases noise. To reduce noise and ringing artifacts, the regularization parameter can be increased. Ringing artifacts also increase with increasing inaccuracy of the blur model. Thus, it is essential to obtain a good fit between the measured radiograph and the blur model prediction as explained in the section :ref:`sec_validate_blur`. 

* The python scripts shown below demonstrate deblurring of radiographs using Wiener filter and RLSD. To obtain the data that is required to run this script, download and unzip the zip file at the link :download:`data <../../demo/data.zip>`. To run the script as is within the current working directory, the files in the zip file must be placed within a folder called ``data``.

.. literalinclude:: ../../demo/deblur_wiener.py
    :caption: Deblurring using Wiener filter

.. literalinclude:: ../../demo/deblur_rlsd.py
    :caption: Deblurring using Regularized Least Squares Deconvolution
