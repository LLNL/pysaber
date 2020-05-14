Input Sanity Check
==================

* The first step to estimating the blur model involves computation of the transmission function, which is the ideal radiograph image that is formed in the absence of X-ray source and detector blurs. This computation of transmission function is performed internally in the function :meth:`pysaber.estimate_blur`, which is used to estimate the blur model by computing the parameters of X-ray source and detector blurs. However, this computation of transmission function is not fail-proof and may result in inaccurate edge localization if certain assumptions made when computing the transmission function are not satisfied.

* Before using :meth:`pysaber.estimate_blur` to estimate blur model, it is recommended to check for accurate edge localization in the transmission function. The transmission function can be computed using the function :meth:`pysaber.get_trans_masks`. 

* The function :meth:`pysaber.get_trans_masks` also returns the mask arrays for the transmission function and radiograph, which are used to include or exclude certain pixels from blur estimation. By default, the radiograph mask only excludes a small number of pixels along the boundary of the radiograph from blur estimation. Additional pixels can be excluded from blur estimation by appropriately setting the input arguments of the functions :meth:`pysaber.get_trans_masks` and :meth:`pysaber.estimate_blur`. The mask for transmission function should also exclude the padded pixels in addition to those pixels excluded by the radiograph mask. Hence, :meth:`pysaber.get_trans_masks` is also useful to check if user expectations for the mask arrays are satisfied.

* Example python scripts that demonstrate the above procedure are shown below. To obtain the data that is required to run this script, download and unzip the zip file at the link :download:`data <../../demo/data.zip>`. To run the script as is within the current working directory, the files in the zip file must be placed within a folder called ``data``.

.. literalinclude:: ../../demo/chk_horz_inp.py
   :caption: Verify transmission function and masks for horizontal edge radiograph.

.. literalinclude:: ../../demo/chk_vert_inp.py
   :caption: Verify transmission function and masks for vertical edge radiograph.
