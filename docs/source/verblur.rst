.. _sec_validate_blur:

Validate Blur Model
===================

* We must ensure that the estimated parameters are indeed a good fit for the measured data. This is done by comparing line profiles across the sharp edge between the measured radiograph and the predicted radiograph from the blur model. The output of the blur model given parameters of source blur, detector blur, and transmission function is computed using the function :meth:`pysaber.get_trans_fit`. The fit must be evaluated for every sharp-edge radiograph that is input to :meth:`pysaber.estimate_blur`.

* Verify the agreement between the measured radiograph and the blur model prediction. Carefully zoom into the region containing the sharp edge and verify if the predicted blur matches with the blur in the measured radiograph. Also, verify the agreement between the measured radiograph values and blur model prediction in regions further away from the sharp edge. The predicted radiograph that is output by the blur model contains additional padding. Hence, it is necessary to account for this padding when comparing with the measured radiograph.

* If the fit is not tight, consider reducing the value of the input argument :attr:`thresh` of the function :meth:`pysaber.estimate_blur` to obtain a better fit. Reducing the convergence threshold, :attr:`thresh`, can improve the agreement between the measured radiograph and the blur model prediction, but will inevitably result in longer run times. A good fit indicates that the blur model is able to accurately model the X-ray source and detector blurs. 

* Example python scripts for line profile comparisons between the blur model prediction and measured radiograph are shown below. To obtain the data that is required to run this script, download and unzip the zip file at the link :download:`data <../../demo/data.zip>`. To run the script as is within the current working directory, the files in the zip file must be placed within a folder called ``data``.

.. literalinclude:: ../../demo/plot_horz_fit.py
   :caption: Line profile comparisons across a horizontal edge radiograph.

.. literalinclude:: ../../demo/plot_vert_fit.py
   :caption: Line profile comparisons across a vertical edge radiograph.
