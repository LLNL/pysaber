Visualize Blur PSF
==================

* For further analysis and visualization, we can also compute the point spread functions (PSF) of source blur and detector blur. The PSF of source blur is computed using the function :meth:`pysaber.get_source_psf` and PSF of detector blur is computed using :meth:`pysaber.get_detector_psf`.

* The function :meth:`pysaber.get_source_psf` is useful to compute PSF either in the plane of the X-ray source or the plane of the detector. Since source blur PSF on the detector plane is a function of the object's source to object distance (SOD) and object to detector distance (ODD), SOD and ODD must be specified when computing source PSF in the plane of the detector. To compute source blur PSF in the source plane, it is sufficient to use the default values for SOD and ODD in :meth:`pysaber.get_source_psf`. 

* The detector blur PSF obtained using :meth:`pysaber.get_detector_psf` models blur due to the scintillator and detector panel. Hence, it is independent of SOD and ODD. 

* Example python scripts that demonstrate visualization of source and detector PSFs are shown below.

.. literalinclude:: ../../demo/plot_source_psf.py
    :caption: Plot X-ray source blur PSF

.. literalinclude:: ../../demo/plot_detector_psf.py
    :caption: Plot X-ray detector blur PSF
