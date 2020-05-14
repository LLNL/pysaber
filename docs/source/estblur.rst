Estimate Blur Model
===================

* To estimate the blur model, we must acquire radiographs of a sharp edge such as a Tungsten rollbar. Each sharp edge radiograph can either contain a single straight edge or two mutually perpendicular intersecting edges. If imaging a single straight edge, then radiographs must be acquired at two different perpendicular orientations of the straight edge. Also, radiographs must be acquired at two different values of SOD/ODD, where SOD is the source to object distance and ODD is the object to detector distance.

* Next, the radiographs must be appropriately normalized. For each radiograph, acquire a bright field image (measurements with X-rays but no sample) and a dark field image (measurements without X-rays). Then, compute the normalized radiograph by dividing the difference between the radiograph and the dark field image with the difference between the bright field and the dark field image.

* Using the normalized radiographs, estimate parameters of X-ray source blur and detector blur using the function :meth:`pysaber.estimate_blur`.

* An example python script that demonstrates blur estimation using radiographs of a single straight edge at various orientations and SOD/ODD values is shown below. To obtain the data that is required to run this script, download and unzip the zip file at the link :download:`data <../../demo/data.zip>`. To run the script as is within the current working directory, the files in the zip file must be placed within a folder called ``data``.

.. literalinclude:: ../../demo/fit_blur_model.py
