import numpy as np #For mathematics on vectors
import matplotlib.pyplot as plt #For displaying images
from matplotlib.colors import LogNorm #To display image values in logarithm scale 
from pysaber import get_detector_psf #To compute PSF of detector blur

pix_wid = 0.675 #Width of each pixel in micrometers
#Parameters of detector blur
#For information on each parameter, run help(get_detector_psf) after importing get_detector_psf 
det_params = {'detector_FWHM_1':2.05, 'detector_FWHM_2':120.72, 'detector_weight_1':0.917, 'cutoff_FWHM_1_multiplier':10, 'cutoff_FWHM_2_multiplier':8}

#Get point spread function (PSF) of detector blur as a 2D numpy array.
detector_psf = get_detector_psf(pix_wid,det_params)
#help(get_detector_psf)
#Uncomment the above line to get help in using the function get_detector_psf


#Display the PSF of detector blur
sz = detector_psf.shape
x = np.arange(-(sz[1]//2),(sz[1]//2)+1,1)*pix_wid
y = np.arange(-(sz[0]//2),(sz[0]//2)+1,1)*pix_wid
plt.pcolormesh(x,y,detector_psf,cmap='gray',norm=LogNorm())
plt.xlabel('micrometers')
plt.ylabel('micrometers')
plt.title('Detector PSF')
plt.colorbar()
plt.show()

