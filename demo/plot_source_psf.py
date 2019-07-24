import numpy as np #For mathematics on vectors
import matplotlib.pyplot as plt #For plotting and showing images
from saber import get_source_psf #To compute PSF of source blur 

pix_wid = 0.675 #Width of each pixel in micrometers
#Parameters of X-ray source blur
#For information on each parameter, run help(get_source_psf) after importing get_source_psf
src_params = {'source_FWHM_x_axis':2.70, 'source_FWHM_y_axis':2.82, 'cutoff_FWHM_multiplier':10}

#Get point spread function (PSF) of source blur in the plane of the X-ray source as a 2D numpy array. Do not supply SOD and SDD if you need PSF in the source plane.
source_psf = get_source_psf(pix_wid,src_params)

#Display the PSF at source plane as an image
sz = source_psf.shape
x = np.arange(-(sz[1]//2),(sz[1]//2)+1,1)*pix_wid
y = np.arange(-(sz[0]//2),(sz[0]//2)+1,1)*pix_wid
plt.pcolormesh(x,y,source_psf,cmap='gray')
plt.xlabel('micrometers')
plt.ylabel('micrometers')
plt.title('X-ray source PSF at source plane')
plt.colorbar()
plt.show()

#To get PSF on the detector plane for a given SOD and SDD, supply SOD and SDD to the function get_source_psf
sod = 25000
sdd = 71000 
source_psf = get_source_psf(pix_wid,src_params,sod,sdd)
#help(get_source_psf)
#Uncomment the above line to get help on using the function get_source_psf

#Display the PSF at detector plane as an image
sz = source_psf.shape
x = np.arange(-(sz[1]//2),(sz[1]//2)+1,1)*pix_wid
y = np.arange(-(sz[0]//2),(sz[0]//2)+1,1)*pix_wid
plt.pcolormesh(x,y,source_psf,cmap='gray')
plt.xlabel('micrometers')
plt.ylabel('micrometers')
plt.title('X-ray source PSF at detector plane')
plt.colorbar()
plt.show()

