import numpy as np
from pysaber import least_squares_deblur #To deblur using regularized least squares deconvolution
import matplotlib.pyplot as plt #To display images
from PIL import Image #To read images in TIFF format

rad_file = 'data/horz_edge_25mm.tif' #Filename of artifact's radiograph
bright_file = 'data/horz_bright.tif' #Bright field image
dark_file = 'data/horz_dark.tif' #Dark field image

sdd = 71003.08 #Source to detector distance (SDD)
sod = 24751.89 #Source to object distance (SOD)
pix_wid = 0.675 #Pixel width in micrometers
reg_param = 0.001 #Regularization parameter

rad = np.asarray(Image.open(rad_file)) #Read radiograph and convert to numpy array
bright = np.asarray(Image.open(bright_file)) #Read bright field image and convert to numpy array
dark = np.asarray(Image.open(dark_file)) #Read dark field image and convert to numpy array
norm_rad = (rad-dark)/(bright-dark) #Normalize radiograph

#X-ray source blur parameters
src_params = {}
src_params['source_FWHM_x_axis'] = 2.70 #Full width half maximum (FWHM) along x-axis (row-wise)
src_params['source_FWHM_y_axis'] = 2.82 #FWHM along y-axis (column-wise)
src_params['cutoff_FWHM_multiplier'] = 10 #PSF is non-zero at upto 10 times the FWHM and clipped to zero outside

#Detector blur parameters
det_params = {}
det_params['detector_FWHM_1'] = 2.05 #FWHM of first exponential
det_params['detector_FWHM_2'] = 120.72 #FWHM of second exponential
det_params['detector_weight_1'] = 0.917 #Weight for first exponential (mixture parameter)
det_params['cutoff_FWHM_1_multiplier'] = 10 
det_params['cutoff_FWHM_2_multiplier'] = 8 

#Deblur the artifact using regularized least squares deconvolution
rlsd_rad = least_squares_deblur(norm_rad,sod,sdd,pix_wid,src_params,det_params,reg_param)

#Display deblurred radiograph of artifact
mag = (sdd-sod)/sod
sz = rlsd_rad.shape
x = np.arange(-(sz[1]//2),(sz[1]//2)+1,1)*pix_wid/mag
y = np.arange(-(sz[0]//2),(sz[0]//2)+1,1)*pix_wid/mag
plt.pcolormesh(x,y,rlsd_rad,cmap='gray')
plt.xlabel('micrometers')
plt.ylabel('micrometers')
plt.title('Wiener deblur')
plt.colorbar()
plt.show()
