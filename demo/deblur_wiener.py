import numpy as np
from saber import wiener_deblur #To deblur using Wiener filtering
import matplotlib.pyplot as plt #To display images
from PIL import Image #To read images in TIFF format

rad_file = 'data/horz_edge_25mm.tif' #Filename of radiograph
bright_file = 'data/horz_bright.tif' #Bright field
dark_file = 'data/horz_dark.tif' #Dark field

sdd = 71003.08 #Source to detector distance (SDD)
sod = 24751.89 #Source to object distance (SOD)
pix_wid = 0.675 #Pixel width in micrometers
reg_param = 0.1 #Regularization parameter

rad = np.asarray(Image.open(rad_file)) #Read radiograph and convert to numpy array
bright = np.asarray(Image.open(bright_file)) #Read bright field image and convert to numpy array
dark = np.asarray(Image.open(dark_file)) #Read dark field image and convert to numpy array
norm_rad = (rad-dark)/(bright-dark) #Normalize radiograph

#X-ray source blur parameters
src_params = {}
src_params['source_FWHM_x_axis'] = 2.70
src_params['source_FWHM_y_axis'] = 2.82
src_params['cutoff_FWHM_multiplier'] = 10

#Detector blur parameters
det_params = {}
det_params['detector_FWHM_1'] = 2.05
det_params['detector_FWHM_2'] = 120.72
det_params['detector_weight_1'] = 0.917
det_params['cutoff_FWHM_1_multiplier'] = 10 
det_params['cutoff_FWHM_2_multiplier'] = 8 

#Deblur the artifact using Wiener filtering
wiener_rad = wiener_deblur(norm_rad,sod,sdd,pix_wid,src_params,det_params,reg_param)

#Display deblurred radiograph of artifact
mag = (sdd-sod)/sod
sz = wiener_rad.shape
x = np.arange(-(sz[1]//2),(sz[1]//2)+1,1)*pix_wid/mag
y = np.arange(-(sz[0]//2),(sz[0]//2)+1,1)*pix_wid/mag
plt.pcolormesh(x,y,wiener_rad,cmap='gray')
plt.xlabel('micrometers')
plt.ylabel('micrometers')
plt.title('Wiener deblur')
plt.colorbar()
plt.show()
