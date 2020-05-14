import numpy as np
from pysaber import wiener_deblur #To deblur using Wiener filtering
import matplotlib.pyplot as plt #To display images
from PIL import Image #To read images in TIFF format

rad_file = 'data/horz_edge_25mm.tif' #Filename of radiograph
bright_file = 'data/horz_bright.tif' #Bright field
dark_file = 'data/horz_dark.tif' #Dark field

sdd = 71003.08 #Source to detector distance (SDD) in micrometers
sod = 24751.89 #Source to object distance (SOD) in micrometers
pix_wid = 0.675 #Pixel width in micrometers
reg_param = 0.1 #Regularization parameter

rad = np.asarray(Image.open(rad_file)) #Read radiograph and convert to numpy array
bright = np.asarray(Image.open(bright_file)) #Read bright field image and convert to numpy array
dark = np.asarray(Image.open(dark_file)) #Read dark field image and convert to numpy array
norm_rad = (rad-dark)/(bright-dark) #Normalize radiograph

#X-ray source blur parameters
src_params = {'source_FWHM_x_axis':2.69,
                'source_FWHM_y_axis':3.01, 
                'norm_power':1.0,
                'cutoff_FWHM_multiplier':10}

#Detector blur parameters
det_params = {'detector_FWHM_1':1.85, 
                'detector_FWHM_2':126.5, 
                'detector_weight_1':0.916, 
                'norm_power':1.0, 
                'cutoff_FWHM_1_multiplier':10, 
                'cutoff_FWHM_2_multiplier':10}

#Deblur the radiograph using Wiener filter
wiener_rad = wiener_deblur(norm_rad,sod,sdd-sod,pix_wid,src_params,det_params,reg_param)

#Display deblurred radiograph
sz = wiener_rad.shape
x = np.arange(-(sz[1]//2),(sz[1]//2)+1,1)*pix_wid
y = np.arange(-(sz[0]//2),(sz[0]//2)+1,1)*pix_wid
plt.pcolormesh(x,y,wiener_rad,cmap='gray')
plt.xlabel('micrometers')
plt.ylabel('micrometers')
plt.title('Wiener deblur')
plt.colorbar()
plt.show()
