import numpy as np
from PIL import Image #To read images in TIFF format
from saber import get_trans_fit #Compute the blurred radiograph as predicted by the blur model given parameters of source and detector blurs 
import matplotlib.pyplot as plt #To display images
    
pix_wid = 0.675 #Width of each pixel in micrometers

#Read a horizontal edge radiograph for which accuracy of fit must be analyzed
rad = Image.open('data/horz_edge_25mm.tif') #Read radiograph
rad = np.asarray(rad) #Convert to numpy array
bright = Image.open('data/horz_bright.tif') #Read bright field
bright = np.asarray(bright) #Convert to numpy array
dark = Image.open('data/horz_dark.tif') #Read dark field
dark = np.asarray(dark) #Convert to numpy array
norm_rad = (rad-dark)/(bright-dark) #Normalize radiograph
sod = 24751.89 #Source to object distance (SOD) of radiograph
sdd = 71003.08 #Source to detector distance (SDD) of radiograph

#Parameters of X-ray source blur
src_params = {'source_FWHM_x_axis':2.70,'source_FWHM_y_axis':2.82,'cutoff_FWHM_multiplier':10}
#Parameters detector blur
det_params = {'detector_FWHM_1':2.05,'detector_FWHM_2':120.72,'detector_weight_1':0.917,'cutoff_FWHM_1_multiplier':10,'cutoff_FWHM_2_multiplier':8}
#Transmission function parameters
trans_params = [0.016,0.979]

#Get the blurry radiograph as predicted or output by the blur model
_,pred_nrad = get_trans_fit(norm_rad,sod,sdd,pix_wid,src_params,det_params,trans_params)

#Show a line plot comparing the measured radiograph and the predicted radiograph at the output of the blur model
sz = norm_rad.shape
coords = np.arange(-(sz[0]//2),sz[0]//2,1)*pix_wid
mid = (pred_nrad.shape[0]//2,pred_nrad.shape[1]//2)

plt.plot(coords,norm_rad[:,sz[1]//2])
plt.plot(coords,pred_nrad[mid[0]-(sz[0]//2):mid[0]+(sz[0]//2),mid[1]])
plt.xlabel('micrometers')
plt.legend(['Measured','Prediction'])
plt.show()
