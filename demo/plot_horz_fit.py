import numpy as np
from PIL import Image #To read images in TIFF format
from pysaber import get_trans_fit #To get blurred radiograph as predicted by the blur model 
import matplotlib.pyplot as plt #To display images
    
pix_wid = 0.675 #Width of each pixel in micrometers

#Read a horizontal edge radiograph for which accuracy of fit must be analyzed
rad = Image.open('data/horz_edge_25mm.tif') #Read radiograph
rad = np.asarray(rad) #Convert to numpy array
bright = Image.open('data/horz_bright.tif') #Read bright field
bright = np.asarray(bright) #Convert to numpy array
dark = Image.open('data/horz_dark.tif') #Read dark field
dark = np.asarray(dark) #Convert to numpy array
nrad = (rad-dark)/(bright-dark) #Normalize radiograph
sod = 24751.89 #Source to object distance (SOD) of radiograph
sdd = 71003.08 #Source to detector distance (SDD) of radiograph

#Parameters of X-ray source blur
src_params = {'source_FWHM_x_axis':2.69,
                'source_FWHM_y_axis':3.01, 
                'norm_power':1.0,
                'cutoff_FWHM_multiplier':10}
#Parameters detector blur
det_params = {'detector_FWHM_1':1.85, 
                'detector_FWHM_2':126.5, 
                'detector_weight_1':0.916, 
                'norm_power':1.0, 
                'cutoff_FWHM_1_multiplier':10, 
                'cutoff_FWHM_2_multiplier':10}
#Transmission function parameters
trans_params = [0.015,0.98]

#Get the blurred radiograph as predicted by the blur model
pred_nrad,_ = get_trans_fit(nrad,sod,sdd-sod,pix_wid,src_params,det_params,trans_params,pad=[3,3],edge='straight-edge')

#Show a line plot comparing the measured radiograph and the predicted blurred radiograph
sz = nrad.shape
coords = np.arange(-(sz[0]//2),sz[0]//2,1)*pix_wid
mid = (pred_nrad.shape[0]//2,pred_nrad.shape[1]//2)

plt.plot(coords,nrad[:,sz[1]//2])
#Due to padding, pred_nrad is three times the size of nrad in each dimension
#For proper alignment in presence of padding, both nrad and pred_nrad are center aligned
#Center alignment is used since an equal amount of padding is applied at both ends of each axis
plt.plot(coords,pred_nrad[mid[0]-(sz[0]//2):mid[0]+(sz[0]//2),mid[1]])
plt.xlabel('micrometers')
plt.legend(['Measured','Prediction'])
plt.show()
