import numpy as np
from PIL import Image #To read images in TIFF format
from pysaber import get_trans_masks #To compute transmission function and masks
import matplotlib.pyplot as plt #To display images
    
pix_wid = 0.675 #Width of each pixel in micrometers

#Read a vertical edge radiograph for which the transmission function must be computed
rad = Image.open('data/vert_edge_25mm.tif') #Read radiograph
rad = np.asarray(rad) #Convert to numpy array 
bright = Image.open('data/vert_bright.tif') #Read bright field
bright = np.asarray(bright) #Convert to numpy array
dark = Image.open('data/vert_dark.tif') #Read dark field
dark = np.asarray(dark) #Convert to numpy array
nrad = (rad-dark)/(bright-dark) #Normalize radiograph

#Get the transmission function and masks 
trans,trans_mask,rad_mask = get_trans_masks(nrad,edge='straight',pad=[3,3])
#Use pad = [1,1] if you do not want padding

#Display the array trans
#Visually check for inaccurate localization of the sharp-edge
plt.imshow(trans,cmap='gray')
plt.colorbar()
plt.show()

#Display and inspect the mask for transmission function
plt.imshow(trans_mask,cmap='gray')
plt.show()

#Display and inspect the mask for radiograph
plt.imshow(rad_mask,cmap='gray')
plt.show()

#Show a line plot comparing the measured radiograph and transmission function
sz = nrad.shape
coords = np.arange(-(sz[1]//2),sz[1]//2,1)*pix_wid
mid = (trans.shape[0]//2,trans.shape[1]//2)

plt.plot(coords,nrad[sz[0]//2,:])
#Due to padding, trans is three times the size of nrad in each dimension
#For proper alignment in the presence of padding, both nrad and trans are center aligned
#Center alignment is used since an equal amount of padding is applied at both ends of each axis
plt.plot(coords,trans[mid[0],mid[1]-(sz[1]//2):mid[1]+(sz[1]//2)])
plt.xlabel('micrometers')
plt.legend(['Measured','Transmission'])
plt.show()
