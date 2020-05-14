import numpy as np #For mathematics on vectors
from PIL import Image #To read images in TIFF format
from pysaber import estimate_blur #To estimate blur PSF parameters

pix_wid = 0.675 #Width of each pixel in micrometers

#Horizontal and vertical edge radiographs
edge_files = ['data/horz_edge_25mm.tif','data/horz_edge_50mm.tif',
                'data/vert_edge_25mm.tif','data/vert_edge_50mm.tif'] 
#Filenames of bright field images for normalization
bright_files = ['data/horz_bright.tif','data/horz_bright.tif',
                    'data/vert_bright.tif','data/vert_bright.tif']
#Filenames of dark field images for normalization
dark_files = ['data/horz_dark.tif','data/horz_dark.tif',
                'data/vert_dark.tif','data/vert_dark.tif']
#Source to object (SOD) distances in micrometers for each radiograph in edge_files
sod = [24751.89,50251.79,24753.05,50253.35] 
#Source to detector (SDD) distances in micrometers for each radiograph in edge_files
sdd = [71003.08,71003.08,71010.86,71010.86] 

rads = [] #List that will contain normalized radiographs
odd = [] #Object to detector distance (ODD) for each radiograph in rads
for i in range(len(edge_files)): #Loop through all the radiograph files
    rad = Image.open(edge_files[i]) #Read radiograph
    rad = np.asarray(rad) #Convert to numpy array
    bright = Image.open(bright_files[i]) #Read bright field
    bright = np.asarray(bright) #Convert to numpy array
    dark = Image.open(dark_files[i]) #Read dark field
    dark = np.asarray(dark) #Convert to numpy array
    nrad = (rad-dark)/(bright-dark) #Normalize radiograph
    rads.append(nrad) #Add normalized radiograph to the list rads
    odd.append(sdd[i]-sod[i]) #Add corresponding ODD to the list odd

#Estimate X-ray source blur, detector blur, and every radiograph's transmission function
#To reduce run time and quickly produce a result, the value for argument thresh can be reduced.
#However, reducing thresh may produce an inaccurate blur model that does not fit the measured data  
src_params,det_params,trans_params = estimate_blur(rads,sod,odd,pix_wid,
                edge='straight',thresh=1e-6,pad=[3,3],power=1.0,save_dir='./')
#src_params is a python dictionary of parameters that quantify X-ray source blur
#det_params is a python dictionary of parameters that quantify blur from the detector panel
#Both src_params and det_params characterize the effective blur and are used during deblurring 
#trans_params is a list of lists, each of which contains the low/high values of transmission function
#trans_params is useful to check accuracy of fit and not useful for deblurring. 

#help(estimate_blur)
#Uncomment above line to get help on using the function estimate_blur

print("-------------------------------------------------------")
print("Source blur model parameters are {}".format(src_params)) 
#Print parameters of source blur
print("Detector blur model parameters are {}".format(det_params)) 
#Print parameters of detector blur
print("Transmission function parameters are {}".format(trans_params)) 
#Print parameters of transmission functions
