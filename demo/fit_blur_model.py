import numpy as np #For mathematics on vectors
from PIL import Image #To read images in TIFF format
from pysaber import get_blur_params #To estimate blur parameters

pix_wid = 0.675 #Width of each pixel in micrometers

#Horizontal edge radiographs
horz_edges = ['data/horz_edge_25mm.tif','data/horz_edge_50mm.tif'] #File names of radiograph images 
horz_bright = 'data/horz_bright.tif' #Bright field image for normalization
horz_dark = 'data/horz_dark.tif' #Dark field image for normalization
horz_sod = [24751.89,50251.79] #List of source to object (SOD) distances for all radiographs in horz_edges
horz_sdd = [71003.08,71003.08] #List of source to detector (SDD) distances for all radiographs in horz_edges

#Vertical edge radiographs
vert_edges = ['data/vert_edge_25mm.tif','data/vert_edge_50mm.tif'] #File names of radiographs
vert_bright = 'data/vert_bright.tif' #Bright field image for normalization
vert_dark = 'data/vert_dark.tif' #Dark field image for normalization
vert_sod = [24753.05,50253.35] #List of source to object (SOD) distances for all radiographs in vert_edges
vert_sdd = [71010.86,71010.86] #List of source to detector (SDD) distances for all radiographs in vert_edges

norm_rads = [] #List that will contain normalized radiographs
sod = [] #SOD for each radiograph in norm_rads
sdd = [] #SDD for each radiograph in norm_rads
for i in range(len(horz_edges)): #For each horizontal edge radiograph
    rad = Image.open(horz_edges[i]) #Read radiograph
    rad = np.asarray(rad) #Convert to numpy array
    bright = Image.open(horz_bright) #Read bright field
    bright = np.asarray(bright) #Convert to numpy array
    dark = Image.open(horz_dark) #Read dark field
    dark = np.asarray(dark) #Convert to numpy array
    nrad = (rad-dark)/(bright-dark) #Normalize radiograph
    norm_rads.append(nrad) #Add normalized radiograph to list
    sod.append(horz_sod[i]) #Add corresponding SOD to list
    sdd.append(horz_sdd[i]) #Add corresponding SDD to list

for i in range(len(vert_edges)): #For each vertical edge radiograph
    rad = Image.open(vert_edges[i]) #Read radiograph
    rad = np.asarray(rad) #Convert to numpy array
    bright = Image.open(vert_bright) #Read bright field
    bright = np.asarray(bright) #Convert to numpy array
    dark = Image.open(vert_dark) #Read dark field
    dark = np.asarray(dark) #Convert to numpy array
    nrad = (rad-dark)/(bright-dark) #Normalize radiograph
    norm_rads.append(nrad) #Add normalized radiograph to list
    sod.append(horz_sod[i]) #Add corresponding SOD to list
    sdd.append(horz_sdd[i]) #Add corresponding SDD to list

#Estimate parameters of X-ray source blur, detector blur, and every radiograph's transmission function
src_params,det_params,trans_params = get_blur_params(norm_rads,sod,sdd,pix_wid,convg_thresh=1e-6,pad_factor=[3,3])
#src_params is a python dictionary of parameters that quantify X-ray source blur
#det_params is a python dictionary of parameters that quantify blur from the detector panel
#Both src_params and det_params characterize the blur and are needed for deblurring operations 
#trans_params is a list of lists of parameters that quantify non-ideal behavior resulting in inaccurate normalization. It is useful to check accuracy of fit and not useful for deblurring. 
#help(get_blur_params)
#For more information, uncomment the above line to get help on using the function get_blur_params

print("-------------------------------------------------------")
print("Source blur model parameters are {}".format(src_params)) #Print parameters of source blur
print("Detector blur model parameters are {}".format(det_params)) #Print parameters of detector blur
print("Transmission function parameters are {}".format(trans_params)) #Print parameters of transmission functions
