import numpy as np
from PIL import Image
import os

horz_edge_files = ['horz_edge_12mm.tif','horz_edge_25mm.tif','horz_edge_38mm.tif','horz_edge_50mm.tif','horz_edge_65mm.tif']
horz_bright_file = 'horz_bright.tif'
horz_dark_file = 'horz_dark.tif'
horz_sod = [12003.38708,24751.88806,37501.93787,50251.78845,65298.23865]
horz_sdd = [71003.07846,71003.07846,71003.07846,71003.07846,71003.07846]

vert_edge_files = ['vert_edge_13mm.tif','vert_edge_25mm.tif','vert_edge_38mm.tif','vert_edge_50mm.tif','vert_edge_60mm.tif']
vert_bright_file = 'vert_bright.tif'
vert_dark_file = 'vert_dark.tif'
vert_sod = [13002.40271,24753.05212,37503.00272,50253.35291,60003.10388]
vert_sdd = [71010.86044,71010.86044,71010.86044,71010.86044,71010.86044]
    
def fetch_data(ddir,idx,orret=False):
    norm_rads = [] #List that will contain normalized radiographs
    orient = []
    sod,sdd = [],[]
    dir_suffix = [] 
 
    rad = np.asarray(Image.open(os.path.join(ddir,horz_edge_files[idx]))) #Read radiograph
    bright = np.asarray(Image.open(os.path.join(ddir,horz_bright_file)))
    dark = np.asarray(Image.open(os.path.join(ddir,horz_dark_file)))
    rad = (rad-dark)/(bright-dark) #Normalize radiograph
    norm_rads.append(rad) #Append normalized radiograph to list norm_rads
    orient.append('horz')
    sod.append(horz_sod[idx]) #Add SOD to list sod
    sdd.append(horz_sdd[idx]) #Add SDD to list sdd
    dir_suffix.append('{:.0f}mm'.format(sod[-1]/1000))       
    #print("Read radiograph {}, SOD of {:.2e}, and SDD of {:.2e}".format(horz_edge_files[idx],sod[-1],sdd[-1]))

    rad = np.asarray(Image.open(os.path.join(ddir,vert_edge_files[idx]))) #Read radiograph
    bright = np.asarray(Image.open(os.path.join(ddir,vert_bright_file)))
    dark = np.asarray(Image.open(os.path.join(ddir,vert_dark_file)))
    rad = (rad-dark)/(bright-dark) #Normalize radiograph
    norm_rads.append(rad) #Append normalized radiograph to list norm_rads
    orient.append('vert')
    sod.append(vert_sod[idx]) #Add SOD to list sod
    sdd.append(vert_sdd[idx]) #Add SDD to list sdd
    dir_suffix.append('{:.0f}mm'.format(sod[-1]/1000))       
    #print("Read radiograph {}, SOD of {:.2e}, and SDD of {:.2e}".format(vert_edge_files[idx],sod[-1],sdd[-1]))

    if orret:
        return norm_rads,sod,sdd,dir_suffix,orient
    else:
        return norm_rads,sod,sdd,dir_suffix

