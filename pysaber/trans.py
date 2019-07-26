
import numpy as np
#import matplotlib.pyplot as plt
from skimage.measure import label,find_contours
#from PIL import Image
#from scipy.ndimage.morphology import distance_transform_edt
import csv
import sys
#from scipy.interpolate import Rbf,interp2d
from skimage.morphology import binary_opening
#from scipy import ndimage
from sklearn.linear_model import RANSACRegressor

def get_contour(rad,thresh):
    """
    Find the edge in the input radiograph.

    Parameters:
        rad (numpy.ndarray): Radiograph of a sharp edge sample
        thresh (float): The value at which a iso-valued contour (contour is the edge) is drawn

    Returns:
        numpy.ndarray: Coordinates along the longest detected contour    
    """
    contours = find_contours(rad,thresh) 
    best_contour = contours[0]
    for contour in contours:
        if(len(contour)>len(best_contour)):
            best_contour = contour
    return(best_contour)

def get_trans(rad,best_contour,trans_min,trans_max,thresh):
    """
    Compute the ideal transmission image.

    Parameters:
        rad (numpy.ndarray): Radiograph of a sharp edge sample
        best_contour (numpy.ndarray): Coordinates of the longest contour that is assumed to be the edge
        trans_min (float): Minimum transmission value
        trans_max (float): Maximum transmission value
        thresh (float): Transmission value for the edge

    Returns:
        numpy.ndarray: Ideal transmission image
    """
    window_interp = 5 #for interpolation. must be odd

    edge_thick = np.ones(rad.shape) #edge pixel will be labeled as 0
    for row,col in best_contour:
        row_floor,col_floor = int(np.floor(row)),int(np.floor(col))
        row_ceil,col_ceil = int(np.ceil(row)),int(np.ceil(col))
        edge_thick[row_floor,col_floor],edge_thick[row_floor,col_ceil] = 0,0
        edge_thick[row_ceil,col_floor],edge_thick[row_ceil,col_ceil] = 0,0

    edge_thick = binary_opening(edge_thick) #erosion followed by dilation. Rids of bright pixels in edge voxels
    rows_edge,cols_edge = np.nonzero(edge_thick==0) #Get edge pixel locations 
    
    labels,num = label(edge_thick,background=0,return_num=True)
    if(num != 2):
        raise ValueError("ERROR: Number of regions detected is {}. Two types of regions must be present in radiographs.".format(num))

    val1 = np.mean(rad[labels==1])
    val2 = np.mean(rad[labels==2])

    trans = np.zeros(rad.shape) #Sample's pixel locations will be labeled as 1
    trans[labels==0] = np.nan
    trans[labels==1] = trans_min if val1<=val2 else trans_max
    trans[labels==2] = trans_max if val1<=val2 else trans_min
      
    for row,col in best_contour:
        trans[int(round(row)),int(round(col))] = thresh

    ideal_trans = trans.copy()
    for row,col in zip(rows_edge,cols_edge):
        if(np.isnan(trans[row,col])):
            norm,ival = 0,0
            for i in range(-int((window_interp-1)/2),int((window_interp-1)/2)+1):
                for j in range(-int((window_interp-1)/2),int((window_interp-1)/2)+1):
                    row_new = row+i
                    col_new = col+j
                    if(i!=0 and j!=0 and row_new>=0 and row_new<trans.shape[0] and col_new>=0 and col_new<trans.shape[1]):
                        if(np.isnan(trans[row_new,col_new]) == False):
                            weight = 1.0/np.sqrt(i*i+j*j)
                            ival += weight*trans[row_new,col_new]
                            norm += weight
            ideal_trans[row,col] = ival/norm if norm != 0 else thresh
            if(norm == 0):
                print("WARNING: No valid value within window for interpolation") 
    return(ideal_trans) 

def get_padded_trans(ideal_trans,bdary_mask_perc,pad_factor,rad_mask):
    """
    Appropriately pad the ideal transmission image and the masks.

    Parameters:
        ideal_trans (numpy.ndarray): Ideal transmission image
        bdary_mask_perc (float): Percentage of image region that must be masked, i.e., excluded from blur estimation, close to the radiograph edges on each side (left, right, top, and bottom). Expressed as a percentage of the radiograph size.
        pad_factor (list [float,float]): Pad factor as expressed in multiples of input radiograph size
        rad_mask (numpy.ndarray): Boolean mask array over the radiograph where blur estimation is done.
    """
    bdary_mask_perc /= 100

    #Solves hw-(h-2*delta)(w-2*delta)=phw where h,w are idea_trans shape, p is bdary_mask_perc, and delta is delta_mask
    a = 4
    b = -2*(ideal_trans.shape[0]+ideal_trans.shape[1])    
    c = bdary_mask_perc*ideal_trans.shape[0]*ideal_trans.shape[1]
    delta_mask = (-b-np.sqrt(b*b-4*a*c))/(2*a)
    if delta_mask < 0:
        raise ValueError("ERROR: delta_mask is negative. This should not occur. Contact the author of this python package.")
#    print("Delta mask is ",delta_mask)

    mask = np.zeros(ideal_trans.shape).astype(bool)
    row_min = int(round(delta_mask))
    row_max = int(round(mask.shape[0]-delta_mask))
    col_min = int(round(delta_mask))
    col_max = int(round(mask.shape[1]-delta_mask))
    mask[row_min:row_max,col_min:col_max] = True
    if rad_mask is not None:
        mask = np.bitwise_and(mask,rad_mask) 
    norm_rad_mask = mask.copy()  
      
    #pad_width0 = int(ideal_trans.shape[0]*(pad_factor[0]-1)/2.0-bdary_mask_perc*mask.shape[0])
    #pad_width1 = int(ideal_trans.shape[1]*(pad_factor[1]-1)/2.0-bdary_mask_perc*mask.shape[1])
    pad_width0 = int(ideal_trans.shape[0]*(pad_factor[0]-1)/2.0)
    pad_width1 = int(ideal_trans.shape[1]*(pad_factor[1]-1)/2.0)
    colidx,rowidx = np.meshgrid(np.arange(-pad_width1,ideal_trans.shape[1]+pad_width1),np.arange(-pad_width0,ideal_trans.shape[0]+pad_width0))
    ideal_trans = np.pad(ideal_trans,((pad_width0,pad_width0),(pad_width1,pad_width1)),mode='constant',constant_values=-1)
    ideal_trans_mask = np.pad(mask,((pad_width0,pad_width0),(pad_width1,pad_width1)),mode='constant',constant_values=False)
    return(ideal_trans,ideal_trans_mask,norm_rad_mask,colidx,rowidx)

def ideal_trans_sharp_edge(norm_rad,bdary_mask_perc=5,pad_factor=[2,2],mask=None):
    """Estimate parameters of transmission model for a radiograph with a single straight sharp edge.

    This function takes as input a normalized radiograph and estimates the ideal transmission image that would have resulted if there was no blur and no noise. This function also pads the transmission image to prevent aliasing during convolution. It also produces masks for the region well within the image edges where blur model estimation is done. The transmission image and masks are packaged into a python dictionary, which is called the transmission model.  

    Parameters:
        norm_rad (numpy.ndarray): Normalized measured radiograph.
        bdary_mask_perc (float): Percentage of image region that must be masked, i.e., excluded from blur estimation, close to radiograph edges
        pad_factor (list [float,float]): Pad factor as expressed in multiples of input radiograph size
        mask (numpy.ndarray): Boolean mask array over the radiograph where blur estimation is done.
 
    Returns:
        trans_model (dict): Python dictionary containing the ideal transmission function, masks, and gradient functions.
        params_init (list): Initial parameters for ideal transmission function
        params_bounds (list of lists): Bounds on transmission function parameters.  
    """

    #pad_factor: pad to multiple of the input image size

    trans_min = max(0.0,np.min(norm_rad))
    trans_max = min(1.0,np.max(norm_rad))
    renorm_rad = (norm_rad-trans_min)/(trans_max-trans_min)
    best_contour = get_contour(renorm_rad,0.5)

    ideal_trans = get_trans(renorm_rad,best_contour,0.0,1.0,0.5)

    x,y = [],[]
    for row,col in best_contour:
        x.append(col)
        y.append(row)
    coeff = np.polyfit(x,y,1)
    line1 = np.poly1d(coeff)   

    ideal_trans,ideal_trans_mask,norm_rad_mask,colidx,rowidx = get_padded_trans(ideal_trans,bdary_mask_perc,pad_factor,mask)    
 
    linediff1 = rowidx - line1(colidx)
            
    for reg in [linediff1>0,linediff1<=0]:
        val = np.mean(ideal_trans[np.bitwise_and(ideal_trans>=0,reg)])
        ideal_trans[np.bitwise_and(ideal_trans==-1,reg)] = 0.0 if val<0.5 else 1.0
        
    if(np.any(np.isnan(ideal_trans))):
        raise ValueError("ERROR: Nan detected in the ideal radiograph image")

    X_rows = ideal_trans[ideal_trans_mask].size
    X0 = 1-ideal_trans[ideal_trans_mask].reshape(X_rows,1)
    X1 = ideal_trans[ideal_trans_mask].reshape(X_rows,1)
    X = np.hstack((X0,X1))
    y = norm_rad[norm_rad_mask].reshape(X_rows,1)
    reg = RANSACRegressor(min_samples=10,residual_threshold=0.1)
    reg.fit(X,y)
    params_init = [float(reg.predict(np.array([[1,0]]))),float(reg.predict(np.array([[0,1]])))]    
    
    z = ideal_trans[ideal_trans_mask]
    z_sq = ideal_trans[ideal_trans_mask]**2
    A = np.zeros((2,2))
    A[0,0] = np.sum(1-2*z+z_sq)
    A[0,1] = np.sum(z-z_sq)
    A[1,0] = A[0,1]
    A[1,1] = np.sum(z_sq)
    b = np.zeros(2)
    b[0] = np.sum(norm_rad[norm_rad_mask]*(1-z))
    b[1] = np.sum(norm_rad[norm_rad_mask]*z)
    #print("old trans init params {}".format(np.matmul(np.linalg.inv(A),b).tolist()))
    params_bounds = [[-1.0,0.5],[0.5,2.0]]

    #print("trans init params {}".format(params_init))

    trans_model = {}
    trans_model['norm_rad_mask'] = norm_rad_mask 
    trans_model['ideal_trans'] = lambda params: params[0]+(params[1]-params[0])*ideal_trans
    trans_model['ideal_trans_mask'] = ideal_trans_mask
    trans_model['ideal_trans_grad'] = lambda params: [1.0-ideal_trans,ideal_trans]

    return trans_model,params_init,params_bounds 

def ideal_trans_perp_corner(norm_rad,bdary_mask_perc=5,pad_factor=[2,2],mask=None):
    """Estimate parameters of transmission model for a radiograph with a sharp perpendicular corner.

    This function takes as input a normalized radiograph and estimates the ideal transmission image that would have resulted if there was no blur and no noise. This function also pads the transmission image to prevent aliasing during convolution. It also produces masks for the region well within the image edges where blur model estimation is done. The transmission image and masks are packaged into a python dictionary, which is called the transmission model.  

    Parameters:
        norm_rad (numpy.ndarray): Normalized measured radiograph.
        bdary_mask_perc (float): Percentage of image region that must be masked, i.e., excluded from blur estimation, close to radiograph edges
        pad_factor (list [float,float]): Pad factor as expressed in multiples of input radiograph size
        mask (numpy.ndarray): Boolean mask array over the radiograph where blur estimation is done.
 
    Returns:
        trans_model (dict): Python dictionary containing the ideal transmission function, masks, and gradient functions.
        params_init (list): Initial parameters for ideal transmission function
        params_bounds (list of lists): Bounds on transmission function parameters.  
    """

    trans_min = max(0.0,np.min(norm_rad))
    trans_max = min(1.0,np.max(norm_rad))
    renorm_rad = (norm_rad-trans_min)/(trans_max-trans_min)
    best_contour = get_contour(renorm_rad,0.5)
    
    ideal_trans = get_trans(renorm_rad,best_contour,0.0,1.0,0.5)

    conlen = len(best_contour)
    x,y = [],[]
    for row,col in best_contour[:conlen//4]:
        x.append(col)
        y.append(row)
    line1_coeff = np.polyfit(x,y,1)
    line1 = np.poly1d(line1_coeff)   

    x,y = [],[]
    for row,col in best_contour[-(conlen//4):]:
        x.append(col)
        y.append(row)
    line2_coeff = np.polyfit(x,y,1)
    line2 = np.poly1d(line2_coeff)   
    
    ideal_trans,ideal_trans_mask,norm_rad_mask,colidx,rowidx = get_padded_trans(ideal_trans,bdary_mask_perc,pad_factor,mask)    
        
    linediff1 = rowidx - line1(colidx)
    linediff2 = rowidx - line2(colidx)
   
    region1 = np.bitwise_and(linediff1<=0,linediff2<=0)
    region2 = np.bitwise_and(linediff1>0,linediff2<=0)
    region3 = np.bitwise_and(linediff1<=0,linediff2>0)
    region4 = np.bitwise_and(linediff1>0,linediff2>0)
            
    for reg in [region1,region2,region3,region4]:
        val = np.mean(ideal_trans[np.bitwise_and(ideal_trans>=0.0,reg)])
        ideal_trans[np.bitwise_and(ideal_trans==-1,reg)] = 0.0 if val<0.5 else 1.0
  
    #perp_mask_perc /= (2*100)
    #col_inter = (line1_coeff[1]-line2_coeff[1])/(line2_coeff[0]-line1_coeff[0])
    #row_inter = line1_coeff[0]*col_inter+line1_coeff[1]
    #circmask = (colidx-col_inter)**2+(rowidx-row_inter)**2<=(perp_mask_perc*max(norm_rad.shape))**2
    #ideal_trans_mask[circmask] = False   
    #colidx,rowidx = np.meshgrid(np.arange(0,norm_rad.shape[1]),np.arange(0,norm_rad.shape[0]))
    #circmask = (colidx-col_inter)**2+(rowidx-row_inter)**2<=(perp_mask_perc*max(norm_rad.shape))**2
    #norm_rad_mask[circmask] = False

    if(np.any(np.isnan(ideal_trans))):
        raise ValueError("ERROR: Nan detected in the ideal radiograph image")

    X_rows = ideal_trans[ideal_trans_mask].size
    X0 = 1-ideal_trans[ideal_trans_mask].reshape(X_rows,1)
    X1 = ideal_trans[ideal_trans_mask].reshape(X_rows,1)
    X = np.hstack((X0,X1))
    y = norm_rad[norm_rad_mask].reshape(X_rows,1)
    reg = RANSACRegressor(min_samples=10,residual_threshold=0.1)
    reg.fit(X,y)
    params_init = [float(reg.predict(np.array([[1,0]]))),float(reg.predict(np.array([[0,1]])))]    
 
    z = ideal_trans[ideal_trans_mask]
    z_sq = ideal_trans[ideal_trans_mask]**2
    A = np.zeros((2,2))
    A[0,0] = np.sum(1-2*z+z_sq)
    A[0,1] = np.sum(z-z_sq)
    A[1,0] = A[0,1]
    A[1,1] = np.sum(z_sq)
    b = np.zeros(2)
    b[0] = np.sum(norm_rad[norm_rad_mask]*(1-z))
    b[1] = np.sum(norm_rad[norm_rad_mask]*z)
#    print("old trans init params {}".format(np.matmul(np.linalg.inv(A),b).tolist()))
#    print("trans init params {}".format(params_init))
    
    trans_model = {}
    trans_model['norm_rad_mask'] = norm_rad_mask 
    trans_model['ideal_trans'] = lambda params: params[0]+(params[1]-params[0])*ideal_trans
    trans_model['ideal_trans_mask'] = ideal_trans_mask
    trans_model['ideal_trans_grad'] = lambda params: [1.0-ideal_trans,ideal_trans]
    params_init = np.matmul(np.linalg.inv(A),b).tolist()
    params_bounds = [[-1.0,0.5],[0.5,2.0]]

    return trans_model,params_init,params_bounds

