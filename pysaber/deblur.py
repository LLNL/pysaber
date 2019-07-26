import numpy as np
from pysaber.models import SourceBlur,DetectorBlur,get_scale,get_FWHM,combine_psfs,convolve_psf
from scipy.signal import fftconvolve
#from scipy.optimize import minimize_scalar
from scipy.optimize import minimize
from copy import deepcopy    
import sys
from skimage.restoration import wiener
#from matplotlib.widgets import RectangleSelector
import matplotlib.pyplot as plt

def regparam_estimate(recon,prior_weights):
    """
    Maximum likelihood estimate of regularization parameter given a initial reconstruction

    Parameters:
        recon (numpy.ndarray): Initial reconstruction (Blurred radiograph is a potential initial reconstruction)
        prior_weights (numpy.ndarray): 3x3 weights for the prior local neighborhood regularization.

    Returns:
        Estimate of regularization parameter  
    """
    (spat_cost,spat_N) = computePriorCost(recon,prior_weights,1.0)
    regparam = 1.0/(1.2*spat_cost/spat_N)
    return(regparam)
    
def computePriorGradient(recon,weights,regparam):
    """
    Compute gradient of prior cost function.

    Parameters:
        recon (numpy.ndarray): Reconstruction (intermediate deblurred radiograph)
        weights (numpy.ndarray): 3x3 weights for the prior local neighborhood regularization.
        regparam (float): Regularization parameter

    Returns:
        numpy.ndarray: Gradient of prior cost with respect to each pixel in the reconstruction
    """
    sz = np.shape(recon)
    grad = np.zeros(sz,dtype=float)
    if (regparam != 0):
        for i in ([-1,0,1]):
            for j in ([-1,0,1]):
                r_xs = 0; r_xe = sz[1]; rn_xs = 0; rn_xe = sz[1]
                r_ys = 0; r_ye = sz[0]; rn_ys = 0; rn_ye = sz[0]
                if (j == -1): r_xs = 1; rn_xe = sz[1]-1;
                if (i == -1): r_ys = 1; rn_ye = sz[0]-1;
                if (j == 1): r_xe = sz[1]-1; rn_xs = 1;
                if (i == 1): r_ye = sz[0]-1; rn_ys = 1;
            
                if (i != 0 or j != 0):
                    Delta = recon[r_ys:r_ye,r_xs:r_xe] - recon[rn_ys:rn_ye,rn_xs:rn_xe]
                    grad[r_ys:r_ye,r_xs:r_xe] += regparam*weights[i+1,j+1]*1.2*np.sign(Delta)*(np.fabs(Delta)**0.2)

    #print ("Avg of recon is ", np.mean(recon))
    return grad

def computePriorLineSearchParams(recon,direc,weights,regparam):
    """
    Vectorized operations for fast computation of line search.

    Parameters:
        recon (numpy.ndarray): Reconstruction (intermediate deblurred radiograph)
        direc (numpy.ndarray): Descent direction
        weights (numpy.ndarray): 3x3 weights for the prior local neighborhood regularization.
        regparam (float): Prior regularization parameter

    Returns:
        numpy.ndarray: Difference between all pairs of a reconstructed pixel and its neighorhood pixel
        numpy.ndarray: Difference between all pairs of a descent direction pixel and its neighorhood pixel
    """
    sz = np.shape(recon)
    off = np.zeros((sz[0],sz[1],3,3),dtype=float)
    slope = np.zeros((sz[0],sz[1],3,3),dtype=float)
    if (regparam != 0):
        for i in ([0,1]):
            if (i == 1): j_sweep = [-1, 0, 1]
            else: j_sweep = [0, 1]
            for j in j_sweep:
                r_xs = 0; r_xe = sz[1]; rn_xs = 0; rn_xe = sz[1]
                r_ys = 0; r_ye = sz[0]; rn_ys = 0; rn_ye = sz[0]
                if (j == -1): r_xs = 1; rn_xe = sz[1]-1;
                if (i == -1): r_ys = 1; rn_ye = sz[0]-1;
                if (j == 1): r_xe = sz[1]-1; rn_xs = 1;
                if (i == 1): r_ye = sz[0]-1; rn_ys = 1;
                
                if (i != 0 or j != 0):
                    off[r_ys:r_ye,r_xs:r_xe,i,j] = (recon[r_ys:r_ye,r_xs:r_xe] - recon[rn_ys:rn_ye,rn_xs:rn_xe])
                    slope[r_ys:r_ye,r_xs:r_xe,i,j] = (direc[r_ys:r_ye,r_xs:r_xe] - direc[rn_ys:rn_ye,rn_xs:rn_xe])
    return (off,slope)

def computePriorCost(recon,weights,regparam):
    """
    Compute prior cost function.

    Parameters:
        recon (numpy.ndarray): Reconstruction (intermediate deblurred radiograph)
        direc (numpy.ndarray): Descent direction
        regparam (float): Prior regularization parameter

    Returns:
        float: Prior cost value
        int: Number of pairs of neighboring pixels
    """
    sz = np.shape(recon)
    cost = 0.0
    N = 0.0
    if (regparam != 0):
        for i in ([0,1]):
            if (i == 1): j_sweep = [-1, 0, 1]
            else: j_sweep = [0, 1]
            for j in j_sweep:
                    r_xs = 0; r_xe = sz[1]; rn_xs = 0; rn_xe = sz[1]
                    r_ys = 0; r_ye = sz[0]; rn_ys = 0; rn_ye = sz[0]
                    if (j == -1): r_xs = 1; rn_xe = sz[1]-1;
                    if (i == -1): r_ys = 1; rn_ye = sz[0]-1;
                    if (j == 1): r_xe = sz[1]-1; rn_xs = 1;
                    if (i == 1): r_ye = sz[0]-1; rn_ys = 1;
                    
                    if (i != 0 or j != 0):
                        diff = np.fabs(recon[r_ys:r_ye,r_xs:r_xe] - recon[rn_ys:rn_ye,rn_xs:rn_xe])
                        N += np.size(diff)
                        cost += regparam*weights[i+1,j+1]*np.sum(diff**1.2)
    return (cost,N)

def LineSearchCost(x,forw_weights,forw_off,forw_slope,prior_weights,prior_regparam,prior_off,prior_slope):
    """
    Cost for line search optimization.

    Parameters:
        x (float): Step size for line search
        forw_weights (numpy.ndarray): Weights for each pixel in the forward model
        forw_off (numpy.ndarray): Offset term within the linear error term of forward cost 
        forw_slope (numpy.ndarray): Slope term within the linear error term of forward cost
        prior_weights (numpy.ndarray): 3x3 weights for the prior local neighborhood regularization.
        prior_regparam (float): Prior regularization parameter
        prior_off (numpy.ndarray): Offset term within the linear error term of prior cost  
        prior_slope (numpy.ndarray): Slope term within the linear error term of prior cost

    Returns:
        float: Total cost which is the sum of forward and prior cost values 
    """
    forw_cost = forw_off + x*forw_slope
    forw_cost = 0.5*np.sum(forw_cost*forw_cost*forw_weights)
    prior_cost = 0
    if (prior_regparam != 0): #Improve speed for unregularized reconstruction
        prior_cost = prior_off + x*prior_slope
        prior_cost = prior_regparam*np.sum(prior_weights*(np.fabs(prior_cost)**1.2))
    return (forw_cost+prior_cost)

def least_squares_deblur(norm_rad,sod,sdd,pix_wid,src_params,det_params,reg_param,init_rad=None,weights=None,convg_thresh=1e-4):
    """
    Function to reduce blur (deblur) in radiographs using a regularized least squares iterative algorithm.     

    Parameters:
        norm_rad (numpy.ndarray): Normalized radiograph to deblur
        sod (float): Source to object distance (SOD) of the radiograph
        sdd (float): Source to detector distance (SDD) of the radiograph
        pix_wid (float): Effective width of each detector pixel. Note that this is the effective pixel size given by dividing the physical width of each detector pixel by the zoom factor of the optical lens.
        src_params (dict): Estimated parameters of X-ray source PSF. It should consist of several key-value pairs. The value for key source_FWHM_x_axis is the full width half maximum (FWHM) of the source PSF along the x-axis (i.e., second array dimension). The value for key source_FWHM_y_axis is the FWHM of source PSF along the y-axis (i.e., first array dimension). All FWHMs are for the source PSF in the plane of the X-ray source (and not the detector plane). The value for key cutoff_FWHM_multiplier decides the non-zero spatial extent of the source blur PSF. The PSF is clipped to zero beginning at a distance, as measured from the PSF's origin, of cutoff_FWHM_multiplier times half the maximum FWHM (maximum of source_FWHM_x_axis and source_FWHM_y_axis).
        det_params (dict): Estimated parameters of detector blur PSF. It should consist of several key-value pairs. The value for key detector_FWHM_1 is the FWHM of the first exponential in the mixture density model for detector blur. The first exponential is the most dominant part of detector blur. The value for key detector_FWHM_2 is the FWHM of the second exponential in the mixture density model. This exponential has the largest FWHM and models the long running tails of the detector blur's point spread function (PSF). The value for key detector_weight_1 is between 0 and 1 and is an approximate measure of the amount of contribution of the first exponential to the detector blur. The values for keys cutoff_FWHM_1_multiplier and cutoff_FWHM_2_multiplier decide the non-zero spatial extent of the detector PSF. The PSF is clipped to zero beginning at a distance, as measured from the PSF's origin, of the maximum of cutoff_FWHM_1_multiplier*detector_FWHM_1/2 and cutoff_FWHM_2_multiplier*detector_FWHM_2/2.
        reg_param (float): Regularization parameter for the least squares deblurring algorithm. Noise and ringing artifacts in the deblurred radiograph decreases with increasing values for reg_param and vice versa. But, note that increasing reg_param can also result in excessive blurring due to over-regulization. It is recommended to empirically choose this parameter by increasing or decreasing it by a factor, greater than one (such as 2 or 10), until the desired image quality is achieved.
        init_rad (numpy.ndarray): Initial estimate for the deblurred radiograph. If set to None, then the blurred radiograph is used as an initial estimate. If not None, then init_rad must be a numpy.ndarray of the same shape as norm_rad. 
        weights (numpy.ndarray): Used to increase or decrease reliance on a particular pixel of norm_rad in the forward model cost function. If set to None, every pixel is assigned the same weight of 1.
        convg_thresh (float): Convergence threshold for the minimizer used to deblur the input radiograph. The iterations stop when the ratio of the reduction in the error function (cost value) and the magnitude of the error function is lower than convg_thresh. This is the parameter ftol that is specified in the options parameter of scipy.optimize.minimize. The optimizer used is L-BFGS-B.
    
    Returns:
        numpy.ndarray: Deblurred radiograph using regularized least squares algorithm. 
    """
    max_wid = pix_wid*min(norm_rad.shape)/2
    if weights is None:
        weights = 1.0
    
    src_mod = SourceBlur(pix_wid,max_wid,sod,sdd-sod,src_params['cutoff_FWHM_multiplier'],param_x=get_scale(src_params['source_FWHM_x_axis']),param_y=get_scale(src_params['source_FWHM_y_axis']))
    det_mod = DetectorBlur(pix_wid,max_wid,det_params['cutoff_FWHM_1_multiplier'],det_params['cutoff_FWHM_2_multiplier'],param_1=get_scale(det_params['detector_FWHM_1']),param_2=get_scale(det_params['detector_FWHM_2']),weight_1=det_params['detector_weight_1']) 
        
    blur_psf = combine_psfs(src_mod.get_psf(),det_mod.get_psf(),are_psf=True)
    
    padw = np.array(blur_psf.shape)//2-1
    prior_weights = np.zeros([3,3])
    for i in ([-1,0,1]):
        for j in ([-1,0,1]):
            if (i != 0 or j != 0):
                prior_weights[i+1,j+1] = 1.0/np.sqrt(i**2+j**2)
    prior_weights = prior_weights/np.sum(prior_weights)

    if init_rad is not None:
        deblurred_rad = deepcopy(init_rad)
    else:
        deblurred_rad = deepcopy(norm_rad)
    padw = np.array(blur_psf.shape)//2-1
    rad_shape = norm_rad.shape
    #rad_size = norm_rad.size 
 
    def forw_cost_func(rad):
        pred = convolve_psf(np.pad(rad,padw,'edge'),blur_psf,padw,is_psf=True,warn=False)[padw[0]:-padw[0],padw[1]:-padw[1]]
        #forw_cost = 0.5*np.sum(((norm_rad-pred)**2)*weights)/rad_size
        forw_cost = 0.5*np.sum(((norm_rad-pred)**2)*weights)
        return forw_cost    

    #prior_cost,prior_size = computePriorCost(deblurred_rad,prior_weights,reg_param)
    #print("Initial cost is {}".format(forw_cost_func(deblurred_rad)+prior_cost/prior_size)) 
    #print("Initial cost is {}".format(forw_cost_func(deblurred_rad)+prior_cost)) 
 
    def gradient_func(x):
        x = x.reshape(rad_shape) 
        pred = convolve_psf(np.pad(x,padw,'edge'),blur_psf,padw,is_psf=True,warn=False)[padw[0]:-padw[0],padw[1]:-padw[1]]
        # Initialize error sinogram, e
        pred = (norm_rad-pred)*weights
        grad = -convolve_psf(np.pad(pred,padw,'edge'),blur_psf,padw,is_psf=True,warn=False)[padw[0]:-padw[0],padw[1]:-padw[1]]
        #grad = grad/rad_size+computePriorGradient(x,prior_weights,reg_param)/prior_size
        grad = grad+computePriorGradient(x,prior_weights,reg_param)
        return grad.ravel()

    def cost_func(x): 
        x = x.reshape(rad_shape) 
        prior_cost,_ = computePriorCost(x,prior_weights,reg_param)
        #cost = forw_cost_func(x)+prior_cost/prior_size
        forw_cost = forw_cost_func(x)
        cost = forw_cost+prior_cost
        print("Cost is {}={}+{}".format(cost,forw_cost,prior_cost))
        return cost 

    res = minimize(cost_func,deblurred_rad.ravel(),method='L-BFGS-B',jac=gradient_func,options={'disp':True,'ftol':convg_thresh})
    return res.x.reshape(rad_shape)

def wiener_deblur(norm_rad,sod,sdd,pix_wid,src_params,det_params,reg_param):
    """
    Function to reduce blur (deblur) in a radiograph using Wiener filtering.     

    Parameters:
        norm_rad (numpy.ndarray): Normalized radiograph to deblur
        sod (float): Source to object distance (SOD) of the radiograph
        sdd (float): Source to detector distance (SDD) of the radiograph
        pix_wid (float): Effective width of each detector pixel. Note that this is the effective pixel size given by dividing the physical width of each detector pixel by the zoom factor of the optical lens.
        src_params (dict): Estimated parameters of X-ray source PSF. It should consist of several key-value pairs. The value for key source_FWHM_x_axis is the full width half maximum (FWHM) of the source PSF along the x-axis (i.e., second array dimension). The value for key source_FWHM_y_axis is the FWHM of source PSF along the y-axis (i.e., first array dimension). All FWHMs are for the source PSF in the plane of the X-ray source (and not the detector plane). The value for key cutoff_FWHM_multiplier decides the non-zero spatial extent of the source blur PSF. The PSF is clipped to zero beginning at a distance, as measured from the PSF's origin, of cutoff_FWHM_multiplier times half the maximum FWHM (maximum of source_FWHM_x_axis and source_FWHM_y_axis).
        det_params (dict): Estimated parameters of detector blur PSF. It should consist of several key-value pairs. The value for key detector_FWHM_1 is the FWHM of the first exponential in the mixture density model for detector blur. The first exponential is the most dominant part of detector blur. The value for key detector_FWHM_2 is the FWHM of the second exponential in the mixture density model. This exponential has the largest FWHM and models the long running tails of the detector blur's point spread function (PSF). The value for key detector_weight_1 is between 0 and 1 and is an approximate measure of the amount of contribution of the first exponential to the detector blur. The values for keys cutoff_FWHM_1_multiplier and cutoff_FWHM_2_multiplier decide the non-zero spatial extent of the detector PSF. The PSF is clipped to zero beginning at a distance, as measured from the PSF's origin, of the maximum of cutoff_FWHM_1_multiplier*detector_FWHM_1/2 and cutoff_FWHM_2_multiplier*detector_FWHM_2/2.
        reg_param (float): Regularization parameter for Wiener filtering deblurring algorithm. Noise and ringing artifacts in the deblurred radiograph decreases with increasing values for reg_param and vice versa. But, note that increasing reg_param can also result in excessive blurring due to over-regulization. It is recommended to empirically choose this parameter by increasing or decreasing it by a factor, greater than one (such as 2 or 10), until the desired image quality is achieved.
    
    Returns:
        numpy.ndarray: Deblurred radiograph using a Wiener filter. 
    """

    max_wid = pix_wid*min(norm_rad.shape)/2
    
    src_mod = SourceBlur(pix_wid,max_wid,sod,sdd-sod,src_params['cutoff_FWHM_multiplier'],param_x=get_scale(src_params['source_FWHM_x_axis']),param_y=get_scale(src_params['source_FWHM_y_axis']))
    det_mod = DetectorBlur(pix_wid,max_wid,det_params['cutoff_FWHM_1_multiplier'],det_params['cutoff_FWHM_2_multiplier'],param_1=get_scale(det_params['detector_FWHM_1']),param_2=get_scale(det_params['detector_FWHM_2']),weight_1=det_params['detector_weight_1']) 
        
    blur_psf = combine_psfs(src_mod.get_psf(),det_mod.get_psf(),are_psf=True)
    
    padw = np.array(blur_psf.shape)//2-1
    norm_rad_pad = np.pad(norm_rad,padw,'edge')
    deblur_rad = wiener(norm_rad_pad,blur_psf,float(reg_param),clip=False)
    return deblur_rad.astype(float)[padw[0]:-padw[0],padw[1]:-padw[1]]

"""
xsel1,ysel1,xsel2,ysel2 = None,None,None,None
def callback(click,release):
    global xsel1,ysel1,xsel2,ysel2
    xsel1,ysel1 = click.xdata,click.ydata
    xsel2,ysel2 = release.xdata,release.ydata
    print("Selected region is from ({:.1f},{:.1f}) to ({:.1f},{:.1f})".format(xsel1,ysel1,xsel2,ysel2))

def selector(event):
    print(None)

def least_squares_deblur_old(norm_rad,sod,sdd,pix_wid,src_params,det_params,reg_param,init_rad=None,weights=None,max_iters=100,func_tol=1e-9,compute_cost=True):
    max_wid = pix_wid*min(norm_rad.shape)/2
    
    src_mod = SourceBlur(pix_wid,max_wid,sod,sdd-sod,src_params['cutoff_FWHM_multiplier'],param_x=get_scale(src_params['source_FWHM_x_axis']),param_y=get_scale(src_params['source_FWHM_y_axis']))
    det_mod = DetectorBlur(pix_wid,max_wid,det_params['cutoff_FWHM_1_multiplier'],det_params['cutoff_FWHM_2_multiplier'],param_1=get_scale(det_params['detector_FWHM_1']),param_2=get_scale(det_params['detector_FWHM_2']),weight_1=det_params['detector_weight_1']) 
        
    blur_psf = combine_psfs(src_mod.get_psf(),det_mod.get_psf(),are_psf=True)
    
    padw = np.array(blur_psf.shape)//2-1
    prior_weights = np.zeros([3,3])
    for i in ([-1,0,1]):
        for j in ([-1,0,1]):
            if (i != 0 or j != 0):
                prior_weights[i+1,j+1] = 1.0/np.sqrt(i**2+j**2)
    prior_weights = prior_weights/np.sum(prior_weights)
    prior_regparam = float(reg_param)
     
    #if(prior_regparam == 'ml'):
    #    print("Select a rectangular region with uniform gray values")
    #    fig, ax = plt.subplots()
    #    plt.imshow(blurred_image.values)
    #    selector.RS = RectangleSelector(ax,callback,drawtype='box',useblit=True,button=[1,3],minspanx=5,minspany=5,spancoords='pixels',interactive=True)
    #    plt.show()
    #    variance = np.var(blurred_image.values[int(ysel1):int(ysel2),int(xsel1):int(xsel2)])
    #    prior_param = regparam_estimate(blurred_image.values,prior_weights)
    #    prior_regparam = variance*prior_param
    #    print("Data variance is {} and prior parameter estimate is {}".format(variance,prior_param))
    #    print("Estimated value of regularization parameter is {}".format(prior_regparam))
    #elif not (isinstance(float(prior_regparam),float)):
    #    print("ERROR: Parameter prior_regparam must be a number") 

    #prior_regparam = float(prior_regparam)
    
    cost = np.zeros(max_iters+1)
    forw_cost = np.zeros(max_iters+1)
    prior_cost = np.zeros(max_iters+1)
    bsh = norm_rad.shape
    error = np.zeros(bsh,dtype=float)
    forw_slope = np.zeros(bsh,dtype=float)
    neggrad = np.zeros(bsh,dtype=float)
    update = np.zeros(bsh,dtype=float)
    if weights is None:
        weights = 1.0

    if init_rad is not None:
        deblurred_rad = deepcopy(init_rad)
    else:
        deblurred_rad = deepcopy(norm_rad)
    padw = np.array(blur_psf.shape)//2-1
    
    neggrad_prev = np.zeros(neggrad.shape)
    blurred_pred = convolve_psf(np.pad(deblurred_rad,padw,'edge'),blur_psf,padw,is_psf=True)[padw[0]:-padw[0],padw[1]:-padw[1]]
    # Initialize error sinogram, e
    error = norm_rad-blurred_pred
    if (compute_cost == True):    
        forw_cost[0] = 0.5*np.sum(error*error*weights)
        (prior_cost[0],NS) = computePriorCost(deblurred_rad,prior_weights,prior_regparam)
        cost[0] = forw_cost[0] + prior_cost[0]
        print("Iteration = " + str(0) + ": Cost is " + str(cost[0]) + ", forward cost is " + str(forw_cost[0]) + ", prior cost is " + str(prior_cost[0]))

    for i in range(1,max_iters):    
        blurred_pred = error*weights
        blurred_pred = convolve_psf(np.pad(blurred_pred,padw,'edge'),blur_psf,padw,is_psf=True)[padw[0]:-padw[0],padw[1]:-padw[1]]
        neggrad = blurred_pred
        neggrad -= computePriorGradient(deblurred_rad,prior_weights,prior_regparam)
        if (i <= 1):
            direc = np.copy(neggrad)
        else:
            beta = np.sum(neggrad*(neggrad-neggrad_prev))/np.sum(neggrad_prev*neggrad_prev)
            direc = neggrad + max(0,beta)*direc
        neggrad_prev = np.copy(neggrad)
        #direc = np.copy(neggrad)
        
        blurred_pred = direc
        blurred_pred = convolve_psf(np.pad(blurred_pred,padw,'edge'),blur_psf,padw,is_psf=True)[padw[0]:-padw[0],padw[1]:-padw[1]]
        forw_slope = -blurred_pred
            
        (prior_off,prior_slope) = computePriorLineSearchParams(deblurred_rad,direc,prior_weights,prior_regparam)

        args = (weights,error,forw_slope,prior_weights,prior_regparam,prior_off,prior_slope)
        res = minimize_scalar(LineSearchCost,args=args,method='Brent')
        step = res.x

        #Update reconstruction
        update = step*direc
        deblurred_rad = deblurred_rad + update

        blurred_pred = deblurred_rad
        blurred_pred = convolve_psf(np.pad(blurred_pred,padw,'edge'),blur_psf,padw,is_psf=True)[padw[0]:-padw[0],padw[1]:-padw[1]]
        error = norm_rad-blurred_pred
        
        #Compute cost
        if (compute_cost == True):
            forw_cost[i] = 0.5*np.sum(error*error*weights)
            (prior_cost[i],NS) = computePriorCost(deblurred_rad,prior_weights,prior_regparam)
            cost[i] = forw_cost[i] + prior_cost[i]
                
            print("Iteration = " + str(i) + ": Cost is " + str(cost[i]) + ", forward cost is " + str(forw_cost[i]) + ", prior cost is " + str(prior_cost[i]))
            if (cost[i] > cost[i-1]):
                print("ERROR - Cost increased!")
                print("Increase in cost is " + str(cost[i]-cost[i-1]) + "; Percentage increase is " + str((cost[i]-cost[i-1])/cost[i-1]*100.0))
                sys.exit()
        #percent_update = np.sqrt(np.sum(update*update)/np.sum(recon*recon))*100
        abs_update = np.max(np.fabs(update))
        #print ("Percentage update is ", percent_update)
        #print (step, np.mean(direc), np.mean(forw_grad), np.mean(prior_grad), np.mean(prior_weights))
            
        #print("Percentage change in reconstruction is " + str(percent_update))
        print("Max change in reconstruction is " + str(abs_update))
        if (abs_update < func_tol):
            print("Least squares recon has converged at iteration = " + str(i))
            break
        
    if(i+1 <  max_iters):
        forw_cost[i+1:] = forw_cost[i]
        prior_cost[i+1:] = prior_cost[i]
        cost[i+1:] = cost[i]
            
    return deblurred_rad,cost,forw_cost,prior_cost
"""
