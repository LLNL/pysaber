import time
import numpy as np
import yaml
from pysaber.trans import ideal_trans_sharp_edge
from pysaber.models import SourceBlur,DetectorBlur,Transmission,get_scale,get_FWHM,combine_psfs,convolve_psf
from pysaber.optim import error_function,jacobian_function,set_model_params
from scipy.optimize import minimize#,check_grad,approx_fprime

def get_blur_params(norm_rads,sod,sdd,pix_wid,convg_thresh=1e-6,bdary_mask_perc=5,pad_factor=[3,3],mask=None,edge_type=None):
    """
        Estimate parameters of point spread functions (PSF) that model X-ray source blur and detector blur from normalized radiographs of a straight sharp edge or two mutually perpendicular sharp edges. 

        This function is used to estimate parameters of the PSFs that model X-ray source blur and detector blur. It takes as input the normalized radiographs at multiple source to object distances (SOD) and source to detector distances (SDD). If each radiograph has a single straight edge, then the measurement must be repeated for two different, preferably perpendicular, orientations of the edge. Currently, only a single straight edge in a radiograph is verified to work. The case of perpendicular edges in a radiograph will be supported soon. For more details, please refer to the document listed in the references.  

        Parameters:
            norm_rads (list): Python list of normalized radiographs, each of type numpy.ndarray, at various SODs and SDDs.
            sod (list): List of source to object distances (SOD), each of type float, at which each normalized radiograph in the list norm_rads was acquired.  
            sdd (list): List of source to detector distances (SDD), each of type float, at which each normalized radiograph in the list norm_rads was acquired.
            pix_wid (float): Effective width of each detector pixel. Note that this is the effective pixel size given by dividing the physical width of each detector pixel by the zoom factor of the optical lens.
            convg_thresh (float): Convergence threshold for the minimizer in the last final step when estimating source, detector, and transmission parameters. The convergence threshold for the minimizer during the first two initialization steps is ten times this value. The iterations stop when the ratio of the reduction in the error function (cost value) and the magnitude of the error function is lower than convg_thresh. This is the parameter ftol that is specified in the options parameter of scipy.optimize.minimize. The optimizer used is L-BFGS-B.
            bdary_mask_perc (float): Percentage of image region in the normalized radiographs that must be excluded from blur estimation. Pixels are excluded (or masked) beginning from the outermost periphery of the image and working inwards until the specified percentage of pixels is reached.
            pad_factor (list): Pad factor is a list of two integers that determine the amount of padding that must be applied to the radiographs to reduce aliasing during convolution. The number of rows/columns after padding is equal to pad_factor[0]/pad_factor[1] times the number of rows/columns in each normalized radiograph before padding. For example, if the first element in pad_factor is 2, then the radiograph is padded to twice its size along the first dimension.  
            mask (numpy.ndarray): Boolean mask of the same shape as the radiograph that is used to exclude pixels from blur estimation. An example use case is if some pixels in the radiographs are bad, then those pixels can be excluded from blur estimation by setting the corresponding entries in mask to false and true otherwise.
            edge_type (str): Used to indicate whether there is a single straight edge or two mutually perpendicular edges in each radiograph. If edge_type is perpendicular, then each radiograph is assumed to have two mutually perpendicular edges and a single straight edge otherwise. Currently, choosing perpendicular as edge_type is not recommended since it isn't a verified functionality and may lead to unstable behavior.  

        Returns:
            dict: Estimated parameters of X-ray source PSF that is returned as a python dictionary. It consists of several key-value pairs. The value for key source_FWHM_x_axis is the full width half maximum (FWHM) of the source PSF along the x-axis (i.e., second array dimension). The value for key source_FWHM_y_axis is the FWHM of source PSF along the y-axis (i.e., first array dimension). All FWHMs are for the source PSF in the plane of the X-ray source (and not the plane of the detector). The value for key cutoff_FWHM_multiplier decides the non-zero spatial extent of the exponential PSF. The PSF is clipped to zero beginning at a distance, as measured from the PSF's origin, of cutoff_FWHM_multiplier times half the maximum FWHM (maximum of source_FWHM_x_axis and source_FWHM_y_axis).
            dict: Estimated parameters of detector PSF that is returned as a python dictionary. It consists of several key-value pairs. The value for key detector_FWHM_1 is the FWHM of the first exponential in the mixture density model for detector blur. The first exponential is the most dominant part of detector blur. The value for key detector_FWHM_2 is the FWHM of the second exponential in the mixture density model. This exponential has the largest FWHM and models the long running tails of the detector blur's PSF. The value for key detector_weight_1 is between 0 and 1 and is an approximate measure of the amount of contribution of the first exponential to the detector blur. The values for keys cutoff_FWHM_1_multiplier and cutoff_FWHM_2_multiplier decide the non-zero spatial extent of the detector PSF. The PSF is clipped to zero beginning at a distance, as measured from the PSF's origin, of the maximum of cutoff_FWHM_1_multiplier*detector_FWHM_1/2 and cutoff_FWHM_2_multiplier*detector_FWHM_2/2.
            list: Estimated parameters of the transmission function for each input radiograph. This return value is a list of lists, where each inner nested list consists of two parameters of type float. These float values give the low and high values respectively of the transmission function. The number of nested lists in the returned list equals the number of input radiographs. Note that the transmission function is the normalized radiograph image that would have resulted in the absence of blur and noise. 
    """
    start_time = time.time()
    trans_models,trans_params,trans_bounds = [],[],[]
    for j in range(len(norm_rads)):
        if edge_type == 'perpendicular':
            trans_dict,params,bounds = ideal_trans_perp_corner(norm_rads[j],bdary_mask_perc=bdary_mask_perc,pad_factor=pad_factor,mask=mask)
        else:
            trans_dict,params,bounds = ideal_trans_sharp_edge(norm_rads[j],bdary_mask_perc=bdary_mask_perc,pad_factor=pad_factor,mask=mask)
            
        #trans_dict has ideal transmission function, masks, etc
        trans_models.append(trans_dict)
        #Add trans_dict to list trans_models 
        trans_params.append(params)
        trans_bounds.append(bounds)
            
        #print("Initial estimates of transmission parameters for radiograph {} are {}.".format(j,params))
        #print("{:.2e} mins has elapsed.".format((time.time()-start_time)/60.0))

    #---------------------------- ESTIMATE BLUR MODEL --------------------------------
    #It is recommended to run the blur model estimation in multiple stages
    #First, only estimate the detector blur using radiographs with the largest SODs
    #Radiographs with the largest SODs are expected to have minimal source blur
    sod_avg = sum(sod)/len(sod) #Compute average of all SODs

    #Only include list elements with SOD > average of all SODs
    norm_rads_est1,trans_models_est1,trans_params_est1,trans_bounds_est1,sod_est1,sdd_est1 = [],[],[],[],[],[]
    for i in range(len(norm_rads)):
        if sod[i] > sod_avg:
            norm_rads_est1.append(norm_rads[i])
            trans_models_est1.append(trans_models[i])
            trans_params_est1.append(trans_params[i])
            trans_bounds_est1.append(trans_bounds[i])
            sod_est1.append(sod[i])
            sdd_est1.append(sdd[i])

    #Estimate detector blur parameters
    print("----------- Init stage: Estimation of only detector blur model parameters -----------")
    _,det_params_est1,_,_,cost_det = estimate_blur_psfs(norm_rads_est1,trans_models_est1,sod_est1,sdd_est1,pix_wid,src_est=False,det_est=True,trans_est=False,trans_params=trans_params_est1,trans_bounds=trans_bounds_est1,convg_thresh=convg_thresh*10)
    print("Estimated detector parameters are {}".format(det_params_est1))
    print("{:.2f} mins has elapsed".format((time.time()-start_time)/60.0))

    #Next, only estimate the source blur using radiographs at the smallest source to object distances
    #Only include list elements with SOD < average of all SODs
    norm_rads_est2,trans_models_est2,trans_params_est2,trans_bounds_est2,sod_est2,sdd_est2 = [],[],[],[],[],[]
    for i in range(len(norm_rads)):
        if sod[i] < sod_avg:
            norm_rads_est2.append(norm_rads[i])
            trans_models_est2.append(trans_models[i])
            trans_params_est2.append(trans_params[i])
            trans_bounds_est2.append(trans_bounds[i])
            sod_est2.append(sod[i])
            sdd_est2.append(sdd[i])

    #Estimate source blur parameters
    print("----------- Init stage: Estimation of only source blur model parameters -----------")
    src_params_est2,_,_,_,cost_src = estimate_blur_psfs(norm_rads_est2,trans_models_est2,sod_est2,sdd_est2,pix_wid,src_est=True,det_est=False,trans_est=False,trans_params=trans_params_est2,trans_bounds=trans_bounds_est2,convg_thresh=convg_thresh*10)
    print("Estimated source parameters are {}".format(src_params_est2))
    print("{:.2f} mins has elapsed".format((time.time() - start_time)/60.0))

    #Write source blur parameters to file
    with open('source_params_init.yml','w') as fid:
        yaml.safe_dump(src_params_est2,fid,default_flow_style=False)

    #Write detector blur parameters to file
    with open('detector_params_init.yml','w') as fid:
        yaml.safe_dump(det_params_est1,fid,default_flow_style=False)

    #Write transmission function parameters to file
    trans_dict = {}
    for i in range(len(norm_rads)):
        label = 'radiograph_{}'.format(i)
        trans_dict[label] = {}
        trans_dict[label]['min param'] = trans_params[i][0] 
        trans_dict[label]['max param'] = trans_params[i][1] 
    with open('transmission_params_init.yml','w') as fid:
        yaml.safe_dump(trans_dict,fid,default_flow_style=False)

    #Finally, refine estimates of both source and detector blur using all radiographs
    print("-------- Final stage: Estimation of source, detector and transmission model parameters --------")
    src_params,det_params,trans_params,trans_bounds,cost = estimate_blur_psfs(norm_rads,trans_models,sod,sdd,pix_wid,src_est=True,det_est=True,trans_est=True,src_params=src_params_est2,det_params=det_params_est1,trans_params=trans_params,trans_bounds=trans_bounds,convg_thresh=convg_thresh)
    print("Blur estimation is complete.")
    print("Estimated source parameters are {}, detector parameters are {}, and transmission function parameters are {}".format(src_params,det_params,trans_params))
    print("{:.2f} mins has elapsed".format((time.time()-start_time)/60.0))

    #Write source blur parameters to file
    with open('source_params.yml','w') as fid:
        yaml.safe_dump(src_params,fid,default_flow_style=False)

    #Write detector blur parameters to file
    with open('detector_params.yml','w') as fid:
        yaml.safe_dump(det_params,fid,default_flow_style=False)

    #Write transmission function parameters to file
    trans_dict = {}
    for i in range(len(norm_rads)):
        label = 'radiograph_{}'.format(i)
        trans_dict[label] = {}
        trans_dict[label]['min param'] = trans_params[i][0] 
        trans_dict[label]['max param'] = trans_params[i][1] 
    with open('transmission_params.yml','w') as fid:
        yaml.safe_dump(trans_dict,fid,default_flow_style=False)

    with open('cost.yml','w') as fid:
        yaml.safe_dump({'error final':cost,'error source init':cost_src,'error detector init':cost_det},fid,default_flow_style=False)

    return src_params,det_params,trans_params

def estimate_blur_psfs(norm_rads,trans_models,sod,sdd,pix_wid,src_est=True,det_est=True,trans_est=True,src_params=None,det_params=None,trans_params=None,trans_bounds=None,convg_thresh=1e-6):
    """
    Estimate parameters of the blur and transmission models.

    Parameters:
        norm_rads (list): List of normalized radiographs
        trans_models (list): List of transmission models
        sod (list): List of source to object distances (SOD)
        sdd (list): List of source to detector distances (SDD)
        pix_wid (float): Width of each detector pixel
        src_est (bool): If true, source parameters are estimated
        det_est (bool): If true, detector parameters are estimated
        trans_est (bool): If true, transmission model parameters are estimated
        src_params (dict): Dictionary of source model parameters
        det_params (dict): Dictionary of detector model parameters
        trans_params (list): List of lists of transmission model parameters
        trans_bounds (list): Bounds on transmission model parameters
        convg_thresh (float): Used to determine convergence of blur estimator (optimizer). It is parameter 'ftol' among optional parameters of function scipy.optimize.minimize.

    Returns:
        dict: Estimated source model parameters
        numpy.ndarray: 2D array of source blur point spread function (PSF)
        dict: Estimated detector model parameters
        numpy.ndarray: 2D array of detector blur point spread function (PSF)
        list: List of lists of transmission model parameters
        list: List of bounds on each of the transmission model parameters for each radiograph
        list: List of predicted images using the blur model
        float: Error function value (cost)     
    """ 
    max_mag = max([(sdd_smpl-sod_smpl)/sod_smpl for sdd_smpl,sod_smpl in zip(sdd,sod)])
    min_mag = min([(sdd_smpl-sod_smpl)/sod_smpl for sdd_smpl,sod_smpl in zip(sdd,sod)])
    if src_params is None:
        src_params = {}
        src_params['source_FWHM_x_axis'] = pix_wid/max_mag if src_est else 0.0
        src_params['source_FWHM_y_axis'] = 2*pix_wid/max_mag if src_est else 0.0
        src_params['cutoff_FWHM_multiplier'] = 10

    if det_params is None:
        det_params = {}
        det_params['detector_FWHM_1'] = pix_wid if det_est else 0.0
        det_params['detector_FWHM_2'] = 10*pix_wid if det_est else 0.0
        det_params['detector_weight_1'] = 0.9 if det_est else 0.0
        det_params['cutoff_FWHM_1_multiplier'] = 10
        det_params['cutoff_FWHM_2_multiplier'] = 10
 
    if trans_bounds is None:
        trans_bounds = []
        if trans_params is None:
            print("Assuming a sharp edge radiograph for bounding transmission parameters")
            for i in range(len(norm_rads)):
                trans_bounds.append([[-1.0,0.5],[0.5,2.0]]) 
        else:
            for param in trans_params:
                trans_bounds.append([[None,None] for i in range(len(param))]) 
 
    if trans_params is None:
        print("Assuming a sharp edge radiograph for transmission parameter initialization")
        trans_params = [[0.0,1.0] for i in range(len(norm_rads))] 

    src_mods,trans_mods = [],[]
    for i in range(len(norm_rads)):
        trans_mods.append(Transmission(trans_models[i],trans_params[i]))
        #print(trans_mods[-1].psf_max_halfsize,pix_wid)
        if i>0:
            assert max_wid == pix_wid*trans_mods[-1].psf_max_halfsize
        max_wid = pix_wid*trans_mods[-1].psf_max_halfsize
        src_mods.append(SourceBlur(pix_wid,max_wid,sod[i],sdd[i]-sod[i],src_params['cutoff_FWHM_multiplier'],param_x=get_scale(src_params['source_FWHM_x_axis']),param_y=get_scale(src_params['source_FWHM_y_axis']),warn=False))
    det_mod = DetectorBlur(pix_wid,max_wid,det_params['cutoff_FWHM_1_multiplier'],det_params['cutoff_FWHM_2_multiplier'],param_1=get_scale(det_params['detector_FWHM_1']),param_2=get_scale(det_params['detector_FWHM_2']),weight_1=det_params['detector_weight_1'],warn=False) 
 
    args_var = []
    args_bounds = []
    if src_est == True:
        args_var.append(get_scale(src_params['source_FWHM_x_axis']))
        args_bounds.append([1e-10,None])
        args_var.append(get_scale(src_params['source_FWHM_y_axis']))
        args_bounds.append([1e-10,None])
        print("Will optimize source blur parameters")
        
    if det_est == True:
        args_var.append(get_scale(det_params['detector_FWHM_1']))
        args_bounds.append([1e-10,None])
        args_var.append(get_scale(det_params['detector_FWHM_2']))
        args_bounds.append([1e-10,None])
        args_var.append(det_params['detector_weight_1'])
        args_bounds.append([0.8,1.0])
        print("Will optimize detector blur parameters")

    if trans_est == True:
        for par in trans_params:
            args_var.extend(par)
        for bound in trans_bounds:
            args_bounds.extend(bound) 
        print("Will optimize transmission function parameters")

    #eps = np.sqrt(np.finfo(float).eps)
    #print("----------------------------")
    #grad_true = jacobian_function(np.array(args_var),src_mods,det_mod,trans_mods,src_est,det_est,trans_est,norm_rads,pix_wid)
    #grad_num = approx_fprime(np.array(args_var),error_function,eps,src_mods,det_mod,trans_mods,src_est,det_est,trans_est,norm_rads,pix_wid)
    #print("eps is {}".format(eps))
    #print("Error in jacobian at {} is {}. Numerical jacobian is {}. True jacobian is {}".format(args_var,np.sqrt(np.sum((grad_num-grad_true)**2)),grad_num,grad_true))
    #print("###########################")

    cost = None
    if(src_est or det_est or trans_est):
        res = minimize(error_function,args_var,args=(src_mods,det_mod,trans_mods,src_est,det_est,trans_est,norm_rads,pix_wid),method='L-BFGS-B',jac=jacobian_function,bounds=args_bounds,options={'disp':True,'ftol':convg_thresh})
        args_opt = res.x.tolist()
        cost = float(res.fun)
    else:
        args_opt = args_var.copy()

    print("Optimized parameters are {}".format(args_opt))    
    set_model_params(args_opt,src_mods,det_mod,trans_mods,src_est,det_est,trans_est) 
    
    args_opt.reverse() 
    if src_est == True:
        src_params['source_FWHM_x_axis'] = float(get_FWHM(args_opt.pop()))
        src_params['source_FWHM_y_axis'] = float(get_FWHM(args_opt.pop()))
        #print("Compare",max_wid,src_params['cutoff_FWHM_multiplier']*max(src_params['source_FWHM_x_axis'],src_params['source_FWHM_y_axis'])*max_mag/2)
        if max_wid < src_params['cutoff_FWHM_multiplier']*max(src_params['source_FWHM_x_axis'],src_params['source_FWHM_y_axis'])*max_mag/2:
            print("WARNING: The maximum single sided width {:.2e}, as determined by the amount of padding in radiographs, that is allowed for source PSF may be insufficient. This may or may not be a problem. Consider increasing the pad factor if results are inconsistent. An example of a inconsistency is if FWHM changes significantly when pad factor is changed. Increasing padding will increase software run time.".format(max_wid));
         
    if det_est == True:
        det_params['detector_FWHM_1'] = float(get_FWHM(args_opt.pop()))
        det_params['detector_FWHM_2'] = float(get_FWHM(args_opt.pop()))
        det_params['detector_weight_1'] = float(args_opt.pop())
        #print("Compare",max_wid,max(det_params['cutoff_FWHM_1_multiplier']*det_params['detector_FWHM_1'],det_params['cutoff_FWHM_2_multiplier']*det_params['detector_FWHM_2'])/2)
        if max_wid < max(det_params['cutoff_FWHM_1_multiplier']*det_params['detector_FWHM_1'],det_params['cutoff_FWHM_2_multiplier']*det_params['detector_FWHM_2'])/2:
            print("WARNING: The maximum single sided width {:.2e}, as determined by the amount of padding in radiographs, that is allowed for detector PSF may be insufficient. This may or may not be a problem. Consider increasing the pad factor if results are inconsistent. An example of a inconsistency is if FWHM changes significantly when pad factor is changed. Increasing padding will increase software run time.".format(max_wid));

    if trans_est == True:
        trans_params = []
        for mod in trans_mods:
            trans_params.append([args_opt.pop() for _ in range(mod.len_params)])

    #src_psf = src_mods[0].get_psf_at_source()
    #det_psf = det_mod.get_psf()
   
    #pred_nrads = []
    #dmod_psf = det_mod.get_psf()
    #for j in range(len(norm_rads)):
    #    rad = norm_rads[j]
    #    radm,tran,tranm = trans_mods[j].get_trans()
    #    smod_psf = src_mods[j].get_psf()
    #    pad_widths = (np.array(tran.shape)-np.array(rad.shape))//2 #assumes symmetric padding
    #    new_psf = combine_psfs(smod_psf,dmod_psf,are_psf=True)
    #    pred = convolve_psf(tran,new_psf,pad_widths,is_psf=True)
    #    pred_nrads.append(pred)
    #return src_params,src_psf,det_params,det_psf,trans_params,trans_bounds,pred_nrads,cost

    return src_params,det_params,trans_params,trans_bounds,cost

def get_source_psf(pix_wid,src_params,sod=1,sdd=2):
    """
    Get point spread function (PSF) of X-ray source blur.

    Parameters:
        pix_wid (float): Effective width of each detector pixel. Note that this is the effective pixel size given by dividing the physical width of each detector pixel by the zoom factor of the optical lens.
        src_params (dict): Estimated parameters of X-ray source PSF. It should consist of several key-value pairs. The value for key source_FWHM_x_axis is the full width half maximum (FWHM) of the source PSF along the x-axis (i.e., second array dimension). The value for key source_FWHM_y_axis is the FWHM of source PSF along the y-axis (i.e., first array dimension). All FWHMs are for the source PSF in the plane of the X-ray source (and not the detector plane). The value for key cutoff_FWHM_multiplier decides the non-zero spatial extent of the source blur PSF. The PSF is clipped to zero beginning at a distance, as measured from the PSF's origin, of cutoff_FWHM_multiplier times half the maximum FWHM (maximum of source_FWHM_x_axis and source_FWHM_y_axis).
        sod (float): Source to object distance (SOD) is necessary to compute PSF of source blur in the plane of the detector. Use the default value to compute source PSF at the plane of the X-ray source. Or, if sdd = 2*sod, then source blur PSF at the detector is same as the source blur PSF at the plane of the X-ray source. 
        sdd (float): Source to detector distance (SDD) is necessary to compute PSF of source blur in the plane of the detector. Use the default value to compute source PSF at the plane of the X-ray source. Or, if sdd = 2*sod, then source blur PSF at the detector is same as the source blur PSF at the plane of the X-ray source.

    Returns:
        numpy.ndarray: 2D array of source blur PSF
    """
    odd = sdd - sod
    if odd < 0:
        raise ValueError('sdd<sod')
    #if max_wid == None:
    max_wid = 0.5*src_params['cutoff_FWHM_multiplier']*max(src_params['source_FWHM_x_axis'],src_params['source_FWHM_y_axis'])*odd/sod
    src_mod = SourceBlur(pix_wid,max_wid,sod,odd,src_params['cutoff_FWHM_multiplier'],param_x=get_scale(src_params['source_FWHM_x_axis']),param_y=get_scale(src_params['source_FWHM_y_axis']),warn=False)
    return src_mod.get_psf()
     
def get_detector_psf(pix_wid,det_params):
    """
    Get point spread function (PSF) of detector blur
    
    Parameters:
        pix_wid (float): Effective width of each detector pixel. Note that this is the effective pixel size given by dividing the physical width of each detector pixel by the zoom factor of the optical lens.
        det_params (dict): Estimated parameters of detector blur PSF. It should consist of several key-value pairs. The value for key detector_FWHM_1 is the FWHM of the first exponential in the mixture density model for detector blur. The first exponential is the most dominant part of detector blur. The value for key detector_FWHM_2 is the FWHM of the second exponential in the mixture density model. This exponential has the largest FWHM and models the long running tails of the detector blur's point spread function (PSF). The value for key detector_weight_1 is between 0 and 1 and is an approximate measure of the amount of contribution of the first exponential to the detector blur. The values for keys cutoff_FWHM_1_multiplier and cutoff_FWHM_2_multiplier decide the non-zero spatial extent of the detector PSF. The PSF is clipped to zero beginning at a distance, as measured from the PSF's origin, of the maximum of cutoff_FWHM_1_multiplier*detector_FWHM_1/2 and cutoff_FWHM_2_multiplier*detector_FWHM_2/2.

    Returns:
        numpy.ndarray: 2D array of detector blur PSF
    """
    #if max_wid == None:
    max_wid = 0.5*max(det_params['cutoff_FWHM_1_multiplier']*det_params['detector_FWHM_1'],det_params['cutoff_FWHM_2_multiplier']*det_params['detector_FWHM_2'])
    det_mod = DetectorBlur(pix_wid,max_wid,det_params['cutoff_FWHM_1_multiplier'],det_params['cutoff_FWHM_2_multiplier'],param_1=get_scale(det_params['detector_FWHM_1']),param_2=get_scale(det_params['detector_FWHM_2']),weight_1=det_params['detector_weight_1'],warn=False)
    return det_mod.get_psf()

def get_effective_psf(pix_wid,src_params,det_params,sod=1,sdd=2): 
    """
    Get point spread function (PSF) of the combined effect of X-ray source and detector blur.

    Parameters:
        pix_wid (float): Effective width of each detector pixel. Note that this is the effective pixel size given by dividing the physical width of each detector pixel by the zoom factor of the optical lens.
        src_params (dict): Estimated parameters of X-ray source PSF. It should consist of several key-value pairs. The value for key source_FWHM_x_axis is the full width half maximum (FWHM) of the source PSF along the x-axis (i.e., second array dimension). The value for key source_FWHM_y_axis is the FWHM of source PSF along the y-axis (i.e., first array dimension). All FWHMs are for the source PSF in the plane of the X-ray source (and not the detector plane). The value for key cutoff_FWHM_multiplier decides the non-zero spatial extent of the source blur PSF. The PSF is clipped to zero beginning at a distance, as measured from the PSF's origin, of cutoff_FWHM_multiplier times half the maximum FWHM (maximum of source_FWHM_x_axis and source_FWHM_y_axis).
        det_params (dict): Estimated parameters of detector blur PSF. It should consist of several key-value pairs. The value for key detector_FWHM_1 is the FWHM of the first exponential in the mixture density model for detector blur. The first exponential is the most dominant part of detector blur. The value for key detector_FWHM_2 is the FWHM of the second exponential in the mixture density model. This exponential has the largest FWHM and models the long running tails of the detector blur's point spread function (PSF). The value for key detector_weight_1 is between 0 and 1 and is an approximate measure of the amount of contribution of the first exponential to the detector blur. The values for keys cutoff_FWHM_1_multiplier and cutoff_FWHM_2_multiplier decide the non-zero spatial extent of the detector PSF. The PSF is clipped to zero beginning at a distance, as measured from the PSF's origin, of the maximum of cutoff_FWHM_1_multiplier*detector_FWHM_1/2 and cutoff_FWHM_2_multiplier*detector_FWHM_2/2.
        sod (float): Source to object distance (SOD) is necessary to compute PSF of source blur in the plane of the detector. Use the default value to compute source PSF at the plane of the X-ray source. Or, if sdd = 2*sod, then source blur PSF at the detector is same as the source blur PSF at the plane of the X-ray source. 
        sdd (float): Source to detector distance (SDD) is necessary to compute PSF of source blur in the plane of the detector. Use the default value to compute source PSF at the plane of the X-ray source. Or, if sdd = 2*sod, then source blur PSF at the detector is same as the source blur PSF at the plane of the X-ray source.

    Returns:
        numpy.ndarray: 2D array of effective blur PSF
    """
    odd = sdd - sod
    if odd < 0:
        raise ValueError('sdd<sod')

    #if max_wid == None:
    max_wid = 0.5*src_params['cutoff_FWHM_multiplier']*max(src_params['source_FWHM_x_axis'],src_params['source_FWHM_y_axis'])*odd/sod
    src_mod = SourceBlur(pix_wid,max_wid,sod,odd,src_params['cutoff_FWHM_multiplier'],param_x=get_scale(src_params['source_FWHM_x_axis']),param_y=get_scale(src_params['source_FWHM_y_axis']))
    max_wid = 0.5*max(det_params['cutoff_FWHM_1_multiplier']*det_params['detector_FWHM_1'],det_params['cutoff_FWHM_2_multiplier']*det_params['detector_FWHM_2'])
    det_mod = DetectorBlur(pix_wid,max_wid,det_params['cutoff_FWHM_1_multiplier'],det_params['cutoff_FWHM_2_multiplier'],param_1=get_scale(det_params['detector_FWHM_1']),param_2=get_scale(det_params['detector_FWHM_2']),weight_1=det_params['detector_weight_1']) 
    smod_psf = src_mod.get_psf()
    dmod_psf = det_mod.get_psf()
    return combine_psfs(smod_psf,dmod_psf,are_psf=True)

def apply_blur_psfs(norm_rad,sod,sdd,pix_wid,src_params,det_params,padded_widths=[0,0],pad_type='constant',pad_constant=0):
    """
    Blur the input radiograph by applying source and detector blur.
    
    Parameters:
        norm_rad (numpy.ndarray): Normalized radiograph to be blurred
        sod (float): Source to object distance (SOD) of the radiograph. If sdd = 2*sod, then source blur PSF at the detector is same as the source blur PSF at the plane of the X-ray source.
        sdd (float): Source to detector distance (SDD) of the radiograph. If sdd = 2*sod, then source blur PSF at the detector is same as the source blur PSF at the plane of the X-ray source.
        pix_wid (float): Effective width of each detector pixel. Note that this is the effective pixel size given by dividing the physical width of each detector pixel by the zoom factor of the optical lens.
        src_params (dict): Estimated parameters of model for X-ray source blur. It is supposed to consist of several key-value pairs. The value for key source_FWHM_x_axis is the full width half maximum (FWHM) of the source blur PSF along the x-axis (i.e., row-wise). The value for key source_FWHM_y_axis is the FWHM of source PSF along the y-axis (i.e., column-wise). All FWHMs are for the source blur PSF in the plane of X-ray source (and not the detector). The value for key cutoff_FWHM_multiplier decides the spatial extent of the exponential PSF as a multiple of the FWHM beyond which the PSF values are assumed to be zero.
        det_params (dict): Estimated parameters of model for detector blur. It is supposed to consist of several key-value pairs. The value for key detector_FWHM_1 is the FWHM of the first exponential in the mixture density model for detector blur. The first exponential is the most dominant part of detector blur. The value for key detector_FWHM_2 is the FWHM of the second exponential in the mixture density. This exponential has the largest FWHM and models the long running tails of the detector blur's point spread function (PSF). The value for key detector_weight_1 is between 0 and 1 and is an approximate measure of the amount of contribution of the first exponential to the detector blur. The value for key cutoff_FWHM_multiplier decides the spatial extent of the exponential PSF as a multiple of the FWHM beyond which the PSF values are assumed to be zero.
        padded_widths (list): Size of padding already applied to input radiograph. It is a list of two integers. The first integer specifies the padding size at the top and bottom of the input (same sized padding at top and bottom). The second specifies the padding size at the left and right of the input (same sized padding at the left and right). 
        pad_type (str): Type of additional padding that will be used if necessary by the function numpy.pad
        pad_constant (float): Constant value that will be used as padding if necessary by the function numpy.pad

    Returns:
        numpy.ndarray: Blurred radiograph    
    """
    input_rad = norm_rad 
    max_wid = pix_wid*(min(padded_widths)//2)
    #Divide by 2 since there are two PSFs (source and detector blur)
    src_mod = SourceBlur(pix_wid,max_wid,sod,sdd-sod,src_params['cutoff_FWHM_multiplier'],param_x=get_scale(src_params['source_FWHM_x_axis']),param_y=get_scale(src_params['source_FWHM_y_axis']))
    det_mod = DetectorBlur(pix_wid,max_wid,det_params['cutoff_FWHM_1_multiplier'],det_params['cutoff_FWHM_2_multiplier'],param_1=get_scale(det_params['detector_FWHM_1']),param_2=get_scale(det_params['detector_FWHM_2']),weight_1=det_params['detector_weight_1']) 
    smod_psf = src_mod.get_psf()
    dmod_psf = det_mod.get_psf()
    new_psf = combine_psfs(smod_psf,dmod_psf,are_psf=True)
    blurred_rad = convolve_psf(input_rad,new_psf,padded_widths,is_psf=True,pad_type=pad_type,pad_constant=pad_constant)
    return blurred_rad
    
def get_trans_fit(norm_rad,sod,sdd,pix_wid,src_params,det_params,trans_params,pad_factor=[3,3],edge_type=None):
    """
        For a measured radiograph consisting of a straight sharp edge or two mutually perpendicular edges, get the ideal transmission function and a prediction from the blur model for the normalized radiograph in the presence of X-ray source and detector blurs. 
    
        Parameters:
            norm_rad (numpy.ndarray): Normalized radiograph
            sod (float): Source to object distance (SOD) for the radiograph norm_rad.
            sdd (float): Source to detector distance (SDD) for the radiograph norm_rad.
            pix_wid (float): Effective width of each detector pixel. Note that this is the effective pixel size given by dividing the physical width of each detector pixel by the zoom factor of the optical lens.
            src_params (dict): Estimated parameters of X-ray source PSF. It should consist of several key-value pairs. The value for key source_FWHM_x_axis is the full width half maximum (FWHM) of the source PSF along the x-axis (i.e., second array dimension). The value for key source_FWHM_y_axis is the FWHM of source PSF along the y-axis (i.e., first array dimension). All FWHMs are for the source PSF in the plane of the X-ray source (and not the detector plane). The value for key cutoff_FWHM_multiplier decides the non-zero spatial extent of the source blur PSF. The PSF is clipped to zero beginning at a distance, as measured from the PSF's origin, of cutoff_FWHM_multiplier times half the maximum FWHM (maximum of source_FWHM_x_axis and source_FWHM_y_axis).
            det_params (dict): Estimated parameters of detector blur PSF. It should consist of several key-value pairs. The value for key detector_FWHM_1 is the FWHM of the first exponential in the mixture density model for detector blur. The first exponential is the most dominant part of detector blur. The value for key detector_FWHM_2 is the FWHM of the second exponential in the mixture density model. This exponential has the largest FWHM and models the long running tails of the detector blur's point spread function (PSF). The value for key detector_weight_1 is between 0 and 1 and is an approximate measure of the amount of contribution of the first exponential to the detector blur. The values for keys cutoff_FWHM_1_multiplier and cutoff_FWHM_2_multiplier decide the non-zero spatial extent of the detector PSF. The PSF is clipped to zero beginning at a distance, as measured from the PSF's origin, of the maximum of cutoff_FWHM_1_multiplier*detector_FWHM_1/2 and cutoff_FWHM_2_multiplier*detector_FWHM_2/2.
            trans_params (list): A two element integer list of transmission function parameters. The first element is the low value and the second element is the high value of the transmission function. Note that the transmission function is the normalized radiograph image that would have resulted in the absence of blur and noise. 
            pad_factor (list): Pad factor is a list of two integers that determine the amount of padding that must be applied to the radiographs to reduce aliasing during convolution. The number of rows/columns after padding is equal to pad_factor[0]/pad_factor[1] times the number of rows/columns in each normalized radiograph before padding. For example, if the first element in pad_factor is 2, then the radiograph is padded to twice its size along the first dimension.  
            edge_type (str): Used to indicate whether there is a single straight edge or two mutually perpendicular edges in each radiograph. If edge_type is perpendicular, then each radiograph is assumed to have two mutually perpendicular edges and a single straight edge otherwise. Currently, choosing perpendicular as edge_type is not recommended since it isn't a verified functionality and may lead to unstable behavior.  
    """
    if edge_type == 'perpendicular':
        trans_dict,params,bounds = ideal_trans_perp_corner(norm_rad,pad_factor=pad_factor)
    else:
        trans_dict,params,bounds = ideal_trans_sharp_edge(norm_rad,pad_factor=pad_factor)
    trans_mod = Transmission(trans_dict,trans_params)
    _,ideal_trans,_ = trans_mod.get_trans()
    pad_widths = (np.array(ideal_trans.shape)-np.array(norm_rad.shape))//2 #assumes symmetric padding
    pred_rad = apply_blur_psfs(ideal_trans,sod,sdd,pix_wid,src_params,det_params,padded_widths=pad_widths)
    return ideal_trans,pred_rad
