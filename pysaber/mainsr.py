__docformat__ = 'reStructuredText'

import time
import numpy as np
import yaml
import os
from pysaber.transsr import ideal_trans_sharp_edge,ideal_trans_perp_corner
from pysaber.modelssr import SourceBlur,DetectorBlur,Transmission,get_scale,get_FWHM,combine_psfs,convolve_psf
from pysaber.optimsr import error_function,jacobian_function,set_model_params
from scipy.optimize import minimize,check_grad,approx_fprime
import matplotlib.pyplot as plt

def estimate_blur(rads,sod,odd,pix_wid,edge,thresh=1e-6,pad=[3,3],masks=None,bdary_mask=5.0,perp_mask=5.0,power=1.0,save_dir='./',only_src=False,only_det=False,mix_det=True):
    """
    Estimate parameters of point spread functions (PSF) that model X-ray source blur and/or detector blur from normalized radiographs of a straight sharp edge or mutually perpendicular intersecting pair of sharp edges. 

    This function is used to estimate parameters of the PSFs that model X-ray source blur and/or detector blur. It takes as input the normalized radiographs at multiple source to object distances (SOD) and object to detector distances (ODD). If each radiograph has a single straight edge, then the measurement must be repeated for two different, preferably perpendicular, orientations of the edge. If the radiograph consists of two intersecting perpendicular edges, then a single radiograph at each specified SOD/ODD is sufficient. Simultaneous estimation of source and detector blur will require radiographs at a minimum of two different value pairs for SOD/ODD. During PSF parameter estimation, the influence of certain regions within each radiograph can be removed by masking. For more details, please read ahead and also refer to the documents listed in :ref:`sec_refs`.  

    Args:
        rads (list): List of radiographs, each of type numpy.ndarray, at various SODs and ODDs. Each radiograph must be normalized using the bright-field (also called flat-field) and dark-field images.
        sod (list): List of source to object distances (SOD), each of type *float*, at which each corresponding radiograph in the list :attr:`rads` was acquired. 
        odd (list): List of object to detector distances (ODD), each of type *float*, at which each corresponding radiograph in the list :attr:`rads` was acquired.
        pix_wid (float): Effective width of each detector pixel. Note that this is the effective pixel size given by dividing the physical width of each detector pixel by the zoom factor of the optical lens.
        edge (str): Used to indicate whether there is a single straight edge or two mutually perpendicular edges in each radiograph. If :attr:`edge` is ``perpendicular``, then each radiograph is assumed to have two mutually perpendicular edges. If it is ``straight``, then each radiograph is assumed to have a single straight edge. Only ``perpendicular`` and ``straight`` are legal choices for :attr:`edge`. 
        thresh (float): Convergence threshold for the minimizer during parameter estimation. The iterations stop when the ratio of the reduction in the error function (cost value) and the magnitude of the error function is lower than :attr:`thresh`. This is the parameter :attr:`ftol` that is specified in the :attr:`options` parameter of ``scipy.optimize.minimize``. The optimizer used is L-BFGS-B. During joint estimation of source and detector blur, the convergence threshold for the minimizer during the first two initialization steps is ten times this value. 
        pad (list): List of two integers that determine the amount of padding that must be applied to the radiographs to reduce aliasing during convolution. The number of rows/columns after padding is equal to :attr:`pad_factor[0]`/:attr:`pad_factor[1]` times the number of rows/columns in each radiograph before padding. For example, if the first element in :attr:`pad_factor` is ``2``, then the radiograph is padded to twice its size along the first dimension. 
        masks (list): List of boolean masks, each of type numpy.ndarray and same shape as the radiograph, that is used to exclude pixels from blur estimation. This is in addition to the masking specified by :attr:`bdary_mask` and :attr:`perp_mask`. An example use case is if some pixels in the radiograph :attr:`rads[i]` are bad, then those pixels can be excluded from blur estimation by setting the corresponding entries in :attr:`masks[i]` to ``False`` and ``True`` otherwise. If None, no user specified mask is used.
        bdary_mask (float): Percentage of image region in the radiographs as measured from the outer edge going inwards that must be excluded from blur estimation. Pixels are excluded (or masked) beginning from the outermost periphery of the image and working inwards until the specified percentage of pixels is reached.
        perp_mask (float): Percentage of circular region to ignore during blur estimation around the intersecting corner of two perpendicular edges. Ignored if :attr:`edge` is ``straight``.
        power (float): Shape parameter of the density function used to model each PSF. For example, choosing a value of one for :attr:`power` creates an exponential (Laplacian) density function. Choosing a value of two for :attr:`power` creates a Gaussian density function.
        save_dir (str): Directory where estimated parameters are saved in *yaml* file format. Source blur parameters are saved in the file ``source_params.yml`` within the folder :attr:`save_dir`. Similary, detector blur and transmission function parameters are saved as ``detector_params.yml`` and ``transmission_params.yml``.
        only_src (bool): If ``True``, only estimate source blur parameters.
        only_det (bool): If ``True``, only estimate detector blur parameters.
        mix_det (bool): If ``True``, do not use mixture model for detector blur.

    Returns:
        tuple: Tuple of objects containing the estimated parameters. If estimating both source and detector blur parameters, returns the three element tuple (``src_pars``, ``det_pars``, ``tran_pars``). If estimating only source blur parameters, returns the two element tuple (``src_pars``, ``tran_pars``). If estimating only detector blur parameters, returns the two element tuple (``det_pars``, ``tran_pars``). ``src_pars`` and ``det_pars`` are python dictionaries. ``tran_pars`` is a list of lists. 

        ``src_pars`` contains the estimated parameters of X-ray source PSF. It consists of several key-value pairs. The value for key ``source_FWHM_x_axis`` is the full width half maximum (FWHM) of the source PSF along the x-axis (i.e., second numpy.ndarray dimension). The value for key ``source_FWHM_y_axis`` is the FWHM of source PSF along the y-axis (i.e., first numpy.ndarray dimension). All FWHMs are for the source PSF in the plane of the X-ray source (and not the plane of the detector). The value for key ``cutoff_FWHM_multiplier`` decides the non-zero spatial extent of the source PSF. The PSF is clipped to zero beginning at a distance, as measured from the PSF's origin, equal to the maximum of ``src_pars['cutoff_FWHM_multiplier']`` times ``src_pars['source_FWHM_x_axis']/2`` and ``src_pars['cutoff_FWHM_multiplier']`` times ``src_pars['source_FWHM_y_axis']/2``. 

        ``det_pars`` contains estimated parameters of detector PSF. It consists of several key-value pairs. The value for key ``detector_FWHM_1`` is the FWHM of the first density function in the mixture density model for detector blur. The first density function is the most dominant part of detector blur. The value for key ``detector_FWHM_2`` is the FWHM of the second density function in the mixture density model. This density function has the largest FWHM and models the long running tails of the detector blur's PSF. The value for key ``detector_weight_1`` is between ``0`` and ``1`` and is a measure of the amount of contribution of the first density function to the detector blur. The values for keys ``cutoff_FWHM_1_multiplier`` and ``cutoff_FWHM_2_multiplier`` decide the non-zero spatial extent of the detector PSF. The PSF is clipped to zero beginning at a distance, as measured from the PSF's origin, equal to the maximum of ``det_pars['cutoff_FWHM_1_multiplier']`` times ``det_pars['detector_FWHM_1']/2`` and ``det_pars['cutoff_FWHM_2_multiplier']`` times ``det_pars['detector_FWHM_2']/2``. If ``mix_det`` is ``False``, then value for key ``detector_weight_1`` is fixed at ``1`` and value for key ``detector_FWHM_2`` is fixed at ``0``. 

        ``tran_pars`` contains estimated parameters of the transmission function for each input radiograph. This return value is a list of lists, where each inner nested list consists of two parameters of type `float`. These `float` values give the low and high values respectively of the transmission function. The number of nested lists in the returned list equals the number of input radiographs. Note that the transmission function is the normalized radiograph image that would have resulted in the absence of blur and noise.
    """
    start_time = time.time()
    est_src = only_src or not only_det
    est_det = only_det or not only_src
    sdd = [i+j for i,j in zip(sod,odd)]

    trans_models,trans_params,trans_bounds = [],[],[]
    for j in range(len(rads)):
        if edge == 'perpendicular':
            mk = None if masks is None else masks[j]
            trans_dict,params,bounds = ideal_trans_perp_corner(rads[j],bdary_mask_perc=bdary_mask,pad_factor=pad,mask=mk,perp_mask_perc=perp_mask)
        elif edge == 'straight':
            mk = None if masks is None else masks[j]
            trans_dict,params,bounds = ideal_trans_sharp_edge(rads[j],bdary_mask_perc=bdary_mask,pad_factor=pad,mask=mk)
        else:
            raise ValueError('Argument edge must be either perpendicular or straight')        

        #trans_dict has ideal transmission function, masks, etc
        trans_models.append(trans_dict)
        #Add trans_dict to list trans_models 
        trans_params.append(params)
        trans_bounds.append(bounds)

        #plt.imshow(norm_rads[j]+trans_dict['norm_rad_mask'].astype(float))                  
        #plt.show()
 
        #print("Initial estimates of transmission parameters for radiograph {} are {}.".format(j,params))
        #print("{:.2e} mins has elapsed.".format((time.time()-start_time)/60.0))

    #---------------------------- ESTIMATE BLUR MODEL --------------------------------
    #It is recommended to run the blur model estimation in multiple stages
    #First, only estimate the detector blur using radiographs with the largest SODs
    #Radiographs with the largest SODs are expected to have minimal source blur
    sod_avg = sum(sod)/len(sod) #Compute average of all SODs

    #Only include list elements with SOD > average of all SODs
    rads_det,trans_models_det,trans_params_det,trans_bounds_det,sod_det,sdd_det = [],[],[],[],[],[]
    for i in range(len(rads)):
        if only_det:
            rads_det.append(rads[i])
            trans_models_det.append(trans_models[i])
            trans_params_det.append(trans_params[i])
            trans_bounds_det.append(trans_bounds[i])
            sod_det.append(sod[i])
            sdd_det.append(sdd[i])
        elif sod[i] > sod_avg and not only_src:
            rads_det.append(rads[i])
            trans_models_det.append(trans_models[i])
            trans_params_det.append(trans_params[i])
            trans_bounds_det.append(trans_bounds[i])
            sod_det.append(sod[i])
            sdd_det.append(sdd[i])

    #Estimate detector blur parameters
    if only_det:
        print("----------- Estimation of only detector blur model parameters -----------")
        _,det_params_det,trans_params_det,_,cost_det = get_blur_psfs(rads_det,trans_models_det,sod_det,sdd_det,pix_wid,src_est=False,det_est=True,trans_est=True,src_params=None,det_params=None,trans_params=trans_params_det,trans_bounds=trans_bounds_det,norm_pow=power,convg_thresh=thresh,mix_det=mix_det)
        trans_params = trans_params_det
        print("Estimated detector parameters are {}".format(det_params_det))
    elif not only_src:
        print("----------- Init stage: Estimation of only detector blur model parameters -----------")
        _,det_params_det,_,_,cost_det = get_blur_psfs(rads_det,trans_models_det,sod_det,sdd_det,pix_wid,src_est=False,det_est=True,trans_est=False,src_params=None,det_params=None,trans_params=trans_params_det,trans_bounds=trans_bounds_det,norm_pow=power,convg_thresh=thresh*10,mix_det=mix_det)
        print("Estimated detector parameters are {}".format(det_params_det))
    print("{:.2f} mins has elapsed".format((time.time()-start_time)/60.0))

    #Next, only estimate the source blur using radiographs at the smallest source to object distances
    #Only include list elements with SOD < average of all SODs
    rads_src,trans_models_src,trans_params_src,trans_bounds_src,sod_src,sdd_src = [],[],[],[],[],[]
    for i in range(len(rads)):
        if only_src:
            rads_src.append(rads[i])
            trans_models_src.append(trans_models[i])
            trans_params_src.append(trans_params[i])
            trans_bounds_src.append(trans_bounds[i])
            sod_src.append(sod[i])
            sdd_src.append(sdd[i])
        elif sod[i] < sod_avg and not only_det:
            rads_src.append(rads[i])
            trans_models_src.append(trans_models[i])
            trans_params_src.append(trans_params[i])
            trans_bounds_src.append(trans_bounds[i])
            sod_src.append(sod[i])
            sdd_src.append(sdd[i])

    #Estimate source blur parameters
    if only_src:
        print("----------- Estimation of only source blur model parameters -----------")
        src_params_src,_,trans_params_src,_,cost_src = get_blur_psfs(rads_src,trans_models_src,sod_src,sdd_src,pix_wid,src_est=True,det_est=False,trans_est=True,src_params=None,det_params=None,trans_params=trans_params_src,trans_bounds=trans_bounds_src,norm_pow=power,convg_thresh=thresh,mix_det=mix_det)
        trans_params = trans_params_src
        print("Estimated source parameters are {}".format(src_params_src))
    elif not only_det:
        print("----------- Init stage: Estimation of only source blur model parameters -----------")
        src_params_src,_,_,_,cost_src = get_blur_psfs(rads_src,trans_models_src,sod_src,sdd_src,pix_wid,src_est=True,det_est=False,trans_est=False,src_params=None,det_params=None,trans_params=trans_params_src,trans_bounds=trans_bounds_src,norm_pow=power,convg_thresh=thresh*10,mix_det=mix_det)
        print("Estimated source parameters are {}".format(src_params_src))
    print("{:.2f} mins has elapsed".format((time.time() - start_time)/60.0))

    #Write source blur parameters to file
    if est_src:
        filen = 'source_params.yml' if only_src else 'source_params_init.yml'
        with open(os.path.join(save_dir,filen),'w') as fid:
            yaml.safe_dump(src_params_src,fid,default_flow_style=False)

    #Write detector blur parameters to file
    if est_det:
        filen = 'detector_params.yml' if only_det else 'detector_params_init.yml'
        with open(os.path.join(save_dir,filen),'w') as fid:
            yaml.safe_dump(det_params_det,fid,default_flow_style=False)

    #Write transmission function parameters to file
    trans_dict = {}
    for i in range(len(rads)):
        label = 'radiograph_{}'.format(i)
        trans_dict[label] = {}
        trans_dict[label]['min param'] = trans_params[i][0] 
        trans_dict[label]['max param'] = trans_params[i][1] 
    filen = 'transmission_params.yml' if only_src or only_det else 'transmission_params_init.yml'
    with open(os.path.join(save_dir,filen),'w') as fid:
        yaml.safe_dump(trans_dict,fid,default_flow_style=False)

    if only_src:
        with open(os.path.join(save_dir,'cost.yml'),'w') as fid:
            yaml.safe_dump({'error final':cost_src},fid,default_flow_style=False)
        return src_params_src,trans_params_src
    if only_det:
        with open(os.path.join(save_dir,'cost.yml'),'w') as fid:
            yaml.safe_dump({'error final':cost_det},fid,default_flow_style=False)
        return det_params_det,trans_params_det

    #Finally, refine estimates of both source and detector blur using all radiographs
    print("-------- Final stage: Estimation of source, detector and transmission model parameters --------")
    src_params,det_params,trans_params,trans_bounds,cost = get_blur_psfs(rads,trans_models,sod,sdd,pix_wid,src_est=True,det_est=True,trans_est=True,src_params=src_params_src,det_params=det_params_det,trans_params=trans_params,trans_bounds=trans_bounds,norm_pow=power,convg_thresh=thresh,mix_det=mix_det)
    print("Blur estimation is complete.")
    print("Estimated source parameters are {}, detector parameters are {}, and transmission function parameters are {}".format(src_params,det_params,trans_params))
    print("{:.2f} mins has elapsed".format((time.time()-start_time)/60.0))

    #Write source blur parameters to file
    with open(os.path.join(save_dir,'source_params.yml'),'w') as fid:
        yaml.safe_dump(src_params,fid,default_flow_style=False)

    #Write detector blur parameters to file
    with open(os.path.join(save_dir,'detector_params.yml'),'w') as fid:
        yaml.safe_dump(det_params,fid,default_flow_style=False)

    #Write transmission function parameters to file
    trans_dict = {}
    for i in range(len(rads)):
        label = 'radiograph_{}'.format(i)
        trans_dict[label] = {}
        trans_dict[label]['min param'] = trans_params[i][0] 
        trans_dict[label]['max param'] = trans_params[i][1] 
    with open(os.path.join(save_dir,'transmission_params.yml'),'w') as fid:
        yaml.safe_dump(trans_dict,fid,default_flow_style=False)

    with open(os.path.join(save_dir,'cost.yml'),'w') as fid:
        yaml.safe_dump({'error final':cost,'error source init':cost_src,'error detector init':cost_det},fid,default_flow_style=False)

    return src_params,det_params,trans_params

def get_blur_psfs(norm_rads,trans_models,sod,sdd,pix_wid,src_est,det_est,trans_est,src_params,det_params,trans_params,trans_bounds,norm_pow,convg_thresh,mix_det):
    max_mag = max([(sdd_smpl-sod_smpl)/sod_smpl for sdd_smpl,sod_smpl in zip(sdd,sod)])
    min_mag = min([(sdd_smpl-sod_smpl)/sod_smpl for sdd_smpl,sod_smpl in zip(sdd,sod)])
    if src_params is None:
        src_params = {}
        src_params['source_FWHM_x_axis'] = pix_wid/max_mag if src_est else 0.0
        src_params['source_FWHM_y_axis'] = 2*pix_wid/max_mag if src_est else 0.0
        src_params['norm_power'] = norm_pow
        src_params['cutoff_FWHM_multiplier'] = 10

    if det_params is None:
        det_params = {}
        det_params['detector_FWHM_1'] = pix_wid if det_est else 0.0
        if mix_det:
            det_params['detector_FWHM_2'] = 10*pix_wid if det_est else 0.0
            det_params['detector_weight_1'] = 0.9 if det_est else 0.0
        else:
            det_params['detector_FWHM_2'] = 0.0
            det_params['detector_weight_1'] = 1.0
        det_params['norm_power'] = norm_pow
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
        src_mods.append(SourceBlur(pix_wid,max_wid,sod[i],sdd[i]-sod[i],src_params['cutoff_FWHM_multiplier'],param_x=get_scale(src_params['source_FWHM_x_axis'],norm_pow),param_y=get_scale(src_params['source_FWHM_y_axis'],norm_pow),param_type='scale',norm_pow=norm_pow,warn=False))
    det_mod = DetectorBlur(pix_wid,max_wid,det_params['cutoff_FWHM_1_multiplier'],det_params['cutoff_FWHM_2_multiplier'],param_1=get_scale(det_params['detector_FWHM_1'],norm_pow),param_2=get_scale(det_params['detector_FWHM_2'],norm_pow),weight_1=det_params['detector_weight_1'],param_type='scale',norm_pow=norm_pow,warn=False) 
 
    args_var = []
    args_bounds = []
    if src_est == True:
        args_var.append(get_scale(src_params['source_FWHM_x_axis'],norm_pow))
        args_bounds.append([1e-10,None])
        args_var.append(get_scale(src_params['source_FWHM_y_axis'],norm_pow))
        args_bounds.append([1e-10,None])
        print("Will optimize source blur parameters")
        
    if det_est == True:
        args_var.append(get_scale(det_params['detector_FWHM_1'],norm_pow))
        args_bounds.append([1e-10,None])
        if mix_det:
            args_var.append(get_scale(det_params['detector_FWHM_2'],norm_pow))
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

#    eps = np.sqrt(np.finfo(float).eps)
#    print("----------------------------")
#    grad_true = jacobian_function(np.array(args_var),src_mods,det_mod,trans_mods,src_est,det_est,trans_est,norm_rads,pix_wid,mix_det)
#    grad_num = approx_fprime(np.array(args_var),error_function,eps,src_mods,det_mod,trans_mods,src_est,det_est,trans_est,norm_rads,pix_wid,mix_det)
#    print("eps is {}".format(eps))
#    print("Error in jacobian at {} is {}. Numerical jacobian is {}. True jacobian is {}".format(args_var,np.sqrt(np.sum((grad_num-grad_true)**2)),grad_num,grad_true))
#    print("###########################")

    cost = None
    if(src_est or det_est or trans_est):
        res = minimize(error_function,args_var,args=(src_mods,det_mod,trans_mods,src_est,det_est,trans_est,norm_rads,pix_wid,mix_det),method='L-BFGS-B',jac=jacobian_function,bounds=args_bounds,options={'disp':True,'ftol':convg_thresh})
        args_opt = res.x.tolist()
        cost = float(res.fun)
    else:
        args_opt = args_var.copy()

    print("Optimized parameters are {}".format(args_opt))    
    set_model_params(args_opt,src_mods,det_mod,trans_mods,src_est,det_est,trans_est,mix_det) 
    
    args_opt.reverse() 
    if src_est == True:
        src_params['source_FWHM_x_axis'] = float(get_FWHM(args_opt.pop(),norm_pow))
        src_params['source_FWHM_y_axis'] = float(get_FWHM(args_opt.pop(),norm_pow))
        #print("Compare",max_wid,src_params['cutoff_FWHM_multiplier']*max(src_params['source_FWHM_x_axis'],src_params['source_FWHM_y_axis'])*max_mag/2)
        if max_wid < src_params['cutoff_FWHM_multiplier']*max(src_params['source_FWHM_x_axis'],src_params['source_FWHM_y_axis'])*max_mag/2:
            print("WARN: The maximum single sided width {:.2e}, as determined by the amount of padding in radiographs, that is allowed for source PSF may be insufficient. This may or may not be a problem. Consider increasing the pad factor if results are inconsistent. An example of a inconsistency is if FWHM changes significantly when pad factor is changed. Increasing padding will increase software run time.".format(max_wid));
         
    if det_est == True:
        det_params['detector_FWHM_1'] = float(get_FWHM(args_opt.pop(),norm_pow))
        if mix_det:
            det_params['detector_FWHM_2'] = float(get_FWHM(args_opt.pop(),norm_pow))
            det_params['detector_weight_1'] = float(args_opt.pop())
        #print("Compare",max_wid,max(det_params['cutoff_FWHM_1_multiplier']*det_params['detector_FWHM_1'],det_params['cutoff_FWHM_2_multiplier']*det_params['detector_FWHM_2'])/2)
        if max_wid < max(det_params['cutoff_FWHM_1_multiplier']*det_params['detector_FWHM_1'],det_params['cutoff_FWHM_2_multiplier']*det_params['detector_FWHM_2'])/2:
            print("WARN: The maximum single sided width {:.2e}, as determined by the amount of padding in radiographs, that is allowed for detector PSF may be insufficient. This may or may not be a problem. Consider increasing the pad factor if results are inconsistent. An example of a inconsistency is if FWHM changes significantly when pad factor is changed. Increasing padding will increase software run time.".format(max_wid));

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

def get_source_psf(pix_wid,src_pars,sod=1.0,odd=1.0):
    """
    Function to compute the point spread function (PSF) of X-ray source blur in the plane of the X-ray source or the detector. 

    If source to object distance (SOD) is equal to object to detector distance (ODD), then the PSF on the detector plane is same as that on the plane of the X-ray source. If PSF on detector plane is desired, it is required to specify the SOD and ODD. If PSF on source plane is desired, use the default values for SOD and ODD. 

    Args:
        pix_wid (float): Effective width of each detector pixel. Note that this is the effective pixel size given by dividing the physical width of each detector pixel by the zoom factor of the optical lens.
        src_pars (dict): Dictionary containing the estimated parameters of X-ray source PSF. It consists of several key-value pairs. The value for key ``source_FWHM_x_axis`` is the full width half maximum (FWHM) of the source PSF along the x-axis (i.e., second numpy.ndarray dimension). The value for key ``source_FWHM_y_axis`` is the FWHM of source PSF along the y-axis (i.e., first numpy.ndarray dimension). All FWHMs are for the source PSF in the plane of the X-ray source (and not the plane of the detector). The value for key ``cutoff_FWHM_multiplier`` decides the non-zero spatial extent of the source PSF. The PSF is clipped to zero beginning at a distance, as measured from the PSF's origin, equal to the maximum of ``src_pars['cutoff_FWHM_multiplier']`` times ``src_pars['source_FWHM_x_axis']/2`` and ``src_pars['cutoff_FWHM_multiplier']`` times ``src_pars['source_FWHM_y_axis']/2``. 
        sod (float): Source to object distance (SOD). 
        odd (float): Object to detector distance (ODD).
    
    Returns:
        numpy.ndarray: PSF of X-ray source blur.
    """

    max_wid = 0.5*src_pars['cutoff_FWHM_multiplier']*max(src_pars['source_FWHM_x_axis'],src_pars['source_FWHM_y_axis'])*odd/sod
    src_mod = SourceBlur(pix_wid,max_wid,sod,odd,src_pars['cutoff_FWHM_multiplier'],param_x=get_scale(src_pars['source_FWHM_x_axis'],src_pars['norm_power']),param_y=get_scale(src_pars['source_FWHM_y_axis'],src_pars['norm_power']),param_type='scale',norm_pow=src_pars['norm_power'],warn=False)
    return src_mod.get_psf()
     
def get_detector_psf(pix_wid,det_pars):
    """
        Function to compute point spread function (PSF) of detector blur. 

        Args:
            pix_wid (float): Effective width of each detector pixel. Note that this is the effective pixel size given by dividing the physical width of each detector pixel by the zoom factor of the optical lens.
            det_pars (dict): Dictionary containing the estimated parameters of detector PSF. It consists of several key-value pairs. The value for key ``detector_FWHM_1`` is the FWHM of the first density function in the mixture density model for detector blur. The first density function is the most dominant part of detector blur. The value for key ``detector_FWHM_2`` is the FWHM of the second density function in the mixture density model. This density function has the largest FWHM and models the long running tails of the detector blur's PSF. The value for key ``detector_weight_1`` is between ``0`` and ``1`` and is a measure of the amount of contribution of the first density function to the detector blur. The values for keys ``cutoff_FWHM_1_multiplier`` and ``cutoff_FWHM_2_multiplier`` decide the non-zero spatial extent of the detector PSF. The PSF is clipped to zero beginning at a distance, as measured from the PSF's origin, equal to the maximum of ``det_pars['cutoff_FWHM_1_multiplier']`` times ``det_pars['detector_FWHM_1']/2`` and ``det_pars['cutoff_FWHM_2_multiplier']`` times ``det_pars['detector_FWHM_2']/2``. 
        
        Returns:
            numpy.ndarray: PSF of detector
    """
    #if max_wid == None:
    max_wid = 0.5*max(det_pars['cutoff_FWHM_1_multiplier']*det_pars['detector_FWHM_1'],det_pars['cutoff_FWHM_2_multiplier']*det_pars['detector_FWHM_2'])
    det_mod = DetectorBlur(pix_wid,max_wid,det_pars['cutoff_FWHM_1_multiplier'],det_pars['cutoff_FWHM_2_multiplier'],param_1=get_scale(det_pars['detector_FWHM_1'],det_pars['norm_power']),param_2=get_scale(det_pars['detector_FWHM_2'],det_pars['norm_power']),weight_1=det_pars['detector_weight_1'],param_type='scale',norm_pow=det_pars['norm_power'],warn=False)
    return det_mod.get_psf()

def get_effective_psf(pix_wid,src_pars,det_pars,sod=1,odd=1): 
    """
        Function to compute the effective point spread function (PSF), which is the convolution of X-ray source and detector PSFs. 
    
        Args:
            pix_wid (float): Effective width of each detector pixel. Note that this is the effective pixel size given by dividing the physical width of each detector pixel by the zoom factor of the optical lens.
            src_pars (dict): Dictionary containing the estimated parameters of X-ray source PSF. It consists of several key-value pairs. The value for key ``source_FWHM_x_axis`` is the full width half maximum (FWHM) of the source PSF along the x-axis (i.e., second numpy.ndarray dimension). The value for key ``source_FWHM_y_axis`` is the FWHM of source PSF along the y-axis (i.e., first numpy.ndarray dimension). All FWHMs are for the source PSF in the plane of the X-ray source (and not the plane of the detector). The value for key ``cutoff_FWHM_multiplier`` decides the non-zero spatial extent of the source PSF. The PSF is clipped to zero beginning at a distance, as measured from the PSF's origin, equal to the maximum of ``src_pars['cutoff_FWHM_multiplier']`` times ``src_pars['source_FWHM_x_axis']/2`` and ``src_pars['cutoff_FWHM_multiplier']`` times ``src_pars['source_FWHM_y_axis']/2``. 
            det_pars (dict): Dictionary containing the estimated parameters of detector PSF. It consists of several key-value pairs. The value for key ``detector_FWHM_1`` is the FWHM of the first density function in the mixture density model for detector blur. The first density function is the most dominant part of detector blur. The value for key ``detector_FWHM_2`` is the FWHM of the second density function in the mixture density model. This density function has the largest FWHM and models the long running tails of the detector blur's PSF. The value for key ``detector_weight_1`` is between ``0`` and ``1`` and is a measure of the amount of contribution of the first density function to the detector blur. The values for keys ``cutoff_FWHM_1_multiplier`` and ``cutoff_FWHM_2_multiplier`` decide the non-zero spatial extent of the detector PSF. The PSF is clipped to zero beginning at a distance, as measured from the PSF's origin, equal to the maximum of ``det_pars['cutoff_FWHM_1_multiplier']`` times ``det_pars['detector_FWHM_1']/2`` and ``det_pars['cutoff_FWHM_2_multiplier']`` times ``det_pars['detector_FWHM_2']/2``. 
            sod (float): Source to object distance (SOD). 
            odd (float): Object to detector distance (ODD).
        
        Returns:
            numpy.ndarray: PSF of effective blur in the plane of detector.
    """
    #if max_wid == None:
    max_wid = 0.5*src_pars['cutoff_FWHM_multiplier']*max(src_pars['source_FWHM_x_axis'],src_pars['source_FWHM_y_axis'])*odd/sod
    src_mod = SourceBlur(pix_wid,max_wid,sod,odd,src_pars['cutoff_FWHM_multiplier'],param_x=get_scale(src_pars['source_FWHM_x_axis']),param_y=get_scale(src_pars['source_FWHM_y_axis']),param_type='scale',norm_pow=src_pars['norm_power'])
    max_wid = 0.5*max(det_pars['cutoff_FWHM_1_multiplier']*det_pars['detector_FWHM_1'],det_pars['cutoff_FWHM_2_multiplier']*det_pars['detector_FWHM_2'])
    det_mod = DetectorBlur(pix_wid,max_wid,det_pars['cutoff_FWHM_1_multiplier'],det_pars['cutoff_FWHM_2_multiplier'],param_1=get_scale(det_pars['detector_FWHM_1']),param_2=get_scale(det_pars['detector_FWHM_2']),weight_1=det_pars['detector_weight_1'],param_type='scale',norm_pow=det_pars['norm_power']) 
    smod_psf = src_mod.get_psf()
    dmod_psf = det_mod.get_psf()
    return combine_psfs(smod_psf,dmod_psf,are_psf=True)

def apply_blur_psfs(rad,sod,odd,pix_wid,src_pars,det_pars,padded_widths=[0,0],pad_type='constant',pad_constant=0):
    """
        Function to blur the input radiograph with point spread functions (PSF) of X-ray source and detector blurs.

        This function blurs the input radiograph with X-ray source blur and detector blur with the specified point spread function (PSF) parameters. This function is useful to observe the effect of source and detector blurs on a simulated radiograph.

        Args:
            rad (numpy.ndarray): Radiograph of type numpy.ndarray that is normalized using the bright-field (also called flat-field) and dark-field images.
            sod (float): Source to object distance (SOD) of radiograph. 
            odd (float): Object to detector distance (ODD) of radiograph.
            pix_wid (float): Effective width of each detector pixel. Note that this is the effective pixel size given by dividing the physical width of each detector pixel by the zoom factor of the optical lens.
            src_pars (dict): Dictionary containing the estimated parameters of X-ray source PSF. It consists of several key-value pairs. The value for key ``source_FWHM_x_axis`` is the full width half maximum (FWHM) of the source PSF along the x-axis (i.e., second numpy.ndarray dimension). The value for key ``source_FWHM_y_axis`` is the FWHM of source PSF along the y-axis (i.e., first numpy.ndarray dimension). All FWHMs are for the source PSF in the plane of the X-ray source (and not the plane of the detector). The value for key ``cutoff_FWHM_multiplier`` decides the non-zero spatial extent of the source PSF. The PSF is clipped to zero beginning at a distance, as measured from the PSF's origin, equal to the maximum of ``src_pars['cutoff_FWHM_multiplier']`` times ``src_pars['source_FWHM_x_axis']/2`` and ``src_pars['cutoff_FWHM_multiplier']`` times ``src_pars['source_FWHM_y_axis']/2``. 
            det_pars (dict): Dictionary containing the estimated parameters of detector PSF. It consists of several key-value pairs. The value for key ``detector_FWHM_1`` is the FWHM of the first density function in the mixture density model for detector blur. The first density function is the most dominant part of detector blur. The value for key ``detector_FWHM_2`` is the FWHM of the second density function in the mixture density model. This density function has the largest FWHM and models the long running tails of the detector blur's PSF. The value for key ``detector_weight_1`` is between ``0`` and ``1`` and is a measure of the amount of contribution of the first density function to the detector blur. The values for keys ``cutoff_FWHM_1_multiplier`` and ``cutoff_FWHM_2_multiplier`` decide the non-zero spatial extent of the detector PSF. The PSF is clipped to zero beginning at a distance, as measured from the PSF's origin, equal to the maximum of ``det_pars['cutoff_FWHM_1_multiplier']`` times ``det_pars['detector_FWHM_1']/2`` and ``det_pars['cutoff_FWHM_2_multiplier']`` times ``det_pars['detector_FWHM_2']/2``. 
            padded_widths (list): List of two integers that specifies the amount of padding already applied to input radiograph. The first integer specifies the padding applied along the first dimension of radiograph. The second integer specifies the padding applied along the second dimension of radiograph. Assumes ``padded_widths[k]/2`` amount of padding is applied at both the left and right extremities of dimension ``k``, where ``k`` is ``0`` or ``1``.
            pad_type (str): Type of additional padding that must be used if amount of padding specified in :attr:`padded_widths` is insufficient. Supported values are ``edge`` and ``constant``.  
            pad_constant (float): If :attr:`pad_type` is `constant`, specify the constant value that must be padded.
        
        Returns:
            numpy.ndarray: Radiograph that is blurred using X-ray source and detector blurs.
    """
    input_rad = rad 
    max_wid = pix_wid*(min(padded_widths)//2)
    #Divide by 2 since there are two PSFs (source and detector blur)
    src_mod = SourceBlur(pix_wid,max_wid,sod,odd,src_pars['cutoff_FWHM_multiplier'],param_x=get_scale(src_pars['source_FWHM_x_axis'],src_pars['norm_power']),param_y=get_scale(src_pars['source_FWHM_y_axis'],src_pars['norm_power']),param_type='scale',norm_pow=src_pars['norm_power'])
    det_mod = DetectorBlur(pix_wid,max_wid,det_pars['cutoff_FWHM_1_multiplier'],det_pars['cutoff_FWHM_2_multiplier'],param_1=get_scale(det_pars['detector_FWHM_1'],det_pars['norm_power']),param_2=get_scale(det_pars['detector_FWHM_2'],det_pars['norm_power']),weight_1=det_pars['detector_weight_1'],param_type='scale',norm_pow=det_pars['norm_power']) 
    smod_psf = src_mod.get_psf()
    dmod_psf = det_mod.get_psf(mix_det=(det_pars['detector_weight_1']!=1))
    new_psf = combine_psfs(smod_psf,dmod_psf,are_psf=True)
    blurred_rad = convolve_psf(input_rad,new_psf,padded_widths,is_psf=True,pad_type=pad_type,pad_constant=pad_constant)
    return blurred_rad
    
def get_trans_fit(rad,sod,odd,pix_wid,src_pars,det_pars,tran_pars,edge,pad=[3,3]):
    """
        Function to compute the blur model prediction and ideal transmission function for a radiograph with a single straight edge or two mutually perpendicular edges. 

        For a measured radiograph consisting of a straight sharp edge or two mutually perpendicular edges, get the ideal transmission function and the predicted radiograph from the blur model. Here, the blur model is used to model the impact of blur due to X-ray source and detector.
    
        Parameters:
            rad (numpy.ndarray): Normalized radiograph of a straight sharp edge or two mutually perpendicular edges.
            sod (float): Source to object distance (SOD) for the radiograph :attr:`rad`.
            odd (float): Object to detector distance (SDD) for the radiograph :attr:`rad`.
            pix_wid (float): Effective width of each detector pixel. Note that this is the effective pixel size given by dividing the physical width of each detector pixel by the zoom factor of the optical lens.
            src_pars (dict): Dictionary containing the estimated parameters of X-ray source PSF. It consists of several key-value pairs. The value for key ``source_FWHM_x_axis`` is the full width half maximum (FWHM) of the source PSF along the x-axis (i.e., second numpy.ndarray dimension). The value for key ``source_FWHM_y_axis`` is the FWHM of source PSF along the y-axis (i.e., first numpy.ndarray dimension). All FWHMs are for the source PSF in the plane of the X-ray source (and not the plane of the detector). The value for key ``cutoff_FWHM_multiplier`` decides the non-zero spatial extent of the source PSF. The PSF is clipped to zero beginning at a distance, as measured from the PSF's origin, equal to the maximum of ``src_pars['cutoff_FWHM_multiplier']`` times ``src_pars['source_FWHM_x_axis']/2`` and ``src_pars['cutoff_FWHM_multiplier']`` times ``src_pars['source_FWHM_y_axis']/2``. 
            det_pars (dict): Dictionary containing the estimated parameters of detector PSF. It consists of several key-value pairs. The value for key ``detector_FWHM_1`` is the FWHM of the first density function in the mixture density model for detector blur. The first density function is the most dominant part of detector blur. The value for key ``detector_FWHM_2`` is the FWHM of the second density function in the mixture density model. This density function has the largest FWHM and models the long running tails of the detector blur's PSF. The value for key ``detector_weight_1`` is between ``0`` and ``1`` and is a measure of the amount of contribution of the first density function to the detector blur. The values for keys ``cutoff_FWHM_1_multiplier`` and ``cutoff_FWHM_2_multiplier`` decide the non-zero spatial extent of the detector PSF. The PSF is clipped to zero beginning at a distance, as measured from the PSF's origin, equal to the maximum of ``det_pars['cutoff_FWHM_1_multiplier']`` times ``det_pars['detector_FWHM_1']/2`` and ``det_pars['cutoff_FWHM_2_multiplier']`` times ``det_pars['detector_FWHM_2']/2``. 
            tran_pars (list): List containing the estimated parameters of the transmission function for the input radiograph. It consists of two parameters of type `float`. These `float` values give the low and high values respectively of the transmission function. Note that the transmission function is the normalized radiograph image that would have resulted in the absence of blur and noise. If not specified (or specified as None), then the best fitting transmission function parameters are estimated using RANSAC regression.
            edge (str): Used to indicate whether there is a single straight edge or two mutually perpendicular edges in each radiograph. If :attr:`edge` is ``perpendicular``, then each radiograph is assumed to have two mutually perpendicular intersecting edges. If it is ``straight``, then each radiograph is assumed to have a single straight edge. Only ``perpendicular`` and ``straight`` are legal choices for :attr:`edge`.
            pad (list): List of two integers that determine the amount of padding that must be applied to the radiographs to reduce aliasing during convolution. The number of rows/columns after padding is equal to :attr:`pad_factor[0]`/:attr:`pad_factor[1]` times the number of rows/columns in each radiograph before padding. For example, if the first element in :attr:`pad_factor` is ``2``, then the radiograph is padded to twice its size along the first dimension. 

        Returns:
            tuple: Tuple of two arrays of type ``numpy.ndarray``. The first array is blurred radiograph as predicted by the blur model. The second array is transmission function, which is the ideal readiograph in the absence of source and detector blur.  
    """
    if edge == 'perpendicular':
        trans_dict,params,bounds = ideal_trans_perp_corner(rad,pad_factor=pad)
    elif edge == 'straight':
        trans_dict,params,bounds = ideal_trans_sharp_edge(rad,pad_factor=pad)
    else:
        raise ValueError('Argument edge must be either perpendicular or straight')        
        
    trans_mod = Transmission(trans_dict,tran_pars)
    _,trans,_ = trans_mod.get_trans()
    pad_widths = (np.array(trans.shape)-np.array(rad.shape))//2 #assumes symmetric padding
    pred_rad = apply_blur_psfs(trans,sod,odd,pix_wid,src_pars,det_pars,padded_widths=pad_widths)
    pred_rad = pred_rad[pad_widths[0]:-pad_widths[0],pad_widths[1]:-pad_widths[1]]
    trans = trans[pad_widths[0]:-pad_widths[0],pad_widths[1]:-pad_widths[1]] 
    return pred_rad,trans

def get_trans_masks(rad,edge,tran_pars=None,pad=[1,1],mask=None,bdary_mask=5.0,perp_mask=5.0):
    """
        Function to compute transmission function and masks for a radiograph with a single straight edge or two mutually perpendicular edges. 

        For a measured radiograph consisting of a straight sharp edge or two mutually perpendicular edges, get the transmission function, mask for transmission function, and mask for radiograph. 
        
        Parameters:
            rad (numpy.ndarray): Normalized radiograph of a straight sharp edge or two mutually perpendicular edges.
            edge (str): Used to indicate whether there is a single straight edge or two mutually perpendicular edges in each radiograph. If :attr:`edge` is ``perpendicular``, then each radiograph is assumed to have two mutually perpendicular edges. If it is ``straight``, then each radiograph is assumed to have a single straight edge. Only ``perpendicular`` and ``straight`` are legal choices for :attr:`edge`.
            tran_pars (list): List containing the estimated parameters of the transmission function for the input radiograph. It consists of two parameters of type `float`. These `float` values give the low and high values respectively of the transmission function. Note that the transmission function is the normalized radiograph image that would have resulted in the absence of blur and noise. If not specified (or specified as None), then the best fitting transmission function parameters are estimated using RANSAC regression. If specified as ``[0,1]``, this function returns the ideal transmission function.
            pad (list): List of two integers that determine the amount of padding that must be applied to the radiographs to reduce aliasing during convolution. The number of rows/columns after padding is equal to :attr:`pad_factor[0]`/:attr:`pad_factor[1]` times the number of rows/columns in each radiograph before padding. For example, if the first element in :attr:`pad_factor` is ``2``, then the radiograph is padded to twice its size along the first dimension. 
            mask (numpy.ndarray): Boolean mask of the same shape as the radiograph that is used to exclude pixels from blur estimation. This is in addition to the masking specified by :attr:`bdary_mask` and :attr:`perp_mask`. An example use case is if some pixels in the radiograph :attr:`rad` are bad, then those pixels can be excluded from blur estimation by setting the corresponding entries in :attr:`mask` to ``False`` and ``True`` otherwise. If None, no user specified mask is used.
            bdary_mask (float): Percentage of image region in the radiographs as measured from the outer edge going inwards that must be excluded from blur estimation. Pixels are excluded (or masked) beginning from the outermost periphery of the image and working inwards until the specified percentage of pixels is reached.
            perp_mask (float): Percentage of circular region to ignore during blur estimation around the intersecting corner of two perpendicular edges. Ignored if :attr:`edge` is ``straight``.

        Returns:
            tuple: Tuple of three arrays each of type ``numpy.ndarray``. The first array is the transmission function, which is the ideal readiograph in the absence of source and detector blur. The second and third arrays are the masks for the transmission function and radiograph respectively. The mask array indicates what pixels must be included (pixel value of ``True``) or excluded (pixel value of ``False``) during blur estimation.  
    """
 
    if edge == 'perpendicular':
        trans_dict,init_params,bounds = ideal_trans_perp_corner(rad,bdary_mask_perc=bdary_mask,pad_factor=pad,mask=mask,perp_mask_perc=perp_mask)
    elif edge == 'straight':
        trans_dict,init_params,bounds = ideal_trans_sharp_edge(rad,bdary_mask_perc=bdary_mask,pad_factor=pad,mask=mask)
    else:
        raise ValueError('Argument edge must be either perpendicular or straight')        

    if tran_pars is None:
        trans_mod = Transmission(trans_dict,init_params)
    else:
        trans_mod = Transmission(trans_dict,tran_pars)
        
    rmask,trans,tmask = trans_mod.get_trans()
    #assert np.all(rmask==tmask)
    return trans,tmask,rmask 
