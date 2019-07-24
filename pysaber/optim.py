import numpy as np
from pysaber.models import combine_psfs,convolve_psf
 
def set_model_params(args_var,src_mods,det_mod,trans_mods,src_est,det_est,trans_est):
    """
    Set parameters of blur and transmission models.

    Parameters:
        args_var (list): List of parameters that must be updated
        src_mods (list): List of objects of class SourceBlur
        det_mod (list): Object of class DetectorBlur
        trans_mods (list): List of objects of class Transmission
        src_est (bool): If true, source parameter estimation is done
        det_est (bool): If true, detector parameter estimation is done
        trans_est (bool): If true, transmission model parameter estimation is done
    """
    args = args_var.copy()
    args.reverse()
    if src_est:
        scale_axis_x = args.pop()
        scale_axis_y = args.pop()
        for model in src_mods:
            model.set_params(scale_axis_x,scale_axis_y)

    if det_est:
        det_mod.set_params(args.pop(),args.pop(),args.pop())

    if trans_est:
        for model in trans_mods:
            model.set_params([args.pop() for _ in range(model.len_params)])

def jacobian_function(args_var,src_mods,det_mod,trans_mods,src_est,det_est,trans_est,norm_rads,pix_wid):
    """
    Compute the Jacobian (derivates) of cost function

    Parameters:
        args_var (numpy.ndarray): Array of parameters of blur and transmission model
        src_mods (list): List of objects of class SourceBlur
        det_mod (list): Object of class DetectorBlur
        trans_mods (list): List of objects of class Transmission
        src_est (bool): If true, source parameter estimation is done
        det_est (bool): If true, detector parameter estimation is done
        trans_est (bool): If true, transmission model parameter estimation is done
        norm_rads (list): List of normalized radiographs
        pix_wid (float): Width of each pixel

    Returns:
        numpy.ndarray: Array of gradients with respect to parameters being optimized         
    """
    args_var = args_var.tolist()
    #print("Evaluating gradient for parameters {}".format(args_var))
    set_model_params(args_var,src_mods,det_mod,trans_mods,src_est,det_est,trans_est)
   
    if src_est: 
        src_grad = [0,0]
    if det_est:
        det_grad = [0,0,0]
    if trans_est:
        trans_grad = []

    det_psf = det_mod.get_psf()
    if det_est:
        det_gradpsf = det_mod.get_grad_psfs()

    for j in range(len(norm_rads)):
        rad = norm_rads[j]
        radm,tran,tranm = trans_mods[j].get_trans()
        pad_widths = (np.array(tran.shape)-np.array(rad.shape))//2 #assumes symmetric padding
        
        src_psf = src_mods[j].get_psf()
        src_det_psf = combine_psfs(src_psf,det_psf,are_psf=True)
        src_det_psfpred = convolve_psf(tran,src_det_psf,pad_widths,is_psf=True,warn=False)
        
        if src_est:
            src_gradpsf = src_mods[j].get_grad_psfs()
            for i in range(len(src_gradpsf)):
                src_det_grad = combine_psfs(src_gradpsf[i],det_psf)
                src_det_gradpred = convolve_psf(tran,src_det_grad,pad_widths,warn=False)
                #src_grad[i] += -np.mean((rad[radm]-src_det_psfpred[tranm])*src_det_gradpred[tranm])
                src_grad[i] += -np.sum((rad[radm]-src_det_psfpred[tranm])*src_det_gradpred[tranm])
        if det_est:
            for i in range(len(det_gradpsf)):
                src_det_grad = combine_psfs(src_psf,det_gradpsf[i])
                src_det_gradpred = convolve_psf(tran,src_det_grad,pad_widths,warn=False)
                #det_grad[i] += -np.mean((rad[radm]-src_det_psfpred[tranm])*src_det_gradpred[tranm])
                det_grad[i] += -np.sum((rad[radm]-src_det_psfpred[tranm])*src_det_gradpred[tranm])
        if trans_est:
            tgrad = trans_mods[j].get_grad()
            for i in range(len(tgrad)):
                temp = convolve_psf(tgrad[i],src_det_psf,pad_widths,warn=False)
                #trans_grad.append(-np.mean((rad[radm]-src_det_psfpred[tranm])*temp[tranm]))
                trans_grad.append(-np.sum((rad[radm]-src_det_psfpred[tranm])*temp[tranm]))
    grad = [] 
    if src_est:
        grad.extend(src_grad) 
    if det_est:
        grad.extend(det_grad)
    if trans_est:
        grad.extend(trans_grad)
    grad = np.array(grad) 
    grad = grad/len(norm_rads)

#    print("Gradient is ", grad)
    return grad    

def error_function(args_var,src_mods,det_mod,trans_mods,src_est,det_est,trans_est,norm_rads,pix_wid):
    """
    Compute error function (also called cost) that measures the magnitude of the difference between the measured radiograph values and its prediction using the blur model.

    Parameters:
        args_var (numpy.ndarray): Array of parameters of blur and transmission model
        src_mods (list): List of objects of class SourceBlur
        det_mod (list): Object of class DetectorBlur
        trans_mods (list): List of objects of class Transmission
        src_est (bool): If true, source parameter estimation is done
        det_est (bool): If true, detector parameter estimation is done
        trans_est (bool): If true, transmission model parameter estimation is done
        norm_rads (list): List of normalized radiographs
        pix_wid (float): Width of each pixel
        
    Returns:
        float: Error function value (also called cost)  
    """
    args_var = args_var.tolist()
    print("Evaluating fit for parameters {}".format(args_var))
    set_model_params(args_var,src_mods,det_mod,trans_mods,src_est,det_est,trans_est)
    
    error = 0
    dmod_psf = det_mod.get_psf()
    for j in range(len(norm_rads)):
        rad = norm_rads[j]
        radm,tran,tranm = trans_mods[j].get_trans()
        pad_widths = (np.array(tran.shape)-np.array(rad.shape))//2 #assumes symmetric padding

        smod_psf = src_mods[j].get_psf()
        new_psf = combine_psfs(smod_psf,dmod_psf,are_psf=True)
        pred = convolve_psf(tran,new_psf,pad_widths,is_psf=True,warn=False)
        #error += np.mean((rad[radm]-pred[tranm])**2)
        error += np.sum((rad[radm]-pred[tranm])**2)
    error = error*0.5
    error = error/len(norm_rads)
    
#    print("Cost is {:.2e}".format(error))
    return error

