import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import next_fast_len

def get_FWHM(scale,norm):
    """
    Compute full width half maximum (FWHM) given scale parameter.

    Parameters:
        scale (float): Scale parameter of density function

    Returns:
        float: Full width half maximum (FWHM)
    """
    if scale == 0.0:
        return np.infty
    else:
        return 2.0*(np.log(2)**(1.0/norm))/np.abs(scale)

def get_scale(FWHM,norm):
    """
    Compute scale parameter given full width half maximum (FWHM)

    Parameters:
        FWHM (float): Full width half maximum (FWHM) of density function

    Returns:
        float: Scale parameter
    """
    if FWHM == 0.0:
        return np.infty
    else:
        return 2.0*(np.log(2)**(1.0/norm))/np.abs(FWHM)

class Blur:
    """Class for modeling blur."""
    def __init__(self,delta):
        """
        Constructor for Blur class.

        Parameters:
            delta (float): Pixel width.
        """
        self.delta = delta

    def set_pars(self,cutoff_width,psf_func,psf_grad_func):
        """
        Set functions that will be used to compute point spread function (PSF) and gradient of PSF. 

        Parameters:
            cutoff_width (float): Maximum width beyond which PSF is assumed/clipped to zero
            psf_func (python function): Function to compute PSF
            psf_grad_func (python function): Function to compute gradient of PSF 
        """
        self.cutoff_width = cutoff_width
        self.psf_func = psf_func
        self.psf_grad_func = psf_grad_func

    def get_psf(self,delta=None,cutoff_width=None,psf_func=None,**kwargs):
        """
            Function to compute point spread function (PSF)

            Parameters:
                delta (float): Pixel width.
                cutoff_width (float): Maximum width beyond which PSF is assumed/clipped to zero.
                psf_func (python function): Function to compute PSF
    
            Returns:
                numpy.ndarray: 2D point spread function (PSF) 
        """
        psf_func = self.psf_func if psf_func is None else psf_func
        delta = self.delta if delta is None else delta
        cutoff_width = self.cutoff_width if cutoff_width is None else cutoff_width

        coord = np.arange(0,cutoff_width,delta)
        coord = np.concatenate((-coord[1:][::-1],coord))
        coord_x,coord_y = np.meshgrid(coord,coord,indexing='xy')
        
        if cutoff_width > delta:
            psf = psf_func(coord_x,coord_y,**kwargs)
        else:
            psf = np.array([[0,0,0],[0,1,0],[0,0,0]])

        return psf
    
    def get_grad_psfs(self,delta=None,cutoff_width=None,grad_func=None,**kwargs):
        """
            Function to compute gradient of point spread function (PSF).

            Parameters:
                delta (float): Pixel width.
                cutoff_width (float): Maximum width beyond which PSF is assumed/clipped to zero.
                grad_func (python function): Function to compute gradient of PSF

            Returns:
                list of numpy.ndarray: Gradient of PSF in 2D w.r.t. each PSF parameter.
        """       
 
        grad_func = self.psf_grad_func if grad_func is None else grad_func
        delta = self.delta if delta is None else delta
        cutoff_width = self.cutoff_width if cutoff_width is None else cutoff_width

        coord = np.arange(0,cutoff_width,delta)
        coord = np.concatenate((-coord[1:][::-1],coord))
        coord_x,coord_y = np.meshgrid(coord,coord,indexing='xy')

        psfs = grad_func(coord_x,coord_y,**kwargs)
        if cutoff_width <= delta:
            for i in range(len(psfs)):
                psfs[i] = np.zeros((3,3),dtype=float) 

        return psfs
    
def combine_psfs(psf_1,psf_2,are_psf=False):
    """
        Function to convolve two 2D arrays.

        Parameters:
            psf_1 (numpy.ndarray): First 2D array
            psf_2 (numpy.ndarray): Second 2D array
            are_psf (bool): If true, the 2D array is a PSF and gradient of PSF otherwise.

        Returns:
            numpy.ndarray: Convolution of the input 2D arrays
    """
    
    shape_1 = np.array(psf_1.shape)
    shape_2 = np.array(psf_2.shape)
    conv_shape = shape_1+shape_2-1
  
    fast_shape = np.array([next_fast_len(int(conv_shape[0])),next_fast_len(int(conv_shape[1]))]) 
 
    new_psf = np.fft.irfft2(np.fft.rfft2(psf_1,s=fast_shape)*np.fft.rfft2(psf_2,s=fast_shape),s=fast_shape)
    new_psf = new_psf[:conv_shape[0],:conv_shape[1]]

    if are_psf:
        assert np.isclose(np.sum(new_psf),1.0),"Sum of PSF is {}".format(np.sum(new_psf))
        assert np.isclose(new_psf[new_psf.shape[0]//2,new_psf.shape[1]//2],np.max(new_psf)),"Difference is {}".format(new_psf[new_psf.shape[0]//2,new_psf.shape[1]//2]-np.max(new_psf))
    return new_psf

def convolve_psf(img,psf,pad_width,is_psf=False,pad_type='edge',pad_constant=0,warn=True):
    """
        Function to convolve an image with a point spread function (PSF) or its gradient.

        Parameters:
            img (numpy.ndarray): Input image
            psf (numpy.ndarray): PSF to convolve
            pad_width (float): Maximum width beyond which PSF is assumed/clipped to zero. Padding already applied to input.
            is_psf (float): If true, 2nd parameter is PSF and gradient of PSF otherwise
            pad_type (str): Type of padding used by the function numpy.pad
            pad_constant (float): Constant value to be used as padding by the function numpy.pad

        Returns:
            numpy.ndarray: Convolution of the first two inputs
    """
    if psf.shape[0]>img.shape[0] or psf.shape[1]>img.shape[1]:
        print(psf.shape,img.shape)

    assert psf.shape[0]<=img.shape[0]
    assert psf.shape[1]<=img.shape[1]
    if is_psf:
        assert np.isclose(psf[psf.shape[0]//2,psf.shape[1]//2],np.max(psf))

    padw = np.array(psf.shape)//2
    paddiff = (max(0,padw[0]-pad_width[0]),max(0,padw[1]-pad_width[1]))
    if(paddiff[0] > 0 and warn):
        print("WARNING: y-axis padding is {}. Expected padding of at least {}. Consider increasing the amount of padding. Will use additional {} padding.".format(pad_width[0],padw[0],pad_type))
    if(paddiff[1] > 0 and warn):
        print("WARNING: x-axis padding is {}. Expected padding of at least {}. Consider increasing the amount of padding. Will use additional {} padding.".format(pad_width[1],padw[1],pad_type))
        
    if pad_type is not None: 
        if paddiff[0]>0 or paddiff[1]>0:
            if pad_type == 'constant':
                img = np.pad(img,((paddiff[0],paddiff[0]),(paddiff[1],paddiff[1])),mode=pad_type,constant_values=pad_constant)
            else:
                img = np.pad(img,((paddiff[0],paddiff[0]),(paddiff[1],paddiff[1])),mode=pad_type)
    
    shape_1 = np.array(img.shape)
    shape_2 = np.array(psf.shape)
    conv_shape = shape_1+shape_2-1
    fast_shape = np.array([next_fast_len(int(conv_shape[0])),next_fast_len(int(conv_shape[1]))]) 

    blurimg = np.fft.irfft2(np.fft.rfft2(img,s=fast_shape)*np.fft.rfft2(psf,s=fast_shape),s=fast_shape)
    blurimg = blurimg[:conv_shape[0],:conv_shape[1]]
    blurimg = blurimg[shape_2[0]//2:-(shape_2[0]//2),shape_2[1]//2:-(shape_2[1]//2)]    
   
    assert np.all(np.array(blurimg.shape)==shape_1)
 
    if pad_type is not None:
        if(paddiff[0] > 0):
            blurimg = blurimg[paddiff[0]:-paddiff[0]]
        if(paddiff[1] > 0):
            blurimg = blurimg[:,paddiff[1]:-paddiff[1]]

    return blurimg 
 
class SourceBlur(Blur):
    """Class for modeling X-ray source blur."""
    def __init__(self,delta,max_width,sod,odd,cutoff_FWHM,param_x,param_y,param_type,norm_pow,warn=True):
        """
            Constructor for creating an object of SourceBlur class.

            Parameters:
                delta (float): Pixel width.
                max_width (float): Maximum width beyond which point spread function (PSF) is assumed/clipped to zero.
                sod (float): X-ray source to object distance
                odd (float): Object to detector distance
                cutoff_FWHM (float): Multiple of FWHM beyond which PSF is assumed/clipped to zero.
                param_x (float): Scale/FWHM parameter along x-axis
                param_y (float): Scale/FWHM parameter along y-axis
                param_type (str): If 'scale', then param_x/param_y are scale parameters and FWHM parameters otherwise 
        """
        self.sod = sod
        self.odd = odd
        self.warn = warn
        self.max_width = max_width
        self.cutoff_FWHM = cutoff_FWHM
        self.norm_pow = norm_pow
        super().__init__(delta)
        self.set_params(param_x,param_y,param_type)
        
    def psf_function(self,sod=None,odd=None,scale_x=None,scale_y=None):
        """
            Creates a function to compute point spread function (PSF)

            Parameters:
                sod (float): X-ray source to object distance
                odd (float): Object to detector distance
                scale_x (float): Scale parameter along x-axis
                scale_y (float): Scale parameter along y-axis

            Returns:
                python function: Function to compute PSF             
        """
        scale_x = self.scale_x if scale_x is None else scale_x
        scale_y = self.scale_y if scale_y is None else scale_y
        sod = self.sod if sod is None else sod
        odd = self.odd if odd is None else odd
        def psf_func(x,y):
            num = np.exp(-((x*scale_x*sod/odd)**2+(y*scale_y*sod/odd)**2)**(self.norm_pow/2.0))
            return num/np.sum(num)
        return psf_func

    def psf_grad_function(self,sod=None,odd=None,scale_x=None,scale_y=None):
        """
            Creates a function to compute gradient of the point spread function (PSF)

            Parameters:
                sod (float): X-ray source to object distance
                odd (float): Object to detector distance
                scale_x (float): Scale parameter along x-axis
                scale_y (float): Scale parameter along y-axis

            Returns:
                python function: Function to compute gradient of PSF             
        """
        scale_x = self.scale_x if scale_x is None else scale_x
        scale_y = self.scale_y if scale_y is None else scale_y
        sod = self.sod if sod is None else sod
        odd = self.odd if odd is None else odd

        dist_func = lambda x,y: ((x*scale_x)**2+(y*scale_y)**2)
        def grad_func(x,y):
            sz = x.shape
            assert x[sz[0]//2,sz[1]//2] == 0
            assert y[sz[0]//2,sz[1]//2] == 0
            dist = dist_func(x,y)
            dist_grad = dist.copy()
            dist_grad[sz[0]//2,sz[1]//2] = 1.0 #To prevent divide by zero.
            dist_grad = dist_grad**(self.norm_pow/2.0-1)
            #dist_grad[sz[0]//2,sz[1]//2] = 1.0
            exp = np.exp(-((sod/odd)**(self.norm_pow))*(dist**(self.norm_pow/2.0)))
            exp_sum = np.sum(exp)
            
            const = self.norm_pow*((sod/odd)**self.norm_pow)
            grad_x = -const*exp*scale_x*(x**2)
            grad_x = grad_x*dist_grad
            assert grad_x[sz[0]//2,sz[1]//2] == 0.0
            grad_quo_x = (exp_sum*grad_x-exp*np.sum(grad_x))/(exp_sum**2)

            grad_y = -const*exp*scale_y*(y**2)
            grad_y = grad_y*dist_grad
            assert grad_y[sz[0]//2,sz[1]//2] == 0.0
            grad_quo_y = (exp_sum*grad_y-exp*np.sum(grad_y))/(exp_sum**2)
        
            assert np.isclose(grad_quo_x[sz[0]//2,sz[1]//2],-np.sum(grad_x)/(exp_sum**2)),(grad_quo_x[sz[0]//2,sz[1]//2]+np.sum(grad_x)/(exp_sum**2)) 
            assert np.isclose(grad_quo_y[sz[0]//2,sz[1]//2],-np.sum(grad_y)/(exp_sum**2)),(grad_quo_y[sz[0]//2,sz[1]//2]+np.sum(grad_y)/(exp_sum**2)) 

            return [grad_quo_x,grad_quo_y] 
        return grad_func
 
    def set_params(self,param_x,param_y,param_type='scale'):
        """
            Function to set parameters.

            Parameters:
                param_x (float): Scale/FWHM parameter along x-axis
                param_y (float): Scale/FWHM parameter along y-axis
                param_type (str): If 'scale', then param_x/param_y are scale parameters and FWHM parameters otherwise 
        """
        param_x = np.abs(param_x)
        param_y = np.abs(param_y)
        if param_type == 'FWHM':
            self.FWHM_x,self.FWHM_y = param_x,param_y
            self.scale_x = get_scale(self.FWHM_x,self.norm_pow)
            self.scale_y = get_scale(self.FWHM_y,self.norm_pow)
        elif param_type == 'scale':
            self.scale_x,self.scale_y = param_x,param_y
            self.FWHM_x = get_FWHM(self.scale_x,self.norm_pow)
            self.FWHM_y = get_FWHM(self.scale_y,self.norm_pow)
        else:
            raise ValueError("param_type is invalid")
    
        cutoff_width = (self.odd/self.sod)*self.cutoff_FWHM*max([self.FWHM_x,self.FWHM_y])/2.0
        #print("src blur: cutoff_width is {}".format(cutoff_width))
        if cutoff_width>self.max_width and self.warn:
            print("WARN: The specified maximum width {} for source PSF is less than the cutoff width {} at SOD {} and ODD {}".format(self.max_width,cutoff_width,self.sod,self.odd))
        cutoff_width = min(cutoff_width,self.max_width)
        #cutoff_width = self.max_width

        psf_func = self.psf_function() 
        psf_grad_funcs = self.psf_grad_function() 
        
        super().set_pars(cutoff_width,psf_func,psf_grad_funcs)

    def get_psf_at_source(self):
        """
            Function to compute PSF at the plane of X-ray source.

            Parameters: 
                None

            Returns:
                numpy.ndarray: 2D PSF at plane of X-ray source
        """
        psf_func = self.psf_function(sod=1.0,odd=1.0)
        cutoff_width = self.cutoff_FWHM*max([self.FWHM_x,self.FWHM_y])/2.0
        psf = self.get_psf(self.delta,cutoff_width,psf_func=psf_func)
        return psf
 
class DetectorBlur(Blur):
    """Class for modeling detector blur."""
    def __init__(self,delta,max_width,cutoff_FWHM_1,cutoff_FWHM_2,param_1,param_2,weight_1,param_type,norm_pow,warn=True):
        """
            Constructor for creating an object of DetectorBlur class.

            Parameters:
                delta (float): Pixel width.
                max_width (float): Maximum width beyond which point spread function (PSF) is assumed/clipped to zero.
                cutoff_FWHM_1 (float): Multiple of FWHM_1 for determining cutoff width.
                cutoff_FWHM_2 (float): Multiple of FWHM_2 for determining cutoff width.
                param_1 (float): Scale/FWHM parameter of first exponential
                param_2 (float): Scale/FWHM parameter of second exponential
                weight_1 (float): Weight for first exponential. 
                param_type (str): If 'scale', then param_x/param_y are scale parameters and FWHM parameters otherwise 
        """
        super().__init__(delta)
        self.max_width = max_width
        self.cutoff_FWHM_1 = cutoff_FWHM_1
        self.cutoff_FWHM_2 = cutoff_FWHM_2
        self.norm_pow = norm_pow
        self.warn = warn
        self.set_params(param_1,param_2,weight_1,param_type)

    def psf_function(self,scale_1=None,scale_2=None,weight_1=None):
        """
            Creates a function to compute point spread function (PSF)

            Parameters:
                scale_1 (float): Scale parameter of first exponential
                scale_2 (float): Scale parameter of second exponential
                weight_1 (float): Weight parameter for first exponential

            Returns:
                python function: Function to compute PSF             
        """
        scale_1 = self.scale_1 if scale_1 is None else scale_1
        scale_2 = self.scale_2 if scale_2 is None else scale_2
        p = self.weight_1 if weight_1 is None else weight_1
        def psf_func(x,y,mix_det=True):
            exp_1 = np.exp(-((scale_1*x)**2+(scale_1*y)**2)**(self.norm_pow/2.0))
            if mix_det:
                exp_2 = np.exp(-((scale_2*x)**2+(scale_2*y)**2)**(self.norm_pow/2.0))
                psf_eff = p*exp_1/np.sum(exp_1)+(1.0-p)*exp_2/np.sum(exp_2)
            else:
                psf_eff = exp_1/np.sum(exp_1)
            #print("det blur: PSF max is {}".format(np.max(psf_eff)))
            return psf_eff
        return psf_func
    
    def psf_grad_function(self,scale_1=None,scale_2=None,weight_1=None):
        """
            Creates a function to compute gradient of the point spread function (PSF)

            Parameters:
                scale_1 (float): Scale parameter of first exponential
                scale_2 (float): Scale parameter of second exponential
                weight_1 (float): Weight parameter for first exponential

            Returns:
                python function: Function to compute gradient of PSF             
        """
        scale_1 = self.scale_1 if scale_1 is None else scale_1
        scale_2 = self.scale_2 if scale_2 is None else scale_2
        p = self.weight_1 if weight_1 is None else weight_1
        
        dist_func = lambda x,y: (x**2+y**2)**(self.norm_pow/2.0)
        def grad_func(x,y,mix_det=True):
            sz = x.shape
           
            dist = dist_func(x,y)
            exp_1 = np.exp(-(scale_1**self.norm_pow)*dist)
            exp_1_sum = np.sum(exp_1)
            grad_1 = -self.norm_pow*exp_1*dist
            grad_1 = grad_1*(scale_1**(self.norm_pow-1)) if self.norm_pow!=1 else grad_1
            assert grad_1[sz[0]//2,sz[1]//2]==0.0
            grad_1 = p*(exp_1_sum*grad_1-exp_1*np.sum(grad_1))/(exp_1_sum**2)

            if mix_det:
                exp_2 = np.exp(-(scale_2**self.norm_pow)*dist)
                exp_2_sum = np.sum(exp_2)
                grad_2 = -self.norm_pow*exp_2*dist
                grad_2 = grad_2*(scale_2**(self.norm_pow-1)) if self.norm_pow!=1 else grad_2
                assert grad_2[sz[0]//2,sz[1]//2]==0.0
                grad_2 = (1-p)*(exp_2_sum*grad_2-exp_2*np.sum(grad_2))/(exp_2_sum**2)

                grad_p = exp_1/exp_1_sum - exp_2/exp_2_sum            
                return [grad_1,grad_2,grad_p]
            else:
                return [grad_1]

        return grad_func 
 
    def set_params(self,param_1,param_2,weight_1,param_type='scale'):
        """
            Function to set parameters.

            Parameters:
                param_1 (float): Scale parameter of first exponential
                param_2 (float): Scale parameter of second exponential
                weight_1 (float): Weight parameter for first exponential
                param_type (str): If 'scale', then param_1/param_2 are scale parameters and FWHM parameters otherwise 
        """
        param_1 = np.abs(param_1)
        if param_type == 'FWHM':
            self.FWHM_1 = param_1
            self.scale_1 = get_scale(self.FWHM_1,self.norm_pow)
        elif param_type == 'scale':
            self.scale_1 = param_1
            self.FWHM_1 = get_FWHM(self.scale_1,self.norm_pow)
        else:
            raise ValueError("param_type is invalid")

        if param_2 is not None:
            param_2 = np.abs(param_2)
            if param_type == 'FWHM':
                self.FWHM_2 = param_2
                self.scale_2 = get_scale(self.FWHM_2,self.norm_pow)
            elif param_type == 'scale':
                self.scale_2 = param_2
                self.FWHM_2 = get_FWHM(self.scale_2,self.norm_pow)
            else:
                raise ValueError("param_type is invalid")
            
        if weight_1 is not None:
            weight_1 = 0.0 if weight_1<0.0 else weight_1
            weight_1 = 1.0 if weight_1>1.0 else weight_1
            self.weight_1 = weight_1   
 
        cutoff_width = max(self.cutoff_FWHM_1*self.FWHM_1,self.cutoff_FWHM_2*self.FWHM_2)/2.0
        if cutoff_width>self.max_width and self.warn:
            print("WARN: The maximum width {} specified for detector PSF is less than the cutoff width {}".format(self.max_width,cutoff_width))
        cutoff_width = min(cutoff_width,self.max_width)

        psf_func = self.psf_function() 
        psf_grad_funcs = self.psf_grad_function() 
        
        super().set_pars(cutoff_width,psf_func,psf_grad_funcs)
        
class Transmission:
    """Class for modeling the X-ray transmission function"""
    def __init__(self,trans_model,params):
        """
            Constructor for initializing an object of Transmission class

            Parameters:
                trans_model (list): Transmission model for the sample
                params (list): Parameters of the transmission model  
        """
        self.norm_rad_mask = trans_model['norm_rad_mask']
        self.ideal_trans = trans_model['ideal_trans']
        self.ideal_trans_mask = trans_model['ideal_trans_mask']
        self.ideal_trans_grad = trans_model['ideal_trans_grad']
        self.len_params = len(params)
        self.psf_max_halfsize = np.min(np.array(self.ideal_trans_mask.shape)-np.array(self.norm_rad_mask.shape))//4 #Max half size of psf
        #Divide by 4 because one divide by 2 since there are two PSFs (source and detector blur) and another divide by 2 to get one-sided max width
        #The impulse response from pixels on one side should not leak into the region of interest (unmasked and optimized) pixels on the other side (reflected side) 
        self.set_params(params)

    def set_params(self,params):
        """
            Set parameters of transmission model.

            Parameters:
                params (list): Parameters of the transmission model
        """
        self.params = params.copy()

    def get_trans(self):
        """
            Returns ideal transmission function and masks for normalized radiograph and ideal transmission function

            Parameters:
                None

            Returns:
                numpy.ndarray,numpy.ndarray,numpy.ndarray: Mask for normalized radiograph, ideal transmission function, mask for ideal transmission function
        """
        return self.norm_rad_mask,self.ideal_trans(self.params),self.ideal_trans_mask

    def get_grad(self):
        """
            Returns gradient of transmission function w.r.t. transmission function parameters

            Parameters:
                None

            Returns:
                numpy.ndarray: Gradient of ideal transmission function
        """
        return self.ideal_trans_grad(self.params) 
