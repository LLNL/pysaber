import yaml
import numpy as np
from plotfig import plot2D,plot1D,plot_srcpsf,plot_detpsf
from pysaber import estimate_blur_psfs,get_source_psf,get_detector_psf,get_effective_psf,apply_blur_psfs

pix_wid = 0.675
SAVE_FOLDER = './figs'

src_params = {}
src_params['source_FWHM_x_axis'] = 2.6585633057348814
src_params['source_FWHM_y_axis'] = 3.0628308228242416
src_params['cutoff_FWHM_multiplier'] = 20

det_params = {}
det_params['detector_FWHM_1'] = 1.8292399839602307
det_params['detector_FWHM_2'] = 148.2381309504106
det_params['detector_weight_1'] = 0.9203666259568399
det_params['cutoff_FWHM_1_multiplier'] = 10 
det_params['cutoff_FWHM_2_multiplier'] = 10

pix_wid = pix_wid/2
src_psf = get_source_psf(pix_wid,src_params)
x = np.arange(-src_psf.shape[1]//2,src_psf.shape[1]//2,1)*pix_wid
y = np.arange(src_psf.shape[0]//2,-src_psf.shape[0]//2,-1)*pix_wid
plot_srcpsf(pix_wid,src_psf,SAVE_FOLDER+'/psfs/srcpsf_atsrc',FWHM_x=src_params['source_FWHM_x_axis'],FWHM_y=src_params['source_FWHM_y_axis'])

det_psf = get_detector_psf(pix_wid,det_params)
x = np.arange(-det_psf.shape[1]//2,det_psf.shape[1]//2,1)*pix_wid
y = np.arange(det_psf.shape[0]//2,-det_psf.shape[0]//2,-1)*pix_wid
plot_detpsf(pix_wid,det_psf,SAVE_FOLDER+'/psfs/detpsf',FWHM_1=det_params['detector_FWHM_1'],FWHM_2=det_params['detector_FWHM_2'],weight_1=det_params['detector_weight_1'])
