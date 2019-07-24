import yaml
import numpy as np
from plotfig import plot2D,plot1D,plot_srcpsf,plot_detpsf
from pysaber import estimate_blur_psfs,get_source_psf,get_detector_psf,get_effective_psf,apply_blur_psfs

pix_wid = 0.675
sod = [24800]
sdd = [71000]
SAVE_FOLDER = '/Users/mohan3/Desktop/Journals/Blur-Modelling/figs'

src_params = {}
src_params['source_FWHM_x_axis'] = 2.71
src_params['source_FWHM_y_axis'] = 2.99
src_params['cutoff_FWHM_multiplier'] = 20

det_params = {}
det_params['detector_FWHM_1'] = 1.84
det_params['detector_FWHM_2'] = 129.4
det_params['detector_weight_1'] = 0.92
det_params['cutoff_FWHM_multiplier'] = 10 

pix_wid = pix_wid/2
src_psf = get_source_psf(pix_wid,src_params)
x = np.arange(-src_psf.shape[1]//2,src_psf.shape[1]//2,1)*pix_wid
y = np.arange(src_psf.shape[0]//2,-src_psf.shape[0]//2,-1)*pix_wid
plot_srcpsf(pix_wid,src_psf,SAVE_FOLDER+'/psfs/srcpsf_atsrc')

for d1,d2 in zip(sod,sdd):
    max_wid = pix_wid*src_params['cutoff_FWHM_multiplier']*max(src_params['source_FWHM_x_axis'],src_params['source_FWHM_y_axis'])/2.0
    src_psf = get_source_psf(pix_wid,src_params,sod=d1,sdd=d2,max_wid=max_wid)
    x = np.arange(-src_psf.shape[1]//2,src_psf.shape[1]//2,1)*pix_wid
    y = np.arange(src_psf.shape[0]//2,-src_psf.shape[0]//2,-1)*pix_wid
    plot_srcpsf(pix_wid,src_psf,SAVE_FOLDER+'/psfs/srcpsf_sod{}'.format(int(d1)))

det_psf = get_detector_psf(pix_wid,det_params)
x = np.arange(-det_psf.shape[1]//2,det_psf.shape[1]//2,1)*pix_wid
y = np.arange(det_psf.shape[0]//2,-det_psf.shape[0]//2,-1)*pix_wid
plot_detpsf(pix_wid,det_psf,SAVE_FOLDER+'/psfs/detpsf')
