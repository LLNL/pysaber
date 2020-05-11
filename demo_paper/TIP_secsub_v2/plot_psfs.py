import yaml
import numpy as np
from plotfig import plot2D,plot1D,plot_srcpsf,plot_detpsf
from pysaber import get_source_psf,get_detector_psf

pix_wid = 0.675
sod = [24800]
sdd = [71000]
SAVE_FOLDER = '.'

src_params = {}
src_params['source_FWHM_x_axis'] = 2.7
src_params['source_FWHM_y_axis'] = 3.0
src_params['norm_power'] = 1.0
src_params['cutoff_FWHM_multiplier'] = 10

det_params = {}
det_params['detector_FWHM_1'] = 1.8
det_params['detector_FWHM_2'] = 135.7
det_params['detector_weight_1'] = 0.92
det_params['norm_power'] = 1.0
det_params['cutoff_FWHM_1_multiplier'] = 10 
det_params['cutoff_FWHM_2_multiplier'] = 6

pix_wid = pix_wid/2
src_psf = get_source_psf(pix_wid,src_params)
x = np.arange(-src_psf.shape[1]//2,src_psf.shape[1]//2,1)*pix_wid
y = np.arange(src_psf.shape[0]//2,-src_psf.shape[0]//2,-1)*pix_wid
plot_srcpsf(pix_wid,src_psf,SAVE_FOLDER+'/results_psfs/srcpsf_atsrc')

for d1,d2 in zip(sod,sdd):
    src_psf = get_source_psf(pix_wid,src_params,sod=d1,odd=d2-d1)
    x = np.arange(-src_psf.shape[1]//2,src_psf.shape[1]//2,1)*pix_wid
    y = np.arange(src_psf.shape[0]//2,-src_psf.shape[0]//2,-1)*pix_wid
    plot_srcpsf(pix_wid,src_psf,SAVE_FOLDER+'/results_psfs/srcpsf_sod{}'.format(int(d1)))

det_psf = get_detector_psf(pix_wid,det_params)
x = np.arange(-det_psf.shape[1]//2,det_psf.shape[1]//2,1)*pix_wid
y = np.arange(det_psf.shape[0]//2,-det_psf.shape[0]//2,-1)*pix_wid
plot_detpsf(pix_wid,det_psf,SAVE_FOLDER+'/results_psfs/detpsf')
