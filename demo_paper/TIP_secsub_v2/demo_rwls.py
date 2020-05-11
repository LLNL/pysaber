import numpy as np
from pysaber import least_squares_deblur
import matplotlib.pyplot as plt
from PIL import Image
import yaml

rad_file = 'data/art_10mm.tif'
bright_file = 'data/art_bright.tif'
dark_file = 'data/art_dark.tif'

sdd = 71012.36
sod = 10003.60
pix_wid = 0.675
rwls_reg = [0.8*1e-3]
convg_thresh = 0.0002

rad = np.asarray(Image.open(rad_file))
bright = np.asarray(Image.open(bright_file))
dark = np.asarray(Image.open(dark_file))
norm_rad = (rad-dark)/(bright-dark)

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
det_params['cutoff_FWHM_2_multiplier'] = 10

for reg in rwls_reg:
    deblur_rad = least_squares_deblur(norm_rad,sod,sdd-sod,pix_wid,src_params,det_params,reg,init_rad=norm_rad,thresh=convg_thresh)
    img = Image.fromarray(deblur_rad)
    img.save('results_art/art_rwls_reg{:.2e}.tif'.format(reg))   

