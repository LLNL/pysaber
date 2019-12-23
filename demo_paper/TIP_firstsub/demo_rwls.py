import numpy as np
from pysaber import least_squares_deblur
import matplotlib.pyplot as plt
from PIL import Image
import yaml

rad_file = 'art_10mm.tif'
bright_file = 'art_bright.tif'
dark_file = 'art_dark.tif'

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
src_params['source_FWHM_x_axis'] = 2.71
src_params['source_FWHM_y_axis'] = 2.99
src_params['cutoff_FWHM_multiplier'] = 20

det_params = {}
det_params['detector_FWHM_1'] = 1.84
det_params['detector_FWHM_2'] = 129.4
det_params['detector_weight_1'] = 0.92
det_params['cutoff_FWHM_1_multiplier'] = 10 
det_params['cutoff_FWHM_2_multiplier'] = 8

for reg in rwls_reg:
    deblur_rad = least_squares_deblur(norm_rad,sod,sdd,pix_wid,src_params,det_params,reg,init_rad=norm_rad,convg_thresh=convg_thresh)
    img = Image.fromarray(deblur_rad)
    img.save('art_rwls_reg{:.2e}.tif'.format(reg))   

