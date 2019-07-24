import numpy as np
from saber import wiener_deblur
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

reg_start = 3.0
reg_mult = 1.05

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
det_params['cutoff_FWHM_multiplier'] = 10 

rwls_stds = []
for reg in rwls_reg:
    rwls_img = np.asarray(Image.open('art_rwls_reg{:.2e}.tif'.format(reg)))
    sh = rwls_img.shape
    rwls_stds.append(np.std(rwls_img[:sh[0]//4,:sh[1]//4]))

print('RWLS stds {}'.format(rwls_stds))

wiener_img = wiener_deblur(norm_rad,sod,sdd,pix_wid,src_params,det_params,reg_start)
wn_stds = [np.std(wiener_img[:sh[0]//4,:sh[1]//4])]
wn_reg = [reg_start] 
print('Wiener reg {:.2e}, std {:.2e}'.format(wn_reg[-1],wn_stds[-1]))
while wn_stds[-1] > min(rwls_stds):
    reg_start = reg_start*reg_mult 
    wiener_img = wiener_deblur(norm_rad,sod,sdd,pix_wid,src_params,det_params,reg_start)
    wn_stds.append(np.std(wiener_img[:sh[0]//4,:sh[1]//4]))
    wn_reg.append(reg_start) 
    img = Image.fromarray(wiener_img)
    img.save('art_wiener_reg{:.2e}.tif'.format(reg_start))   
    print('Wiener reg {:.2e}, std {:.2e}'.format(wn_reg[-1],wn_stds[-1]))

for reg,std in zip(rwls_reg,rwls_stds):
    idx = np.argmin(np.fabs(wn_stds-std))
    print('RWLS (reg,std)=({:.2e},{:.2e}), Wiener (reg,std)=({:.2e},{:.2e})'.format(reg,std,wn_reg[idx],wn_stds[idx])) 
