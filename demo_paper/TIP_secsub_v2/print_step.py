import time
import yaml
import numpy as np
import sys
import os
ddir = 'data'
sys.path.insert(0,os.path.abspath(ddir))
from files import *

start_time = time.time()
   
index = [2,3]
suff,horz_sod,vert_sod = '',[],[]
for k in index:
    _,sod_smpl,sdd_smpl,suff_smpl,ort_smpl = fetch_data(ddir,k,orret=True)
    for ort,sod in zip(ort_smpl,sod_smpl):
        if ort=='horz':
            horz_sod.append(sod)
        if ort=='vert':
            vert_sod.append(sod)
    for s in suff_smpl:
        suff += s
sdir = 'multi_psf/exp_srcdetpsf_'+suff 
    
with open(os.path.join(sdir,'source_params_init.yml'),'r') as fid:
    src_params_init = yaml.safe_load(fid)

with open(os.path.join(sdir,'detector_params_init.yml'),'r') as fid:
    det_params_init = yaml.safe_load(fid)

with open(os.path.join(sdir,'source_params.yml'),'r') as fid:
    src_params = yaml.safe_load(fid)

with open(os.path.join(sdir,'detector_params.yml'),'r') as fid:
    det_params = yaml.safe_load(fid)

print('Before step (3)')
print('horz & vert & src x & src y & det 1 & det 2 & det weight \\\\\\hline')
print('{:.1f},{:.1f} & {:.1f},{:.1f} & {:.1f} & {:.1f} & {:.1f} & {:.1f} & {:.2f} \\\\\\hline'.format(horz_sod[0]/1000,horz_sod[1]/1000,vert_sod[0]/1000,vert_sod[1]/1000,src_params_init['source_FWHM_x_axis'],src_params_init['source_FWHM_y_axis'],det_params_init['detector_FWHM_1'],det_params_init['detector_FWHM_2'],det_params_init['detector_weight_1']))
print('-------------------------------------------------------------------')
print('After step (3)')
print('horz & vert & src x & src y & det 1 & det 2 & det weight \\\\\\hline')
print('{:.1f},{:.1f} & {:.1f},{:.1f} & {:.1f} & {:.1f} & {:.1f} & {:.1f} & {:.2f} \\\\\\hline'.format(horz_sod[0]/1000,horz_sod[1]/1000,vert_sod[0]/1000,vert_sod[1]/1000,src_params['source_FWHM_x_axis'],src_params['source_FWHM_y_axis'],det_params['detector_FWHM_1'],det_params['detector_FWHM_2'],det_params['detector_weight_1']))

print("Total of {:.2f} mins has elapsed".format((time.time() - start_time)/60.0))
