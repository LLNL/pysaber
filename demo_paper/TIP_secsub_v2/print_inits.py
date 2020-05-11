import time
import yaml
import numpy as np
import sys
import os
ddir = 'data'
sys.path.insert(0,os.path.abspath(ddir))
from files import *

start_time = time.time()
   
indices_run = [0,1,2,3,4]
horz_sod = [[] for _ in indices_run]
vert_sod = [[] for _ in indices_run]

horz_sod,vert_sod = np.zeros(len(indices_run)),np.zeros(len(indices_run))
src_FWHM_x,src_FWHM_y = np.zeros(len(indices_run)),np.zeros(len(indices_run))
det_FWHM_1,det_FWHM_2,det_weight = np.zeros(len(indices_run)),np.zeros(len(indices_run)),np.zeros(len(indices_run))
for k in indices_run:
    suff = ''
    _,sod_smpl,sdd_smpl,suff_smpl,ort_smpl = fetch_data(ddir,k,orret=True)
    assert len(ort_smpl)==2
    for ort,sod in zip(ort_smpl,sod_smpl):
        if ort=='horz':
            horz_sod[k] = sod
        if ort=='vert':
            vert_sod[k] = sod
    for s in suff_smpl:
        suff += s

    sdir = 'one_psf/exp_srcpsf_'+suff 
    with open(os.path.join(sdir,'source_params.yml'),'r') as fid:
        src_params = yaml.safe_load(fid)

    src_FWHM_x[k] = src_params['source_FWHM_x_axis']
    src_FWHM_y[k] = src_params['source_FWHM_y_axis']
    
    sdir = 'one_psf/exp_detpsf_'+suff 
    with open(os.path.join(sdir,'detector_params.yml'),'r') as fid:
        det_params = yaml.safe_load(fid)
    
    det_FWHM_1[k] = det_params['detector_FWHM_1']
    det_FWHM_2[k] = det_params['detector_FWHM_2']
    det_weight[k] = det_params['detector_weight_1']

print('horz & vert & src x & src y')
for k in range(len(indices_run)):
    sod = [horz_sod[k],vert_sod[k]]
    print('{:.1f} & {:.1f} & {:.1f} & {:.1f} \\\\\\hline'.format(horz_sod[k]/1000,vert_sod[k]/1000,src_FWHM_x[k],src_FWHM_y[k]))

print('\nhorz & vert & det 1 & det 2 & det weight')
for k in range(len(indices_run)):
    sod = [horz_sod[k],vert_sod[k]]
    print('{:.1f} & {:.1f} & {:.1f} & {:.1f} & {:.2f} \\\\\\hline'.format(horz_sod[k]/1000,vert_sod[k]/1000,det_FWHM_1[k],det_FWHM_2[k],det_weight[k]))

print("Total of {:.2f} mins has elapsed".format((time.time() - start_time)/60.0))
