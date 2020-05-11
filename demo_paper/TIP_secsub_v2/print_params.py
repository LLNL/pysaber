import time
import yaml
import numpy as np
import sys
import os
ddir = 'data'
sys.path.insert(0,os.path.abspath(ddir))
from files import *

start_time = time.time()
   
indices_run = [[0,1],[0,2],[0,3],[0,4],[1,2],[1,3],[1,4],[2,3],[2,4],[3,4]]
src_FWHM_x,src_FWHM_y = np.zeros(len(indices_run)),np.zeros(len(indices_run))
det_FWHM_1,det_FWHM_2,det_weight = np.zeros(len(indices_run)),np.zeros(len(indices_run)),np.zeros(len(indices_run))
horz_sod = [[] for _ in indices_run]
vert_sod = [[] for _ in indices_run]

for i,index in enumerate(indices_run):
    suff = ''
    for k in index:
        _,sod_smpl,sdd_smpl,suff_smpl,ort_smpl = fetch_data(ddir,k,orret=True)
        for ort,sod in zip(ort_smpl,sod_smpl):
            if ort=='horz':
                horz_sod[i].append(sod)
            if ort=='vert':
                vert_sod[i].append(sod)
        for s in suff_smpl:
            suff += s
    sdir = 'multi_psf/exp_srcdetpsf_'+suff 
        
    with open(os.path.join(sdir,'source_params.yml'),'r') as fid:
        src_params = yaml.safe_load(fid)

    with open(os.path.join(sdir,'detector_params.yml'),'r') as fid:
        det_params = yaml.safe_load(fid)
    
    src_FWHM_x[i] = src_params['source_FWHM_x_axis']
    src_FWHM_y[i] = src_params['source_FWHM_y_axis']
    det_FWHM_1[i] = det_params['detector_FWHM_1']
    det_FWHM_2[i] = det_params['detector_FWHM_2']
    det_weight[i] = det_params['detector_weight_1']

print('horz & vert & src x & src y & det 1 & det 2 & det weight \\\\\\hline')
for i,index in enumerate(indices_run):
    print('{:.1f},{:.1f} & {:.1f},{:.1f} & {:.1f} & {:.1f} & {:.1f} & {:.1f} & {:.2f} \\\\\\hline'.format(horz_sod[i][0]/1000,horz_sod[i][1]/1000,vert_sod[i][0]/1000,vert_sod[i][1]/1000,src_FWHM_x[i],src_FWHM_y[i],det_FWHM_1[i],det_FWHM_2[i],det_weight[i]))

print('\n\n & src x & src y & det 1 & det 2 & det weight \\\\\\hline')
print('Mean & {:.1f} & {:.1f} & {:.1f} & {:.1f} & {:.2f} \\\\\\hline'.format(np.mean(src_FWHM_x),np.mean(src_FWHM_y),np.mean(det_FWHM_1),np.mean(det_FWHM_2),np.mean(det_weight))) 
print('Std Dev & {:.2f} & {:.2f} & {:.2f} & {:.1f} & {:.2f} \\\\\\hline'.format(np.std(src_FWHM_x),np.std(src_FWHM_y),np.std(det_FWHM_1),np.std(det_FWHM_2),np.std(det_weight))) 
print('Std Dev/Mean*100 & {:.1f} & {:.1f} & {:.1f} & {:.1f} & {:.2f} \\\\\\hline'.format(100*round(np.std(src_FWHM_x),2)/round(np.mean(src_FWHM_x),1),100*round(np.std(src_FWHM_y),2)/round(np.mean(src_FWHM_y),1),100*round(np.std(det_FWHM_1),2)/round(np.mean(det_FWHM_1),1),100*round(np.std(det_FWHM_2),1)/round(np.mean(det_FWHM_2),1),100*round(np.std(det_weight),2)/round(np.mean(det_weight),2))) 
 
print("Total of {:.2f} mins has elapsed".format((time.time() - start_time)/60.0))
