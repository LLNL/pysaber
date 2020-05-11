import time
import sys
import os
import yaml
ddir = '../data'
sys.path.insert(0,os.path.abspath(ddir))
from files import *
from PIL import Image
from pysaber import get_source_psf,get_detector_psf

start_time = time.time()

#---------------------------- READ AND NORMALIZE RADIOGRAPHS --------------------------------
indices_run = [[0,1],[0,2],[0,3],[0,4],[1,2],[1,3],[1,4],[2,3],[2,4],[3,4]]
pix_wid = 0.675
log_dir = 'exp_srcdetpsf_'

src_params,det_params = [],[]
src_params_mean = {'source_FWHM_x_axis':0.0,
            'source_FWHM_y_axis':0.0,
            'norm_power':1.0,
            'cutoff_FWHM_multiplier':10}
det_params_mean = {'detector_FWHM_1':0.0,
            'detector_FWHM_2':0.0,
            'detector_weight_1':0.0,
            'norm_power':1.0,
            'cutoff_FWHM_1_multiplier':10,
            'cutoff_FWHM_2_multiplier':10}

for ind_train in indices_run:
    rads,sod,sdd,suff,orret = [],[],[],'',[]
    for k in ind_train:
        rads_smpl,sod_smpl,sdd_smpl,suff_smpl,orret_smpl = fetch_data(ddir,k,orret=True)
        rads += rads_smpl
        sod += sod_smpl
        sdd += sdd_smpl
        orret += orret_smpl
        for s in suff_smpl:
            suff += s
    sdir = log_dir+suff
    
    with open(os.path.join(sdir,'source_params.yml'),'r') as fid:
        src_params.append(yaml.safe_load(fid))
    src_params[-1]['cutoff_FWHM_multiplier'] = 11
    
    with open(os.path.join(sdir,'detector_params.yml'),'r') as fid:
        det_params.append(yaml.safe_load(fid))
    det_params[-1]['cutoff_FWHM_1_multiplier'] = 11
    det_params[-1]['cutoff_FWHM_2_multiplier'] = 12

    src_params_mean['source_FWHM_x_axis'] += src_params[-1]['source_FWHM_x_axis']/len(indices_run) 
    src_params_mean['source_FWHM_y_axis'] += src_params[-1]['source_FWHM_y_axis']/len(indices_run)          
    det_params_mean['detector_FWHM_1'] += det_params[-1]['detector_FWHM_1']/len(indices_run) 
    det_params_mean['detector_FWHM_2'] += det_params[-1]['detector_FWHM_2']/len(indices_run) 
    det_params_mean['detector_weight_1'] += det_params[-1]['detector_weight_1']/len(indices_run)

src_psf_mean = get_source_psf(pix_wid,src_params_mean)
src_sh = np.array(src_psf_mean.shape,dtype=int)
det_psf_mean = get_detector_psf(pix_wid,det_params_mean)
det_sh = np.array(det_psf_mean.shape,dtype=int)

src_err,det_err = np.zeros(len(indices_run),dtype=float),np.zeros(len(indices_run),dtype=float)
for i in range(len(indices_run)):
    src_psf = get_source_psf(pix_wid,src_params[i])
    off = (np.array(src_psf.shape,dtype=int)-src_sh)//2
    print(off)
    src_psf = src_psf[off[0]:-off[0],off[1]:-off[1]]
    src_err[i] = np.sqrt(np.mean((src_psf-src_psf_mean)**2)) 

    det_psf = get_detector_psf(pix_wid,det_params[i])
    off = (np.array(det_psf.shape,dtype=int)-det_sh)//2
    print(off)
    det_psf = det_psf[off[0]:-off[0],off[1]:-off[1]]
    det_err[i] = np.sqrt(np.mean((det_psf-det_psf_mean)**2)) 
      
print('Source mean = {}, std = {}, 100*std/mean',np.mean(src_err),np.std(src_err),100*np.std(src_err)/np.mean(src_err))
print('Detector mean = {}, std = {}, 100*std/mean',np.mean(det_err),np.std(det_err),100*np.std(det_err)/np.mean(det_err))

print("{:.2e} mins has elapsed".format((time.time()-start_time)/60.0))

    
