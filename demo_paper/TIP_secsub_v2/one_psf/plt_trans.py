import time
import sys
import os
import yaml
ddir = '../data'
sys.path.insert(0,os.path.abspath(ddir))
from files import *
from pysaber import get_trans_fit
from PIL import Image

start_time = time.time()

#---------------------------- READ AND NORMALIZE RADIOGRAPHS --------------------------------
ind_train = [3,3,2,2]
ind_test = [1,4]
pix_wid = 0.675
log_dirs = ['exp_detpsf_','gauss_detpsf_','exp_srcpsf_','gauss_srcpsf_']

errors = []
for lidx,lgdir in enumerate(log_dirs):
    rads,sod,sdd,suff,orret = fetch_data(ddir,ind_train[lidx],orret=True)
    sdir = lgdir   
    for s in suff: 
        sdir = sdir+s
    assert sdir!=lgdir   
 
    if 'src' in lgdir:
        with open(os.path.join(sdir,'source_params.yml'),'r') as fid:
            src_params = yaml.safe_load(fid)
    else:
        src_params = {}
        src_params['source_FWHM_x_axis'] = 0.0       
        src_params['source_FWHM_y_axis'] = 0.0       
        src_params['norm_power'] = 1.0       
        src_params['cutoff_FWHM_multiplier'] = 10.0

    if 'det' in lgdir: 
        with open(os.path.join(sdir,'detector_params.yml'),'r') as fid:
            det_params = yaml.safe_load(fid)
    else:
        det_params = {}
        det_params['detector_FWHM_1'] = 0.0
        det_params['detector_FWHM_2'] = 0.0
        det_params['detector_weight_1'] = 1.0
        det_params['norm_power'] = 1.0
        det_params['cutoff_FWHM_1_multiplier'] = 10.0       
        det_params['cutoff_FWHM_2_multiplier'] = 10.0       
    with open(os.path.join(sdir,'transmission_params.yml'),'r') as cfg:
        dt = yaml.safe_load(cfg)

    trans_horz,trans_vert = np.zeros(2,dtype=float),np.zeros(2,dtype=float)
    num_horz,num_vert = 0,0
    for k in range(len(dt.keys())):
        key = 'radiograph_{}'.format(k)
        if orret[k] == 'horz':
            trans_horz[0] += dt[key]['min param']
            trans_horz[1] += dt[key]['max param']
            num_horz += 1
        if orret[k] == 'vert':
            trans_vert[0] += dt[key]['min param']
            trans_vert[1] += dt[key]['max param']
            num_vert += 1
    trans_horz /= num_horz
    trans_vert /= num_vert

    err,num = 0.0,0
    for j in range(len(ind_test)):
        rads,sod,sdd,suff,ort = fetch_data(ddir,ind_test[j],orret=True)
        for k in range(len(rads)):
            trans_params = trans_horz if ort[k]=='horz' else trans_vert
            pred,_ = get_trans_fit(rads[k],sod[k],sdd[k]-sod[k],pix_wid,src_params,det_params,trans_params,edge='straight-edge')
         
            img = Image.fromarray(pred.astype(np.float32))
            img.save(os.path.join(sdir,'pred_{}_{}.tif'.format(ort[k],suff[k]))) 
            img = Image.fromarray(rads[k].astype(np.float32))
            img.save(os.path.join(sdir,'rad_{}_{}.tif'.format(ort[k],suff[k]))) 
            err += np.mean(np.absolute(rads[k]-pred))    
            num += 1 
    err = err/num
    errors.append(err)
   
for lgdir,err in zip(log_dirs,errors): 
    print('Log Dir {}, Error {}'.format(lgdir,err))

print("{:.2e} mins has elapsed".format((time.time()-start_time)/60.0))

        
