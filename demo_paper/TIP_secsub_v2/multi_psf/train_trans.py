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
ind_train = [2,3]
pix_wid = 0.675
log_dirs = ['exp_srcdetpsf_','gauss_srcdetpsf_','exp_srcdetpsf_nomix_']

errors,errors_init = [],[]
for lidx,lgdir in enumerate(log_dirs):
    rads,sod,sdd,suff = [],[],[],''
    for k in ind_train:
        rads_smpl,sod_smpl,sdd_smpl,suff_smpl = fetch_data(ddir,k)
        rads += rads_smpl
        sod += sod_smpl
        sdd += sdd_smpl
        for s in suff_smpl:
            suff += s
    sdir = lgdir+suff
    
    with open(os.path.join(sdir,'source_params.yml'),'r') as fid:
        src_params = yaml.safe_load(fid)
    
    with open(os.path.join(sdir,'source_params_init.yml'),'r') as fid:
        src_params_init = yaml.safe_load(fid)

    with open(os.path.join(sdir,'detector_params.yml'),'r') as fid:
        det_params = yaml.safe_load(fid)
    
    with open(os.path.join(sdir,'detector_params_init.yml'),'r') as fid:
        det_params_init = yaml.safe_load(fid)

    with open(os.path.join(sdir,'transmission_params.yml'),'r') as cfg:
        dt = yaml.safe_load(cfg)
    
    trans_params = []
    for k in range(len(dt.keys())):
        key = 'radiograph_{}'.format(k)
        trans_params.append([dt[key]['min param'],dt[key]['max param']])

    with open(os.path.join(sdir,'transmission_params_init.yml'),'r') as cfg:
        dt = yaml.safe_load(cfg)
    
    trans_params_init = []
    for k in range(len(dt.keys())):
        key = 'radiograph_{}'.format(k)
        trans_params_init.append([dt[key]['min param'],dt[key]['max param']])

    err,err_init,num = 0.0,0.0,0
    for j in range(len(ind_train)):
        rads,sod,sdd,suff,ort = fetch_data(ddir,ind_train[j],orret=True)
        for k in range(len(rads)):
            pred,_ = get_trans_fit(rads[k],sod[k],sdd[k]-sod[k],pix_wid,src_params,det_params,trans_params[k],edge='straight-edge')
     
            img = Image.fromarray(pred.astype(np.float32))
            img.save(os.path.join(sdir,'pred_{}_{}.tif'.format(ort[k],suff[k]))) 
            img = Image.fromarray(rads[k].astype(np.float32))
            img.save(os.path.join(sdir,'rad_{}_{}.tif'.format(ort[k],suff[k]))) 
            err += np.mean(np.absolute(rads[k]-pred))
 
            pred_init,_ = get_trans_fit(rads[k],sod[k],sdd[k]-sod[k],pix_wid,src_params_init,det_params_init,trans_params_init[k],edge='straight-edge')
     
            img = Image.fromarray(pred_init.astype(np.float32))
            img.save(os.path.join(sdir,'pred_init_{}_{}.tif'.format(ort[k],suff[k]))) 
            img = Image.fromarray(rads[k].astype(np.float32))
            img.save(os.path.join(sdir,'rad_init_{}_{}.tif'.format(ort[k],suff[k]))) 
            err_init += np.mean(np.absolute(rads[k]-pred_init))

            num += 1 

    err = err/num
    err_init = err_init/num
    errors.append(err)
    errors_init.append(err_init)

for lgdir,err in zip(log_dirs,errors):
    print('Log Dir {}, Error {}'.format(lgdir,err))

for lgdir,err_init in zip(log_dirs,errors_init):
    print('Log Dir {}, Error Init {}'.format(lgdir,err_init))

print("{:.2e} mins has elapsed".format((time.time()-start_time)/60.0))

    
