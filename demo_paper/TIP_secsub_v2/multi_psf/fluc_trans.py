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
indices_run = [[0,1],[0,2],[0,3],[0,4],[1,2],[1,3],[1,4],[2,3],[2,4],[3,4]]
ind_test = [2]
pix_wid = 0.675
log_dir = 'exp_srcdetpsf_'

errors = []
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
        src_params = yaml.safe_load(fid)

    with open(os.path.join(sdir,'detector_params.yml'),'r') as fid:
        det_params = yaml.safe_load(fid)

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
            pred,_ = get_trans_fit(rads[k],sod[k],sdd[k],pix_wid,src_params,det_params,trans_params,edge_type='straight-edge')
     
            img = Image.fromarray(pred.astype(np.float32))
            img.save(os.path.join(sdir,'pred_{}_{}.tif'.format(ort[k],suff[k]))) 
            img = Image.fromarray(rads[k].astype(np.float32))
            img.save(os.path.join(sdir,'rad_{}_{}.tif'.format(ort[k],suff[k]))) 
            err += np.mean(np.absolute(rads[k]-pred))    
            num += 1 
    err = err/num
    errors.append(err)

for ind_train,err in zip(indices_run,errors):
    print('Ind train {}, Error {}'.format(ind_train,err))

print("{:.2e} mins has elapsed".format((time.time()-start_time)/60.0))

    
