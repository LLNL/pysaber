import time
import sys
import os
ddir = '../data'
sys.path.insert(0,os.path.abspath(ddir))
from files import *
from pysaber import estimate_blur

start_time = time.time()

#---------------------------- READ AND NORMALIZE RADIOGRAPHS --------------------------------
indices_run = [[2,3]]
pix_wid = 0.675
norm_pow = 2.0
log_dir = 'gauss_srcdetpsf_'

for index in indices_run:
    rads,sod,sdd,suff = [],[],[],''

    for k in index:
        rads_smpl,sod_smpl,sdd_smpl,suff_smpl = fetch_data(ddir,k)
        rads += rads_smpl
        sod += sod_smpl
        sdd += sdd_smpl
        for s in suff_smpl:
            suff += s

    sdir = log_dir+suff
    if not os.path.exists(sdir):
        os.mkdir(sdir)

    odd = [sdd[i]-sod[i] for i in range(len(sod))]
    src_params,det_params,trans_params = estimate_blur(rads,sod,odd,pix_wid,thresh=1e-6,pad=[3,3],edge='straight-edge',power=norm_pow,save_dir=sdir)
    print("{:.2e} mins has elapsed".format((time.time()-start_time)/60.0))

    
