import time
import sys
import os
ddir = '../data'
sys.path.insert(0,os.path.abspath(ddir))
from files import *
from pysaber import estimate_blur

start_time = time.time()

#---------------------------- READ AND NORMALIZE RADIOGRAPHS --------------------------------
indices_run = [0,1,2,3,4]
pix_wid = 0.675
norm_pow = 2.0
log_dir = 'gauss_srcpsf_'

for j in indices_run:
    rads,sod,sdd,suff = fetch_data(ddir,j)

    sdir = log_dir
    for s in suff:
        sdir = sdir+s
    if not os.path.exists(sdir):
        os.mkdir(sdir)

    odd = [sdd[i]-sod[i] for i in range(len(sod))]
    det_params,trans_params = estimate_blur(rads,sod,odd,pix_wid,thresh=1e-6,pad=[3,3],edge='straight-edge',power=norm_pow,save_dir=sdir,only_src=True)
    print("{:.2e} mins has elapsed".format((time.time()-start_time)/60.0))

    
