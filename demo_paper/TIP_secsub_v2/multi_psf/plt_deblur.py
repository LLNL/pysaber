import time
import sys
import os
import yaml
ddir = '../data'
sys.path.insert(0,os.path.abspath(ddir))
from files import *
from pysaber import wiener_deblur,get_trans_masks
from PIL import Image

start_time = time.time()

#---------------------------- READ AND NORMALIZE RADIOGRAPHS --------------------------------
ind_train = [2,3]
ind_test = [1,4]
log_dirs = ['exp_srcdetpsf_','gauss_srcdetpsf_','exp_srcdetpsf_nomix_']

pix_wid = 0.675
#noise_std = [0.01,0.006,0.002]
noise_std = [0.003]
#noise_std = 0.001
reg_start = 0.01
reg_mult = 1.1
reg_max = 1e6

def comp_mean_std(img):
    sh = img.shape
    assert len(sh)==2
    p1 = img[:sh[0]//4,:sh[1]//4]
    p2 = img[:sh[0]//4,3*sh[1]//4:]
    p3 = img[3*sh[0]//4:,:sh[1]//4]
    p4 = img[3*sh[0]//4:,3*sh[1]//4:]
    m1,m2,m3,m4 = np.mean(p1),np.mean(p2),np.mean(p3),np.mean(p4)
    s1,s2,s3,s4 = np.std(p1),np.std(p2),np.std(p3),np.std(p4)
    return np.array([m1,m2,m3,m4],dtype=float),np.array([s1,s2,s3,s4],dtype=float)

for lidx,lgdir in enumerate(log_dirs):
    rads,sod,sdd,suff,orret = [],[],[],'',[]
    for k in ind_train:
        _,_,_,suff_smpl,orret_smpl = fetch_data(ddir,k,orret=True)
        orret += orret_smpl
        for s in suff_smpl:
            suff += s

    sdir = lgdir+suff
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

    for j in range(len(ind_test)):
        rads,sod,sdd,suff,ort = fetch_data(ddir,ind_test[j],orret=True)
        for k in range(len(rads)):
            smpl_mean,smpl_std = comp_mean_std(rads[k])
            print('***********************')
            print('Rad Std is {}'.format(np.mean(smpl_std)))
            print('***********************')
            reg = reg_start
            for std in noise_std:
                while reg < reg_max:
#                    print(reg,std)
                    deb = wiener_deblur(rads[k],sod[k],sdd[k]-sod[k],pix_wid,src_params,det_params,reg_par=reg)
                    smpl_mean,smpl_std = comp_mean_std(deb)
                    smpl_std_mean = np.mean(smpl_std)
                    if smpl_std_mean < std:
                        print('-------------------------------------------')
                        print('smpl_mean {}'.format(smpl_mean))
                        print('Final Std is {} < {} at reg of {}'.format(smpl_std_mean,std,reg))
                        print('-------------------------------------------')
                        img = Image.fromarray(deb.astype(np.float32))
                        img.save(os.path.join(sdir,'deblur_{}_{}_sd{}.tif'.format(ort[k],suff[k],std)))
                        break 
                    reg = reg*reg_mult

            trans_params = trans_horz if ort[k]=='horz' else trans_vert
            trans,mask = get_trans_masks(rads[k],trans_params,edge='straight-edge')
            img = Image.fromarray(trans.astype(np.float32))
            img.save(os.path.join(sdir,'trans_{}_{}.tif'.format(ort[k],suff[k])))
            
print("{:.2e} mins has elapsed".format((time.time()-start_time)/60.0))

    
