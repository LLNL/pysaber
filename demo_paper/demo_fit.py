import time
import yaml
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import LogFormatterMathtext,FuncFormatter

from pysaber.trans import ideal_trans_sharp_edge 
from pysaber import estimate_blur_psfs

start_time = time.time()

#---------------------------- READ AND NORMALIZE RADIOGRAPHS --------------------------------
horz_edge_files = ['horz_edge_12mm.tif','horz_edge_25mm.tif','horz_edge_38mm.tif','horz_edge_50mm.tif','horz_edge_65mm.tif']
horz_bright_file = 'horz_bright.tif'
horz_dark_file = 'horz_dark.tif'
horz_sod = [12003.38708,24751.88806,37501.93787,50251.78845,65298.23865]
horz_sdd = [71003.07846,71003.07846,71003.07846,71003.07846,71003.07846]

vert_edge_files = ['vert_edge_13mm.tif','vert_edge_25mm.tif','vert_edge_38mm.tif','vert_edge_50mm.tif','vert_edge_60mm.tif']
vert_bright_file = 'vert_bright.tif'
vert_dark_file = 'vert_dark.tif'
vert_sod = [13002.40271,24753.05212,37503.00272,50253.35291,60003.10388]
vert_sdd = [71010.86044,71010.86044,71010.86044,71010.86044,71010.86044]

#List that will contain source to detector distances
indices_run = [[0,1],[0,2],[0,3],[0,4],[1,2],[1,3],[1,4],[2,3],[2,4],[3,4]]
#indices_run = [[1,4]]
pix_wid = 0.675

for index in indices_run:
    norm_rads = [] #List that will contain normalized radiographs
    trans_models,trans_params,trans_bounds = [],[],[]
    sod,sdd = [],[]
    edge_files = []
    for j in index:
        edge_files.append(horz_edge_files[j])
        rad = np.asarray(Image.open(horz_edge_files[j])) #Read radiograph
        bright = np.asarray(Image.open(horz_bright_file))
        dark = np.asarray(Image.open(horz_dark_file))
        rad = (rad-dark)/(bright-dark) #Normalize radiograph
        norm_rads.append(rad) #Append normalized radiograph to list norm_rads
        trans_dict,params,bounds = ideal_trans_sharp_edge(rad,pad_factor=[2,2]) #trans_dict has ideal transmission function, masks, etc
        #trans_dict has ideal transmission function, masks, etc
        trans_models.append(trans_dict)
        #Add trans_dict to list trans_models 
        trans_params.append(params)
        trans_bounds.append(bounds)
        sod.append(horz_sod[j]) #Add SOD to list sod
        sdd.append(horz_sdd[j]) #Add SDD to list sdd
        print("Read radiograph {} with pixel width of {:.2e}, SOD of {:.2e}, and SDD of {:.2e}".format(horz_edge_files[j],pix_wid,sod[-1],sdd[-1]))
        
        edge_files.append(vert_edge_files[j])
        rad = np.asarray(Image.open(vert_edge_files[j])) #Read radiograph
        bright = np.asarray(Image.open(vert_bright_file))
        dark = np.asarray(Image.open(vert_dark_file))
        rad = (rad-dark)/(bright-dark) #Normalize radiograph
        norm_rads.append(rad) #Append normalized radiograph to list norm_rads
        trans_dict,params,bounds = ideal_trans_sharp_edge(rad,pad_factor=[2,2]) #trans_dict has ideal transmission function, masks, etc
        #trans_dict has ideal transmission function, masks, etc
        trans_models.append(trans_dict)
        #Add trans_dict to list trans_models 
        trans_params.append(params)
        trans_bounds.append(bounds)
        sod.append(vert_sod[j]) #Add SOD to list sod
        sdd.append(vert_sdd[j]) #Add SDD to list sdd
        print("Read radiograph {} with pixel width of {:.2e}, SOD of {:.2e}, and SDD of {:.2e}".format(vert_edge_files[j],pix_wid,sod[-1],sdd[-1]))

    print("Initial estimates for transmission parameters are {}".format(trans_params))
    print("Total of {:.2e} mins has elapsed since start of program".format((time.time() - start_time)/60.0))

    #---------------------------- ESTIMATE BLUR MODEL --------------------------------
    #It is recommended to run the blur model estimation in multiple installments
    #First, only estimate the detector blur using radiographs with the largest SODs
    #Radiographs with the largest SODs are expected to have minimal source blur
    sod_avg = sum(sod)/len(sod) #Compute average of all SODs

    #Only include list elements with SOD > average of all SODs
    norm_rads_est1,trans_models_est1,trans_params_est1,trans_bounds_est1,sod_est1,sdd_est1 = [],[],[],[],[],[]
    for i in range(len(norm_rads)):
        if sod[i] > sod_avg:
            norm_rads_est1.append(norm_rads[i])
            trans_models_est1.append(trans_models[i])
            trans_params_est1.append(trans_params[i])
            trans_bounds_est1.append(trans_bounds[i])
            sod_est1.append(sod[i])
            sdd_est1.append(sdd[i])

    #Estimate detector blur parameters
    _,det_params_est1,_,_,cost_det = estimate_blur_psfs(norm_rads_est1,trans_models_est1,sod_est1,sdd_est1,pix_wid,src_est=False,det_est=True,trans_est=False,trans_params=trans_params_est1,trans_bounds=trans_bounds_est1,convg_thresh=1e-5)
    print("Finished stage 1 of blur estimation")
    print("Estimated detector parameters are {}".format(det_params_est1))
    print("Total of {:.2f} mins has elapsed since start of program".format((time.time() - start_time)/60.0))

    #Next, only estimate the source blur using radiographs at the smallest source to object distances
    #Only include list elements with SOD < average of all SODs
    norm_rads_est2,trans_models_est2,trans_params_est2,trans_bounds_est2,sod_est2,sdd_est2 = [],[],[],[],[],[]
    for i in range(len(norm_rads)):
        if sod[i] < sod_avg:
            norm_rads_est2.append(norm_rads[i])
            trans_models_est2.append(trans_models[i])
            trans_params_est2.append(trans_params[i])
            trans_bounds_est2.append(trans_bounds[i])
            sod_est2.append(sod[i])
            sdd_est2.append(sdd[i])

    #Estimate source blur parameters
    src_params_est2,_,_,_,cost_src = estimate_blur_psfs(norm_rads_est2,trans_models_est2,sod_est2,sdd_est2,pix_wid,src_est=True,det_est=False,trans_est=False,trans_params=trans_params_est2,trans_bounds=trans_bounds_est2,convg_thresh=1e-5)
    print("Finished stage 2 of blur estimation")
    print("Estimated source parameters are {}".format(src_params_est2))
    print("Total of {:.2f} mins has elapsed since start of program".format((time.time() - start_time)/60.0))

    file_suffix = ''
    for d in sod: 
        file_suffix += '_'+str(int(round(d/1000)))+'mm'

    #Write source blur parameters to file
    with open('src_init'+file_suffix+'.yml','w') as fid:
        yaml.safe_dump(src_params_est2,fid,default_flow_style=False)

    #Write detector blur parameters to file
    with open('det_init'+file_suffix+'.yml','w') as fid:
        yaml.safe_dump(det_params_est1,fid,default_flow_style=False)

    #Write transmission function parameters to file
    trans_dict = {}
    for i in range(len(norm_rads)):
        trans_dict[edge_files[i]] = {}
        trans_dict[edge_files[i]]['minimum'] = trans_params[i][0] 
        trans_dict[edge_files[i]]['maximum'] = trans_params[i][1] 
    with open('trans_init'+file_suffix+'.yml','w') as fid:
        yaml.safe_dump(trans_dict,fid,default_flow_style=False)

    #Finally, refine estimates of both source and detector blur using all radiographs
    src_params,det_params,trans_params,trans_bounds,cost = estimate_blur_psfs(norm_rads,trans_models,sod,sdd,pix_wid,src_est=True,det_est=True,trans_est=True,src_params=src_params_est2,det_params=det_params_est1,trans_params=trans_params,trans_bounds=trans_bounds,convg_thresh=1e-6)
    print("Finished all stages of blur estimation")
    print("Estimated source parameters are {}, detector parameters are {}, and transmission function parameters are {}".format(src_params,det_params,trans_params))
    print("Total of {:.2f} mins has elapsed since start of program".format((time.time() - start_time)/60.0))

    #Write source blur parameters to file
    with open('src'+file_suffix+'.yml','w') as fid:
        yaml.safe_dump(src_params,fid,default_flow_style=False)

    #Write detector blur parameters to file
    with open('det'+file_suffix+'.yml','w') as fid:
        yaml.safe_dump(det_params,fid,default_flow_style=False)

    #Write transmission function parameters to file
    trans_dict = {}
    for i in range(len(norm_rads)):
        trans_dict[edge_files[i]] = {}
        trans_dict[edge_files[i]]['minimum'] = trans_params[i][0] 
        trans_dict[edge_files[i]]['maximum'] = trans_params[i][1] 
    with open('trans'+file_suffix+'.yml','w') as fid:
        yaml.safe_dump(trans_dict,fid,default_flow_style=False)

    with open('cost'+file_suffix+'.yml','w') as fid:
        yaml.safe_dump({'cost_opt_all':cost,'cost_opt_source':cost_src,'cost_opt_detector':cost_det},fid,default_flow_style=False)
