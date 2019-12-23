import time
import yaml
import numpy as np

start_time = time.time()
   
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

indices_run = [[0,1],[0,2],[0,3],[0,4],[1,2],[1,3],[1,4],[2,3],[2,4],[3,4]]

print('horz & vert & src x & src y & det 1 & det 2 & det weight')
src_FWHM_x,src_FWHM_y = np.zeros(len(indices_run)),np.zeros(len(indices_run))
det_FWHM_1,det_FWHM_2,det_weight = np.zeros(len(indices_run)),np.zeros(len(indices_run)),np.zeros(len(indices_run))
for k,index in enumerate(indices_run):
    sod = []
    for i in index:
        sod.append(horz_sod[i])
        sod.append(vert_sod[i])

    file_suffix = ''
    for d in sod: 
        file_suffix += '_'+str(int(round(d/1000)))+'mm'

    with open('src'+file_suffix+'.yml','r') as fid:
        src_params = yaml.safe_load(fid)
    
    with open('det'+file_suffix+'.yml','r') as fid:
        det_params = yaml.safe_load(fid)

    src_FWHM_x[k] = src_params['source_FWHM_x_axis']
    src_FWHM_y[k] = src_params['source_FWHM_y_axis']
    det_FWHM_1[k] = det_params['detector_FWHM_1']
    det_FWHM_2[k] = det_params['detector_FWHM_2']
    det_weight[k] = det_params['detector_weight_1']

    print('{:.1f},{:.1f} & {:.1f},{:.1f} & {:.2f} & {:.2f} & {:.2f} & {:.1f} & {:.2f}'.format(horz_sod[index[0]]/1000,horz_sod[index[1]]/1000,vert_sod[index[0]]/1000,vert_sod[index[1]]/1000,src_FWHM_x[k],src_FWHM_y[k],det_FWHM_1[k],det_FWHM_2[k],det_weight[k]))

print('\n\n & src x & src y & det 1 & det 2 & det weight')
print('mean & {:.2f} & {:.2f} & {:.2f} & {:.1f} & {:.2f}'.format(np.mean(src_FWHM_x),np.mean(src_FWHM_y),np.mean(det_FWHM_1),np.mean(det_FWHM_2),np.mean(det_weight))) 
print('variance & {:.2f} & {:.2f} & {:.2f} & {:.1f} & {:.2f}'.format(np.std(src_FWHM_x),np.std(src_FWHM_y),np.std(det_FWHM_1),np.std(det_FWHM_2),np.std(det_weight))) 
 
print("Total of {:.2f} mins has elapsed".format((time.time() - start_time)/60.0))
