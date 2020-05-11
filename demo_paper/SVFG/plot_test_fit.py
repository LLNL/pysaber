import yaml
import numpy as np
from PIL import Image
from pysaber.trans import ideal_trans_sharp_edge 
from pysaber import apply_blur_psfs
from plotfig import plot2D,plot1D,plot_srcpsf,plot_detpsf
import os

SOURCE_FOLDER = '../TIP_firstsub'
SAVE_FOLDER = './figs'

horz_edge_files = ['horz_edge_25mm.tif']
horz_bright_file = 'horz_bright.tif'
horz_dark_file = 'horz_dark.tif'
horz_sod = [24751.88806]
horz_sdd = [71003.07846]

vert_edge_files = ['vert_edge_25mm.tif']
vert_bright_file = 'vert_bright.tif'
vert_dark_file = 'vert_dark.tif'
vert_sod = [24753.05212]
vert_sdd = [71010.86044]

pix_wid = 0.675

with open(os.path.join(SOURCE_FOLDER,'src_12mm_13mm_65mm_60mm.yml'),'r') as fid:
    src_params = yaml.safe_load(fid)

with open(os.path.join(SOURCE_FOLDER,'det_12mm_13mm_65mm_60mm.yml'),'r') as fid:
    det_params = yaml.safe_load(fid)

horz_nrads,horz_tmods = [],[]
for i,filen in enumerate(horz_edge_files): 
    rad = np.asarray(Image.open(os.path.join(SOURCE_FOLDER,filen)))
    bright = np.asarray(Image.open(os.path.join(SOURCE_FOLDER,horz_bright_file)))
    dark = np.asarray(Image.open(os.path.join(SOURCE_FOLDER,horz_dark_file)))
    rad = (rad-dark)/(bright-dark)
    horz_nrads.append(rad)
    trans_dict,params,bounds = ideal_trans_sharp_edge(rad,pad_factor=[3,3])
    horz_tmods.append(trans_dict) 

vert_nrads,vert_tmods = [],[]
for i,filen in enumerate(vert_edge_files): 
    rad = np.asarray(Image.open(os.path.join(SOURCE_FOLDER,filen)))
    bright = np.asarray(Image.open(os.path.join(SOURCE_FOLDER,vert_bright_file)))
    dark = np.asarray(Image.open(os.path.join(SOURCE_FOLDER,vert_dark_file)))
    rad = (rad-dark)/(bright-dark)
    vert_nrads.append(rad)
    trans_dict,params,bounds = ideal_trans_sharp_edge(rad,pad_factor=[3,3])
    vert_tmods.append(trans_dict) 

with open(os.path.join(SOURCE_FOLDER,'trans_12mm_13mm_65mm_60mm.yml'),'r') as cfg:
    trans_dict = yaml.safe_load(cfg)

horz_tpars = np.array([0,0],dtype=float) 
for filen in ['horz_edge_12mm.tif','horz_edge_65mm.tif']: 
    horz_tpars[0] += trans_dict[filen]['minimum']
    horz_tpars[1] += trans_dict[filen]['maximum']
horz_tpars = horz_tpars/2
print(horz_tpars)

vert_tpars = np.array([0,0],dtype=float) 
for filen in ['vert_edge_13mm.tif','vert_edge_60mm.tif']: 
    vert_tpars[0] += trans_dict[filen]['minimum']
    vert_tpars[1] += trans_dict[filen]['maximum']
vert_tpars = vert_tpars/2
print(vert_tpars)

trans_img = vert_tmods[0]['ideal_trans']([0,1])
pad_widths = (np.array(trans_img.shape)-np.array(vert_nrads[0].shape))//2
linesty = ['r-','c-']
for i in range(len(vert_edge_files)): 
    trans_img = vert_tmods[i]['ideal_trans'](vert_tpars)
    prad = apply_blur_psfs(trans_img,vert_sod[i],vert_sdd[i],pix_wid,src_params,det_params,pad_widths,pad_type='edge')
    
    img = vert_nrads[i]                     
    x = np.arange(-img.shape[1]//2,img.shape[1]//2,1)*pix_wid
    y = np.arange(img.shape[0]//2,-img.shape[0]//2,-1)*pix_wid
    y_cen,x_cen = img.shape[0]//2,img.shape[1]//2
    y_off,x_off = int(img.shape[0]*0.0),np.arange(-int(0.5*img.shape[1]),int(0.5*img.shape[1]))
    lines =  [[x[x_cen+x_off],np.ones(x_off.size)*y[y_cen+y_off]]]
    x_list,y_list = [x[x_cen+x_off]],[img[y_cen+y_off,x_cen+x_off]]
    plot2D(x,y,img,SAVE_FOLDER+'/fit/vert_nrad_sod{:.0f}.png'.format(vert_sod[i]),lines=lines,linesty=linesty)    

    img = prad
    x = np.arange(-img.shape[1]//2,img.shape[1]//2,1)*pix_wid
    y = np.arange(img.shape[0]//2,-img.shape[0]//2,-1)*pix_wid
    y_cen,x_cen = img.shape[0]//2,img.shape[1]//2
    x_list.append(x[x_cen+x_off]),y_list.append(img[y_cen+y_off,x_cen+x_off])

    linestyle = ['b-','r-']
    legend = [r'True',r'Predicted']
    plot1D(x_list,y_list,SAVE_FOLDER+'/fit/vert_line_sod{:.0f}.png'.format(vert_sod[i]),aspect=0.8,legend=legend,linesty=linestyle,markstyle=None,title=r'SOD={:.2f}mm,SDD={:.2f}mm'.format(vert_sod[i]/1000,vert_sdd[i]/1000))

    print('Vertical sod {} sdd {} RMSE {}'.format(vert_sod[i],vert_sdd[i],np.sqrt(np.mean((y_list[0]-y_list[1])**2))))

trans_img = horz_tmods[0]['ideal_trans']([0,1])
pad_widths = (np.array(trans_img.shape)-np.array(horz_nrads[0].shape))//2
for i in range(len(horz_edge_files)): 
    trans_img = horz_tmods[i]['ideal_trans'](horz_tpars)
    prad = apply_blur_psfs(trans_img,horz_sod[i],horz_sdd[i],pix_wid,src_params,det_params,pad_widths,pad_type='edge')
    
    img = horz_nrads[i]                     
    x = np.arange(-img.shape[1]//2,img.shape[1]//2,1)*pix_wid
    y = np.arange(img.shape[0]//2,-img.shape[0]//2,-1)*pix_wid
    y_cen,x_cen = img.shape[0]//2,img.shape[1]//2
    y_off,x_off = np.arange(int(-0.5*img.shape[0]),int(0.5*img.shape[0])),int(img.shape[1]*0.0)
    lines = [[np.ones(y_off.size)*x[x_cen+x_off],y[y_cen+y_off]]]
    x_list,y_list = [y[y_cen+y_off]],[img[y_cen+y_off,x_cen+x_off]]
    plot2D(x,y,img,SAVE_FOLDER+'/fit/horz_nrad_sod{:.0f}.png'.format(horz_sod[i]),lines=lines,linesty=linesty)    

    img = prad
    x = np.arange(-img.shape[1]//2,img.shape[1]//2,1)*pix_wid
    y = np.arange(img.shape[0]//2,-img.shape[0]//2,-1)*pix_wid
    y_cen,x_cen = img.shape[0]//2,img.shape[1]//2
    x_list.append(y[y_cen+y_off]),y_list.append(img[y_cen+y_off,x_cen+x_off])

    linestyle = ['b-','r-']
    legend = [r'True',r'Predicted']
    plot1D(x_list,y_list,SAVE_FOLDER+'/fit/horz_line_sod{:.0f}.png'.format(horz_sod[i]),aspect=0.8,legend=legend,linesty=linestyle,markstyle=None,title=r'SOD={:.2f}mm,SDD={:.2f}mm'.format(horz_sod[i]/1000,horz_sdd[i]/1000))
    
    print('Horizontal sod {} sdd {} RMSE {}'.format(horz_sod[i],horz_sdd[i],np.sqrt(np.mean((y_list[0]-y_list[1])**2))))

