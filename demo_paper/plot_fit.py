import yaml
import numpy as np
from PIL import Image
from saber.trans import ideal_trans_sharp_edge 
from saber import apply_blur_psfs
from plotfig import plot2D,plot1D,plot_srcpsf,plot_detpsf

SAVE_FOLDER = '/Users/mohan3/Desktop/Journals/Blur-Modelling/figs'

horz_edge_files = ['horz_edge_50mm.tif','horz_edge_65mm.tif']
horz_bright_file = 'horz_bright.tif'
horz_dark_file = 'horz_dark.tif'
horz_sod = [50251.78845,65298.23865]
horz_sdd = [71003.07846,71003.07846]

vert_edge_files = ['vert_edge_50mm.tif','vert_edge_60mm.tif']
vert_bright_file = 'vert_bright.tif'
vert_dark_file = 'vert_dark.tif'
vert_sod = [50253.35291,60003.10388]
vert_sdd = [71010.86044,71010.86044]

pix_wid = 0.675

file_suffix = ''
for d1,d2 in zip(horz_sod,vert_sod): 
    file_suffix += '_'+str(int(round(d1/1000)))+'mm'
    file_suffix += '_'+str(int(round(d2/1000)))+'mm'

with open('src_init'+file_suffix+'.yml','r') as fid:
    src_params_init = yaml.safe_load(fid)

with open('src'+file_suffix+'.yml','r') as fid:
    src_params = yaml.safe_load(fid)

with open('det_init'+file_suffix+'.yml','r') as fid:
    det_params_init = yaml.safe_load(fid)

with open('det'+file_suffix+'.yml','r') as fid:
    det_params = yaml.safe_load(fid)

horz_nrads,horz_tmods = [],[]
for i,filen in enumerate(horz_edge_files): 
    rad = np.asarray(Image.open(filen))
    bright = np.asarray(Image.open(horz_bright_file))
    dark = np.asarray(Image.open(horz_dark_file))
    rad = (rad-dark)/(bright-dark)
    horz_nrads.append(rad)
    trans_dict,params,bounds = ideal_trans_sharp_edge(rad,pad_factor=[2,2])
    horz_tmods.append(trans_dict) 

vert_nrads,vert_tmods = [],[]
for i,filen in enumerate(vert_edge_files): 
    rad = np.asarray(Image.open(filen))
    bright = np.asarray(Image.open(vert_bright_file))
    dark = np.asarray(Image.open(vert_dark_file))
    rad = (rad-dark)/(bright-dark)
    vert_nrads.append(rad)
    trans_dict,params,bounds = ideal_trans_sharp_edge(rad,pad_factor=[2,2])
    vert_tmods.append(trans_dict) 

with open('trans_init'+file_suffix+'.yml','r') as cfg:
    trans_dict = yaml.safe_load(cfg)

horz_tpars_init = [] 
for filen in horz_edge_files: 
    minm = trans_dict[filen]['minimum']
    maxm = trans_dict[filen]['maximum']
    horz_tpars_init.append([minm,maxm])

vert_tpars_init = [] 
for filen in vert_edge_files: 
    minm = trans_dict[filen]['minimum']
    maxm = trans_dict[filen]['maximum']
    vert_tpars_init.append([minm,maxm])

with open('trans'+file_suffix+'.yml','r') as cfg:
    trans_dict = yaml.safe_load(cfg)

horz_tpars = [] 
for filen in horz_edge_files: 
    minm = trans_dict[filen]['minimum']
    maxm = trans_dict[filen]['maximum']
    horz_tpars.append([minm,maxm])

vert_tpars = [] 
for filen in vert_edge_files: 
    minm = trans_dict[filen]['minimum']
    maxm = trans_dict[filen]['maximum']
    vert_tpars.append([minm,maxm])

trans_img = vert_tmods[0]['ideal_trans']([0,1])
pad_widths = (np.array(trans_img.shape)-np.array(vert_nrads[0].shape))//2
linesty = ['m-','r-','c-']
for i in range(len(vert_edge_files)): 
    trans_img = vert_tmods[i]['ideal_trans'](vert_tpars_init[i])
    prad_init = apply_blur_psfs(trans_img,vert_sod[i],vert_sdd[i],pix_wid,src_params_init,det_params_init,pad_widths,pad_type='edge')

    trans_img = vert_tmods[i]['ideal_trans'](vert_tpars[i])
    prad = apply_blur_psfs(trans_img,vert_sod[i],vert_sdd[i],pix_wid,src_params,det_params,pad_widths,pad_type='edge')
    
    img = vert_nrads[i]                     
    x = np.arange(-img.shape[1]//2,img.shape[1]//2,1)*pix_wid
    y = np.arange(img.shape[0]//2,-img.shape[0]//2,-1)*pix_wid
    y_cen,x_cen = img.shape[0]//2,img.shape[1]//2
#    y_off0,x_off0 = -int(img.shape[0]*0.3),np.arange(-int(0.5*img.shape[1]),int(0.5*img.shape[1]))
    y_off1,x_off1 = -int(img.shape[0]*0.25),np.arange(-int(0.5*img.shape[1]),int(0.0*img.shape[1]))
    y_off2,x_off2 = int(img.shape[0]*0.0),np.arange(-int(0.001*img.shape[1]),int(0.015*img.shape[1]))
    y_off3,x_off3 = int(img.shape[0]*0.25),np.arange(-int(0.005*img.shape[1]),int(0.5*img.shape[1]))
    #lines = [[x[x_cen+x_off0],np.ones(x_off0.size)*y[y_cen+y_off0]],
    lines =  [[x[x_cen+x_off1],np.ones(x_off1.size)*y[y_cen+y_off1]],
             [x[x_cen+x_off2],np.ones(x_off2.size)*y[y_cen+y_off2]],
             [x[x_cen+x_off3],np.ones(x_off3.size)*y[y_cen+y_off3]]]
    #x_list0,y_list0 = [x[x_cen+x_off0]],[img[y_cen+y_off0,x_cen+x_off0]]
    x_list1,y_list1 = [x[x_cen+x_off1]],[img[y_cen+y_off1,x_cen+x_off1]]
    x_list2,y_list2 = [x[x_cen+x_off2]],[img[y_cen+y_off2,x_cen+x_off2]]
    x_list3,y_list3 = [x[x_cen+x_off3]],[img[y_cen+y_off3,x_cen+x_off3]]
    plot2D(x,y,img,SAVE_FOLDER+'/fit/vert_nrad_sod{:.0f}.png'.format(vert_sod[i]),lines=lines,linesty=linesty)    

    img = prad_init
    x = np.arange(-img.shape[1]//2,img.shape[1]//2,1)*pix_wid
    y = np.arange(img.shape[0]//2,-img.shape[0]//2,-1)*pix_wid
    y_cen,x_cen = img.shape[0]//2,img.shape[1]//2
    #x_list0.append(x[x_cen+x_off0]),y_list0.append(img[y_cen+y_off0,x_cen+x_off0])
    x_list1.append(x[x_cen+x_off1]),y_list1.append(img[y_cen+y_off1,x_cen+x_off1])
    x_list2.append(x[x_cen+x_off2]),y_list2.append(img[y_cen+y_off2,x_cen+x_off2])
    x_list3.append(x[x_cen+x_off3]),y_list3.append(img[y_cen+y_off3,x_cen+x_off3])

    img = prad
    x = np.arange(-img.shape[1]//2,img.shape[1]//2,1)*pix_wid
    y = np.arange(img.shape[0]//2,-img.shape[0]//2,-1)*pix_wid
    y_cen,x_cen = img.shape[0]//2,img.shape[1]//2
    #x_list0.append(x[x_cen+x_off0]),y_list0.append(img[y_cen+y_off0,x_cen+x_off0])
    x_list1.append(x[x_cen+x_off1]),y_list1.append(img[y_cen+y_off1,x_cen+x_off1])
    x_list2.append(x[x_cen+x_off2]),y_list2.append(img[y_cen+y_off2,x_cen+x_off2])
    x_list3.append(x[x_cen+x_off3]),y_list3.append(img[y_cen+y_off3,x_cen+x_off3])

    linestyle = ['g-','b-','r-']
    markstyle = [None,'d','*']
    legend = [r"$I_k(i,j)$",r"$\bar{I}_k(i,j)$ before step $3$", r"$\bar{I}_k(i,j)$ after step $3$"]
    #plot1D(x_list0,y_list0,SAVE_FOLDER+'/fit/line_sod{}_0.png'.format(file_suffix),aspect=0.8,legend=legend)
    plot1D(x_list1,y_list1,SAVE_FOLDER+'/fit/vert_line_sod{:.0f}_1.png'.format(vert_sod[i]),aspect=0.8,legend=legend,linesty=linestyle,markstyle=markstyle,markevery=200)
    markstyle = ['s','d','*']
    legend = [r"$I_k(i,j)$",r"$\bar{I}_k(i,j)$"+"\n"+r"before"+"\n"+r"step $3$", r"$\bar{I}_k(i,j)$"+"\n"+r"after"+"\n"+r"step $3$"]
    plot1D(x_list2,y_list2,SAVE_FOLDER+'/fit/vert_line_sod{:.0f}_2.png'.format(vert_sod[i]),aspect=0.8,legend=legend,linesty=linestyle,markstyle=markstyle,markevery=5)
    legend = [r"$I_k(i,j)$",r"$\bar{I}_k(i,j)$ before step $3$", r"$\bar{I}_k(i,j)$ after step $3$"]
    markstyle = [None,'d','*']
    plot1D(x_list3,y_list3,SAVE_FOLDER+'/fit/vert_line_sod{:.0f}_3.png'.format(vert_sod[i]),aspect=0.8,legend=legend,linesty=linestyle,markstyle=markstyle,markevery=200)

trans_img = horz_tmods[0]['ideal_trans']([0,1])
pad_widths = (np.array(trans_img.shape)-np.array(horz_nrads[0].shape))//2
for i in range(len(horz_edge_files)): 
    trans_img = horz_tmods[i]['ideal_trans'](horz_tpars_init[i])
    prad_init = apply_blur_psfs(trans_img,horz_sod[i],horz_sdd[i],pix_wid,src_params_init,det_params_init,pad_widths,pad_type='edge')

    trans_img = horz_tmods[i]['ideal_trans'](horz_tpars[i])
    prad = apply_blur_psfs(trans_img,horz_sod[i],horz_sdd[i],pix_wid,src_params,det_params,pad_widths,pad_type='edge')
    
    img = horz_nrads[i]                     
    x = np.arange(-img.shape[1]//2,img.shape[1]//2,1)*pix_wid
    y = np.arange(img.shape[0]//2,-img.shape[0]//2,-1)*pix_wid
    y_cen,x_cen = img.shape[0]//2,img.shape[1]//2
#    y_off0,x_off0 = -int(img.shape[0]*0.3),np.arange(-int(0.5*img.shape[1]),int(0.5*img.shape[1]))
    y_off1,x_off1 = np.arange(-int(0.5*img.shape[0]),-int(0.002*img.shape[0])),-int(img.shape[1]*0.25)
    y_off2,x_off2 = np.arange(-int(0.015*img.shape[0]),int(0.005*img.shape[0])),int(img.shape[1]*0.0)
    y_off3,x_off3 = np.arange(-int(0.005*img.shape[0]),int(0.5*img.shape[0])),int(img.shape[1]*0.25)
    #lines = [[x[x_cen+x_off0],np.ones(x_off0.size)*y[y_cen+y_off0]],
    lines = [[np.ones(y_off1.size)*x[x_cen+x_off1],y[y_cen+y_off1]],
             [np.ones(y_off2.size)*x[x_cen+x_off2],y[y_cen+y_off2]],
             [np.ones(y_off3.size)*x[x_cen+x_off3],y[y_cen+y_off3]]]
    #x_list0,y_list0 = [x[x_cen+x_off0]],[img[y_cen+y_off0,x_cen+x_off0]]
    x_list1,y_list1 = [y[y_cen+y_off1]],[img[y_cen+y_off1,x_cen+x_off1]]
    x_list2,y_list2 = [y[y_cen+y_off2]],[img[y_cen+y_off2,x_cen+x_off2]]
    x_list3,y_list3 = [y[y_cen+y_off3]],[img[y_cen+y_off3,x_cen+x_off3]]
    plot2D(x,y,img,SAVE_FOLDER+'/fit/horz_nrad_sod{:.0f}.png'.format(horz_sod[i]),lines=lines,linesty=linesty)    

    img = prad_init
    x = np.arange(-img.shape[1]//2,img.shape[1]//2,1)*pix_wid
    y = np.arange(img.shape[0]//2,-img.shape[0]//2,-1)*pix_wid
    y_cen,x_cen = img.shape[0]//2,img.shape[1]//2
    #x_list0.append(x[x_cen+x_off0]),y_list0.append(img[y_cen+y_off0,x_cen+x_off0])
    x_list1.append(y[y_cen+y_off1]),y_list1.append(img[y_cen+y_off1,x_cen+x_off1])
    x_list2.append(y[y_cen+y_off2]),y_list2.append(img[y_cen+y_off2,x_cen+x_off2])
    x_list3.append(y[y_cen+y_off3]),y_list3.append(img[y_cen+y_off3,x_cen+x_off3])

    img = prad
    x = np.arange(-img.shape[1]//2,img.shape[1]//2,1)*pix_wid
    y = np.arange(img.shape[0]//2,-img.shape[0]//2,-1)*pix_wid
    y_cen,x_cen = img.shape[0]//2,img.shape[1]//2
    #x_list0.append(x[x_cen+x_off0]),y_list0.append(img[y_cen+y_off0,x_cen+x_off0])
    x_list1.append(y[y_cen+y_off1]),y_list1.append(img[y_cen+y_off1,x_cen+x_off1])
    x_list2.append(y[y_cen+y_off2]),y_list2.append(img[y_cen+y_off2,x_cen+x_off2])
    x_list3.append(y[y_cen+y_off3]),y_list3.append(img[y_cen+y_off3,x_cen+x_off3])

    legend = [r"$I_k(i,j)$",r"$\bar{I}_k(i,j)$ before step $3$", r"$\bar{I}_k(i,j)$ after step $3$"]
    #plot1D(x_list0,y_list0,SAVE_FOLDER+'/fit/line_sod{}_0.png'.format(file_suffix),aspect=0.8,legend=legend)
    markstyle = [None,'d','*']
    plot1D(x_list1,y_list1,SAVE_FOLDER+'/fit/horz_line_sod{:.0f}_1.png'.format(horz_sod[i]),aspect=0.8,legend=legend,linesty=linestyle,markstyle=markstyle,markevery=200)
    legend = [r"$I_k(i,j)$",r"$\bar{I}_k(i,j)$"+"\n"+r"before"+"\n"+r"step $3$", r"$\bar{I}_k(i,j)$"+"\n"+r"after"+"\n"+r"step $3$"]
    markstyle = ['s','d','*']
    plot1D(x_list2,y_list2,SAVE_FOLDER+'/fit/horz_line_sod{:.0f}_2.png'.format(horz_sod[i]),aspect=0.8,legend=legend,linesty=linestyle,markstyle=markstyle,markevery=5)
    legend = [r"$I_k(i,j)$",r"$\bar{I}_k(i,j)$ before step $3$", r"$\bar{I}_k(i,j)$ after step $3$"]
    markstyle = [None,'d','*']
    plot1D(x_list3,y_list3,SAVE_FOLDER+'/fit/horz_line_sod{:.0f}_3.png'.format(horz_sod[i]),aspect=0.8,legend=legend,linesty=linestyle,markstyle=markstyle,markevery=200)

