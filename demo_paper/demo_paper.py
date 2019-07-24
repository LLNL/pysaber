import time
import yaml
import numpy as np
from PIL import Image
import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import LogFormatterMathtext,FuncFormatter
from plotfig import plot2D,plot1D,plot_srcpsf,plot_detpsf
from pysaber.trans import ideal_trans_perp_corner  
from pysaber import estimate_blur_psfs,get_source_psf,get_detector_psf,get_effective_psf,apply_blur_psfs

SAVE_FOLDER = '/Users/mohan3/Desktop/Journals/Blur-Modelling/figs'

FONTSZ = 20
LINEIND_WIDTH = 2.0
plt.rc('font', size=FONTSZ)          # controls default text sizes
plt.rc('axes', titlesize=FONTSZ)     # fontsize of the axes title
plt.rc('axes', labelsize=FONTSZ)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=FONTSZ)    # fontsize of the tick labels
plt.rc('ytick', labelsize=FONTSZ)    # fontsize of the tick labels
plt.rc('legend', fontsize=FONTSZ)    # legend fontsize
plt.rc('figure', titlesize=FONTSZ)  # fontsize of the figure title

start_time = time.time()

#---------------------------- READ AND NORMALIZE RADIOGRAPHS --------------------------------
with open('config.yml','r') as cfg:
    config = yaml.safe_load(cfg) #Get radiograph file names along with parameters of experiment

conf_keys = list(config.keys())
filenames,sod,sdd = [],[],[]
pix_wid = None
for ky in conf_keys:
    filenames.append(config[ky]['radiograph'])
    sod.append(float(config[ky]['sod'])) #Add SOD to list sod
    sdd.append(float(config[ky]['sdd'])) #Add SDD to list sdd
    if pix_wid is None:
        pix_wid = float(config[ky]['pix_wid'])
    elif pix_wid != config[ky]['pix_wid']:
        raise ValueError("ERROR: Pixel width must be same for all images")

with open('src_params_init.yml','r') as cfg:
    src_params_init = yaml.safe_load(cfg) 

with open('src_params.yml','r') as cfg:
    src_params = yaml.safe_load(cfg) 

with open('det_params_init.yml','r') as cfg:
    det_params_init = yaml.safe_load(cfg) 

with open('det_params.yml','r') as cfg:
    det_params = yaml.safe_load(cfg) 

with open('trans_params_init.yml','r') as cfg:
    trans_dict = yaml.safe_load(cfg)
trans_params_init = [] 
for i in range(len(conf_keys)):
    minm = trans_dict[filenames[i]]['minimum']
    maxm = trans_dict[filenames[i]]['maximum']
    trans_params_init.append([minm,maxm])

with open('trans_params.yml','r') as cfg:
    trans_dict = yaml.safe_load(cfg)
trans_params = [] 
for i in range(len(conf_keys)):
    minm = trans_dict[filenames[i]]['minimum']
    maxm = trans_dict[filenames[i]]['maximum']
    trans_params.append([minm,maxm])

norm_rads = [] #List that will contain normalized radiographs
trans_models,pred_rads_init,pred_rads = [],[],[]
for i in range(len(conf_keys)):
    ky = conf_keys[i]
    rad = np.asarray(Image.open(config[ky]['radiograph'])) #Read radiograph
    bright = np.asarray(Image.open(config[ky]['bright']))
    dark = np.asarray(Image.open(config[ky]['dark']))
    rad = (rad*bright-dark)/(bright-dark) #Normalize radiograph
    norm_rads.append(rad) #Append normalized radiograph to list norm_rads
    mcent = [int(v) for v in config[ky]['mask_center'].split(',')]
    mwid = [int(v) for v in config[ky]['mask_width'].split(',')]
    mask = np.ones(rad.shape)
    mask[mcent[0]-mwid[0]//2:mcent[0]+mwid[0]//2,mcent[1]-mwid[1]//2:mcent[1]+mwid[1]//2] = 0
    trans_dict,_,_ = ideal_trans_perp_corner(rad,pad_factor=[3,3],mask=mask.astype(bool)) #trans_dict has ideal transmission function, masks, etc
    trans_models.append(trans_dict) #Add trans_dict to list trans_models 
    trans_img = trans_dict['ideal_trans'](trans_params_init[i])
    pad_widths = (np.array(trans_img.shape)-np.array(rad.shape))//2
    pred_rads_init.append(apply_blur_psfs(trans_img,sod[i],sdd[i],pix_wid,src_params_init,det_params_init,pad_widths,pad_type='edge'))
    trans_img = trans_dict['ideal_trans'](trans_params[i])
    pad_widths = (np.array(trans_img.shape)-np.array(rad.shape))//2
    pred_rads.append(apply_blur_psfs(trans_img,sod[i],sdd[i],pix_wid,src_params,det_params,pad_widths,pad_type='edge'))

linesty = ['r-','m-','c-']
idx = np.arange(len(filenames))[np.array(sod)==min(sod)][0]
img = norm_rads[idx]
x = np.arange(-img.shape[1]//2,img.shape[1]//2,1)*pix_wid
y = np.arange(img.shape[0]//2,-img.shape[0]//2,-1)*pix_wid
y_cen,x_cen = img.shape[0]//2,img.shape[1]//2
y_off1,x_off1 = img.shape[0]//8,np.arange(int(0.16*img.shape[1]),int(0.3*img.shape[1]))
lines = [[x[x_cen+x_off1],np.ones(x_off1.size)*y[y_cen+y_off1]]]
x_list1,y_list1 = [x[x_cen+x_off1]],[img[y_cen+y_off1,x_cen+x_off1]]
y_off2,x_off2 = img.shape[0]//4,np.arange(int(0.26*img.shape[1]),img.shape[1]//2)
x_list2,y_list2 = [x[x_cen+x_off2]],[img[y_cen+y_off2,x_cen+x_off2]]
lines.append([x[x_cen+x_off2],np.ones(x_off2.size)*y[y_cen+y_off2]])
plot2D(x,y,img,SAVE_FOLDER+'/transimg/blurred_rad.png',lines=lines,linesty=linesty)

img = trans_models[idx]['ideal_trans']([0.0,1.0])
x = np.arange(-img.shape[1]//2,img.shape[1]//2,1)*pix_wid
y = np.arange(img.shape[0]//2,-img.shape[0]//2,-1)*pix_wid
y_cen,x_cen = img.shape[0]//2,img.shape[1]//2
lines = [[x[x_cen+x_off1],np.ones(x_off1.size)*y[y_cen+y_off1]]]
lines.append([x[x_cen+x_off2],np.ones(x_off2.size)*y[y_cen+y_off2]])
x_list1.append(x[x_cen+x_off1]),y_list1.append(img[y_cen+y_off1,x_cen+x_off1])
x_list2.append(x[x_cen+x_off2]),y_list2.append(img[y_cen+y_off2,x_cen+x_off2])
plot2D(x,y,img,SAVE_FOLDER+'/transimg/ideal_rad.png',lines=lines,linesty=linesty)

img = trans_models[idx]['norm_rad_mask'].astype(float)
x = np.arange(-img.shape[1]//2,img.shape[1]//2,1)*pix_wid
y = np.arange(img.shape[0]//2,-img.shape[0]//2,-1)*pix_wid
#y_cen,x_cen = img.shape[0]//2,img.shape[1]//2
#lines = [[x[x_cen+x_off1],np.ones(x_off1.size)*y[y_cen+y_off1]]]
#lines.append([x[x_cen+x_off2],np.ones(x_off2.size)*y[y_cen+y_off2]])
#x_list1.append(x[x_cen+x_off1]),y_list1.append(img[y_cen+y_off1,x_cen+x_off1])
#x_list2.append(x[x_cen+x_off2]),y_list2.append(img[y_cen+y_off2,x_cen+x_off2])
plot2D(x,y,img,SAVE_FOLDER+'/transimg/rad_mask.png')

img = trans_models[idx]['ideal_trans_mask'].astype(float)
x = np.arange(-img.shape[1]//2,img.shape[1]//2,1)*pix_wid
y = np.arange(img.shape[0]//2,-img.shape[0]//2,-1)*pix_wid
#y_cen,x_cen = img.shape[0]//2,img.shape[1]//2
#lines = [[x[x_cen+x_off1],np.ones(x_off1.size)*y[y_cen+y_off1]]]
#lines.append([x[x_cen+x_off2],np.ones(x_off2.size)*y[y_cen+y_off2]])
#x_list1.append(x[x_cen+x_off1]),y_list1.append(img[y_cen+y_off1,x_cen+x_off1])
#x_list2.append(x[x_cen+x_off2]),y_list2.append(img[y_cen+y_off2,x_cen+x_off2])
plot2D(x,y,img,SAVE_FOLDER+'/transimg/trans_mask.png')

legend1 = [r'$I_k(i,j)$',r'$\tilde{T}_k(i,j)$']
legend2 = [r'$I_k(i,j)$',r'$\tilde{T}_k(i,j)$']
plot1D(x_list1,y_list1,SAVE_FOLDER+'/transimg/rad_lineprof1.png',aspect=0.9,legend=legend1)
plot1D(x_list2,y_list2,SAVE_FOLDER+'/transimg/rad_lineprof2.png',aspect=0.9,legend=legend2)

src_params['cutoff_FWHM_multiplier'] = 20
src_psf = get_source_psf(pix_wid,src_params)
x = np.arange(-src_psf.shape[1]//2,src_psf.shape[1]//2,1)*pix_wid
y = np.arange(src_psf.shape[0]//2,-src_psf.shape[0]//2,-1)*pix_wid
plot_srcpsf(pix_wid,src_psf,SAVE_FOLDER+'/psfs/srcpsf_atsrc')

for d1,d2 in zip(sod,sdd):
    max_wid = pix_wid*src_params['cutoff_FWHM_multiplier']*max(src_params['source_FWHM_x_axis'],src_params['source_FWHM_y_axis'])/2.0
    src_psf = get_source_psf(pix_wid,src_params,sod=d1,sdd=d2,max_wid=max_wid)
    x = np.arange(-src_psf.shape[1]//2,src_psf.shape[1]//2,1)*pix_wid
    y = np.arange(src_psf.shape[0]//2,-src_psf.shape[0]//2,-1)*pix_wid
    plot_srcpsf(pix_wid,src_psf,SAVE_FOLDER+'/psfs/srcpsf_sod{}'.format(int(d1)))

det_psf = get_detector_psf(pix_wid/2,det_params)
x = np.arange(-det_psf.shape[1]//2,det_psf.shape[1]//2,1)*pix_wid/2
y = np.arange(det_psf.shape[0]//2,-det_psf.shape[0]//2,-1)*pix_wid/2
plot_detpsf(pix_wid/2,det_psf,SAVE_FOLDER+'/psfs/detpsf')

i = np.arange(len(norm_rads))[np.array(sod)==min(sod)][0]
img = norm_rads[i]
x = np.arange(-img.shape[1]//2,img.shape[1]//2,1)*pix_wid
y = np.arange(img.shape[0]//2,-img.shape[0]//2,-1)*pix_wid
y_cen,x_cen = img.shape[0]//2,img.shape[1]//2
y_off1,x_off1 = img.shape[0]//8,np.arange(int(0.2*img.shape[1]),int(0.28*img.shape[1]))
y_off2,x_off2 = img.shape[0]//6,np.arange(int(0.25*img.shape[1]),int(img.shape[1]//2))
y_off3,x_off3 = img.shape[0]//4,np.arange(int(-0.1*img.shape[1]),int(0.22*img.shape[1]))
lines = [[x[x_cen+x_off1],np.ones(x_off1.size)*y[y_cen+y_off1]],
         [x[x_cen+x_off2],np.ones(x_off2.size)*y[y_cen+y_off2]],
         [x[x_cen+x_off3],np.ones(x_off3.size)*y[y_cen+y_off3]]]
x_list1,y_list1 = [x[x_cen+x_off1]],[img[y_cen+y_off1,x_cen+x_off1]]
x_list2,y_list2 = [x[x_cen+x_off2]],[img[y_cen+y_off2,x_cen+x_off2]]
x_list3,y_list3 = [x[x_cen+x_off3]],[img[y_cen+y_off3,x_cen+x_off3]]
plot2D(x,y,img,SAVE_FOLDER+'/fit/inp_rad{}.png'.format(int(sod[i])),lines=lines,linesty=linesty)    

img = pred_rads_init[i]
img = img[img.shape[0]//2-y_cen:img.shape[0]//2+y_cen,img.shape[1]//2-x_cen:img.shape[1]//2+x_cen]
img = img.reshape(norm_rads[i].shape)
x = np.arange(-img.shape[1]//2,img.shape[1]//2,1)*pix_wid
y = np.arange(img.shape[0]//2,-img.shape[0]//2,-1)*pix_wid
y_cen,x_cen = img.shape[0]//2,img.shape[1]//2
lines = [[x[x_cen+x_off1],np.ones(x_off1.size)*y[y_cen+y_off1]],
         [x[x_cen+x_off2],np.ones(x_off2.size)*y[y_cen+y_off2]],
         [x[x_cen+x_off3],np.ones(x_off3.size)*y[y_cen+y_off3]]]
x_list1.append(x[x_cen+x_off1]),y_list1.append(img[y_cen+y_off1,x_cen+x_off1])
x_list2.append(x[x_cen+x_off2]),y_list2.append(img[y_cen+y_off2,x_cen+x_off2])
x_list3.append(x[x_cen+x_off3]),y_list3.append(img[y_cen+y_off3,x_cen+x_off3])
plot2D(x,y,img,SAVE_FOLDER+'/fit/pred_rad_init{}.png'.format(int(sod[i])),lines=lines,linesty=linesty)

img = pred_rads[i]
img = img[img.shape[0]//2-y_cen:img.shape[0]//2+y_cen,img.shape[1]//2-x_cen:img.shape[1]//2+x_cen]
img = img.reshape(norm_rads[i].shape)
x = np.arange(-img.shape[1]//2,img.shape[1]//2,1)*pix_wid
y = np.arange(img.shape[0]//2,-img.shape[0]//2,-1)*pix_wid
y_cen,x_cen = img.shape[0]//2,img.shape[1]//2
lines = [[x[x_cen+x_off1],np.ones(x_off1.size)*y[y_cen+y_off1]],
         [x[x_cen+x_off2],np.ones(x_off2.size)*y[y_cen+y_off2]],
         [x[x_cen+x_off3],np.ones(x_off3.size)*y[y_cen+y_off3]]]
x_list1.append(x[x_cen+x_off1]),y_list1.append(img[y_cen+y_off1,x_cen+x_off1])
x_list2.append(x[x_cen+x_off2]),y_list2.append(img[y_cen+y_off2,x_cen+x_off2])
x_list3.append(x[x_cen+x_off3]),y_list3.append(img[y_cen+y_off3,x_cen+x_off3])
plot2D(x,y,img,SAVE_FOLDER+'/fit/pred_rad{}.png'.format(int(sod[i])),lines=lines,linesty=linesty)

legend = [r"$I(i,j)$",r"$\bar{I}(i,j)$"+"\n"+r" before step $3$", r"$\bar{I}(i,j)$"+"\n"+" after step $3$"]
plot1D(x_list1,y_list1,SAVE_FOLDER+'/fit/line_sod{}_1.png'.format(int(sod[i])),aspect=0.8,legend=legend)
plot1D(x_list2,y_list2,SAVE_FOLDER+'/fit/line_sod{}_2.png'.format(int(sod[i])),aspect=0.8,legend=legend)
plot1D(x_list3,y_list3,SAVE_FOLDER+'/fit/line_sod{}_3.png'.format(int(sod[i])),aspect=0.8,legend=legend)

'''
i = np.arange(len(norm_rads))[np.array(sod)==max(sod)][0]
img = norm_rads[i]
x = np.arange(-img.shape[1]//2,img.shape[1]//2,1)*pix_wid
y = np.arange(img.shape[0]//2,-img.shape[0]//2,-1)*pix_wid
y_cen,x_cen = img.shape[0]//2,img.shape[1]//2
y_off1,x_off1 = img.shape[0]//10,np.arange(int(0.14*img.shape[1]),int(0.155*img.shape[1]))
y_off2,x_off2 = img.shape[0]//4,np.arange(int(0.15*img.shape[1]),int(img.shape[1]//2))
lines = [[x[x_cen+x_off1],np.ones(x_off1.size)*y[y_cen+y_off1]],
         [x[x_cen+x_off2],np.ones(x_off2.size)*y[y_cen+y_off2]]]
x_list1,y_list1 = [x[x_cen+x_off1]],[img[y_cen+y_off1,x_cen+x_off1]]
x_list2,y_list2 = [x[x_cen+x_off2]],[img[y_cen+y_off2,x_cen+x_off2]]
plot2D(x,y,img,SAVE_FOLDER+'/fit/inp_rad{}.png'.format(int(sod[i])),lines=lines,linesty=linesty)    

img = pred_rads_init[i]
img = img[img.shape[0]//2-y_cen:img.shape[0]//2+y_cen,img.shape[1]//2-x_cen:img.shape[1]//2+x_cen]
img = img.reshape(norm_rads[i].shape)
x = np.arange(-img.shape[1]//2,img.shape[1]//2,1)*pix_wid
y = np.arange(img.shape[0]//2,-img.shape[0]//2,-1)*pix_wid
y_cen,x_cen = img.shape[0]//2,img.shape[1]//2
lines = [[x[x_cen+x_off1],np.ones(x_off1.size)*y[y_cen+y_off1]],
         [x[x_cen+x_off2],np.ones(x_off2.size)*y[y_cen+y_off2]]]
x_list1.append(x[x_cen+x_off1]),y_list1.append(img[y_cen+y_off1,x_cen+x_off1])
x_list2.append(x[x_cen+x_off2]),y_list2.append(img[y_cen+y_off2,x_cen+x_off2])
plot2D(x,y,img,SAVE_FOLDER+'/fit/pred_rad_init{}.png'.format(int(sod[i])),lines=lines,linesty=linesty)

img = pred_rads[i]
img = img[img.shape[0]//2-y_cen:img.shape[0]//2+y_cen,img.shape[1]//2-x_cen:img.shape[1]//2+x_cen]
img = img.reshape(norm_rads[i].shape)
x = np.arange(-img.shape[1]//2,img.shape[1]//2,1)*pix_wid
y = np.arange(img.shape[0]//2,-img.shape[0]//2,-1)*pix_wid
y_cen,x_cen = img.shape[0]//2,img.shape[1]//2
lines = [[x[x_cen+x_off1],np.ones(x_off1.size)*y[y_cen+y_off1]],
         [x[x_cen+x_off2],np.ones(x_off2.size)*y[y_cen+y_off2]]]
x_list1.append(x[x_cen+x_off1]),y_list1.append(img[y_cen+y_off1,x_cen+x_off1])
x_list2.append(x[x_cen+x_off2]),y_list2.append(img[y_cen+y_off2,x_cen+x_off2])
plot2D(x,y,img,SAVE_FOLDER+'/fit/pred_rad{}.png'.format(int(sod[i])),lines=lines,linesty=linesty)

legend = [r'$I(i,j)$',r'$\bar{I}(i,j)$ before step $3$', r'$\bar{I}(i,j)$ after step $3$']
plot1D(x_list1,y_list1,SAVE_FOLDER+'/fit/line_sod{}_1.png'.format(int(sod[i])),aspect=1.0,legend=legend)
plot1D(x_list2,y_list2,SAVE_FOLDER+'/fit/line_sod{}_2.png'.format(int(sod[i])),aspect=1.0,legend=legend)
'''

with open('config_val.yml','r') as cfg:
    config = yaml.safe_load(cfg) #Get radiograph file names along with parameters of experiment

conf_keys = list(config.keys())
filenames,sod,sdd = [],[],[]
pix_wid = None
for ky in conf_keys:
    filenames.append(config[ky]['radiograph'])
    sod.append(float(config[ky]['sod'])) #Add SOD to list sod
    sdd.append(float(config[ky]['sdd'])) #Add SDD to list sdd
    if pix_wid is None:
        pix_wid = float(config[ky]['pix_wid'])
    elif pix_wid != config[ky]['pix_wid']:
        raise ValueError("ERROR: Pixel width must be same for all images")

with open('trans_params_val.yml','r') as cfg:
    trans_dict = yaml.safe_load(cfg)
trans_params_val = [] 
for i in range(len(conf_keys)):
    minm = trans_dict[filenames[i]]['minimum']
    maxm = trans_dict[filenames[i]]['maximum']
    trans_params_val.append([minm,maxm])

norm_rads = [] #List that will contain normalized radiographs
trans_models,pred_rads_val = [],[]
for i in range(len(conf_keys)):
    ky = conf_keys[i]
    rad = np.asarray(Image.open(config[ky]['radiograph'])) #Read radiograph
    bright = np.asarray(Image.open(config[ky]['bright']))
    dark = np.asarray(Image.open(config[ky]['dark']))
    rad = (rad*bright-dark)/(bright-dark) #Normalize radiograph
    norm_rads.append(rad) #Append normalized radiograph to list norm_rads
    trans_dict,_,_ = ideal_trans_perp_corner(rad,pad_factor=[3,3]) #trans_dict has ideal transmission function, masks, etc
    trans_models.append(trans_dict) #Add trans_dict to list trans_models 
    trans_img = trans_dict['ideal_trans'](trans_params_val[i])
    pad_widths = (np.array(trans_img.shape)-np.array(rad.shape))//2
    pred_rads_val.append(apply_blur_psfs(trans_img,sod[i],sdd[i],pix_wid,src_params,det_params,pad_widths,pad_type='edge'))

i = np.arange(len(conf_keys))[np.array(sod)==min(sod)][0]
img = norm_rads[i]
x = np.arange(-img.shape[1]//2,img.shape[1]//2,1)*pix_wid
y = np.arange(img.shape[0]//2,-img.shape[0]//2,-1)*pix_wid
y_cen,x_cen = img.shape[0]//2,img.shape[1]//2
y_off1,x_off1 = int(1.5*img.shape[0]//4),np.arange(int(0.15*img.shape[1]),int(0.35*img.shape[1]))
y_off2,x_off2 = int(img.shape[0]//4),np.arange(int(0.25*img.shape[1]),int(img.shape[1]//2))
y_off3,x_off3 = int(1.25*img.shape[0]//4),np.arange(int(-0.1*img.shape[1]),int(0.22*img.shape[1]))
lines = [[x[x_cen+x_off1],np.ones(x_off1.size)*y[y_cen+y_off1]],
         [x[x_cen+x_off2],np.ones(x_off2.size)*y[y_cen+y_off2]],
         [x[x_cen+x_off3],np.ones(x_off3.size)*y[y_cen+y_off3]]]
x_list1,y_list1 = [x[x_cen+x_off1]],[img[y_cen+y_off1,x_cen+x_off1]]
x_list2,y_list2 = [x[x_cen+x_off2]],[img[y_cen+y_off2,x_cen+x_off2]]
x_list3,y_list3 = [x[x_cen+x_off3]],[img[y_cen+y_off3,x_cen+x_off3]]
plot2D(x,y,img,SAVE_FOLDER+'/fit/inp_rad_val{}.png'.format(int(sod[i])),lines=lines,linesty=linesty)    

img = pred_rads_val[i]
img = img[img.shape[0]//2-y_cen:img.shape[0]//2+y_cen,img.shape[1]//2-x_cen:img.shape[1]//2+x_cen]
img = img.reshape(norm_rads[i].shape)
x = np.arange(-img.shape[1]//2,img.shape[1]//2,1)*pix_wid
y = np.arange(img.shape[0]//2,-img.shape[0]//2,-1)*pix_wid
y_cen,x_cen = img.shape[0]//2,img.shape[1]//2
lines = [[x[x_cen+x_off1],np.ones(x_off1.size)*y[y_cen+y_off1]],
         [x[x_cen+x_off2],np.ones(x_off2.size)*y[y_cen+y_off2]],
         [x[x_cen+x_off3],np.ones(x_off3.size)*y[y_cen+y_off3]]]
x_list1.append(x[x_cen+x_off1]),y_list1.append(img[y_cen+y_off1,x_cen+x_off1])
x_list2.append(x[x_cen+x_off2]),y_list2.append(img[y_cen+y_off2,x_cen+x_off2])
x_list3.append(x[x_cen+x_off3]),y_list3.append(img[y_cen+y_off3,x_cen+x_off3])
plot2D(x,y,img,SAVE_FOLDER+'/fit/pred_rad_val{}.png'.format(int(sod[i])),lines=lines,linesty=linesty)

legend = [r"$I(i,j)$",r"$\bar{I}(i,j)$"+"\n"+r" before step $3$", r"$\bar{I}(i,j)$"+"\n"+" after step $3$"]
plot1D(x_list1,y_list1,SAVE_FOLDER+'/fit/line_val_sod{}_1.png'.format(int(sod[i])),aspect=0.8,legend=legend)
plot1D(x_list2,y_list2,SAVE_FOLDER+'/fit/line_val_sod{}_2.png'.format(int(sod[i])),aspect=0.8,legend=legend)
plot1D(x_list3,y_list3,SAVE_FOLDER+'/fit/line_val_sod{}_3.png'.format(int(sod[i])),aspect=0.8,legend=legend)
