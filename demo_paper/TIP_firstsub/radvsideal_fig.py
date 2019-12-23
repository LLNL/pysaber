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
from pysaber.trans import ideal_trans_sharp_edge 
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
edge_files = ['vert_edge_13mm.tif']
bright_files = ['vert_bright.tif']
dark_files = ['vert_dark.tif']
sod = [13002.40271]
sdd = [71010.86044]
pix_wid = 0.675

for i in range(len(edge_files)):
    rad = np.asarray(Image.open(edge_files[i])) #Read radiograph
    bright = np.asarray(Image.open(bright_files[i]))
    dark = np.asarray(Image.open(dark_files[i]))

    norm_rad = (rad-dark)/(bright-dark) #Normalize radiograph
    trans_dict,_,_ = ideal_trans_sharp_edge(norm_rad,pad_factor=[2,2])
    #trans_dict has ideal transmission function, masks, etc
    ideal_trans = trans_dict['ideal_trans']([0,1])
    rad_mask = trans_dict['norm_rad_mask'].astype(float)
    ideal_mask = trans_dict['ideal_trans_mask'].astype(float)

    linesty = ['r-','m-','y-','c-']
    img = norm_rad
    x = np.arange(-img.shape[1]//2,img.shape[1]//2,1)*pix_wid
    y = np.arange(img.shape[0]//2,-img.shape[0]//2,-1)*pix_wid
    y_cen,x_cen = img.shape[0]//2,img.shape[1]//2

    y_off0,x_off0 = -int(img.shape[0]*0.3),np.arange(-int(0.5*img.shape[1]),int(0.5*img.shape[1]))
    x_list0,y_list0 = [x[x_cen+x_off0]],[img[y_cen+y_off0,x_cen+x_off0]]
    lines = [[x[x_cen+x_off0],np.ones(x_off0.size)*y[y_cen+y_off0]]]

    y_off1,x_off1 = -int(img.shape[0]*0.1),np.arange(-int(0.07*img.shape[1]),int(0.09*img.shape[1]))
    x_list1,y_list1 = [x[x_cen+x_off1]],[img[y_cen+y_off1,x_cen+x_off1]]
    lines.append([x[x_cen+x_off1],np.ones(x_off1.size)*y[y_cen+y_off1]])

    y_off2,x_off2 = int(img.shape[0]*0.1),np.arange(-int(0.5*img.shape[1]),-int(0.0*img.shape[1]))
    x_list2,y_list2 = [x[x_cen+x_off2]],[img[y_cen+y_off2,x_cen+x_off2]]
    lines.append([x[x_cen+x_off2],np.ones(x_off2.size)*y[y_cen+y_off2]])

    y_off3,x_off3 = int(img.shape[0]*0.3),np.arange(int(0.01*img.shape[1]),int(0.5*img.shape[1]))
    x_list3,y_list3 = [x[x_cen+x_off3]],[img[y_cen+y_off3,x_cen+x_off3]]
    lines.append([x[x_cen+x_off3],np.ones(x_off3.size)*y[y_cen+y_off3]])

    plot2D(x,y,img,SAVE_FOLDER+'/transimg/norm_rad_sod{:.0f}_sdd{:.0f}.png'.format(sod[i],sdd[i]),lines=lines,linesty=linesty)

    img = ideal_trans
    x = np.arange(-img.shape[1]//2,img.shape[1]//2,1)*pix_wid
    y = np.arange(img.shape[0]//2,-img.shape[0]//2,-1)*pix_wid
    y_cen,x_cen = img.shape[0]//2,img.shape[1]//2
    lines = [[x[x_cen+x_off0],np.ones(x_off0.size)*y[y_cen+y_off0]]]
    lines.append([x[x_cen+x_off1],np.ones(x_off1.size)*y[y_cen+y_off1]])
    lines.append([x[x_cen+x_off2],np.ones(x_off2.size)*y[y_cen+y_off2]])
    lines.append([x[x_cen+x_off3],np.ones(x_off3.size)*y[y_cen+y_off3]])
    x_list0.append(x[x_cen+x_off0]),y_list0.append(img[y_cen+y_off0,x_cen+x_off0])
    x_list1.append(x[x_cen+x_off1]),y_list1.append(img[y_cen+y_off1,x_cen+x_off1])
    x_list2.append(x[x_cen+x_off2]),y_list2.append(img[y_cen+y_off2,x_cen+x_off2])
    x_list3.append(x[x_cen+x_off3]),y_list3.append(img[y_cen+y_off3,x_cen+x_off3])
    plot2D(x,y,img,SAVE_FOLDER+'/transimg/ideal_trans_sod{:.0f}_sdd{:.0f}.png'.format(sod[i],sdd[i]),lines=lines,linesty=linesty)

    img = rad_mask
    x = np.arange(-img.shape[1]//2,img.shape[1]//2,1)*pix_wid
    y = np.arange(img.shape[0]//2,-img.shape[0]//2,-1)*pix_wid
    plot2D(x,y,img,SAVE_FOLDER+'/transimg/rad_mask_sod{:.0f}_sdd{:.0f}.png'.format(sod[i],sdd[i]))

    img = ideal_mask
    x = np.arange(-img.shape[1]//2,img.shape[1]//2,1)*pix_wid
    y = np.arange(img.shape[0]//2,-img.shape[0]//2,-1)*pix_wid
    plot2D(x,y,img,SAVE_FOLDER+'/transimg/ideal_mask_sod{:.0f}_sdd{:.0f}.png'.format(sod[i],sdd[i]))

    legend0 = [r'$I_k(i,j)$',r'$\tilde{T}_k(i,j)$']
    legend1 = [r'$I_k(i,j)$',r'$\tilde{T}_k(i,j)$']
    legend2 = [r'$I_k(i,j)$',r'$\tilde{T}_k(i,j)$']
    legend3 = [r'$I_k(i,j)$',r'$\tilde{T}_k(i,j)$']
    plot1D(x_list0,y_list0,SAVE_FOLDER+'/transimg/rad_lineprof0_sod{:.0f}_sdd{:.0f}.png'.format(sod[i],sdd[i]),aspect=0.9,legend=legend0)
    plot1D(x_list1,y_list1,SAVE_FOLDER+'/transimg/rad_lineprof1_sod{:.0f}_sdd{:.0f}.png'.format(sod[i],sdd[i]),aspect=0.9,legend=legend1)
    plot1D(x_list2,y_list2,SAVE_FOLDER+'/transimg/rad_lineprof2_sod{:.0f}_sdd{:.0f}.png'.format(sod[i],sdd[i]),aspect=0.9,legend=legend2)
    plot1D(x_list3,y_list3,SAVE_FOLDER+'/transimg/rad_lineprof3_sod{:.0f}_sdd{:.0f}.png'.format(sod[i],sdd[i]),aspect=0.9,legend=legend3)

