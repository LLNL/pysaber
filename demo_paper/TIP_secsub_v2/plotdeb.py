import numpy as np
import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import LogFormatterMathtext,FuncFormatter
import matplotlib.patches as patches
import sys
from PIL import Image
from plotfig import plot1D

FONTSZ = 20
LINEIND_WIDTH = 1.0
plt.rc('font', size=FONTSZ)          # controls default text sizes
plt.rc('axes', titlesize=FONTSZ)     # fontsize of the axes title
plt.rc('axes', labelsize=FONTSZ)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=FONTSZ)    # fontsize of the tick labels
plt.rc('ytick', labelsize=FONTSZ)    # fontsize of the tick labels
plt.rc('legend', fontsize=FONTSZ)    # legend fontsize
plt.rc('figure', titlesize=FONTSZ)  # fontsize of the figure title

def logfmt(x, pos):
    a, b = '{:.0e}'.format(x).split('e')
    b = int(b)
    return r'${} \times 10^{{{}}}$'.format(a, b)

def plot2D(x,y,psf,filename,crop=1.0,type='linear',vmin=None,vmax=None,xlabel=r'Distance ($\mu m)$',ylabel=r'Distance ($\mu m)$',tick='sci',colorbar=True,draw_rect=False,lines=None,linesty=['r-'],notick=False):
    plt.figure()
    x_min = int(x.size//2-crop*(x.size//2))
    x_max = int(x.size//2+crop*(x.size//2))+(1 if x.size % 2 != 0 else 0)
    y_min = int(y.size//2-crop*(y.size//2))
    y_max = int(y.size//2+crop*(y.size//2))+(1 if y.size % 2 != 0 else 0)
    if(type is 'linear'):
        img = psf[y_min:y_max,x_min:x_max]
        vmin = np.min(img) if vmin is None else vmin
        vmax = np.max(img) if vmax is None else vmax
        im = plt.pcolormesh(x[x_min:x_max],y[y_min:y_max],img,cmap='gray',vmin=vmin,vmax=vmax)
        if colorbar:
            if(np.max(img) < 0.1):
                plt.colorbar(im,format=FuncFormatter(logfmt))
            else:
                plt.colorbar(im)
        if(lines is not None):
            lines = [lines] if not isinstance(lines,list) else lines
            for line,sty in zip(lines,linesty):
                plt.plot(line[0],line[1],sty,markersize=LINEIND_WIDTH,linewidth=LINEIND_WIDTH)
    elif(type is 'log'):
        img = psf[y_min:y_max,x_min:x_max]
        vmin = np.min(img) if vmin is None else vmin
        vmax = np.max(img) if vmax is None else vmax
        im = plt.pcolormesh(x[x_min:x_max],y[y_min:y_max],img,cmap='gray',norm=LogNorm(),vmin=vmin,vmax=vmax)
        if colorbar:
            plt.colorbar(im,format=LogFormatterMathtext())
        if(lines is not None):
            lines = [lines] if not isinstance(lines,list) else lines
            for line,sty in zip(lines,linesty):
                plt.plot(line[0],line[1],sty,markersize=LINEIND_WIDTH,linewidth=LINEIND_WIDTH)
    else:
        print('Plot type is not recognized')
        sys.exit(-1)
    if draw_rect is not False:
        rect = patches.Rectangle(draw_rect[0],draw_rect[1][0],draw_rect[1][1],linewidth=2,edgecolor='r',facecolor='none')
        plt.axes(plt.gca()).add_patch(rect)
    plt.axes(plt.gca()).set_aspect('equal')
    if notick:
        plt.gca().tick_params(axis='x',labelbottom=False)
        plt.gca().tick_params(axis='y',labelleft=False)
    else: 
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        if(tick == 'sci'):
            plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
            plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plt.tight_layout()
    plt.savefig(filename,dpi=128,bbox_inches='tight')
    plt.close()

def plot_artimages(out_folder,delta,art_tem,art_orig,art_wiener,reg_wiener,art_rwls,reg_rwls):
    frac = [0.73,0.73]
    
    mid = [518-70,444+70]
    minsz = min([mid[0],art_tem.shape[0]-mid[0],mid[1],art_tem.shape[1]-mid[1]])
    
    filename = out_folder + 'art_gtruth'
    #yc = [mid[0]-minsz,mid[0]+minsz]
    #xc = [mid[1]-minsz,mid[1]+minsz]
    #art_tem = art_tem[yc[0]:yc[1],xc[0]:xc[1]]
    pix = 0.145
    #pix = 0.16
    y = np.arange(art_tem.shape[0]//2,-art_tem.shape[0]//2,-1)*pix
    x = np.arange(-art_tem.shape[1]//2,art_tem.shape[1]//2,1)*pix
    plot2D(x,y,art_tem,filename+'.png',colorbar=False)
    plot2D(x[mid[1]:mid[1]+int(frac[1]*minsz)],y[mid[0]-int(frac[0]*minsz):mid[0]],art_tem[mid[0]-int(frac[0]*minsz):mid[0],mid[1]:mid[1]+int(frac[1]*minsz)],filename+'_zoom.png',colorbar=False)

    frac = [0.63,0.63]
    #mid = [950,1050]    
    mid = [975,975]    

    filename = out_folder+'art_meas'
    img = art_orig
    minsz = min([mid[0],img.shape[0]-mid[0],mid[1],img.shape[1]-mid[1]])
    yc = [mid[0]-minsz,mid[0]+minsz]
    xc = [mid[1]-minsz,mid[1]+minsz]
    img = img[yc[0]:yc[1],xc[0]:xc[1]]
    y = np.arange(img.shape[0]//2,-img.shape[0]//2,-1)*delta
    x = np.arange(-img.shape[1]//2,img.shape[1]//2,1)*delta
    lines = [[x,np.ones(x.size)*y[3*img.shape[0]//8]]]
    #plot2D(x,y,img,filename+'.png',vmin=0.7,vmax=1.0,draw_rect=[(0,0),(img.shape[0]//6,img.shape[1]//6)],lines=lines)
    plot2D(x,y,img,filename+'.png',vmin=0.7,vmax=1.0)
    plot2D(x[minsz:minsz+int(frac[1]*minsz)],y[minsz-int(frac[0]*minsz):minsz],img[minsz-int(frac[0]*minsz):minsz,minsz:minsz+int(frac[1]*minsz)],filename+'_zoom.png',vmin=0.7,vmax=1.0)
    x_list = [x]
    y_list = [img[3*img.shape[0]//8]]    
    legend = ['Input']

    filename = out_folder+'art_wiener'
    for i in range(len(art_wiener)):
        img = art_wiener[i][yc[0]:yc[1],xc[0]:xc[1]]
        plot2D(x,y,img,filename+'_reg{}.png'.format(str(reg_wiener[i]).replace('.','_')),vmin=0.7,vmax=1.0,draw_rect=[(x[0],y[img.shape[1]//4]),(delta*(img.shape[0]//4),delta*(img.shape[1]//4))])
        plot2D(x[minsz:minsz+int(frac[1]*minsz)],y[minsz-int(frac[0]*minsz):minsz],img[minsz-int(frac[0]*minsz):minsz,minsz:minsz+int(frac[1]*minsz)],filename+'_zoom_reg{}.png'.format(str(reg_wiener[i]).replace('.','_')),vmin=0.7,vmax=1.0)
        x_list.append(x)
        y_list.append(img[3*img.shape[0]//8])
        legend.append('Wiener')
 
    filename = out_folder+'art_rwls'
    for i in range(len(art_rwls)):
        img = art_rwls[i][yc[0]:yc[1],xc[0]:xc[1]]
        plot2D(x,y,img,filename+'_reg{}.png'.format(str(reg_rwls[i]).replace('.','_')),vmin=0.7,vmax=1.0,draw_rect=[(x[0],y[img.shape[1]//4]),(delta*(img.shape[0]//4),delta*(img.shape[1]//4))])
        plot2D(x[minsz:minsz+int(frac[1]*minsz)],y[minsz-int(frac[0]*minsz):minsz],img[minsz-int(frac[0]*minsz):minsz,minsz:minsz+int(frac[1]*minsz)],filename+'_zoom_reg{}.png'.format(str(reg_rwls[i]).replace('.','_')),vmin=0.7,vmax=1.0)
        x_list.append(x)
        y_list.append(img[3*img.shape[0]//8])
        legend.append('RLSD')

    plot1D(x_list,y_list,out_folder+'/deblur_line_prof.png',legend=legend,aspect=0.5) 
