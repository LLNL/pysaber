import numpy as np
import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import LogFormatterMathtext,FuncFormatter
import matplotlib.patches as patches
from matplotlib.figure import figaspect
import sys
from PIL import Image

FONTSZ = 20
LINEIND_WIDTH = 3.0
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
        for pars in draw_rect:
            rect = patches.Rectangle(pars[0],pars[1][0],pars[1][1],linewidth=2,edgecolor='b',facecolor='none')
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

def plot1D(xin,psf,filename,legend=None,crop=1.0,type='linear',xlabel=r'Distance ($\mu m)$',ylabel='Density',aspect=1.0,linesty=None,markstyle=None,markevery=1): 
    psf = psf if isinstance(psf,list) else [psf]
    xin = xin if isinstance(xin,list) else [xin]

    w,h = figaspect(aspect)
    plt.figure(figsize=(w,h))
    if(type is 'linear'):
        for i,(x,p) in enumerate(zip(xin,psf)): 
            x_min = int(x.size//2-crop*(x.size//2))
            x_max = int(x.size//2+crop*(x.size//2))+(1 if x.size % 2 != 0 else 0)
            if linesty is not None:
                plt.plot(x[x_min:x_max],p[x_min:x_max],linesty[i],linewidth=LINEIND_WIDTH,marker=markstyle[i],markevery=markevery,markersize=20)
            else:
                plt.plot(x[x_min:x_max],p[x_min:x_max],linewidth=LINEIND_WIDTH)
        if(legend is not None):
            plt.legend(legend)
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    elif(type is 'log'):
        for i,(x,p) in enumerate(zip(xin,psf)): 
            x_min = int(x.size//2-crop*(x.size//2))
            x_max = int(x.size//2+crop*(x.size//2))+(1 if x.size % 2 != 0 else 0)
            if linesty is not None:
                plt.loglog(x[x_min:x_max],p[x_min:x_max],linesty[i],linewidth=LINEIND_WIDTH)
            else:
                plt.loglog(x[x_min:x_max],p[x_min:x_max],linewidth=LINEIND_WIDTH)
        if(legend is not None):
            plt.legend(legend)
    else:
        print('Plot type is not recognized')
        sys.exit(-1)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(filename,dpi=128,bbox_inches='tight')
    plt.close()

def get_full_psf(psf):
    full_psf = np.concatenate((psf[1:][::-1],psf),axis=0)
    full_psf = np.concatenate((full_psf[:,1:][:,::-1],full_psf),axis=1)
    return(full_psf) 

def plot_srcpsf(delta,full_psf,fileprefix):
    filename = fileprefix
    
    y_hwid,x_hwid = delta*(full_psf.shape[0]//2),delta*(full_psf.shape[1]//2)
    x,y = np.arange(-x_hwid,x_hwid+delta/2,delta),np.arange(-y_hwid,y_hwid+delta/2,delta)
    
    plot2D(x,y,full_psf,filename+'.png',crop=0.75,tick='lin')
    plot1D(y,full_psf[:,full_psf.shape[1]//2],filename+'_vert.png',crop=0.75,aspect=0.85)
    plot1D(x,full_psf[full_psf.shape[0]//2],filename+'_horz.png',crop=0.75,aspect=0.85)
    
    filename = fileprefix + '_mtf'
    full_psf_shift = np.fft.ifftshift(full_psf)

    mtf = np.absolute(np.fft.fftshift(np.fft.fft2(full_psf_shift)))
    vdelta = 1.0/(full_psf.shape[0]*delta)
    udelta = 1.0/(full_psf.shape[1]*delta)
    v_hwid,u_hwid = vdelta*(mtf.shape[0]//2),udelta*(mtf.shape[1]//2)
    u,v = np.arange(-u_hwid,u_hwid+udelta/2,udelta),np.arange(-v_hwid,v_hwid+vdelta/2,vdelta)

    plot2D(u,v,mtf,filename+'.png',xlabel=r'Frequency ($\mu m^{-1}$)',ylabel=r'Frequency ($\mu m^{-1}$)',crop=0.1,tick='lin')
    plot1D(v,mtf[:,mtf.shape[1]//2],filename+'_vert.png',xlabel=r'Frequency ($\mu m^{-1}$)',ylabel='',crop=0.1)
    plot1D(u,mtf[mtf.shape[0]//2],filename+'_horz.png',xlabel=r'Frequency ($\mu m^{-1}$)',ylabel='',crop=0.1)

def plot_detpsf(delta,full_psf,fileprefix):
    global LINEIND_WIDTH
    filename = fileprefix
    y_hwid,x_hwid = delta*(full_psf.shape[0]//2),delta*(full_psf.shape[1]//2)
    x,y = np.arange(0,x_hwid,delta),np.arange(0,y_hwid,delta)
    x_full,y_full = np.arange(-x_hwid+delta,x_hwid-delta/2,delta),np.arange(-y_hwid+delta,y_hwid-delta/2,delta)

    line = np.vstack((x,np.zeros(y.size)))
    LINEIND_WIDTH = 2.0
    plot2D(x_full,y_full,full_psf,filename+'_log.png',type='log',lines=line,vmax=10**-5)
    LINEIND_WIDTH = 1.0
    plot1D(y,full_psf[full_psf.shape[0]//2:,full_psf.shape[1]//2],filename+'_log_vert.png',type='log',aspect=0.85)
    plot1D(x,full_psf[full_psf.shape[0]//2,full_psf.shape[1]//2:],filename+'_log_horz.png',type='log',aspect=0.85)
        
    plot2D(x_full,y_full,full_psf,filename+'.png',lines=line)
    plot1D(y,full_psf[full_psf.shape[0]//2:,full_psf.shape[1]//2],filename+'_vert.png')
    plot1D(x,full_psf[full_psf.shape[0]//2,full_psf.shape[1]//2:],filename+'_horz.png')
        
    filename = fileprefix+'_mtf'
    full_psf_shift = np.fft.ifftshift(full_psf)

    mtf = np.absolute(np.fft.fftshift(np.fft.fft2(full_psf_shift)))
    vdelta = 1.0/(full_psf.shape[0]*delta)
    udelta = 1.0/(full_psf.shape[1]*delta)
    v_hwid,u_hwid = vdelta*(mtf.shape[0]//2),udelta*(mtf.shape[1]//2)
    u,v = np.arange(-u_hwid,u_hwid+udelta/2,udelta),np.arange(-v_hwid,v_hwid+vdelta/2,vdelta)

    plot2D(u,v,mtf,filename+'.png',xlabel=r'Frequency ($\mu m^{-1}$)',ylabel=r'Frequency ($\mu m^{-1}$)',crop=0.1,tick='lin')
    plot1D(v,mtf[:,mtf.shape[1]//2],filename+'_vert.png',xlabel=r'Frequency ($\mu m^{-1}$)',ylabel='',crop=0.1)
    plot1D(u,mtf[mtf.shape[0]//2],filename+'_horz.png',xlabel=r'Frequency ($\mu m^{-1}$)',ylabel='',crop=0.1)

def plot_truthvspredfit(delta,true_horz,ideal_horz,pred_horz,true_vert,ideal_vert,pred_vert):
    filenameh = out_folder+'fig_horz_'
    filenamev = out_folder+'fig_vert_'

    for i in range(len(true_horz)):
        shape = true_horz[i].shape
        y = np.arange(shape[0]-1,-1,-1)*delta
        x = np.arange(0,shape[1],1)*delta
        plot2D(x,y,true_horz[i],filenameh+'true_dist{}.png'.format(i),vmin=0,vmax=1)
        plot2D(x,y,pred_horz[i],filenameh+'pred_dist{}.png'.format(i),vmin=0,vmax=1)
        hrows,hcols = shape[0]//2,shape[1]//2
        midrow,midcol = ideal_horz[i].shape[0]//2,ideal_horz[i].shape[1]//2
        plot2D(x,y,ideal_horz[i][midrow-hrows:midrow+hrows,midcol-hcols:midcol+hcols],filenameh+'ideal_dist{}.png'.format(i),vmin=0,vmax=1)
        line_cols = [int(1/3*shape[1]),int(2/3*shape[1])]
        line_x = np.concatenate((line_cols[0]*np.ones(shape[0]//2),line_cols[1]*np.ones(shape[0]//2)))*delta
        line_y = np.concatenate((y[:shape[0]//2],y[:shape[0]//2])) 
        plot2D(x,y,true_horz[i],filenameh+'lineprof_locs_dist{}.png'.format(i),line=np.vstack((line_x,line_y)),vmin=0,vmax=1)
        for j in range(len(line_cols)):
            plot1D(y[:shape[0]//2],[true_horz[i][:shape[0]//2,line_cols[j]],pred_horz[i][:shape[0]//2,line_cols[j]]],filenameh+'lineprof_dist{}_line{}.png'.format(i,j),legend=['input','prediction'])            

    for i in range(len(true_vert)):
        y = np.arange(true_vert[i].shape[0]-1,-1,-1)*delta
        x = np.arange(0,true_vert[i].shape[1],1)*delta
        plot2D(x,y,true_vert[i],filenamev+'true_dist{}.png'.format(i),vmin=0,vmax=1)
        plot2D(x,y,pred_vert[i],filenamev+'pred_dist{}.png'.format(i),vmin=0,vmax=1)
        hrows,hcols = true_horz[i].shape[0]//2,true_horz[i].shape[1]//2
        midrow,midcol = ideal_horz[i].shape[0]//2,ideal_horz[i].shape[1]//2
        plot2D(x,y,ideal_vert[i][midrow-hrows:midrow+hrows,midcol-hcols:midcol+hcols],filenamev+'ideal_dist{}.png'.format(i),vmin=0,vmax=1)
        line_rows = [int(1/3*shape[0]),int(2/3*shape[0])]
        line_y = np.concatenate((line_rows[0]*np.ones(shape[1]//2),line_rows[1]*np.ones(shape[1]//2)))*delta
        line_x = np.concatenate((x[shape[1]//2:],x[shape[1]//2:])) 
        plot2D(x,y,true_vert[i],filenamev+'lineprof_locs_dist{}.png'.format(i),line=np.vstack((line_x,line_y)),vmin=0,vmax=1)
        for j in range(len(line_cols)):
            plot1D(x[shape[1]//2:],[true_vert[i][line_rows[j],shape[1]//2:],pred_vert[i][line_rows[j],shape[1]//2:]],filenamev+'lineprof_dist{}_line{}.png'.format(i,j),legend=['input','prediction'])            

