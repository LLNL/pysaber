import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import seaborn as sns
import os
from line_plot import horz_edge_lineplot,vert_edge_lineplot

direc = 'multi_psf/exp_srcdetpsf_38mm38mm50mm50mm'
files = ['{}_horz_38mm','{}_vert_38mm',
         '{}_horz_50mm','{}_vert_50mm']
horz_x,horz_y = [499,998,1497],[range(0,1010),range(970,1030),range(1003,1996)]
vert_x,vert_y = [range(0,1037),range(990,1040),range(1003,1996)],[499,998,1497]

legends = [[r"$I_k(i,j)$",r"$\bar{I}_k(i,j)$ before step $3$", r"$\bar{I}_k(i,j)$ after step $3$"],
           [r"$I_k(i,j)$",r"$\bar{I}_k(i,j)$"+"\n"+"before step $3$", r"$\bar{I}_k(i,j)$"+"\n"+"after step $3$"],
           [r"$I_k(i,j)$",r"$\bar{I}_k(i,j)$ before step $3$", r"$\bar{I}_k(i,j)$ after step $3$"]]
linestyle = ['g-','b-','r-']
markstyle = [[None,'d','*'],
             ['s','d','*'],
             [None,'d','*']]
markevery = [200,10,200]
errors = np.zeros(len(files),dtype=float)
errors_init = np.zeros(len(files),dtype=float)

for i,filen in enumerate(files):
    rad_init = np.asarray(Image.open(os.path.join(direc,filen.format('rad_init')+'.tif'))) 
    images = [rad_init]
    pred_init = np.asarray(Image.open(os.path.join(direc,filen.format('pred_init')+'.tif')))
    err_init = np.absolute(pred_init-rad_init)
    img = Image.fromarray(err_init)
    img.save(os.path.join(direc,filen.format('err_init')+'.tif')) 
    errors_init[i] = np.sqrt(np.mean(err_init**2))
    print('Init & {} & {:.3e}\\\\'.format(filen,errors_init[i]))  
    images.append(pred_init)

    rad = np.asarray(Image.open(os.path.join(direc,filen.format('rad')+'.tif'))) 
    pred = np.asarray(Image.open(os.path.join(direc,filen.format('pred')+'.tif')))
    err = np.absolute(pred-rad)
    img = Image.fromarray(err)
    img.save(os.path.join(direc,filen.format('err')+'.tif')) 
    errors[i] = np.sqrt(np.mean(err**2))
    print('Final & {} & {:.3e}\\\\'.format(filen,errors[i]))  
    images.append(pred)

    if 'horz' in filen and '38mm' in filen:
        horz_edge_lineplot(images,os.path.join('results',filen.format('pred_init')),legends,linestyle,markstyle,markevery,horz_x,horz_y)
    if 'vert' in filen and '38mm' in filen:
        vert_edge_lineplot(images,os.path.join('results',filen.format('pred_init')),legends,linestyle,markstyle,markevery,vert_x,vert_y)
