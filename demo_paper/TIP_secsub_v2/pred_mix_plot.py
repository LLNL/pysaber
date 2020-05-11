import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import seaborn as sns
import os
from line_plot import horz_edge_lineplot,vert_edge_lineplot

direcs = ['multi_psf/exp_srcdetpsf_nomix_38mm38mm50mm50mm',
          'multi_psf/exp_srcdetpsf_38mm38mm50mm50mm']
files = ['{}_horz_38mm','{}_vert_38mm',
         '{}_horz_50mm','{}_vert_50mm']
horz_x,horz_y = [499,998,1497],[range(0,1010),range(970,1030),range(1003,1996)]
vert_x,vert_y = [range(0,1037),range(990,1040),range(1003,1996)],[499,998,1497]

legend = [[r'$I_k(i,j)$',r'$\bar{I}_k(i,j)$ with fixed $q=1$',r'$\bar{I}_k(i,j)$ with estimated $q$'],
            [r'$I_k(i,j)$',r'$\bar{I}_k(i,j)$ with'+'\n'+'fixed $q=1$',r'$\bar{I}_k(i,j)$ with'+'\n'+'estimated $q$'],
            [r'$I_k(i,j)$',r'$\bar{I}_k(i,j)$ with fixed $q=1$',r'$\bar{I}_k(i,j)$ with estimated $q$']]
linestyle = ['g-','b-','r-']
markstyle = [[None,'d','*'],
             ['s','d','*'],
             [None,'d','*']]
markevery = [200,10,200]
errors = np.zeros((len(files),len(direcs)),dtype=float)

for i,filen in enumerate(files):
    images = []
    for j,DIR in enumerate(direcs):
        rad = np.asarray(Image.open(os.path.join(DIR,filen.format('rad')+'.tif'))) 
        pred = np.asarray(Image.open(os.path.join(DIR,filen.format('pred')+'.tif')))
        err = np.absolute(pred-rad)
        img = Image.fromarray(err)
        img.save(os.path.join(DIR,filen.format('err')+'.tif')) 
        errors[i,j] = np.sqrt(np.mean(err**2))
        print('{} & {} & {:.3e}\\\\'.format(DIR,filen,errors[i,j]))  
        images.append(pred)
    images = [rad]+images
    if 'horz' in filen and '38mm' in filen:
        horz_edge_lineplot(images,os.path.join('results',filen.format('pred_mix')),legend,linestyle,markstyle,markevery,horz_x,horz_y)
    if 'vert' in filen and '38mm' in filen:
        vert_edge_lineplot(images,os.path.join('results',filen.format('pred_mix')),legend,linestyle,markstyle,markevery,vert_x,vert_y)
