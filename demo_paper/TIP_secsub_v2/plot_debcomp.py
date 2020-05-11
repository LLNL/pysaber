import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import seaborn as sns
import os
from line_plot import horz_edge_lineplot,vert_edge_lineplot

direcs = ['multi_psf/gauss_srcdetpsf_38mm38mm50mm50mm',
          'multi_psf/exp_srcdetpsf_38mm38mm50mm50mm']
noise_std = [0.01,0.006]
files = ['{}_horz_25mm','{}_vert_25mm']

linestyle = ['b-','k-','g-','r-']
markstyle = [['^',None,'d','*'],
            ['^',None,'d','*'],
            ['^',None,'d','*']]
markevery = [20,20]
legends = [[r'Rad $I_k(i,j)$',r'Ideal $T_k(i,j)$',r'Gauss, $r=2$',r'Exp, $r=1$'],
           [r'Rad $I_k(i,j)$',r'Ideal $T_k(i,j)$',r'Gauss, $r=2$',r'Exp, $r=1$'],
           [r'Rad $I_k(i,j)$',r'Ideal $T_k(i,j)$',r'Gauss, $r=2$',r'Exp, $r=1$']]
#horz_x,horz_y = [499,998,1497],[range(0,1015),range(950,1120),range(1010,1996)]
#vert_x,vert_y = [range(0,1040),range(980,1050),range(998,1996)],[499,998,1497]
horz_x,horz_y = [998],[range(950,1120)]
vert_x,vert_y = [range(980,1050)],[998]

errors_deb = np.zeros((len(files),len(noise_std),len(direcs)),dtype=float)
for i,filen in enumerate(files):
    for k,std in enumerate(noise_std):
        images = []
        for j,DIR in enumerate(direcs):
            rad = np.asarray(Image.open(os.path.join(DIR,filen.format('rad')+'.tif'))) 
            trans = np.asarray(Image.open(os.path.join(DIR,filen.format('trans')+'.tif'))) 
            deb = np.asarray(Image.open(os.path.join(DIR,filen.format('deblur')+'_sd{}.tif'.format(std))))
            err_deb = np.absolute(deb-trans)
            img = Image.fromarray(err_deb)
            img.save(os.path.join(DIR,filen.format('deberr')+'_sd{}.tif'.format(std))) 
            errors_deb[i,k,j] = np.sqrt(np.mean(err_deb**2))
            if 'srcdetpsf' in DIR: 
                images.append(deb)        
            
        images = [rad,trans]+images
        if 'horz_25mm' in filen:
            horz_edge_lineplot(images,os.path.join('results',filen.format('deb_sd{:.0e}'.format(std))),legends,linestyle,markstyle,markevery,horz_x,horz_y,mark2D=(k==0))
        if 'vert_25mm' in filen:
            vert_edge_lineplot(images,os.path.join('results',filen.format('deb_sd{:.0e}'.format(std))),legends,linestyle,markstyle,markevery,vert_x,vert_y,mark2D=(k==0))

