import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import seaborn as sns
import os
from line_plot import horz_edge_lineplot,vert_edge_lineplot

direcs = ['one_psf/gauss_srcpsf_38mm38mm',
          'one_psf/exp_srcpsf_38mm38mm',
          'one_psf/gauss_detpsf_50mm50mm',
          'one_psf/exp_detpsf_50mm50mm',
          'multi_psf/gauss_srcdetpsf_38mm38mm50mm50mm', 
          'multi_psf/exp_srcdetpsf_38mm38mm50mm50mm']
files = [['{}_horz_25mm','{}_vert_25mm'],
         ['{}_horz_65mm','{}_vert_60mm']]
noise_std = [0.01,0.006]

errors_pred = np.zeros((len(files),len(direcs)),dtype=float)
errors_deb = np.zeros((len(files),len(noise_std),len(direcs)),dtype=float)

for i,(filen1,filen2) in enumerate(files):
    for j,DIR in enumerate(direcs):
        rad = np.asarray(Image.open(os.path.join(DIR,filen1.format('rad')+'.tif'))) 
        pred = np.asarray(Image.open(os.path.join(DIR,filen1.format('pred')+'.tif')))
        err1 = np.absolute(pred-rad)
        img = Image.fromarray(err1)
        img.save(os.path.join(DIR,filen1.format('err')+'.tif')) 
        
        rad = np.asarray(Image.open(os.path.join(DIR,filen2.format('rad')+'.tif'))) 
        pred = np.asarray(Image.open(os.path.join(DIR,filen2.format('pred')+'.tif')))
        err2 = np.absolute(pred-rad)
        img = Image.fromarray(err2)
        img.save(os.path.join(DIR,filen2.format('err')+'.tif')) 
       
        err = np.stack((err1,err2),axis=0) 
        errors_pred[i,j] = np.sqrt(np.mean(err**2))
       
        for k,std in enumerate(noise_std): 
            trans = np.asarray(Image.open(os.path.join(DIR,filen1.format('trans')+'.tif'))) 
            deb = np.asarray(Image.open(os.path.join(DIR,filen1.format('deblur')+'_sd{}.tif'.format(std))))
            err1_deb = np.absolute(deb-trans)
            img = Image.fromarray(err1_deb)
            img.save(os.path.join(DIR,filen1.format('deberr')+'_sd{}.tif'.format(std))) 
            
            trans = np.asarray(Image.open(os.path.join(DIR,filen2.format('trans')+'.tif'))) 
            deb = np.asarray(Image.open(os.path.join(DIR,filen2.format('deblur')+'_sd{}.tif'.format(std))))
            err2_deb = np.absolute(deb-trans)
            img = Image.fromarray(err2_deb)
            img.save(os.path.join(DIR,filen2.format('deberr')+'_sd{}.tif'.format(std))) 
            
            err_deb = np.stack((err1_deb,err2_deb),axis=0)
            errors_deb[i,k,j] = np.sqrt(np.mean(err_deb**2))
       
prstr = ''
for i,(filen1,filen2) in enumerate(files):
    prstr += ' & {},{}'.format(filen1.format('pred'),filen2.format('pred'))
for k in range(len(noise_std)):
    for i,(filen1,filen2) in enumerate(files):
        prstr += ' & {},{},sd{}'.format(filen1.format('deblur'),filen2.format('deblur'),noise_std[k])
prstr += '\\\\'
print(prstr)

min_pred = np.min(errors_pred,axis=-1) 
min_deb = np.min(errors_deb,axis=-1) 
for j,DIR in enumerate(direcs):
    prstr = '{}'.format(DIR)
    for i in range(len(files)):
        if errors_pred[i,j]==min_pred[i]:
            #prstr += ' & \\textbf{'+'{:.2e}'.format(errors_pred[i,j])+'}'
            prstr += ' & \\textbf{'+'{:.4f}'.format(errors_pred[i,j])+'}'
        else:
            #prstr += ' & {:.2e}'.format(errors_pred[i,j])
            prstr += ' & {:.4f}'.format(errors_pred[i,j])
    for k in range(len(noise_std)):
        for i in range(len(files)):
            if errors_deb[i,k,j]==min_deb[i,k]:
                #prstr += ' & \\textbf{'+'{:.2e}'.format(errors_deb[i,k,j])+'}'
                prstr += ' & \\textbf{'+'{:.4f}'.format(errors_deb[i,k,j])+'}'
            else:
                #prstr += ' & {:.2e}'.format(errors_deb[i,k,j])
                prstr += ' & {:.4f}'.format(errors_deb[i,k,j])
    prstr += '\\\\\\hline'
    print(prstr)
