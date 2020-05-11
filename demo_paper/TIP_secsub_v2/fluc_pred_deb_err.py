import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import seaborn as sns
import os
from line_plot import horz_edge_lineplot,vert_edge_lineplot

#direcs = ['exp_srcdetpsf_12mm13mm25mm25mm',
#          'exp_srcdetpsf_12mm13mm38mm38mm',
#          'exp_srcdetpsf_12mm13mm50mm50mm',
#          'exp_srcdetpsf_12mm13mm65mm60mm',
#          'exp_srcdetpsf_25mm25mm38mm38mm',
#          'exp_srcdetpsf_25mm25mm50mm50mm',
#          'exp_srcdetpsf_25mm25mm65mm60mm',
#          'exp_srcdetpsf_38mm38mm50mm50mm',
#          'exp_srcdetpsf_38mm38mm65mm60mm',
#          'exp_srcdetpsf_50mm50mm65mm60mm']
direcs = ['exp_srcdetpsf_12mm13mm38mm38mm',
          'exp_srcdetpsf_12mm13mm50mm50mm',
          'exp_srcdetpsf_12mm13mm65mm60mm',
          'exp_srcdetpsf_38mm38mm50mm50mm',
          'exp_srcdetpsf_38mm38mm65mm60mm',
          'exp_srcdetpsf_50mm50mm65mm60mm']
#files = [['{}_horz_38mm','{}_vert_38mm']]
files = [['{}_horz_25mm','{}_vert_25mm']]
noise_std = [0.01,0.006]

#errors_pred = np.zeros((len(files),len(direcs)),dtype=float)
errors_deb = np.zeros((len(files),len(noise_std),len(direcs)),dtype=float)

for i,(filen1,filen2) in enumerate(files):
    for j,DIR in enumerate(direcs):
        DIR = 'multi_psf/'+DIR
#        rad = np.asarray(Image.open(os.path.join(DIR,filen1.format('rad')+'.tif'))) 
#        pred = np.asarray(Image.open(os.path.join(DIR,filen1.format('pred')+'.tif')))
#        err1 = np.absolute(pred-rad)
#        img = Image.fromarray(err1)
#        img.save(os.path.join(DIR,filen1.format('err')+'.tif')) 
        
#        rad = np.asarray(Image.open(os.path.join(DIR,filen2.format('rad')+'.tif'))) 
#        pred = np.asarray(Image.open(os.path.join(DIR,filen2.format('pred')+'.tif')))
#        err2 = np.absolute(pred-rad)
#        img = Image.fromarray(err2)
#        img.save(os.path.join(DIR,filen2.format('err')+'.tif')) 
       
#        err = np.stack((err1,err2),axis=0)
#        errors_pred[i,j] = np.sqrt(np.mean(err**2))
#        print(filen1,filen2,DIR,errors_pred[i,j]) 
       
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
            print(filen1,filen2,DIR,std,errors_deb[i,k,j]) 
       
prstr = ''
#for i,(filen1,filen2) in enumerate(files):
#    prstr += ' & {},{}'.format(filen1.format('pred'),filen2.format('pred'))
for i,(filen1,filen2) in enumerate(files):
    for k in range(len(noise_std)):
        prstr += ' & {},{},sd{}'.format(filen1.format('deblur'),filen2.format('deblur'),noise_std[k])
prstr += '\\\\'
print(prstr)

#mean_pred = np.min(errors_pred,axis=-1) 
mean_deb = np.min(errors_deb,axis=-1) 
#std_pred = np.std(errors_pred,axis=-1) 
std_deb = np.std(errors_deb,axis=-1) 
for j,DIR in enumerate(direcs):
    prstr = '{}'.format(DIR)
#    for i in range(len(files)):
#        prstr += ' & {:.3e}'.format(errors_pred[i,j])
    for i in range(len(files)):
        for k in range(len(noise_std)):
            #prstr += ' & {:.2e}'.format(errors_deb[i,k,j])
            prstr += ' & {:.4f}'.format(errors_deb[i,k,j])
    prstr += '\\\\\\hline'
    print(prstr)

print('MEAN')    
prstr = ''
#for i in range(len(files)):
#    prstr += ' & {:.3e}'.format(mean_pred[i])
for i in range(len(files)):
    for k in range(len(noise_std)):
        #prstr += ' & {:.2e}'.format(mean_deb[i,k])
        prstr += ' & {:.4f}'.format(mean_deb[i,k])
prstr += '\\\\\\hline'
print(prstr)

print('STDEV')    
prstr = ''
#for i in range(len(files)):
#    prstr += ' & {:.3e}'.format(std_pred[i])
for i in range(len(files)):
    for k in range(len(noise_std)):
        #prstr += ' & {:.2e}'.format(std_deb[i,k])
        prstr += ' & {:.4f}'.format(std_deb[i,k])
prstr += '\\\\\\hline'
print(prstr)

print('100*STDEV/MEAN')   
prstr = ''
#for i in range(len(files)):
#    prstr += ' & {:.3e}'.format(100*std_pred[i]/mean_pred[i])
for i in range(len(files)):
    for k in range(len(noise_std)):
        #prstr += ' & {:.2e}'.format(100*std_deb[i,k]/mean_deb[i,k])
        prstr += ' & {:.2f}'.format(100*round(std_deb[i,k],4)/round(mean_deb[i,k],4))
prstr += '\\\\\\hline'
print(prstr)
 
