import numpy as np
import os
from plotfig import plot_motpsf

pix_wid = 0.5e-4 #um
pixels = 1000
SAVE_FOLDER = './figs/psfs'

FWHM_x = 3.7e-3 #um
FWHM_y = 3.0e-3 #um

const = 2*np.sqrt(2*np.log(2))
sd_x = FWHM_x/const
sd_y = FWHM_y/const

x = np.arange(-(pixels//2),pixels//2,1)*pix_wid
y = np.arange(-(pixels//2),pixels//2,1)*pix_wid
x,y = np.meshgrid(x,y,indexing='xy')
psf = np.exp(-0.5*(x*x/(sd_x*sd_x)+y*y/(sd_y*sd_y)))
psf = psf/np.sum(psf)

plot_motpsf(pix_wid,psf,os.path.join(SAVE_FOLDER,'motion_psf'),FWHM_x,FWHM_y)
