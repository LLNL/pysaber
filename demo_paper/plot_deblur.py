import numpy as np
from PIL import Image
from plotdeb import *

SAVE_FOLDER = '/Users/mohan3/Desktop/Journals/Blur-Modelling/figs/deblur/'

pix_wid = 0.675/7.1 #um
wiener_reg = [5.39e+00]
rwls_reg = [8.00e-04]

art_tem = np.asarray(Image.open('res_target_low_res_simens.tif'))

rad = np.asarray(Image.open('art_10mm.tif'))
bright = np.asarray(Image.open('art_bright.tif'))
dark = np.asarray(Image.open('art_dark.tif'))
art_orig = (rad-dark)/(bright-dark)

art_wiener = []
for reg in wiener_reg:
    art_wiener.append(np.asarray(Image.open('art_wiener_reg{:.2e}.tif'.format(reg))))

art_rwls = []
for reg in rwls_reg:
    art_rwls.append(np.asarray(Image.open('art_rwls_reg{:.2e}.tif'.format(reg))))

plot_artimages(SAVE_FOLDER,pix_wid,art_tem,art_orig,art_wiener,wiener_reg,art_rwls,rwls_reg)
