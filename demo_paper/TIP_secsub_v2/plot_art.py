import numpy as np
from PIL import Image
from plotdeb import *

SAVE_FOLDER = './results_art/'

pix_wid = 0.675/7.1 #um
wiener_reg = [5.13e+00]
rwls_reg = [8.00e-04]

art_tem = np.asarray(Image.open('data/res_target_low_res_simens.tif'))

rad = np.asarray(Image.open('data/art_10mm.tif'))
bright = np.asarray(Image.open('data/art_bright.tif'))
dark = np.asarray(Image.open('data/art_dark.tif'))
art_orig = (rad-dark)/(bright-dark)

art_wiener = []
for reg in wiener_reg:
    art_wiener.append(np.asarray(Image.open('results_art/art_wiener_reg{:.2e}.tif'.format(reg))))

art_rwls = []
for reg in rwls_reg:
    art_rwls.append(np.asarray(Image.open('results_art/art_rwls_reg{:.2e}.tif'.format(reg))))

plot_artimages(SAVE_FOLDER,pix_wid,art_tem,art_orig,art_wiener,wiener_reg,art_rwls,rwls_reg)
