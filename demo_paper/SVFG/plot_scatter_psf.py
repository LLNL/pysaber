import numpy as np
from PIL import Image
import csv
from plotfig import plot_scatpsf
import os

Z_min = 1
Z_max = 85

delta = 10  #um
pixels = 1024
odd = 71003.07846-65298.23865 #um
folder = 'figs/psfs/'
spec_w = 'Weights_160kV.csv'

with open(spec_w,'r') as csv_file:
    reader = csv.reader(csv_file)
    energies,weights = [],[]
    for i,row in enumerate(reader):
        if row[0]!='Energy (keV)' and row[1]!='Weight':
            energies.append(float(row[0]))
            weights.append(float(row[1]))    
energies = np.array(energies)
weights = np.array(weights)

def linfunc(x,m):
    return(m*x)

line_params = np.fromfile('linearFitToScatExp.bin',dtype=float).reshape((2,Z_max-Z_min+1))

Z_smpl = int(input('Choose a Z: '))

scatter_exponent = line_params[0,Z_smpl-1]*energies
eff_psf = []
for exp,weigh in zip(scatter_exponent,weights):
    psf_func = lambda x:np.exp(-exp*np.arctan(x/odd))/(1+x**2/odd**2)
    max_width = delta*pixels/2
    coord = np.arange(0,max_width,delta)
    coord_x,coord_y = np.meshgrid(coord,coord)
    coordinates = np.sqrt(coord_x**2+coord_y**2)
    psf = psf_func(coordinates)
    psf /= (psf[0,0]+2*(np.sum(psf[1:,0])+np.sum(psf[0,1:]))+4*np.sum(psf[1:,1:]))
    full_psf = np.concatenate((psf[1:][::-1],psf),axis=0)
    full_psf = np.concatenate((full_psf[:,1:][:,::-1],full_psf),axis=1)
    eff_psf.append(weigh*full_psf)
eff_psf = np.sum(np.stack(eff_psf,axis=0),axis=0)
line_prof = eff_psf[eff_psf.shape[0]//2]

maxval = np.max(line_prof)
maxidx = np.argmin(np.abs(line_prof-maxval))
lowidx = np.argmin(np.abs(line_prof[:maxidx]-maxval/2))
highidx = maxidx+np.argmin(np.abs(line_prof[maxidx:]-maxval/2))
FWHM = (highidx-lowidx)*delta

plot_scatpsf(delta,eff_psf,os.path.join(folder,'scatter_psf'),FWHM)
