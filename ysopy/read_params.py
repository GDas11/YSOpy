import numpy as np
import corner
import matplotlib.pyplot as plt
from pypeit.core import wave
import astropy.units as u
from astropy.io import ascii
from mcmc import total_spec
import sys

arr = np.load("params_2.npy")
x = np.zeros(shape=(3600,9))
for i in range(200):
    for j in range(18):
        x[i*18+j] = arr[i,j,:]

print(x.shape)
# print(x)
# print("shape",arr.shape)
pars = []
for i in range(9):
    pars.append(np.median(x[:,i]))
print(pars)

#read the data
path_to_valid = "../../../../validation_files/"
data = ascii.read(path_to_valid+'HIRES_sci_42767_1/KOA_42767/HIRES/extracted/tbl/ccd0/flux/HI.20030211.26428_0_02_flux.tbl.gz')
data = [data['wave'],data['Flux']/np.median(data['Flux']),data['Error']/np.median(data['Flux'])]
#vac to air correction for given data
wavelengths_air = wave.vactoair(data[0]*u.AA)
data[0] = wavelengths_air
best_fit = total_spec(pars,data[0]*u.AA)

fig = plt.figure(figsize=(16,10))
ax = fig.add_subplot(111)
ax.plot(data[0],data[1],label="observed")
ax.plot(data[0],best_fit,label="model")
plt.legend()
plt.show()

sys.exit(0)
figure = corner.corner(
    x,
    labels=[
        r'$M$',
        r'log $\dot M$',
        r'$B$', r'$\theta$',
        r'log $n_e$',
        r'$R_{\ast}$',
        r'$T_o$',
        r'$T_{slab}$',
        r'$\tau$'
    ],
    quantiles=[0.16,0.5,0.84],
    show_titles=True,
    smooth=0.5
)

plt.show()
#print(arr)