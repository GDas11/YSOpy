import numpy as np
import matplotlib.pyplot as plt

flux_data = np.load("obs_fits.npy")
wavelength = np.load("wavelength.npy")
flux_train = np.load("obs_viscous_disk_flux.npy")
plt.plot(wavelength, flux_train)

line, = plt.plot(wavelength, flux_data[0])
for i in range(1,len(flux_data)):

    line.set_ydata(flux_data[i])
    plt.pause(.3)
# if c%200==0:			# code for plotting
# this is commented , but if we want plot this comment has to be removed
plt.show()