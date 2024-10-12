from itertools import repeat
import time

import numpy as np
import pandas as pd
from astropy.modeling.models import BlackBody
import astropy.constants as const
import astropy.units as u
import os
import base_funcs as bf
from H_emission_refac import get_l_slab

config = bf.config_read("config_file.cfg")
save = config["save"]

# define the constants
sigma = const.sigma_sb
c = const.c
h = const.h
k = const.k_B
m_e = const.m_e
Z = 1  # number of protons in the nucleus # here it is Hydrogen


t_h_slab = config["t_slab"]
n_e = config["n_e"]
n_i = n_e  # for the H slab ne = ni = nH
v_o = 3.28795e15 * u.Hertz  # ionisation frequency of H

# Defining the blackbody SEDs
bb_v = BlackBody(temperature=t_h_slab)
bb_lam = BlackBody(temperature=t_h_slab, scale=1.0 * u.erg / (u.cm ** 2 * u.AA * u.s * u.sr))

def generate_kappa_fb_arr(config_file):
    """Generates the kappa (i.e. absorption coefficient) for free-bound case.
    Parameters
    ----------
    config_file:    dict
        Configuration dictionary
    Returns
    -------
    kappa_h_l_fb_arr:   astropy.units.Quantity
        kappa (i.e absorption coefficient) for free-bound case
    """
    wavelength = np.logspace(np.log10(config_file['l_min'].value),
                      np.log10(config_file['l_max'].value), config_file['n_h_minus']) * u.AA
    t_slab = config_file['t_slab']

    # Parameters for the photo-detachment cross-section calculation
    phto_detach_coeff = pd.DataFrame({'n': [1, 2, 3, 4, 5, 6]})
    phto_detach_coeff['C_n'] = [152.519, 49.534, -118.858, 92.536, -34.194, 4.982]

    lamb_0 = 1.6419 * u.micrometer # photo-detachment threshold wavelength
    alpha_const = 1.439e8 * u.AA * u.K

    wavelength_fb = np.extract(wavelength < lamb_0.to(u.AA), wavelength)

    #code to avoid for loop
    fb_l = np.zeros(wavelength_fb.shape[0]) * u.cm **2
    l_micro = wavelength_fb.to(u.micrometer)
    for n in range(1, 7):
        term = phto_detach_coeff.loc[n - 1][1] * np.power((1 / l_micro.value - 1 / lamb_0.value),((n - 1) / 2)) * (
            u.cm) ** 2  # nth row and 1st column is the coefficient # eqn 2.28 Manara
        fb_l += term
    sigma_lamb = 1e-18 * l_micro.value ** 3 * np.power((1 / l_micro.value - 1 / lamb_0.value), 1.5) * fb_l  # unit same as 2.28
    kappa_h_l_fb_arr = (0.750 * t_slab.value ** (-5 / 2) * np.exp(alpha_const / (lamb_0 * t_slab)) * (
        1 - np.exp(-alpha_const / (l_micro * t_slab))) * sigma_lamb).value
    kappa_h_l_fb_arr = kappa_h_l_fb_arr * (u.cm) ** 4 * (u.dyne) ** (-1)

    return kappa_h_l_fb_arr


def generate_kappa_ff_arr(config_file):
    l = np.logspace(np.log10(config_file['l_min'].value),
                    np.log10(config_file['l_max'].value), config_file['n_h_minus']) * u.AA
    t_slab = config_file['t_slab']

    # Parameters for free-free absorption coefficient
    # For 1820 Angstrom to 3645 angstrom
    free_free_coeff_1 = pd.DataFrame({'n': [1, 2, 3, 4, 5, 6]})
    free_free_coeff_1['A_n'] = [518.1021, 473.2636, -482.2089, 115.5291, 0, 0]
    free_free_coeff_1['B_n'] = [-734.8667, 1443.4137, -737.1616, 169.6374, 0, 0]
    free_free_coeff_1['C_n'] = [1021.1755, -1977.3395, 1096.8827, -245.6490, 0, 0]
    free_free_coeff_1['D_n'] = [-479.0721, 922.3575, -521.1341, 114.2430, 0, 0]
    free_free_coeff_1['E_n'] = [93.1373, -178.9275, 101.7963, -21.9972, 0, 0]
    free_free_coeff_1['F_n'] = [-6.4285, 12.3600, -7.0571, 1.5097, 0, 0]
    # for greater than 3645 angstrom
    free_free_coeff_2 = pd.DataFrame({'n': [1, 2, 3, 4, 5, 6]})
    free_free_coeff_2['A_n'] = [0, 2483.3460, -3449.8890, 2200.0400, -696.2710, 88.2830]
    free_free_coeff_2['B_n'] = [0, 285.827, -1158.3820, 2427.7190, -1841.4000, 444.5170]
    free_free_coeff_2['C_n'] = [0, -2054.2910, 8746.5230, -13651.1050, 8624.9700, -1863.8640]
    free_free_coeff_2['D_n'] = [0, 2827.7760, -11485.6320, 16755.5240, -10051.5300, 2095.2880]
    free_free_coeff_2['E_n'] = [0, -1341.5370, 5303.6090, -7510.4940, 4400.0670, -901.7880]
    free_free_coeff_2['F_n'] = [0, 208.9520, -812.9390, 1132.7380, -655.0200, 132.9850]

    # kappa_ff_arr = []
    # print('entering for loop')

    #Alternative code to avoid the for loop
    kappa_ff_arr = np.zeros(config_file['n_h_minus'])

    #wavelengths less than 0.3645 microns
    wavelengths_micron = l.to(u.micrometer).value
    l_micro = wavelengths_micron[np.where(wavelengths_micron<0.3645)]
    ff_l = np.zeros(l_micro.shape[0])
    for n in range(1, 7):
        term = (5040 / t_slab.value) ** ((n + 1) / 2) * (
                l_micro ** 2 * free_free_coeff_1.loc[n - 1][1] + free_free_coeff_1.loc[n - 1][2] +
                free_free_coeff_1.loc[n - 1][3] / l_micro + free_free_coeff_1.loc[n - 1][4] / l_micro ** 2 +
                free_free_coeff_1.loc[n - 1][5] / l_micro ** 3 + free_free_coeff_1.loc[n - 1][6] / l_micro ** 4)
        ff_l += term
    kappa_ff_arr[:l_micro.shape[0]] = ff_l * 1e-29
    lower_end = l_micro.shape[0]

    #wavelengths larger than 0.3645 microns
    l_micro = wavelengths_micron[np.where(wavelengths_micron>=0.3645)]
    ff_l = np.zeros(l_micro.shape[0])
    for n in range(1, 7):
        term = (5040 / t_slab.value) ** ((n + 1) / 2) * (
                l_micro ** 2 * free_free_coeff_2.loc[n - 1][1] + free_free_coeff_2.loc[n - 1][2] +
                free_free_coeff_2.loc[n - 1][3] / l_micro + free_free_coeff_2.loc[n - 1][
                    4] / l_micro ** 2 +
                free_free_coeff_2.loc[n - 1][5] / l_micro ** 3 + free_free_coeff_2.loc[n - 1][
                    6] / l_micro ** 4)
        ff_l += term
    kappa_ff_arr[lower_end:] = ff_l * 1e-29

    kappa_ff_arr = np.array(kappa_ff_arr) * u.cm ** 4 / u.dyne
    # print("kappa ff len: ", len(kappa_ff_arr))
    return kappa_ff_arr


def h_min_inten_wavelength(config_file):
    """
    This function calculates the h-emission case intensity taking wavelength array parameters from
    config file. It also checks if the parameters are within range of the model or not. If not then
    the program doesn't break rather puts a warning.
    """
    wavelength = np.logspace(np.log10(config_file['l_min'].value),np.log10(config_file['l_max'].value), config_file['n_h_minus']) * u.AA
    # print(len(wavelength))
    t_slab = config_file['t_slab']
    ne = config_file['n_e']
    save_grid = config_file["save_grid_data"]

    # warnings
    if 7000 * u.K > t_slab or t_slab > 11000 * u.K:
        print(f"Warning!! Temperature value of {t_slab} is out of theoretical bounds of this model")
    if 11 > np.log10(ne.value) or np.log10(ne.value) > 16:
        print(f"Warning!! Density value of {np.log10(ne.value)} is out of theoretical bounds of this model")
    if 0.1 > tau or tau > 5:
        print(f"Warning!! Optical depth value of {tau} is out of theoretical bounds of this model")
    
    # wavelength_fb = np.extract(wavelength < 16419 * u.AA, wavelength)
    kappa_fb_arr = generate_kappa_fb_arr(config_file)
    kappa_ff_arr = generate_kappa_ff_arr(config_file)
    k_l_arr = np.zeros(config_file['n_h_minus']) * kappa_fb_arr[0].unit
    k_l_arr[:kappa_fb_arr.shape[0]] = kappa_fb_arr
    k_l_arr += kappa_ff_arr

    print("Len of k_l_arr: ", len(k_l_arr))
    # finding the number density at quantum level n = 2
    # n = 2 we get from the fact that all transitions come and eqm is there at the Balmer level
    # else Lyman alpha emission will be there which is not so favoured? I don't know?
    n = 1
    # so finally we see that because the optical depth for H- emission case is less, around e-10
    # What happens is the emission that happens when a H atom releases a lymann alpha photon is
    # checked this? How do we choose the value of the nth level
    kappa_h_l_tot = k_l_arr * h ** 3 / (2 * np.pi * k * m_e) ** (3 / 2) * n ** 2 * t_slab ** (-3 / 2) * \
                    np.exp(h * v_o / (n ** 2 * k * t_slab)) * ne ** 2 * ne * k * t_slab
    # the above things are in wavelength thing
    # now have to convert everything into frequency regime
    # print(kappa_h_l_tot.unit)
    kappa_h_l_tot = kappa_h_l_tot.to(u.cm ** (-1))
    # print(kappa_h_l_tot)
    j_h_minus_l = kappa_h_l_tot * bb_lam(wavelength)
    # print(j_h_minus_l.unit)
    j_h_minus_l = j_h_minus_l.to(u.erg / (u.AA * u.s * u.sr * u.cm ** 3))
    # print(j_h_minus_l.unit, j_h_minus_l)

    #to get l_slab, we need the calculation from the H emission
    l_slab = get_l_slab(config_file)
    print("slab length: ", l_slab)
    tau_v_arr_h_minus = kappa_h_l_tot * l_slab
    # print(tau_v_arr_h_minus)
    beta_h_minus_v_arr = (1 - np.exp(-tau_v_arr_h_minus)) / tau_v_arr_h_minus
    # print(beta_h_minus_v_arr)
    intensity_h_minus_l = j_h_minus_l * l_slab * beta_h_minus_v_arr
    print(len(intensity_h_minus_l))

    if save_grid:
        h_min_grid_path = config_file["h_min_grid_path"]
        name = f"temp_{int(config_file['t_slab'].value)}_tau_{np.round(config_file['tau'],1)}_" \
               f"e{int(np.log10(config_file['n_e'].value))}_len_{config_file['n_h_minus']}"
        dtls_wrte = str(f"\n****** Constants Used *******\n"
                        f"h : {h}\nk : {k}\n"
                        f"m_e : {m_e}\nZ : {Z}\t number of protons in the nucleus, here it is Hydrogen\n"
                        f"v_o = 3.28795e15 Hz\t ionisation frequency of H\n"
                        f"\n****** Parameters from Config File ******\n"
                        f"t_slab = {config_file['t_slab']}\t Temperature of the slab\n"
                        f"tau = {config_file['tau']}\t optical depth\n"
                        f"n_e : 10**({np.log10(n_e.value)})\t Density of electrons\n"
                        f"l_l_slab = {config_file['l_l_slab']}\t ref wavelength for "
                        f"length of slab\n"
                        f"v = {(const.c / config_file['l_l_slab']).to(u.Hz)}\t ref frequency for "
                        f"length of slab\n"
                        f"l_min = {config_file['l_min']}\n"
                        f"l_max = {config_file['l_max']}\n"
                        f"n_h_minus = {config_file['n_h_minus']}\tlength of energy axis\n"
                        f"\n\n----- Some important parameters -----\n\n"
                        f"l_slab : {l_slab}\tlength of the slab calculated\n")
        if os.path.exists(f"{h_min_grid_path}/{name}"):
            print("grid for given configuration of t_slab, tau, n_e and len_h_minus exists!!")
        else:
            print('Grid DNE so creating')
            os.mkdir(f"{h_min_grid_path}/{name}")
            np.save(f"{h_min_grid_path}/{name}/Flux_wav.npy", intensity_h_minus_l.value)
            np.save(f"{h_min_grid_path}/{name}/j_h_tot.npy", j_h_minus_l.value)
            np.save(f"{h_min_grid_path}/{name}/kappa_h_l_tot.npy", kappa_h_l_tot.value)
            with open(f'{h_min_grid_path}/{name}/details.txt', 'w+') as f:
                f.write(dtls_wrte)

    return intensity_h_minus_l


def generate_grid_h_min(config_file, t_slab, log_n_e, tau):
    """
    This function helps generate the grid for H-emission.
    Parameters
    -----------
    config_file:    dict
                    the frequency range is taken from the config file
    t_slab:         int
                    the required value of temperature in Kelvin. If want to find
                    for temperature suppose 8000 K then this parameter should be 8000.
    log_n_e:        int
                    required electron density . e.g., if we want for 10^15 cm^(-3)
                    then put this parameter as 15.
    tau:            float
                    This is the optical density at 3000 Angstrom. e.g., for
                    optical depth of 1.2 we have to put 1.2.
    """
    config_file["t_slab"] = t_slab * u.K
    config_file["n_e"] = 10 ** log_n_e * (u.cm ** (-3))
    config_file["tau"] = tau
    inten = h_min_inten_wavelength(config_file)
    saving = config_file["save_grid_data"]
    print(f"save : {saving}")


if __name__ == "__main__":
    for temp in range(8000,11000,100):
        for tau in [1.0,]:
            for n in [13,]:
                print(temp, tau, n)
                st = time.time()
                generate_grid_h_min(config_file=config, t_slab=temp, log_n_e=n, tau=tau)
                print("---------------------*******-------------------")
                print(f"time taken : {time.time() - st}")
