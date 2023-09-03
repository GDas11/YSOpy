import argparse
from configparser import ConfigParser

import astropy
import astropy.constants as const
import astropy.units as u
import numpy as np
from astropy.io.votable import parse
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from astropy.modeling.physical_models import BlackBody
from scipy.integrate import trapezoid
import os
from dust_extinction.parameter_averages import F19
from dust_extinction.averages import G21_MWAvg
from scipy.integrate import dblquad
import numpy.ma as ma


def config_read(path):
    """This function reads the config file given a path"""
    config = ConfigParser()
    config.read(path)
    config_data = config['Default']
    dict_config = dict(config_data)
    dict_config["b"] = float(dict_config["b"]) * u.kilogauss
    dict_config["m"] = float(dict_config["m"]) * const.M_sun
    dict_config["m_dot"] = float(dict_config["m_dot"]) * const.M_sun / (1 * u.year).to(u.s)
    dict_config["r_star"] = float(dict_config["r_star"]) * const.R_sun
    dict_config["inclination"] = float(dict_config["inclination"]) * u.degree
    dict_config["d_star"] = float(dict_config["d_star"]) * u.parsec
    dict_config["t_star"] = float(dict_config["t_star"]) * u.K
    dict_config["t_0"] = float(dict_config["t_0"]) * u.K
    dict_config["av"] = float(dict_config["av"])
    dict_config["rv"] = float(dict_config["rv"])
    dict_config["l_0"] = float(dict_config["l_0"]) * u.AA
    dict_config["t_slab"] = float(dict_config["t_slab"]) * u.K
    dict_config["n_e"] = float(dict_config["n_e"]) * u.cm ** (-3)
    dict_config["tau"] = float(dict_config["tau"])
    dict_config["l_min"] = float(dict_config["l_min"]) * u.AA
    dict_config["l_max"] = float(dict_config["l_max"]) * u.AA
    dict_config["n_data"] = int(dict_config["n_data"])
    dict_config["n_disk"] = int(dict_config["n_disk"])
    dict_config["n_h"] = int(dict_config["n_h"])
    dict_config["l_l_slab"] = float(dict_config["l_l_slab"]) * u.AA
    dict_config["n_h_minus"] = int(dict_config["n_h_minus"])

    if dict_config["save"] == "True":
        dict_config["save"] = True
    else:
        dict_config["save"] = False

    if dict_config["plot"] == "True":
        dict_config["plot"] = True
    else:
        dict_config["plot"] = False

    if dict_config["save_grid_data"] == "True":
        dict_config["save_grid_data"] = True
    else:
        dict_config["save_grid_data"] = False
    # print(dict_config)
    return dict_config


'''
def unif_reinterpolate2(x, y):
    f = interp1d(x, y)
    l = np.linspace(np.log10(3000), np.log10(50000), 420000)
    y_final = f(l) * u.erg / (u.cm * u.cm * u.s * u.AA)
    return l, y_final


def unif_reinterpolate(x, y):
    """interpolate the datasets having very low sampling"""
    f = interp1d(x, y)
    wav = np.linspace(2995, 50005, 420000, endpoint=True) * u.AA
    return wav, (f(wav) * u.erg / (u.cm * u.cm * u.s * u.AA))
'''


def read_bt_settl(config, temperature: int, logg: float):
    """read the stored BT-Settl model spectra

    Parameters
    ----------
    config : dict
        dictionary containing system parameters

    temperature : int
        t // 100, where t is the t_eff of the model in Kelvin

    logg : float
        log of surface gravity of the atmosphere model

    Returns
    ----------
    trimmed_wave : astropy.units.Quantity
        array having the wavelength axis of the read BT-Settl data, in units of Angstrom,
        trimmed to the region of interest, a padding of 10 A is left to avoid errors in interpolation

    trimmed_flux : astropy.units.Quantity
        array of flux values, in units of erg / (cm^2 s A), trimmed to region of interest
    """

    loc = config['bt_settl_path']
    l_min = config['l_min']
    l_max = config['l_max']
    n_data = config['n_data']

    if temperature >= 100:
        address = f"{loc}/lte{temperature}-{logg}-0.0a+0.0.BT-Settl.7.dat.xml"
    elif (temperature > 25) and temperature < 100:
        address = f"{loc}/lte0{temperature}-{logg}-0.0a+0.0.BT-Settl.7.dat.xml"
    elif (temperature >= 20) and temperature < 100:
        address = f"{loc}/lte0{temperature}-{logg}-0.0.BT-Settl.7.dat.xml"
    else:
        address = f"{loc}/lte0{temperature}-{logg}-0.0.BT-Settl.7.dat.xml"
    table = parse(address)
    data = table.get_first_table().array

    # trim data to region of interest
    # extra region left for re-interpolation
    trimmed_data = np.extract(data['WAVELENGTH'] > l_min.value - 10, data)
    trimmed_data = np.extract(trimmed_data['WAVELENGTH'] < l_max.value + 10, trimmed_data)
    trimmed_wave = trimmed_data['WAVELENGTH'].astype(np.float64) * u.AA
    trimmed_flux = trimmed_data['FLUX'].astype(np.float64) * (u.erg / (u.cm * u.cm * u.s * u.AA))

    # for faulty data make a linear re-interpolation
    if 20 <= temperature <= 25:
        x, y = unif_reinterpolate(config, trimmed_wave, trimmed_flux)
        f = interp1d(x, y)
        trimmed_wave = np.linspace(l_min.value - 5, l_max.value + 5, n_data, endpoint=True) * u.AA
        trimmed_flux = f(trimmed_wave) * u.erg / (u.cm * u.cm * u.s * u.AA)
        # trimmed_data = ma.array([x, y], dtype=[('WAVELENGTH', 'float'), ('FLUX', 'float')])
    return trimmed_wave, trimmed_flux


def unif_reinterpolate(config, x, y):
    """interpolate the datasets having very low sampling
    Parameters
    ----------
    config : dict
             dictionary containing system parameters
    x : astropy.units.Quantity
        array of wavelength values, in units of Angstrom
    y : astropy.units.Quantity
        array of flux values, in units of erg / (cm^2 s A)
    Returns
    ---------
    wav : astropy.units.Quantity
          new wavelength axis over which the flux values are interpolated
    f(wav) : astropy.units.Quantity
             interpolated flux values, in units of erg / (cm^2 s A)
    """
    l_min = config['l_min']
    l_max = config['l_max']
    n_data = config['n_data']
    f = interp1d(x, y)
    wav = np.linspace(l_min.value - 5, l_max.value + 5, n_data, endpoint=True) * u.AA
    return wav, (f(wav) * u.erg / (u.cm * u.cm * u.s * u.AA))


def interpolate_conv(config, wavelength, flux, sampling, v_red):
    """Interpolate the given data to a logarithmic scale in the wavelength axis,
    to account for a variable kernel during convolution.

    Parameters
    ----------
    config : dict
             dictionary containing system parameters
    wavelength : astropy.units.Quantity
                 array of wavelength values, in units of Angstrom
    flux : astropy.units.Quantity
           array of flux values, in units of erg / (cm^2 s A)
    sampling : int
               desired number of points in the range (l0 - l_max, l0 + l_max)
    v_red : float
            reduced velocity, i.e. v_kep * sin(i) / c

    Returns
    ----------
    kernel_length : [int] number of points in the kernel
    wavelength_log : [numpy.ndarray] or [astropy.Quantity object] array of wavelengths in logarithmic scale
    flux_interpolated :
    """
    l0 = config['l_0'].value
    l_min = config['l_min']
    l_max = config['l_max']

    # determine the number of points in interpolated axis
    # spacing of points in interpolated axis must match that of kernel
    x_log = np.log10(wavelength.value)
    k = (1 + v_red) / (1 - v_red)
    n_points = sampling / np.log10(k) * np.log10(wavelength[-1] / wavelength[0])

    wavelength_log = np.logspace(x_log[0], x_log[-1], int(n_points))
    wavelength_log = np.extract(wavelength_log > l_min.value - 5, wavelength_log)
    wavelength_log = np.extract(wavelength_log < l_max.value + 5, wavelength_log)
    f_log = interp1d(wavelength, flux)
    flux_interpolated = f_log(wavelength_log)

    # determine number of points to be taken in kernel
    l_around = np.extract(wavelength_log > (l0 * (1 - v_red)), wavelength_log)
    l_around = np.extract(l_around < (l0 * (1 + v_red)), l_around)
    kernel_length = (len(l_around) // 2) * 2 + 1  # this gives the number of data points in the kernel

    return kernel_length, wavelength_log, flux_interpolated


def logspace_reinterp(config, wavelength, flux):
    """interpolates the given wavelength-flux data and interpolates to a logarithmic axis in the wavelength,
    used to convert all the SEDs to a common wavelength axis, so that they can be added

    Parameters
    ----------
    config : dict
        dictionary containing system parameters

    wavelength : astropy.units.Quantity
        wavelength array

    flux : astropy.units.Quantity
        flux array to be interpolated

    Returns
    ----------
    wavelength_req : astropy.units.Quantity
        new wavelength axis

    flux_final : astropy.units.Quantity
        interpolated flux array
    """
    l_min = config['l_min']
    l_max = config['l_max']
    n_data = config['n_data']
    f = interp1d(wavelength, flux)
    wavelength_req = np.logspace(np.log10(l_min.value), np.log10(l_max.value), n_data)
    flux_final = f(wavelength_req) * u.erg / (u.cm * u.cm * u.s * u.AA)

    return wavelength_req, flux_final


def ker(x, l_0, l_max):
    """Defines the kernel for the convolution. A kernel for a rotating ring is taken

    Parameters
    ----------
    x : float or astropy.units.Quantity
        value at which kernel function is to be evaluated

    l_0 : float of astropy.units.Quantity
        central wavelength around which kernel function is evaluated

    l_max : float or astropy.units.Quantity
        maximum deviation from l_0 up to which the kernel function is well-defined"""
    return 1 / np.sqrt(1 - ((x - l_0) / l_max) ** 2)


def generate_kernel(config: dict, sampling: int, v_red: astropy.units.Quantity):
    """generates the kernel in the form of an array,
    to be used for convolution of the flux in subsequent steps.

    Parameters
    ----------
    config : dict
        dictionary containing system parameters

    sampling : int
        Number of points in the kernel array

    v_red : astropy.units.Quantity
        Ratio of L_max to L_0, i.e., v_kep * sin(i) / c

    Returns
    ----------
        kernel_arr : numpy.ndarray
                     numpy array of the kernel
    """
    l0 = config['l_0'].value

    # since data is uniformly sampled in log wavelength space, kernel has to be done similarly
    log_ax = np.logspace(np.log10(l0 * (1 - v_red.value)), np.log10(l0 * (1 + v_red.value)), sampling, endpoint=False)
    kernel_arr = ma.masked_invalid(ker(log_ax, l0, l0 * v_red))
    kernel_arr = ma.filled(kernel_arr, 0)
    # log_ax = np.delete(log_ax,0)
    kernel_arr = np.delete(kernel_arr, 0)

    # normalize kernel
    norm = np.sum(kernel_arr)
    kernel_arr = kernel_arr / norm
    return kernel_arr


def temp_visc(config: dict, r, r_in):
    """Define the temperature profile for the viscously heated disk

    Parameters
    ----------
    config : dict
        dictionary containing system parameters

    r : astropy.units.Quantity
        value of the radius at which the temperature is to be calculated

    r_in : astropy.units.Quantity
        inner truncation radius of the viscously heated disk

    Returns
    ----------
    t : astropy.units.Quantity
        temperature at the given radius
    """
    m = config['m']
    m_dot = config['m_dot']
    if r > 49 / 36 * r_in:
        t = ((3 * const.G * m * m_dot) * (1 - np.sqrt(r_in / r)) / (8 * np.pi * const.sigma_sb * r ** 3)) ** 0.25
    else:
        t = ((3 * const.G * m * m_dot) * (1 / 7) / (8 * np.pi * const.sigma_sb * (49 / 36 * r_in) ** 3)) ** 0.25
    return t.to(u.K)


def generate_temp_arr(config):  # ask if len r will be user defined
    """Calculate r_in, generate the temperature vs radius arrays, and bundle into a dictionary
    Parameters
    ----------
    config : dict
             dictionary containing system parameters

    Returns
    ---------
    dr : astropy.units.Quantity
        thickness of each annulus
    t_max : int
        maximum value of the temperature // 100 which is required to be called from BT-Settl database
    d : dict
        dictionary having radii of the annuli as keys and the temperature rounded to the nearest
        available BT-Settl spectra as the values
    r_in : astropy.units.Quantity
        inner truncation radius of the viscously heated disk
    r_sub : astropy.units.Quantity
        radius at which t_visc = 1400 K, formal boundary of the viscously heated disk
    """
    m_sun_yr = const.M_sun / (1 * u.yr).to(u.s)
    plot = config['plot']
    m = config['m']
    r_star = config['r_star']
    m_dot = config['m_dot']
    n_disk = config["n_disk"]
    print("Num of Annuli", n_disk)
    r_in = 7.186 * (r_star / (2 * const.R_sun)) ** (5 / 7) / (
            (m_dot / (1e-8 * m_sun_yr)) ** (2 / 7) * (m / (0.5 * const.M_sun)) ** (1 / 7)) * r_star
    r_in = r_in / 2.0  # correction factor taken 0.5, ref Long, Romanova, Lovelace 2005
    print(r_in)
    print(r_star)
    r_in = max([r_in, r_star])

    # estimate R_sub
    r_sub_approx = ((3 * const.G * m * m_dot) / (8 * np.pi * const.sigma_sb * (1400 * u.K) ** 4)) ** (1 / 3)
    print('Here R_in: ', r_in.to(u.m))
    print('Here R_sub: ', r_sub_approx.to(u.m))

    r_visc = np.linspace(r_in, r_sub_approx, n_disk)  # taking 10000 points, sufficient for high accretion case
    t_visc = np.zeros(n_disk) * u.K

    for i in range(len(t_visc)):
        t_visc[i] = temp_visc(config, r_visc[i], r_in)

    t_visc = ma.masked_less(t_visc, 1400 * u.K)
    r_visc = ma.masked_where(ma.getmask(t_visc), r_visc)

    t_visc = ma.compressed(t_visc)
    r_visc = ma.compressed(r_visc)
    d = {}
    print('Max temp:', max(t_visc), min(r_visc.to(u.m)))
    for i in range(len(r_visc)):
        t_int = int(np.round(t_visc[i].value / 100))
        if t_int < 71:
            d[r_visc[i].value] = int(np.round(t_visc[i].value / 100))
        elif 120 >= t_int > 70: # and t_int % 2 == 1:  # As per temperatures in BT-Settl data
            d[r_visc[i].value] = int(
                np.round(t_visc[i].value / 200)) * 2
        elif 120 < t_int: # and t_int % 5 != 0:
            d[r_visc[i].value] = int(
                np.round(t_visc[i].value / 500)) * 5
    temp_arr = []
    for i in r_visc:
        temp_arr.append(d[i.value])
    temp_arr = np.array(temp_arr)
    if len(t_visc) == 0:
        r_sub = r_in
        t_max = 14
        dr = None
    else:
        t_max = int(max(d.values()))
        r_sub = r_visc[-1]
        dr = r_visc[1] - r_visc[0]

    if plot:
        plt.plot(r_visc / const.R_sun, t_visc)
        plt.ylabel('Temperature [Kelvin]')
        plt.xlabel(r'Radius $R_{sun}$')
        plt.show()
    r_visc = r_visc.to(u.m)
    np.save("/Users/tusharkantidas/NIUS/testing/Contribution/r_visc.npy", r_visc.value)
    np.save("/Users/tusharkantidas/NIUS/testing/Contribution/temp_arr.npy", temp_arr)
    # print(temp_arr)
    return dr, t_max, d, r_in, r_sub


def generate_visc_flux(config, d: dict, t_max, dr):
    """Generate the flux contributed by the viscously heated disk
    Parameters
    ----------
    config : dict
        dictionary containing system parameters
    d : dict
        dictionary produced by generate_temp_arr, having the radii and their
        corresponding temperatures reduced to the integer values
    t_max : int
        maximum temperature of the viscously heated disk, reduced to nearest int BT-Settl value
    dr : astropy.units.Quantity
        thickness of each annulus

    Returns
    ----------
    wavelength : astropy.units.Quantity
        wavelength array in units of Angstrom
    obs_viscous_disk_flux : astropy.units.Quantity
        observed flux from the viscous disk, in units of erg / (cm^2 s A)
    """
    plot = config['plot']
    save = config['save']
    save_loc = config['save_loc']
    d_star = config['d_star']
    inclination = config['inclination']
    m = config['m']
    l_min = config['l_min']
    l_max = config['l_max']
    n_data = config['n_data']
    viscous_disk_flux = np.zeros(n_data) * (u.erg * u.m ** 2 / (u.cm ** 2 * u.s * u.AA))  # total number of data points

    for int_temp in range(14, t_max + 1):
        # to store total flux contribution from annuli of this temperature
        temp_flux = np.zeros(n_data) * (u.erg / (u.s * u.AA))

        radii = np.array([r for r, t in d.items() if t == int_temp])
        radii = radii * u.m
        if len(radii) == 0:
            print(f"completed for temperature of {int_temp * 100}\nNumber of rings included: {len(radii)}")
            continue

        radii = sorted(radii)

        if int_temp in range(14, 20):  # constrained by availability of BT-Settl models
            logg = 3.5
        else:
            logg = 1.5

        wavelength, flux = read_bt_settl(config, int_temp, logg)

        for r in radii:
            if inclination.value == 0:
                x_throw, y_final = logspace_reinterp(config, wavelength, flux)
            else:
                v_kep = np.sqrt(const.G * m / r)
                v_red = v_kep * np.sin(inclination) / const.c
                interp_samp, wavelength_new, flux_new = interpolate_conv(config, wavelength, flux, 100, v_red)
                kernel = generate_kernel(config, interp_samp, v_red)

                convolved_spectra = np.convolve(flux_new, kernel, mode="same")

                x_throw, y_final = logspace_reinterp(config, wavelength_new, convolved_spectra)
            temp_flux += y_final * np.pi * (2 * r * dr + dr ** 2)
        viscous_disk_flux += temp_flux
        print("completed for temperature of", int_temp, "\nnumber of rings included:", len(radii))
        if save:
            np.save(f'{save_loc}/{int_temp}_flux.npy', temp_flux.value)
            temp_flux_obs = (temp_flux * np.cos(inclination) / (np.pi * d_star ** 2)).to(u.erg / (u.cm ** 2 * u.s * u.AA))
            np.save(f'{save_loc}/{int_temp}_flux_obs.npy', temp_flux_obs.value)
    wavelength = np.logspace(np.log10(l_min.value), np.log10(l_max.value), n_data) * u.AA
    obs_viscous_disk_flux = viscous_disk_flux * np.cos(inclination) / (np.pi * d_star ** 2)
    obs_viscous_disk_flux = obs_viscous_disk_flux.to(u.erg / (u.cm ** 2 * u.s * u.AA))

    if save:
        np.save(f'{save_loc}/disk_component.npy', obs_viscous_disk_flux.value)
    if plot:
        plt.plot(wavelength, obs_viscous_disk_flux)
        plt.show()

    return wavelength, obs_viscous_disk_flux


# photosphere
def generate_photosphere_flux(config):
    """generate the flux from the stellar photosphere

    Parameters
    ----------
    config : dict
        dictionary containing system parameters

    Returns
    ----------
    obs_star_flux : astropy.units.Quantity
        flux array due to the stellar photosphere
    """
    l_min = config['l_min']
    l_max = config['l_max']
    log_g_star = config['log_g_star']  # ADD THIS TO CONFIG CODE
    t_star = config["t_star"]
    r_star = config["r_star"]
    d_star = config["d_star"]

    int_star_temp = int(np.round(t_star / (100 * u.K)))
    star_data = parse(f"{config['bt_settl_path']}/lte0{int_star_temp}-{log_g_star}-0.0a+0.0.BT-Settl.7.dat.xml")
    data_table = star_data.get_first_table().array
    trimmed_data2 = np.extract(data_table['WAVELENGTH'] > l_min.value - 10, data_table)
    trimmed_data2 = np.extract(trimmed_data2['WAVELENGTH'] < l_max.value + 10, trimmed_data2)
    x2 = trimmed_data2['WAVELENGTH'].astype(np.float64)
    y2 = trimmed_data2['FLUX'].astype(np.float64)

    wavelength, y_new_star = logspace_reinterp(config, x2, y2)
    obs_star_flux = y_new_star * (r_star.si / d_star.si) ** 2
    if config['plot']:
        plt.plot(wavelength, obs_star_flux)
        plt.title('Photospheric flux')
        plt.xlabel('wavelength')
        plt.ylabel('observed flux')
        plt.show()
    if config['save']:
        np.save(f"{config['save_loc']}/stellar_component.npy", obs_star_flux.value)

    return obs_star_flux


def cos_gamma_func(phi, theta, incl):
    """Calculate the dot product between line-of-sight unit vector and area unit vector"""
    cos_gamma = np.sin(theta) * np.cos(phi) * np.sin(incl) + np.cos(theta) * np.cos(incl)
    if cos_gamma >= 0:
        return cos_gamma * np.sin(theta)
    else:
        return 0


def magnetospheric_component(config, r_in):
    """Calculate the flux contribution due to the magnetospheric accretion columns

    Parameters
    ----------
    config : dict
             dictionary containing system parameters

    r_in : float
        Inner truncation radius of the viscously heated disk

    Returns
    ----------
    obs_mag_flux : astropy.units.Quantity
        flux due to the magnetospheric accretion of the YSO
    """

    t_slab = config['t_slab']
    tau = config['tau']
    n_e = config['n_e']
    inclination = config['inclination']
    m = config['m']
    m_dot = config['m_dot']
    r_star = config['r_star']
    d_star = config['r_star']
    save = config["save"]
    plot = config["plot"]
    l_min = config['l_min']
    l_max = config['l_max']
    n_data = config['n_data']

    if config['mag_comp'] == "hslab":
        h_flux = np.load(
            f"{config['h_grid_loc']}/{int(t_slab.value)}_tau_{np.round(tau, 1)}_len_{config['n_h']}/Flux_wav.npy")
        print(len(h_flux))
        h_minus_flux = np.load(
            f"{config['h_min_grid_loc']}/{int(t_slab.value)}_tau_{np.round(tau, 1)}_e{int(np.log10(n_e.value))}_"
            f"len_{config['n_h_minus']}/Flux_wav.npy")
        print(len(h_minus_flux))
        h_slab_flux = (h_flux + h_minus_flux) * u.erg / (u.cm ** 2 * u.s * u.AA)
        wav_slab = np.logspace(np.log10(1250), np.log10(5e4), 5000) * u.AA
        wav2 = np.logspace(np.log10(5e4), np.log10(1e6), 250) * u.AA
        # a blackbody SED is a good approximation for the Hydrogen slab beyond 50000 Angstroms

        bb_int = BlackBody(temperature=t_slab, scale=1 * u.erg / (u.cm ** 2 * u.s * u.AA * u.sr))
        bb_spec = bb_int(wav2)
        bb_spec = bb_spec * u.sr
        h_slab_flux = np.append(h_slab_flux, bb_spec[1:])
        wav_slab = np.append(wav_slab, wav2[1:])
        integrated_flux = trapezoid(h_slab_flux, wav_slab)
        l_mag = const.G * m * m_dot * (1 / r_star - 1 / r_in)
        area_shock = l_mag / integrated_flux

    elif config['mag_comp'] == 'blackbody':
        # calculate shock area for blackbody model
        l_mag = const.G * m * m_dot * (1 / r_star - 1 / r_in)
        area_shock = l_mag / (const.sigma_sb * (8000.0 * u.K) ** 4)
    else:
        raise ValueError("Only accepted magnetosphere models are \'blackbody\' and \'hslab\'")

    if (area_shock / (4 * np.pi * r_star ** 2)).si > 1:
        print(f"fraction of area {(area_shock / (4 * np.pi * r_star ** 2)).si}")
        if save:
            with open(f"{config['save_loc']}/{config['dir_name']}/details.txt", 'a+') as f:
                f.write("WARNING/nTotal area of black body required is more than stellar surface area")
    else:
        print(f"fraction of area {(area_shock / (4 * np.pi * r_star ** 2)).si}")
        if save:
            with open(f"{config['save_loc']}//details.txt", 'a+') as f:
                f.write(
                    f"ratio of area of shock to stellar surface area =  {(area_shock / (4 * np.pi * r_star ** 2)).si}")

    # corresponding theta max and min
    th_max = np.arcsin(np.sqrt(r_star / r_in))
    if area_shock / (4 * np.pi * r_star ** 2) + np.cos(th_max) > 1:
        print('Theta min not well defined')
        th_min = 0 * u.rad

        if save:
            with open(f"{config['save_loc']}/{config['dir_name']}/details.txt", 'a+') as f:
                f.write(f"Theta_min not well defined")
    else:
        th_min = np.arccos(area_shock / (4 * np.pi * r_star ** 2) + np.cos(
            th_max))  # ## min required due to high area of magnetosphere in some cases

    print(f"The values are \nth_min : {th_min.to(u.degree)}\nth_max : {th_max.to(u.degree)}")
    # integrate
    intg_val1, err = dblquad(cos_gamma_func, th_min.value, th_max.value, 0, 2 * np.pi, args=(inclination.value,))
    intg_val2, err = dblquad(cos_gamma_func, np.pi - th_max.value, np.pi - th_min.value, 0, 2 * np.pi,
                             args=(inclination.value,))
    intg_val = intg_val1 + intg_val2
    print(f"integral val : {intg_val}, error : {err}")

    if save:
        with open(f"{config['save_loc']}/details.txt", 'a+') as f:
            f.write(f"integral val : {intg_val}, error : {err}")
            f.write(f"The values are \nth_min : {th_min.to(u.degree)}\nth_max : {th_max.to(u.degree)}")
    # scale unit to derive flux in wavelength
    if config['mag_comp'] == "hslab":
        # read Hydrogen slab SED
        # T_slab  = 11000 * u.K
        # n_e = 1e15 * (u.cm)**-3
        # tau = 0.2

        # retrieve stored H_slab SEDs
        h_flux = np.load(
            f"{config['h_grid_loc']}/{int(t_slab.value)}_tau_{np.round(tau, 1)}_len_{config['n_h']}/Flux_wav.npy")
        h_minus_flux = np.load(
            f"{config['h_min_grid_loc']}/{int(t_slab.value)}_tau_{np.round(tau, 1)}_e{int(np.log10(n_e.value))}_len"
            f"_{config['n_h_minus']}/Flux_wav.npy")
        h_slab_flux = (h_flux + h_minus_flux)

        wav_slab = np.logspace(np.log10(1250), np.log10(5e4), 5000)  # this is defined in the H-slab databases

        # interpolate to required wavelength axis

        func_slab = interp1d(wav_slab, h_slab_flux)
        print(l_min, l_max, n_data)
        wav_ax = np.logspace(np.log10(l_min.value), np.log10(l_max.value), n_data)
        h_slab_flux_interp = func_slab(wav_ax)
        h_slab_flux_interp = h_slab_flux_interp * u.erg / (u.cm ** 2 * u.s * u.AA)
        obs_mag_flux = h_slab_flux_interp * (r_star / d_star) ** 2 * intg_val
    elif config['mag_comp'] == "blackbody":
        # calculate blackbody spectrum
        scale_unit = u.erg / (u.cm ** 2 * u.s * u.AA * u.sr)
        t_slab = config["t_slab"]  # here taking the blackbody temp == slab temp # may be diff also
        bb = BlackBody(t_slab, scale=1 * scale_unit)
        l_bb = np.linspace(np.log10(l_min.value), np.log10(l_max.value), n_data) * u.AA
        flux_bb = bb(l_bb)

        obs_bb_flux = (flux_bb * (r_star / d_star) ** 2 * intg_val)
        obs_mag_flux = obs_bb_flux.to(1 * (u.erg) / (u.cm ** 2 * u.s * u.AA * u.sr)) * u.sr
    else:
        raise ValueError("Only accepted magnetosphere models are \'blackbody\' and \'hslab\'")

    if plot:
        wav_ax = np.logspace(np.log10(l_min.value), np.log10(l_max.value), n_data)
        plt.plot(wav_ax, obs_mag_flux)
        plt.show()
    if save:
        np.save(f"{config['save_loc']}/mag_component.npy", obs_mag_flux.value)
    return obs_mag_flux


def t_eff_dust(r, config):
    """Define the temperature profile in the passively heated dusty disk

    Parameters
    ----------
    r : astropy.units.Quantity
        radius at which temperature is to be evaluated

    config : dict
        dictionary containing system parameters

    Returns
    ------------
    t : astropy.units.Quantity
        temperature value at r

    """
    r_star = config['r_star']
    t_star = config['t_star']
    m = config['m']
    t_0 = config['t_0']
    alpha_0 = 0.003 * (r_star / (1.6 * const.R_sun)) / (r / const.au) + 0.05 * (t_star.value / 3400) ** (4 / 7) * (
            r_star / (1.6 * const.R_sun)) ** (2 / 7) * (r / const.au) ** (2 / 7) / (m / (0.3 * const.M_sun)) ** (
                      4 / 7)
    t = (alpha_0 / 2) ** 0.25 * (r_star / r) ** 0.5 * t_0
    return t


def generate_dusty_disk_flux(config, r_in, r_sub):
    """Generates the SED of the dusty disk component, as worked out by Liu et al. 2022, assuming each annulus to emit in
    the form a blackbody, having a temperature profile that is either radiation dominated (transition layer present),
    or has region of viscously heated dust (transition layer absent).

    Parameters
    ----------
    config : dict
        dictionary containing system parameters

    r_in : astropy.units.Quantity
        inner truncation radius of the viscously heated disk

    r_sub : astropy.units.Quantity
        radius at which t_visc = 1400 K, i.e. where dust sublimation begins, sets a formal
        outer boundary of the viscously heated disk

    Returns
    ----------
    obs_dust_flux : astropy.units.Quantity
        array of observed flux from the dusty disk component
    """
    plot = config['plot']
    save = config['save']
    save_loc = config['save_loc']
    l_min = config['l_min']
    l_max = config['l_max']
    n_data = config['n_data']
    d_star = config['d_star']
    inclination = config['inclination']

    r_dust = np.logspace(np.log10(r_sub.si.value), np.log10(270 * const.au.value), 5000) * u.m
    t_dust_init = t_eff_dust(r_dust, config)

    if t_eff_dust(r_sub, config) > 1400 * u.K:
        t_dust = ma.masked_greater(t_dust_init.value, 1400)
        t_dust = t_dust.filled(1400)
        t_dust = t_dust * u.K
    else:
        t_visc_dust = np.zeros(len(r_dust)) * u.K
        for i in range(len(r_dust)):
            t_visc_dust[i] = temp_visc(config, r_dust[i], r_in)  # has to be done using for loop to avoid ValueError
        print(t_visc_dust)
        t_dust = np.maximum(t_dust_init, t_visc_dust)

    if plot:
        plt.plot(r_dust / const.au, t_dust.value)
        plt.show()
    print(t_dust)
    dust_flux = np.zeros(n_data) * u.erg / (u.cm * u.cm * u.s * u.AA * u.sr) * (u.m * u.m)
    wavelength = np.logspace(np.log10(l_min.value), np.log10(l_max.value), n_data) * u.AA
    for i in range(len(r_dust) - 1):
        scale_unit = u.erg / (u.cm ** 2 * u.s * u.AA * u.sr)
        dust_bb = BlackBody(t_dust[i], scale=1 * scale_unit)
        dust_bb_flux = dust_bb(wavelength)
        dust_flux += dust_bb_flux * np.pi * (r_dust[i + 1] ** 2 - r_dust[i] ** 2)
        if i % 100 == 0:  # print progress after every 100 annuli
            print(r_dust[i])
            print(f"done temperature {i}")

    obs_dust_flux = dust_flux * np.cos(inclination) / (np.pi * d_star.si ** 2) * u.sr
    if save:
        np.save(f'{save_loc}/dust_component.npy', obs_dust_flux.value)
    if plot:
        plt.plot(wavelength, obs_dust_flux)
        plt.show()
    return obs_dust_flux


def dust_extinction_flux(config, wavelength, obs_viscous_disk_flux, obs_star_flux, obs_mag_flux,
                         obs_dust_flux):  # , obs_viscous_disk_flux, obs_dust_flux, obs_mag_flux):
    """Redden the spectra with the Milky Way extinction curves. Ref. Gordon et. al 2021, Fitzpatrick et. al 2019
    Parameters :
            wavelength: array_like
                        wavelength array, typically in a logspace
            details: dict
                     dictionary containing the input parameters of the model
            obs_star_flux: array_like
            obs_viscous_disk_flux: array_like
            obs_mag_flux: array_like
            obs_dust_flux: array_like
    returns:
            total_flux: array_like
                        spectra reddened as per the given parameters of a_v and r_v in details
    """
    r_v = config['rv']
    a_v = config['av']
    save = config['save']
    plot = config['plot']
    save_loc = config['save_loc']

    wav1 = np.extract(wavelength < 33e3 * u.AA, wavelength)
    wav2 = np.extract(wavelength >= 33e3 * u.AA, wavelength)

    print("Disk", obs_viscous_disk_flux)
    print("Dust", obs_dust_flux)
    print("Mag", obs_mag_flux)
    print("Photo", obs_star_flux)
    total = obs_star_flux + obs_viscous_disk_flux + obs_dust_flux + obs_mag_flux

    total_flux_1 = total[:len(wav1)]
    total_flux_2 = total[len(wav1):]
    ext1 = F19(Rv=r_v)
    ext2 = G21_MWAvg()  # Gordon et al. 2021, milky way average curve

    exting_spec_1 = total_flux_1 * ext1.extinguish(wav1, Av=a_v)
    exting_spec_2 = total_flux_2 * ext2.extinguish(wav2, Av=a_v)

    total_flux = np.append(exting_spec_1, exting_spec_2)
    if save:
        np.save(f'{save_loc}/extinguished_spectra.npy', total_flux.value)
    if plot:
        plt.plot(wavelength, total_flux.value, label='extinguished spectrum')
        plt.xlabel(r'Wavelength [$\AA$]')
        plt.ylabel(r'Flux [erg / cm^2 s A]')
        plt.legend()
        plt.show()
    return total_flux


def parse_args(raw_args=None):
    """Take config file location from the command line"""
    parser = argparse.ArgumentParser(description="YSO Spectrum generator")
    parser.add_argument('ConfigfileLocation', type=str,
                        help="Path to config file")
    args = parser.parse_args(raw_args)
    return args


def main(raw_args=None):
    """Calls the base functions sequentially, and finally generates extinguished spectra for the given system"""
    args = parse_args(raw_args)
    dict_config = config_read(args.ConfigfileLocation)
    # dict_config = config_read("/Users/tusharkantidas/NIUS/refactoring/config_file.cfg")
    dr, t_max, d, r_in, r_sub = generate_temp_arr(dict_config)
    wavelength, obs_viscous_disk_flux = generate_visc_flux(dict_config, d, t_max, dr)
    print('Viscous disk done')
    obs_mag_flux = magnetospheric_component(dict_config, r_in)
    print("Magnetic component done")
    obs_dust_flux = generate_dusty_disk_flux(dict_config, r_in, r_sub)
    print("Dust component done")
    obs_star_flux = generate_photosphere_flux(dict_config)
    print("Photospheric component done")
    total_flux = dust_extinction_flux(dict_config, wavelength, obs_viscous_disk_flux, obs_star_flux, obs_mag_flux,
                                      obs_dust_flux)

    # sys.exit(0)
    if dict_config['plot']:
        plt.plot(wavelength, np.log10(total_flux.value))
        plt.xlabel("Wavelength [Angstrom]")
        plt.ylabel("Flux [erg / cm^2 s A]")
        plt.show()
    print("done")


if __name__ == "__main__":
    main(raw_args=None)

