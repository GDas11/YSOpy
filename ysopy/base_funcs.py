import astropy.units
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
from scipy.integrate import dblquad
from scipy.integrate import trapezoid
from scipy.interpolate import interp1d
import astropy.constants as const
import astropy.units as u
from astropy.io.votable import parse
from astropy.modeling.physical_models import BlackBody
from dust_extinction.parameter_averages import F19
from dust_extinction.averages import G21_MWAvg
from configparser import ConfigParser
import argparse
import time


def config_read(path):
    """Read data from config file and cast to expected data types

    Parameters
    ----------
    path : str
        path to the config file

    Returns
    ----------
    dict_config : dict
        dictionary containing the parameters of the system in the expected units
    """
    config = ConfigParser()
    config.read(path)
    config_data = config['Parameters']
    dict_config = dict(config_data)

    # convert to the required astropy units
    dict_config["l_min"] = float(dict_config["l_min"]) * u.AA
    dict_config["l_max"] = float(dict_config["l_max"]) * u.AA
    dict_config["n_data"] = int(dict_config["n_data"])

    dict_config["b"] = float(dict_config["b"]) * u.kilogauss
    dict_config["m"] = float(dict_config["m"]) * const.M_sun
    dict_config["m_dot"] = float(dict_config["m_dot"]) * const.M_sun / (1 * u.year).to(u.s)
    dict_config["r_star"] = float(dict_config["r_star"]) * const.R_sun
    dict_config["inclination"] = float(dict_config["inclination"]) * u.degree
    dict_config["n_disk"] = int(dict_config["n_disk"])
    dict_config["n_dust_disk"] = int(dict_config["n_dust_disk"])
    dict_config["d_star"] = float(dict_config["d_star"]) * const.pc
    dict_config["t_star"] = float(dict_config["t_star"]) * u.K
    dict_config["log_g_star"] = float(dict_config["log_g_star"])
    dict_config["t_0"] = float(dict_config["t_0"]) * u.K
    dict_config["av"] = float(dict_config["av"])
    dict_config["rv"] = float(dict_config["rv"])
    dict_config["l_0"] = float(dict_config["l_0"]) * u.AA
    dict_config["t_slab"] = float(dict_config["t_slab"]) * u.K
    dict_config["n_e"] = float(dict_config["n_e"]) * u.cm ** (-3)
    dict_config["tau"] = float(dict_config["tau"])
    dict_config["n_h"] = int(dict_config["n_h"])
    dict_config["l_l_slab"] = float(dict_config["l_l_slab"]) * u.AA
    dict_config["n_h_minus"] = int(dict_config["n_h_minus"])
    
    for param in ["save", "save_each", "plot", "save_grid_data", "verbose"]:
        if dict_config[param] == "True":
            dict_config[param] = True
        elif dict_config[param] == "False":
            dict_config[param] = False

    if dict_config['save']:
        with open(f"{dict_config['save_loc']}/details.txt", 'a+') as f:
            f.write(str(dict_config))

    return dict_config
#trial comment

def read_bt_settl_npy(config, temperature: int, logg: float, r_in=None):
    """read the stored BT-Settl model spectra from .npy format, for the supplied temperature and logg values

    Parameters
    ----------
    config : dict
        dictionary containing system parameters

    temperature : int
        t // 100, where t is the t_eff of the model in Kelvin

    logg : float
        log of surface gravity of the atmosphere model

    r_in : astropy.units.Quantity or None
        if supplied, calculates the padding required

    Returns
    ----------
    trimmed_wave : astropy.units.Quantity
        array having the wavelength axis of the read BT-Settl data, in units of Angstrom,
        trimmed to the region of interest, a padding of 10 A is left to avoid errors in interpolation

    trimmed_flux : astropy.units.Quantity
        array of flux values, in units of erg / (cm^2 s A), trimmed to region of interest
    """

    loc = config['bt_settl_path']
    m = config['m']
    inclination = config['inclination']
    l_min = config['l_min']
    l_max = config['l_max']
    n_data = config['n_data']

    if temperature >= 100:
        address = f"{loc}/lte{temperature}-{logg}-0.0a+0.0.BT-Settl.7.dat.npy"
    elif (temperature > 25) and temperature < 100:
        address = f"{loc}/lte0{temperature}-{logg}-0.0a+0.0.BT-Settl.7.dat.npy"
    elif (temperature >= 20) and temperature < 100:
        address = f"{loc}/lte0{temperature}-{logg}-0.0.BT-Settl.7.dat.npy"
    else:
        address = f"{loc}/lte0{temperature}-{logg}-0.0.BT-Settl.7.dat.npy"
    
    data = np.load(address)

    # trim data to region of interest
    # extra region left for re-interpolation
    l_pad = 20
    if r_in is not None:
        v_max = np.sqrt(const.G.value * m.value / r_in.value) * np.sin(inclination) / const.c.value
        l_pad = l_max.value * v_max
        # print(l_pad)

    cond1 = l_min.value - 1.5 * l_pad < data[0]
    cond2 = data[0] < l_max.value + 1.5 * l_pad
    trimmed_ids = np.logical_and(cond1,cond2)
    trimmed_wave = data[0][trimmed_ids].astype(np.float64) * u.AA
    trimmed_flux = data[1][trimmed_ids].astype(np.float64) * (u.erg / (u.cm * u.cm * u.s * u.AA))

    # for faulty data make a linear re-interpolation
    if 20 <= temperature <= 25:
        x, y = unif_reinterpolate(config, trimmed_wave, trimmed_flux, l_pad)
        f = interp1d(x, y)
        trimmed_wave = np.linspace(l_min.value - l_pad, l_max.value + l_pad, n_data, endpoint=True) * u.AA
        trimmed_flux = f(trimmed_wave) * u.erg / (u.cm * u.cm * u.s * u.AA)

    return trimmed_wave, trimmed_flux


def read_bt_settl(config, temperature: int, logg: float, r_in=None):
    """read the stored BT-Settl model spectra from VOtable format, for the supplied temperature and logg values

    Parameters
    ----------
    config : dict
        dictionary containing system parameters

    temperature : int
        t // 100, where t is the t_eff of the model in Kelvin

    logg : float
        log of surface gravity of the atmosphere model

    r_in : astropy.units.Quantity or None
        if supplied, calculates the padding required

    Returns
    ----------
    trimmed_wave : astropy.units.Quantity
        array having the wavelength axis of the read BT-Settl data, in units of Angstrom,
        trimmed to the region of interest, a padding of 10 A is left to avoid errors in interpolation

    trimmed_flux : astropy.units.Quantity
        array of flux values, in units of erg / (cm^2 s A), trimmed to region of interest
    """

    loc = config['bt_settl_path']
    m = config['m']
    inclination = config['inclination']
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
    l_pad = 20
    if r_in is not None:
        v_max = np.sqrt(const.G.value * m.value / r_in.value) * np.sin(inclination) / const.c.value
        l_pad = l_max.value * v_max
        # print(l_pad)

    trimmed_data = np.extract(data['WAVELENGTH'] > l_min.value - 1.5 * l_pad, data)
    trimmed_data = np.extract(trimmed_data['WAVELENGTH'] < l_max.value + 1.5 * l_pad, trimmed_data)
    trimmed_wave = trimmed_data['WAVELENGTH'].astype(np.float64) * u.AA
    trimmed_flux = trimmed_data['FLUX'].astype(np.float64) * (u.erg / (u.cm * u.cm * u.s * u.AA))

    # for faulty data make a linear re-interpolation
    if 20 <= temperature <= 25:
        x, y = unif_reinterpolate(config, trimmed_wave, trimmed_flux, l_pad)
        f = interp1d(x, y)
        trimmed_wave = np.linspace(l_min.value - l_pad, l_max.value + l_pad, n_data, endpoint=True) * u.AA
        trimmed_flux = f(trimmed_wave) * u.erg / (u.cm * u.cm * u.s * u.AA)
        # trimmed_data = ma.array([x, y], dtype=[('WAVELENGTH', 'float'), ('FLUX', 'float')])
    return trimmed_wave, trimmed_flux


def unif_reinterpolate(config, x, y, l_pad):
    """interpolate the datasets having very low sampling
    Parameters
    ----------
    config : dict
             dictionary containing system parameters
    x : astropy.units.Quantity
        array of wavelength values, in units of Angstrom
    y : astropy.units.Quantity
        array of flux values, in units of erg / (cm^2 s A)
    l_pad : float
        padding in wavelength axis (in Angstrom)
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
    wav = np.linspace(l_min.value - 1.2 * l_pad, l_max.value + 1.2 * l_pad, n_data, endpoint=True) * u.AA
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

    flux : astropy.units.Quantity or numpy.ndarray
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
    n_disk = config['n_disk']
    b = config["b"]
    r_in = 7.186 * (b / (1 * u.kG)) ** (4 / 7) * (r_star / (2 * const.R_sun)) ** (5 / 7) / (
            (m_dot / (1e-8 * m_sun_yr)) ** (2 / 7) * (m / (0.5 * const.M_sun)) ** (1 / 7)) * r_star
    r_in = r_in / 2.0  # correction factor taken 0.5, ref Long, Romanova, Lovelace 2005
    r_in = max([r_in, r_star])

    # estimate R_sub
    r_sub_approx = ((3 * const.G * m * m_dot) / (8 * np.pi * const.sigma_sb * (1400 * u.K) ** 4)) ** (1 / 3)

    r_visc = np.linspace(r_in, r_sub_approx, n_disk)
    t_visc = np.zeros(n_disk) * u.K

    for i in range(len(t_visc)):
        t_visc[i] = temp_visc(config, r_visc[i], r_in)


    # truncate at T < 1400 K, i.e. sublimation temperature of the dust
    t_visc = ma.masked_less(t_visc, 1400 * u.K)
    r_visc = ma.masked_where(ma.getmask(t_visc), r_visc)
    t_visc = ma.compressed(t_visc)
    r_visc = ma.compressed(r_visc)
    d = {}
    for i in range(len(r_visc)):
        t_int = int(np.round(t_visc[i].value / 100))
        if t_int < 71:
            d[r_visc[i].value] = int(np.round(t_visc[i].value / 100))
        elif 120 >= t_int > 70 and t_int % 2 == 1:  # As per temperatures in BT-Settl data
            d[r_visc[i].value] = int(np.round(t_visc[i].value / 200)) * 2
        elif 120 < t_int:  # and t_int % 5 != 0:
            d[r_visc[i].value] = int(np.round(t_visc[i].value / 500)) * 5
    if len(t_visc) == 0:
        r_sub = r_in
        t_max = 14
        dr = None
    else:
        t_max = int(max(d.values()))
        r_sub = r_visc[-1]
        dr = r_visc[1] - r_visc[0]
    if config['save']:
        np.save("radius_arr.npy", r_visc.si.value)
        np.save("temp_arr.npy", t_visc.si.value)

    if plot:
        plt.plot(r_visc / const.R_sun, t_visc)
        plt.ylabel('Temperature [Kelvin]')
        plt.xlabel(r'Radius $R_{sun}$')
        plt.show()
    return dr, t_max, d, r_in, r_sub


def generate_temp_arr_planet(config, mass_p, dist_p, d):
    # mass and distance of planet fix such that it is within the viscous disk
    r_visc = np.array([radius for radius, t in d.items()]) * u.m
    t_visc = np.array([t for radius, t in d.items()]) * u.Kelvin
    mass_plnt = mass_p * u.jupiterMass
    dist_plnt = dist_p * u.AU
    m = config["m"]
    # Distance of star to L1 from star
    print("********************************************************************************")
    print(mass_plnt.unit)
    print(m.unit)
    low_plnt_lim = dist_plnt * (1 - np.sqrt(mass_plnt / (3 * m)))
    up_plnt_lim = dist_plnt * (1 + np.sqrt(mass_plnt / (3 * m)))
    print("Check this ******************")
    print('Position of L1: ', low_plnt_lim.to(u.AU))
    print('Position of L2: ', up_plnt_lim.to(u.AU))
    print('Last element of radius array: ', r_visc.to(u.AU)[-1])
    print("*********************")
    print(f"planet's influence: {low_plnt_lim} to {up_plnt_lim}\n")
    r_new = []
    for r in r_visc:
        if r > low_plnt_lim:
            if r < up_plnt_lim:
                r_new.append(r.to(u.AU).value)
    r_new = np.array(r_new) * u.AU

    terms = np.where(low_plnt_lim < r_visc)
    terms2 = np.where(up_plnt_lim > r_visc)
    terms_act = []

    for i in terms[0]:
        for j in terms2[0]:
            if i == j:
                terms_act.append(i)
    # removing this radius from r_visc
    for r in r_visc:
        if r in r_new:
            r_visc = np.delete(r_visc, np.where(r_visc == r))
    # print(r_visc.to(u.AU))
    # print(temp_arr, len(temp_arr))
    # temp_arr = list(temp_arr.value)
    t_visc = list(t_visc.value)
    for i in terms_act:
        # temp_arr.remove(temp_arr[terms_act[0]])
        t_visc.remove(t_visc[terms_act[0]])
    # temp_arr = np.array(temp_arr) * u.Kelvin
    t_visc = np.array(t_visc) * u.Kelvin
    # print(temp_arr, len(temp_arr))
    # plt.plot(r_visc.to(u.AU), t_visc)
    # plt.show()
    d_new = {}
    for i in range(len(r_visc)):
        t_int = int(np.round(t_visc[i].value))
        if t_int < 71:
            d_new[r_visc[i].value] = int(np.round(t_visc[i].value))
        elif 120 >= t_int > 70 and t_int % 2 == 1:  # As per temperatures in BT-Settl data
            d_new[r_visc[i].value] = int(
                np.round(t_visc[i].value)) * 2
        elif 120 < t_int:  # and t_int % 5 != 0:
            d_new[r_visc[i].value] = int(
                np.round(t_visc[i].value)) * 5
    return r_visc, t_visc, d_new  # , temp_arr


def generate_visc_flux(config, d: dict, t_max, dr, r_in=None):
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
    dr : astropy.units.Quantity or None
        thickness of each annulus
    r_in : astropy.units.Quantity
        inner truncation radius, needed to estimate padding

    Returns
    ----------
    wavelength : astropy.units.Quantity
        wavelength array in units of Angstrom
    obs_viscous_disk_flux : astropy.units.Quantity
        observed flux from the viscous disk, in units of erg / (cm^2 s A)
    """
    plot = config['plot']
    save = config['save']
    save_each = config['save_each']
    save_loc = config['save_loc']
    d_star = config['d_star']
    inclination = config['inclination']
    m = config['m']
    l_min = config['l_min']
    l_max = config['l_max']
    n_data = config['n_data']
    viscous_disk_flux = np.zeros(n_data) * (u.erg * u.m ** 2 / (u.cm ** 2 * u.s * u.AA))  # total number of data points

    for int_temp in range(t_max, 13, -1):
        # to store total flux contribution from annuli of this temperature
        temp_flux = np.zeros(n_data) * (u.erg / (u.s * u.AA))

        radii = np.array([r for r, t in d.items() if t == int_temp])
        radii = radii * u.m
        radii = sorted(radii, reverse=True)

        if int_temp in range(14, 20):  # constrained by availability of BT-Settl models
            logg = 3.5
        else:
            logg = 1.5

        if len(radii) != 0:
            wavelength, flux = read_bt_settl_npy(config, int_temp, logg, r_in)

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
                if save_each:
                    np.save(f'{save_loc}/radius_{r}_flux.npy', y_final.value)
            temp_flux += y_final * np.pi * (2 * r * dr + dr ** 2)
        viscous_disk_flux += temp_flux
        if config['verbose']:
            print("completed for temperature of", int_temp, "\nnumber of rings included:", len(radii))
        if save:
            np.save(f'{save_loc}/{int_temp}_flux.npy', temp_flux.value)
    wavelength = np.logspace(np.log10(l_min.value), np.log10(l_max.value), n_data) * u.AA
    obs_viscous_disk_flux = viscous_disk_flux * np.cos(inclination) / (np.pi * d_star ** 2)
    obs_viscous_disk_flux = obs_viscous_disk_flux.to(u.erg / (u.cm ** 2 * u.s * u.AA))
    if save:
        np.save(f'{save_loc}/disk_component.npy', obs_viscous_disk_flux.value)
    if plot:
        plt.plot(wavelength, obs_viscous_disk_flux)
        plt.xlabel("Wavelength in Angstrom ----->")
        plt.ylabel("Flux [erg / ($cm^{2}$ s angstrom)] ----->")
        plt.title("Viscous Disk SED")
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

    #change to npy
    #star_data = parse(f"{config['bt_settl_path']}/lte0{int_star_temp}-{log_g_star}-0.0a+0.0.BT-Settl.7.dat.xml")
    address = f"{config['bt_settl_path']}/lte0{int_star_temp}-{log_g_star}-0.0a+0.0.BT-Settl.7.dat.npy"
    data = np.load(address)
    cond1 = l_min.value - 10 < data[0]
    cond2 = data[0] < l_max.value + 10
    trimmed_ids = np.logical_and(cond1,cond2)
    x2 = data[0][trimmed_ids].astype(np.float64)
    y2 = data[1][trimmed_ids].astype(np.float64)

    #data_table = star_data.get_first_table().array
    #trimmed_data2 = np.extract(data_table['WAVELENGTH'] > l_min.value - 10, data_table)
    #trimmed_data2 = np.extract(trimmed_data2['WAVELENGTH'] < l_max.value + 10, trimmed_data2)
    #x2 = trimmed_data2['WAVELENGTH'].astype(np.float64)
    #y2 = trimmed_data2['FLUX'].astype(np.float64)

    wavelength, y_new_star = logspace_reinterp(config, x2, y2)
    obs_star_flux = y_new_star * (r_star.si / d_star.si) ** 2
    if config['plot']:
        plt.plot(wavelength, obs_star_flux)
        plt.xlabel("Wavelength in $\AA$ ----->")
        plt.ylabel("Flux [erg / ($cm^{2}$ s $\AA$)] ----->")
        plt.title("Stellar Photosphere SED")
        plt.show()
    if config['save']:
        np.save(f"{config['save_loc']}/stellar_component.npy", obs_star_flux.value)

    return obs_star_flux


def cos_gamma_func(phi, theta, incl):
    """Calculate the dot product between line-of-sight unit vector and area unit vector
    This is required only for a blackbody magnetosphere,as done by Liu et al.
    """
    cos_gamma = np.sin(theta) * np.cos(phi) * np.sin(incl) + np.cos(theta) * np.cos(incl)
    if cos_gamma < 0:
        return 0
    else:
        return cos_gamma * np.sin(theta)

def magnetospheric_component_calculate(config, r_in):
    """Calculte the H-slab component on the fly
    """



def magnetospheric_component(config, r_in):
    """Retireve the flux for the H-slab from the stored grid (if H-slab is enabled),
    or calculate the magnetospheric component as a blackbody

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
    d_star = config['d_star']
    l_min = config['l_min']
    l_max = config['l_max']
    n_data = config['n_data']
    save = config["save"]
    plot = config["plot"]

    # retrieve the stored H-slab flux from the saved location
    if config['mag_comp'] == "hslab":
        h_flux = np.load(f"{config['h_grid_path']}/temp_{int(t_slab.value)}_tau_{np.round(tau, 1)}_len_{config['n_h']}/Flux_wav.npy")
        h_minus_flux = np.load(f"{config['h_min_grid_path']}/temp_{int(t_slab.value)}_tau_{np.round(tau, 1)}_e{int(np.log10(n_e.value))}_len_{config['n_h_minus']}/Flux_wav.npy")
        h_slab_flux = (h_flux + h_minus_flux) * u.erg / (u.cm ** 2 * u.s * u.AA)

        # two wavelength regimes are used
        wav_slab = np.logspace(np.log10(1250), np.log10(5e4), 5000) * u.AA
        wav2 = np.logspace(np.log10(5e4), np.log10(1e6), 250) * u.AA

        # a blackbody SED is a good approximation for the Hydrogen slab beyond 50000 Angstroms
        bb_int = BlackBody(temperature=t_slab, scale=1 * u.erg / (u.cm ** 2 * u.s * u.AA * u.sr))
        bb_spec = bb_int(wav2)
        bb_spec = bb_spec * u.sr
        h_slab_flux = np.append(h_slab_flux, bb_spec[1:])
        wav_slab = np.append(wav_slab, wav2[1:])
        # calculate the total luminosity to get the area of the shock
        # this is the approach taken by Liu et al., however  for low accretion rates, this yields a very large covering fraction
        integrated_flux = trapezoid(h_slab_flux, wav_slab)
        l_mag = const.G * m * m_dot * (1 / r_star - 1 / r_in)
        area_shock = l_mag / integrated_flux
        if config['verbose']:
            print(f"shock area : {area_shock.si}")

    elif config['mag_comp'] == 'blackbody':
        l_mag = const.G * m * m_dot * (1 / r_star - 1 / r_in)
        area_shock = l_mag / (const.sigma_sb * t_slab ** 4)  # allowing the effective temperature of the shock to vary
    else:
        raise ValueError("Only accepted magnetosphere models are \'blackbody\' and \'hslab\'")

    # shock area fraction warning
    fraction = (area_shock / (4 * np.pi * r_star ** 2)).si
    if config['verbose']:
        print(f"fraction of area {fraction}")
    if fraction > 1:
        if save:
            with open(f"{config['save_loc']}/details.txt", 'a+') as f:
                f.write("WARNING/nTotal area of shock required is more than stellar surface area")
    else:
        if save:
            with open(f"{config['save_loc']}//details.txt", 'a+') as f:
                f.write(f"ratio of area of shock to stellar surface area =  {fraction}")

    # getting the geometry of the shocked region, if it is less than the stellar photosphere area
    # calculate corresponding theta max and min
    th_max = np.arcsin(np.sqrt(r_star / r_in))
    if area_shock / (4 * np.pi * r_star ** 2) + np.cos(th_max) > 1:
        if config['verbose']:
            print('Theta min not well defined')
        th_min = 0 * u.rad

        if save:
            with open(f"{config['save_loc']}/details.txt", 'a+') as f:
                f.write(f"Theta_min not well defined")
    else:
        # min required due to high area of magnetosphere in some cases
        th_min = np.arccos(area_shock / (4 * np.pi * r_star ** 2) + np.cos(th_max))

    if config['verbose']:
        print(f"The values are \nth_min : {th_min.to(u.degree)}\nth_max : {th_max.to(u.degree)}")
    # integrate
    intg_val1, err = dblquad(cos_gamma_func, th_min.value, th_max.value, 0, 2 * np.pi, args=(inclination.value,))
    intg_val2, err = dblquad(cos_gamma_func, np.pi - th_max.value, np.pi - th_min.value, 0, 2 * np.pi, args=(inclination.value,))
    intg_val = intg_val1 + intg_val2
    if config['verbose']:
        print(f"integral val : {intg_val}, error : {err}")

    if save:
        with open(f"{config['save_loc']}/details.txt", 'a+') as f:
            f.write(f"integral val : {intg_val}, error : {err}")
            f.write(f"The values are \nth_min : {th_min.to(u.degree)}\nth_max : {th_max.to(u.degree)}")

    if config['mag_comp'] == "hslab":
        # retrieve stored H_slab SEDs
        h_flux = np.load(
            f"{config['h_grid_path']}/temp_{int(t_slab.value)}_tau_{np.round(tau, 1)}_len_{config['n_h']}/Flux_wav.npy")
        h_minus_flux = np.load(
            f"{config['h_min_grid_path']}/temp_{int(t_slab.value)}_tau_{np.round(tau, 1)}_e{int(np.log10(n_e.value))}_len"
            f"_{config['n_h_minus']}/Flux_wav.npy")
        h_slab_flux = (h_flux + h_minus_flux)

        wav_slab = np.logspace(np.log10(1250), np.log10(5e4), config['n_h'])  # this is defined in the H-slab databases

        # interpolate to required wavelength axis,same as the other components
        func_slab = interp1d(wav_slab, h_slab_flux)
        wav_ax = np.logspace(np.log10(l_min.value), np.log10(l_max.value), n_data)
        h_slab_flux_interp = func_slab(wav_ax)
        h_slab_flux_interp = h_slab_flux_interp * u.erg / (u.cm ** 2 * u.s * u.AA)
        obs_mag_flux = h_slab_flux_interp * (r_star / d_star) ** 2 * intg_val

    elif config['mag_comp'] == "blackbody":
        # calculate blackbody spectrum, defined at a temperature t_slab
        scale_unit = u.erg / (u.cm ** 2 * u.s * u.AA * u.sr)
        bb = BlackBody(t_slab, scale=1 * scale_unit)
        wavelength = np.logspace(np.log10(l_min.value), np.log10(l_max.value), n_data) * u.AA
        flux_bb = bb(wavelength)

        obs_bb_flux = flux_bb * (r_star / d_star) ** 2 * intg_val
        obs_mag_flux = obs_bb_flux.to(1 * u.erg / (u.cm ** 2 * u.s * u.AA * u.sr)) * u.sr
    else:
        raise ValueError("Only accepted magnetosphere models are \'blackbody\' and \'hslab\'")

    if plot:
        wav_ax = np.logspace(np.log10(l_min.value), np.log10(l_max.value), n_data)
        plt.plot(wav_ax, obs_mag_flux)
        plt.xlabel("Wavelength in $\AA$ ----->")
        plt.ylabel("Flux [erg / ($cm^{2}$ s $\AA$)] ----->")
        plt.title("Magnetospheric Shock Region SED")
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
    n_dust_disk = config['n_dust_disk']
    r_dust = np.logspace(np.log10(r_sub.si.value), np.log10(270 * const.au.value), n_dust_disk) * u.m
    t_dust_init = t_eff_dust(r_dust, config)

    if t_eff_dust(r_sub, config) > 1400 * u.K:
        t_dust = ma.masked_greater(t_dust_init.value, 1400)
        t_dust = t_dust.filled(1400)
        t_dust = t_dust * u.K
    else:
        t_visc_dust = np.zeros(len(r_dust)) * u.K
        for i in range(len(r_dust)):
            t_visc_dust[i] = temp_visc(config, r_dust[i], r_in)  # has to be done using for loop to avoid ValueError
        #print(t_visc_dust)
        t_dust = np.maximum(t_dust_init, t_visc_dust)

    if plot:
        plt.plot(r_dust / const.au, t_dust.value)
        plt.xlabel("Radial distance (in AU) ----->")
        plt.ylabel("Temperature (in Kelvin) ----->")
        plt.title("Dusty Disk Radial Temperature Variation")
        plt.show()
    #print(t_dust)
    dust_flux = np.zeros(n_data) * u.erg / (u.cm * u.cm * u.s * u.AA * u.sr) * (u.m * u.m)
    wavelength = np.logspace(np.log10(l_min.value), np.log10(l_max.value), n_data) * u.AA
    for i in range(len(r_dust) - 1):
        scale_unit = u.erg / (u.cm ** 2 * u.s * u.AA * u.sr)
        dust_bb = BlackBody(t_dust[i], scale=1 * scale_unit)
        dust_bb_flux = dust_bb(wavelength)
        dust_flux += dust_bb_flux * np.pi * (r_dust[i + 1] ** 2 - r_dust[i] ** 2)
        if i % 100 == 0:  # print progress after every 100 annuli
            #print(r_dust[i])
            if config['verbose']:
                print(f"done temperature {i}")

    obs_dust_flux = dust_flux * np.cos(inclination) / (np.pi * d_star.si ** 2) * u.sr
    if save:
        np.save(f'{save_loc}/dust_component.npy', obs_dust_flux.value)
    if plot:
        plt.plot(wavelength, obs_dust_flux)
        plt.xlabel("Wavelength in $\AA$ ----->")
        plt.ylabel("Flux [erg / ($cm^{2}$ s $\AA$)] ----->")
        plt.title("Dust Dominated Disk SED")
        plt.show()
    return obs_dust_flux


def dust_extinction_flux(config, wavelength, obs_viscous_disk_flux, obs_star_flux, obs_mag_flux, obs_dust_flux):
    """Redden the spectra with the Milky Way extinction curves. Ref. Gordon et al. 2021, Fitzpatrick et. al 2019

    Parameters
    ----------
    config : dict
             dictionary containing system parameters

    wavelength: astropy.units.Quantity
        wavelength array, in units of Angstrom

    obs_star_flux: astropy.units.Quantity
    obs_viscous_disk_flux: astropy.units.Quantity
    obs_mag_flux: astropy.units.Quantity
    obs_dust_flux: astropy.units.Quantity

    Returns
    ----------
    total_flux: astropy.units.Quantity
        spectra reddened as per the given parameters of a_v and r_v in details
    """
    r_v = config['rv']
    a_v = config['av']
    save = config['save']
    plot = config['plot']
    save_loc = config['save_loc']

    wav1 = np.extract(wavelength < 33e3 * u.AA, wavelength)
    wav2 = np.extract(wavelength >= 33e3 * u.AA, wavelength)

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
        plt.xlabel("Wavelength in $\AA$ ----->")
        plt.ylabel("Flux [erg / ($cm^{2}$ s $\AA$)] ----->")
        plt.title("Extinguished Spectra")
        plt.legend()
        plt.show()
    return total_flux


def parse_args(raw_args=None):
    """Take config file location from the command line"""
    parser = argparse.ArgumentParser(description="YSO Spectrum generator")
    parser.add_argument('ConfigfileLocation', type=str, help="Path to config file")
    args = parser.parse_args(raw_args)
    return args


def contribution(raw_args=None):
    """find the contribution of the various annuli towards a particular line/group of lines
    """
    args = parse_args(raw_args)
    config = config_read(args.ConfigfileLocation)
    dr, t_max, d, r_in, r_sub = generate_temp_arr(config)
    inclination = config['inclination']
    m = config['m']
    n_data = config['n_data']
    l_min = config['l_min']
    l_max = config['l_max']

    arr = []  # to store the cumulative flux arrays
    t_max = max(d.values())
    wav = np.logspace(np.log10(l_min.value), np.log10(l_max.value), n_data)
    wav = (wav - 22930) / 22930 * 3e5
    wav = np.extract(wav > -3000, wav)
    wav = np.extract(wav < 3000, wav)

    viscous_disk_flux = np.zeros(len(wav)) * (u.erg * u.m ** 2 / (u.cm ** 2 * u.s * u.AA))
    cumulative_flux = np.zeros(len(wav)) * (u.erg / (u.cm ** 2 * u.s * u.AA))
    # annulus_flux = np.zeros(len(wav)) * (u.erg / (u.cm ** 2 * u.s * u.AA))
    flag = 0
    z_val = []
    # loop over the temperatures
    for int_temp in range(t_max, 14, -1):

        # to store total flux contribution from annuli of this temperature
        temp_flux = np.zeros(len(wav)) * (u.erg / (u.s * u.AA))
        radii = np.array([r for r, t in d.items() if t == int_temp])
        if len(radii) == 0:
            continue
        radii = radii * u.m
        if int_temp in range(14, 20):  # constrained by availability of BT-Settl models
            logg = 3.5
        else:
            logg = 1.5

        wavelength, flux = read_bt_settl_npy(config, int_temp, logg)
        radii = sorted(radii)
        if len(radii) == 0:
            if config['verbose']:
                print("no radii at this temp")
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

                # convert to velocity space and trim to required region
                x_throw = (x_throw - 22930) / 22930 * 3e5
                y_final = np.extract(x_throw > -3000, y_final)
                x_throw = np.extract(x_throw > -3000, x_throw)
                y_final = np.extract(x_throw < 3000, y_final)
                cumulative_flux += y_final

                if flag % 50 == 0:
                    arr.append(np.log10(cumulative_flux.copy().value * 2 * np.pi * r.value))
                    z_val.append(r.value.copy())
                flag += 1
            temp_flux += y_final * np.pi * (2 * r * dr + dr ** 2)
        viscous_disk_flux += temp_flux
        if config['verbose']:
            print(f"done temp {int_temp}")
    # trim to CO line
    wavelength = np.logspace(np.log10(l_min.value), np.log10(l_max.value), n_data)
    wavelength = (wavelength - 22930) / 22930 * 3e5
    wavelength = np.extract(wavelength > -3000, wavelength)
    wavelength = np.extract(wavelength < 3000, wavelength)

    # plot the heat map
    fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
    ax.imshow(arr, aspect='auto')
    plt.show()

    for i in range(len(arr)):
        fl = arr[i]
        z = np.ones(len(fl)) * z_val[i]
        ax.set_xlabel("Velocity (km/s)")
        ax.set_ylabel("Flux (erg / cm^2 s A)")
        ax.set_zlabel("Extent of integration")
        ax.plot(wavelength, fl, z, label=f'i = {i}')
    plt.show()


def total_spec(dict_config):
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
    return wavelength, total_flux


def main(raw_args=None):
    """Calls the base functions sequentially, and finally generates extinguished spectra for the given system"""

    st = time.time()
    args = parse_args(raw_args)
    dict_config = config_read(args.ConfigfileLocation)
    # dict_config = config_read("/home/arch/yso/config_file.das")
    dr, t_max, d, r_in, r_sub = generate_temp_arr(dict_config)
    # control line for planetesimal
    #garb1, garb2, d = generate_temp_arr_planet(dict_config, 2, 0.03, d)
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

    et = time.time()
    print(f"Total time taken : {et - st}")
    if dict_config['plot']:
        plt.plot(wavelength, obs_star_flux, label="Stellar photosphere")
        plt.plot(wavelength, total_flux, label="Total")
        plt.plot(wavelength, obs_viscous_disk_flux, label="Viscous Disk")
        plt.plot(wavelength, obs_mag_flux, label="Magnetosphere")
        plt.plot(wavelength, obs_dust_flux, label="Dusty disk")
        plt.legend()
        plt.xlabel("Wavelength [Angstrom]")
        plt.ylabel("Flux [erg / cm^2 s A]")
        plt.show()

    print("done")


def new_contribution():
    config = config_read("config_file.das")
    dr, t_max, d, r_in, r_sub = generate_temp_arr(config)

    save_loc = config['save_loc']
    l_min = config['l_min']
    l_max = config['l_max']
    n_data = config['n_data']
    r_star = config['r_star']
    print(d)
    r_visc = np.array([r for r, t in d.items()])
    r_visc = sorted(r_visc) * u.m
    d_star = config['d_star']
    inclination = config['inclination']

    fig, ax = plt.subplots()
    arr = []
    radii = []

    # trim wavelength axis to region of interest
    wavelength = np.logspace(np.log10(l_min.value), np.log10(l_max.value), n_data)
    wav_new = ma.masked_where(wavelength < 10000, wavelength)
    wav_new = ma.masked_where(wav_new > 25000, wav_new)
    wav_new = wav_new.compressed()
    print(wav_new)
    print(r_in)
    print((r_sub / r_in).si)

    for i in range(0, len(r_visc), 7):
        r = r_visc[i]
        flux = np.load(f"{save_loc}/radius_{r}_flux.npy")
        flux = flux * (u.erg / (u.cm ** 2 * u.s * u.AA)) * r * dr
        # trim to region of interest
        wavelength = np.logspace(np.log10(l_min.value), np.log10(l_max.value), n_data)
        flux = np.extract(wavelength < 25000, flux)
        wavelength = np.extract(wavelength < 25000, wavelength)
        flux = np.extract(wavelength > 10000, flux)
        wavelength = np.extract(wavelength > 10000, wavelength)
        # correct for distance
        flux *= np.cos(inclination) / (np.pi * d_star ** 2)
        flux = flux.to(u.erg / (u.cm ** 2 * u.s * u.AA))

        ax.plot(wav_new, np.log10(flux.value) - 0.05 * i,
                label=f"r={np.round(r / const.R_sun, 2)} R_sun, T={d[r.value] * 100} K")
        arr.append(flux.value)

    plt.xlabel("Wavelength [Angstrom]")
    plt.ylabel("log_10 Flux (+ offset) [erg / cm^2 s A]")
    plt.legend()
    plt.show()
#def total_spec():



if __name__ == "__main__":
    main(raw_args=None)
