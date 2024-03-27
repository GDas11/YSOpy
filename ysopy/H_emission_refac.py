from itertools import repeat
import astropy.units
import numpy as np
from astropy.modeling.models import BlackBody
import astropy.constants as const
import astropy.units as u
from multiprocessing import Pool
import os
import base_funcs as bf

# define the constants
sigma = const.sigma_sb
G = const.G
c = const.c
h = const.h
k = const.k_B
m_e = const.m_e
Z = 1  # number of protons in the nucleus # here it is Hydrogen

# for the H slab ne = ni = nH
i_h = 13.602 * u.eV
v_o = 3.28795e15 * u.Hertz  # ionisation frequency of H

# defining the infinite series summation function

def infinisum(f, m: int):
    """This calculates the infinite sum of a function f
    given some accuracy (by default 1e-8)
    Parameters
    ----------
    f : function
        the function upon which we want to do infinite sum
    m : int
        parameter for the limits of the infinite sum
    Returns
    ----------
    res: float
        infinite sum for given parameters

    """
    n = m
    res = sum(f(k) for k in range(n, 2 ** n))
    while True:
        term = sum(f(k) for k in range(2 ** n, 2 ** (n + 1)))
        if term < 1e-8:  # should it be a parameter??????
            break
        n, res = n + 1, res + term
        # print(term, res)
    return res


# calculating free-bound emissivity
def j_h_fb_calc(config_file, v):
    """
    This function calculates the emissivity parameter for free-bound case
    of H emission. adopted from Manara Ph.D. Thesis (2014)
    Parameters
    ----------
    config_file:    dict
                    The config dictionary
    v:  astropy.units.Quantity
        Frequency to be given in units of Hz
    Returns
    ----------
    j_h_fb_v:astropy.units.Quantity
            Emissivity parameter for free-bound case of H emission in
            frequency space

    """
    t_slab = config_file["t_slab"]
    n_e = config_file["n_e"]
    n_i = n_e
    t_fb_v = v / (v_o * Z ** 2)  # this term is recurring in
    # the expressions so putting it here to make loading easier
    m = int((1 / t_fb_v) ** 0.5 + 1)  # m parameter for the limits of the infinite sum
    f = lambda n: (1 / n ** 3) * np.exp((h * v_o) / (k * t_slab * n ** 2)) * (
            1 + 0.1728 * ((t_fb_v) ** (1 / 3) - (2 / n ** 2) * (t_fb_v) ** (-2 / 3)) - 0.0496 * (
            (t_fb_v) ** (2 / 3) - (2 / (3 * n ** 2)) * (t_fb_v) ** (-1 / 3) +
            (2 / (3 * n ** 4)) * (t_fb_v) ** (-4 / 3)))
    fb_vt = infinisum(f, m) * (2 * h * v_o * Z ** 2 / (k * t_slab))
    j_h_fb_v = 5.44 * 10 ** (-39) * Z ** 2 / (t_slab.value) ** (1 / 2) * n_e.value * n_i.value * np.exp(
        (-h * v) / (k * t_slab)) * fb_vt * u.erg * u.cm ** (-3) * u.s ** (-1) * u.Hertz ** (-1) * u.sr ** (-1)
    return j_h_fb_v


'''def f_sum(t, n):
    """
    This is a function that we have to replace in the above function because the used 'f' lambda
    function is violating PEP-8 rule
    Parameters
    --------
    t:  float
        the recurring term given by t_fb_v in j_h_fb_v
    n:  int
        sum will be performed for this value of n
    Returns
    --------
    sum_func: astropy.units.Quantity
        One term of Gaunt-factor for a given frequency and integer
    """
    sum_func = (1 / n ** 3) * np.exp((h * v_o) / (k * t_h_slab * n ** 2)) * (
            1 + 0.1728 * (t ** (1 / 3) - (2 / n ** 2) * t ** (-2 / 3)) - 0.0496 * (
            t ** (2 / 3) - (2 / (3 * n ** 2)) * t ** (-1 / 3) +
            (2 / (3 * n ** 4)) * t ** (-4 / 3)))
    return sum_func
'''

# calculating free-free emissivity

def j_h_ff_calc(config_file, v):
    """

    :param config_file:
    :param v: astropy.units.Quantity
    It gives te
    :return:
    """
    t_slab = config_file["t_slab"]
    n_e = config_file["n_e"]
    n_i = n_e
    t_fb_v = v / (v_o * Z ** 2)
    g_ff_v = 1 + 0.1728 * (t_fb_v) ** (1 / 3) * (1 + (2 * k * t_slab / (h * v))) - 0.0496 * (t_fb_v) ** (2 / 3) * (
            1 + (2 * k * t_slab / (3 * h * v)) + (4 / 3) * (k * t_slab / (h * v)) ** 2)
    j_h_ff_v = 5.44 * 10 ** (-39) * Z ** 2 / (t_slab.value) ** (1 / 2) * n_e.value * n_i.value * np.exp(
        (-h * v) / (k * t_slab)) * g_ff_v * u.erg * u.cm ** (-3) * u.s ** (-1) * u.Hertz ** (-1) * u.sr ** (-1)
    return j_h_ff_v


def main(config_file: dict, v: astropy.units.Quantity):
    """
    This function takes in the config_file and the frequency and gives the total emissivity parameter
    for H emission case. Note this function will not save anything. For saving we have to
    use generate_grid_h.
    :param config_file: This parameter is to give the configuration file
    :param v: frequency in units of hertz
    :return: emissivity parameter (j) at that frequency
    """
    h_fb_arr, h_ff_arr = [], []
    p = Pool()
    result = p.starmap(j_h_fb_calc, zip(repeat(config_file), v))
    for i in result:
        # print(i)
        h_fb_arr.append(i.value)
    result2 = p.starmap(j_h_ff_calc, zip(repeat(config_file), v))
    for i in result2:
        # print(i)
        h_ff_arr.append(i.value)
    h_fb_arr = np.array(h_fb_arr) * (u.erg / (u.cm ** 3 * u.Hz * u.s * u.sr))
    h_ff_arr = np.array(h_ff_arr) * (u.erg / (u.cm ** 3 * u.Hz * u.s * u.sr))
    j_h_total: astropy.units.Quantity = h_fb_arr + h_ff_arr
    return j_h_total


def get_l_slab(config_file: dict):
    """
    Function to calculate the length of the slab
    :param config_file: input config file
    :return: length of the slab in meters
    """
    tau = config_file["tau"]
    v = const.c / config_file["l_l_slab"]
    # total emissivity
    j_h_fb_v = j_h_fb_calc(config_file, v)
    j_h_ff_v = j_h_ff_calc(config_file, v)
    j_h_arr = j_h_fb_v + j_h_ff_v
    # define L_slab
    bb_v = BlackBody(temperature=config_file["t_slab"])
    l_slab: astropy.units.Quantity = tau * bb_v(v) / j_h_arr  # L slab in frequency space
    # print('L_slab : ', L_slab.si)

    return l_slab.si


def generate_grid_h(config_file, t_slab_para, den, opti_depth):
    """
    This function is to generate a grid of h emission spectra. It also checks if a file has
    been created earlier and stored. If not then it goes to calculate the emissivity parameter
    array from scratch given the parameters.
    Note for case of H emission there is not a need to calculate for different ne values
    as intensity doesn't depend on that factor.
    Parameters
    -----------
    config_file:    dict
                    the frequency range is taken from the config file
    t_slab_para:    int
                    the required value of temperature in Kelvin. If want to find
                    for temperature suppose 8000 K then this parameter should be 8000.
    den:            int
                    required electron density . e.g., if we want for 10^15 cm^(-3)
                    then put this parameter as 15.
    opti_depth:     float
                    This is the optical density at 3000 Angstrom. e.g., for
                    optical depth of 1.2 we have to put 1.2.
    Returns
    -----------
    astropy.units.Quantity
    An intensity array in units of (u.erg / (u.cm ** 2 * u.s * u.AA * u.sr)) [wavelength space]


    """
    lam = np.logspace(np.log10(config_file['l_min'].value),
                      np.log10(config_file['l_max'].value), config_file['n_h']) * u.AA
    v = const.c / lam
    h_grid_path = config_file["h_grid_path"]
    # print(lam)
    config_file["t_slab"] = t_slab_para * u.K
    config_file["n_e"] = 10 ** den * (u.cm ** (-3))
    config_file["tau"] = opti_depth
    saving = config_file["save_grid_data"]
    if os.path.exists(f"{h_grid_path}/{t_slab_para}_tau_{tau}_len_{config_file['n_h']}"):
        j_h_arr = np.load(f"{h_grid_path}/{t_slab_para}_tau_{tau}_len_{config_file['n_h']}/j_h_tot.npy") * (u.erg / (u.cm ** 3 * u.Hz * u.s * u.sr))
        if len(j_h_arr) == config_file["n_h"]:
            print('True, the grid exists so not going for multiprocess')
            print(f"{t_slab_para}_{opti_depth}_{len(j_h_arr)} exists")
        else:
            print('False: this has to go for multiprocess grid not found')
            print(f'Length of the data set present is incompatible : {config_file["n_h"]} given : {len(j_h_arr)} having')
            j_h_arr = main(config_file, v)
            if saving:
                os.mkdir(f"{h_grid_path}/{t_slab_para}_tau_{tau}_len_{config_file['n_h']}")
                np.save(f"{h_grid_path}/{t_slab_para}_tau_{tau}_len_{config_file['n_h']}/j_h_tot.npy", j_h_arr.value)

    else:
        print('False: this has to go for multiprocess grid not found')
        j_h_arr = main(config_file, v)
        if saving:
            os.mkdir(f"{h_grid_path}/{t_slab_para}_tau_{tau}_len_{config_file['n_h']}")
            np.save(f"{h_grid_path}/{t_slab_para}_tau_{tau}_len_{config_file['n_h']}/j_h_tot.npy", j_h_arr.value)
    t_slab = config_file["t_slab"]
    bb_freq = BlackBody(temperature=t_slab)  # blackbody thing to be used in freq case
    l_slab = get_l_slab(config_file)

    tau_v_arr_h = j_h_arr * l_slab / bb_freq(v)
    beta_h_v_arr = (1 - np.exp(-tau_v_arr_h)) / tau_v_arr_h
    intensity_h_l = (j_h_arr * l_slab * beta_h_v_arr * (c / (lam ** 2))).to(
        u.erg / (u.cm ** 2 * u.s * u.AA * u.sr))
    # print(intensity_h_l)

    if saving:
        np.save(f"{h_grid_path}/{t_slab_para}_tau_{tau}_len_{config_file['n_h']}/Flux_wav.npy", intensity_h_l.value)
        dtls_wrte = str(f"\n****** Constants Used *******\n"
                        f"G : {G}\nc : {c}\n"
                        f"h : {h}\nk : {k}\n"
                        f"m_e : {m_e}\n"
                        f"Z : {Z}\t number of protons in the nucleus, here it is Hydrogen\n"
                        f"Equilibrium Quantum Level : {n}\n"
                        f"v_o = 3.28795e15 Hz\t ionisation frequency of H\n"
                        f"\n****** Parameters from Config File ******\n"
                        f"t_slab = {config_file['t_slab']}\t Temperature of the slab\n"
                        f"tau = {config_file['tau']}\t optical depth\n"
                        f"l_l_slab = {config_file['l_l_slab']}\t ref wavelength for "
                        f"length of slab\n"
                        f"v = {(const.c / config_file['l_l_slab']).to(u.Hz)}\t ref frequency for "
                        f"length of slab\n"
                        f"l_min = {config_file['l_min']}\n"
                        f"l_max = {config_file['l_max']}\n"
                        f"n_h_minus = {config_file['n_h_minus']}\tlength of energy axis\n"
                        f"\n\n----- Some important parameters -----\n\n"
                        f"l_slab : {l_slab}\tlength of the slab calculated\n")
        details = {f'T_h_slab ({t_slab.unit})': t_slab.value,
                   f'ne ': config_file["n_e"],
                   f'tau': tau,
                   f"l_init ({config_file['l_min'].unit})": config_file['l_min'],
                   f"l_final ({config_file['l_max'].unit})": config_file['l_max'],
                   'len_w': config_file['n_h'],
                   f'L_slab ({l_slab.unit})': l_slab.value,
                   'Equilibrium Quantum Level ': n}
        with open(f"{h_grid_path}/{t_slab_para}_tau_{tau}_len_{config_file['n_h']}/details.txt", 'w+') as f:
            f.write(str(dtls_wrte))
    else:
        print('Data not saving!!')
    return intensity_h_l


# define multi-core process


if __name__ == '__main__':
    config = bf.config_read("config_file.cfg")
    for temp in range(8000, 8500, 500):
        for tau in [0.01]:
            for n in [14]:
                print(temp, tau, n)
                inten = generate_grid_h(config_file=config, t_slab_para=temp, opti_depth=tau, den=n)
