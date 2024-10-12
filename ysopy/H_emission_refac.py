import numpy as np
from astropy.modeling.models import BlackBody
import astropy.constants as const
import astropy.units as u
import os
import base_funcs as bf

# define the constants
c = const.c
h = const.h
k = const.k_B
m_e = const.m_e # only written in save
Z = 1  # number of protons in the nucleus # here it is Hydrogen

# for the H slab ne = ni = nH
i_h = 13.602 * u.eV
v_o = 3.28795e15 * u.Hertz  # ionisation frequency of H

    
def f_sum(t, m, t_slab, large_num = 1000):
    """Calculates the free-bound Gaunt factor, performing a summation over the states

    Parameters
    --------
    t:  float
        the recurring term given by t_fb_v in j_h_fb_calc
    m:  int
        sum will be performed starting from this integer
    t_slab: astropy.units.Quantity
        The slab temperature in Kelvin
    large_num: int
        the endpoint upto which the summation is carried out,
        theoretically this is an infifnite sum, but taking ~100 is good enough

    Returns
    --------
    sum_func: astropy.units.Quantity
        One term of Gaunt-factor for a given frequency and integer
    """
    n = np.arange(m, large_num)
    terms = (1 / n ** 3) * np.exp((h * v_o) / (k * t_slab * n ** 2)) * (
            1 + 0.1728 * (t ** (1 / 3) - (2 / n ** 2) * t ** (-2 / 3)) - 0.0496 * (
            t ** (2 / 3) - (2 / (3 * n ** 2)) * t ** (-1 / 3) +
            (2 / (3 * n ** 4)) * t ** (-4 / 3)))  # ref. Manara eq. 2.15
    result = np.sum(terms)
    return result

# Alternative to j_h_fb_calc
def j_h_fb_calc(config_file, v):
    """
    Calculates the emissivity parameter for free-bound case
    of H emission. Adopted from C. F. Manara's Ph.D. Thesis (2014)
    Parameters
    ----------
    config_file:    dict
                    The configuration dictionary
    v:  astropy.units.Quantity
        Frequency array to be given in units of Hz
    Returns
    ----------
    j_h_fb_v:astropy.units.Quantity
            Emissivity parameter for free-bound case of H emission in
            frequency space

    """
    t_slab = config_file["t_slab"]
    n_e = config_file["n_e"]
    n_i = n_e
    t_fb_v = v / (v_o * Z ** 2)  # this term is recurring in the expressions so putting it here to make loading easier
    m = (1 / t_fb_v) ** 0.5 + 1
    m = m.astype('int32')  # m parameter for the lower limit of the infinite sum, ref. Manara eq. 2.14
    
    #calculate the Gaunt factor
    g_fb_vt = np.zeros(t_fb_v.shape[0])
    for i in range(t_fb_v.shape[0]):
        g_fb_vt[i] = f_sum(t_fb_v[i], m[i], t_slab, large_num=100)

    # some lines for comparison, large num = 100 is sufficient
    # print("for 50 steps :",g_fb_vt)
    # g_fb_vt_2 = np.zeros(t_fb_v.shape[0])
    # for i in range(t_fb_v.shape[0]):
    #     g_fb_vt_2[i] = f_sum(t_fb_v[i], m[i], t_slab, large_num=50000)
    # print("for 50000 steps :",g_fb_vt)
    # max_err = np.max(np.abs(g_fb_vt - g_fb_vt_2))
    # print(f"max err {max_err}")
    
    g_fb_vt = g_fb_vt * (2 * h * v_o * Z ** 2 / (k * t_slab))

    j_h_fb_v = 5.44 * 10 ** (-39) * Z ** 2 / (t_slab.value) ** (1 / 2) * n_e.value * n_i.value * np.exp(
        (-h * v) / (k * t_slab)) * g_fb_vt * u.erg * u.cm ** (-3) * u.s ** (-1) * u.Hertz ** (-1) * u.sr ** (-1)
    
    return j_h_fb_v


# calculating free-free emissivity
def j_h_ff_calc(config_file, v):
    """
    Calculate the free-free emissivity
    Parameters
    ----------
    config_file:  dict
        Configuration dictionary
    v: astropy.units.Quantity
        frequency array on which the emissivity is evaluated

    Returns
    -------
    j_h_ff_v: astropy.units.Quantity
        emissivity in units of erg / (cm^3 s Hz sr)
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


def get_l_slab(config_file: dict):
    """Calculates the length of the slab, using the optical depth at 3000 Angstroms.
    This assumes the emission at 3000 A to be entirely from the H component
    (See the discussion following eq. 2.4 in Manara's thesis)

    Parameters
    ----------
    config_file: dict
        Configuration dictionary

    Returns
    -------
    l_slab: astropy.units.Quantity
        length of the slab in meters
    """
    tau = config_file["tau"]
    v = const.c / config_file["l_l_slab"]
    v = np.array([v.si.value]) * u.Hz # for consistency

    # total emissivity
    j_h_fb_v = j_h_fb_calc(config_file, v)
    j_h_ff_v = j_h_ff_calc(config_file, v)
    j_h_tot = j_h_fb_v + j_h_ff_v
    
    # define L_slab
    bb_v = BlackBody(temperature=config_file["t_slab"])
    l_slab = tau * bb_v(v) / j_h_tot
    l_slab = l_slab.si

    return l_slab


def generate_grid_h(config_file, t_slab, log_n_e, tau):
    """
    This function is to generate a grid of h emission spectra. It also checks if a file has
    been created earlier and stored. If not then it goes to calculate the emissivity parameter
    array from scratch given the parameters.
    Note for case of H emission there is not a need to calculate for different n_e values
    as intensity doesn't depend on that factor.
    Parameters
    -----------
    config_file:    dict
                    the frequency range is taken from the config file
    t_slab:    int
                    the required value of temperature in Kelvin. If want to find
                    for temperature suppose 8000 K then this parameter should be 8000.
    log_n_e:        int
                    required electron density . e.g., if we want for 10^15 cm^(-3)
                    then put this parameter as 15.
    tau:            float
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
    config_file["t_slab"] = t_slab * u.K
    config_file["n_e"] = 10 ** log_n_e * (u.cm ** (-3))
    config_file["tau"] = tau
    saving = config_file["save_grid_data"]

    # changed the code to remove overchecking
    # the check for n_h in the directory name is (should be) sufficient
    # directory name is of the form: temp_5000_tau_1.0_len_5000
    dirname = f"temp_{t_slab}_tau_{tau}_len_{config_file['n_h']}"

    if os.path.exists(f"{h_grid_path}/{dirname}"):
        
        j_h_arr = np.load(f"{h_grid_path}/{dirname}/j_h_tot.npy") * (u.erg / (u.cm ** 3 * u.Hz * u.s * u.sr))
        print('True, the grid exists so not going for multiprocess')
        print(f"{t_slab}_{tau}_{len(j_h_arr)} exists")
    
    else:
        print('False: this has to go for multiprocess, grid not found')

        print("Starting free-bound calculation")
        j_h_fb_arr = j_h_fb_calc(config_file,v)

        print("starting free-free calculation")
        j_h_ff_arr = j_h_ff_calc(config_file, v)

        j_h_total= j_h_fb_arr + j_h_ff_arr

        if saving:
            os.makedirs(f"{h_grid_path}/{dirname}")
            np.save(f"{h_grid_path}/{dirname}/j_h_tot.npy", j_h_total.value)

    bb_freq = BlackBody(temperature=t_slab*u.K)  # blackbody thing to be used in freq case
    l_slab = get_l_slab(config_file)

    # calculate specific intensity in the wavelength domain, using c / lambda**2
    # see Manara eq. 2.19, 2.20
    tau_v_arr_h = j_h_arr * l_slab / bb_freq(v)
    beta_h_v_arr = (1 - np.exp(-tau_v_arr_h)) / tau_v_arr_h
    intensity_h_l = (j_h_arr * l_slab * beta_h_v_arr * (c / (lam ** 2))).to(u.erg / (u.cm ** 2 * u.s * u.AA * u.sr))

    if saving:
        np.save(f"{h_grid_path}/{dirname}/Flux_wav.npy", intensity_h_l.value)
        dtls_wrte = str(f"\n****** Constants Used *******\n"
                        f"h : {h}\nk : {k}\n"
                        f"m_e : {m_e}\n"
                        f"Z : {Z}\t number of protons in the nucleus, here it is Hydrogen\n"
                        #f"Equilibrium Quantum Level : {n}\n"       #Where does this come from?
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
                   f'L_slab ({l_slab.unit})': l_slab.value}
                   #'Equilibrium Quantum Level ': n}        ## Where does this come from?
        with open(f"{h_grid_path}/{dirname}/details.txt", 'w+') as f:
            f.write(str(dtls_wrte))
    else:
        print('Data not saving!! Details were not stored.')

    return intensity_h_l

def get_h_intensity(config_file):
    """Generates the intensity from the H-component"""
    wavelength = np.logspace(np.log10(config_file['l_min'].value),
                      np.log10(config_file['l_max'].value), config_file['n_h']) * u.AA
    v = const.c / wavelength

    if config_file['verbose']:
        print("Starting free-bound calculation")
    j_h_fb_arr = j_h_fb_calc(config_file,v)
    if config_file['verbose']:
        print("starting free-free calculation")
    j_h_ff_arr = j_h_ff_calc(config_file, v)
    j_h_total= j_h_fb_arr + j_h_ff_arr

    l_slab = get_l_slab(config_file)
    bb_freq = BlackBody(temperature=config_file['t_slab'])
    tau_v_arr_h = j_h_total * l_slab / bb_freq(v)
    beta_h_v_arr = (1 - np.exp(-tau_v_arr_h)) / tau_v_arr_h
    
    intensity_h_l = (j_h_total * l_slab * beta_h_v_arr * (c / (wavelength ** 2))).to(u.erg / (u.cm ** 2 * u.s * u.AA * u.sr))

    return intensity_h_l

if __name__ == '__main__':
    config = bf.config_read("config_file.cfg")
    for temp in (8000,):
        for tau in [1.0,]:
            for log_n_e in [13,]:
                print(temp, tau, log_n_e)
                inten = generate_grid_h(config_file=config, t_slab=temp, tau=tau, log_n_e=log_n_e)
