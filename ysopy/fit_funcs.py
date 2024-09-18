import base_funcs as bf
import astropy.constants as const
import astropy.units as u
import numpy as np
from scipy.optimize import least_squares
import pandas as pd


def residual_calc(params, obs_data_path):
    """
    Calculates the vector of residuals, given the parameters which are allowed to vary
    Parameters
    ----------
    params: list
            [b,m,m_dot,inclination,t_0,
    obs_data_path: str

    Returns
    -------
    err: numpy.ndarray
    """
    dict_config = bf.config_read('/home/arch/yso/YSOpy_codes/ver_23_03_2024/ysopy/ysopy/config_file.cfg')

    # overwrite the required parameters
    dict_config['b'] = params[0] * u.kilogauss
    dict_config['m_dot'] = params[2] * const.M_sun / (1 * u.year).to(u.s)
    dict_config['inclination'] = params[3] * u.degree

    dict_config['t_0'] = params[4]

    # take the H-slab parameters to the closest available value on the grid
    # for now the grid is fixed to the following: (change by hand)
    t_slab_grid = np.array(range(8000, 11000, 500))
    n_e_grid = np.array([12, 13, 14])
    tau_grid = np.array([0.5, 1.0, 1.5, 2.0])
    dict_config['t_slab'] = t_slab_grid[np.argmin(np.abs(t_slab_grid-params['t_slab']))] * u.K
    dict_config['t_slab'] = n_e_grid[np.argmin(np.abs(n_e_grid - params['n_e']))]
    dict_config['tau'] = tau_grid[np.min(np.abs(tau_grid - params['tau']))]

    # read Baraffe's model for t_star, r_star, log_g_star
    # get closest match in stellar mass
    pms_data = pd.read_csv('/home/arch/yso/baraffe-data.csv')
    match_id = np.argmin(np.abs(pms_data['m'] - params[1]))
    dict_config['m'] = pms_data['m'][match_id] * const.M_sun
    dict_config['r_star'] = pms_data['r_star'] * const.R_sun

    # take the available values from BT Settl
    t_star = np.round(pms_data['t_star'][match_id] / 100, 1) * 100 * u.K
    dict_config['t_star'] = t_star
    dict_config['log_g_star'] = 3.5

    # generate theoretical spectra for the modified dict_config
    wavelength, theoretical_spectra = bf.total_spec(dict_config)

    # read data from a given set location
    obs_spectra = np.load(obs_data_path)

    # add code for interpolating or restricting to the same wavelength range

    # assuming the observed data and theoretical spectra have the same x-axis
    err = np.abs(obs_spectra - theoretical_spectra)

    return err


def fitter():
    return None


err_arr = residual_calc({'m': 1.0, 'm_dot': 1e-6, 'inclination': 45.0, 'b': 1.0}, '/home/arch/yso/fit_test/data_spec.npy')
print(err_arr)
print("Run completed successfully")