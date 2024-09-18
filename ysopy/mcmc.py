import numpy as np

import base_funcs as bf
import emcee
from configparser import ConfigParser


def config_reader(filepath):
    parser = ConfigParser()
    parser.read(filepath)
    config_data = dict(parser['Parameters'])
    config_data['m_u'] = float(config_data['m_u'])
    config_data['m_l'] = float(config_data['m_l'])
    config_data['m_dot_u'] = float(config_data['m_dot_u'])
    config_data['m_dot_l'] = float(config_data['m_dot_l'])

    return config_data


def total_spec(theta,wavelength):
    """
    Generates the model spectra by running ysopy for the given parameters in theta array
    Parameters
    ----------
    theta
    wavelength

    Returns
    -------

    """
    # modify config file
    m, m_dot = theta
    config = bf.config_read('config_file.cfg')
    config['m'] = m * const.M
    config['m_dot'] = m_dot
    
    flux, wave = bf.



def log_prior(theta):
    m, m_dot = theta
    config_data = config_reader('mcmc_config.cfg')
    if config_data['m_l']<m<config_data['m_u'] and config_data['m_dot_l']<m_dot<config_data['m_dot_u']:
        return 0.0
    return -np.inf


def log_likelihood(theta, x, y, yerr):
    m, b, log_f = theta
    model = m * x + b
    sigma2 = yerr**2 + model**2 * np.exp(2 * log_f)
    return -0.5 * np.sum((y - model) ** 2 / sigma2 + np.log(sigma2))


def log_probability(theta, x, y, yerr):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, x, y, yerr)
