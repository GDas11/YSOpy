import numpy as np
from scipy.interpolate import interp1d
import base_funcs as bf
import astropy.constants as const
import astropy.units as u
from astropy.io import ascii
from pypeit.core import wave
import emcee
from configparser import ConfigParser
import matplotlib.pyplot as plt
import time

from multiprocessing import Pool
import os

os.environ["OMP_NUM_THREADS"] = "1"


def config_reader(filepath):
    """
    Read the config file containing the bounds for each parameter
    """
    parser = ConfigParser()
    parser.read(filepath)
    config_data = dict(parser['Parameters'])

    config_data['m_u'] = float(config_data['m_u'])
    config_data['m_l'] = float(config_data['m_l'])

    config_data['log_m_dot_u'] = float(config_data['log_m_dot_u'])
    config_data['log_m_dot_l'] = float(config_data['log_m_dot_l'])

    config_data['b_u'] = float(parser['Parameters']['b_u'])
    config_data['b_l'] = float(parser['Parameters']['b_l'])
    
    config_data['r_star_u'] = float(parser['Parameters']['r_star_u'])
    config_data['r_star_l'] = float(parser['Parameters']['r_star_l'])
    
    config_data['inclination_u'] = float(parser['Parameters']['inclination_u'])
    config_data['inclination_l'] = float(parser['Parameters']['inclination_l'])
    
    config_data['t_0_u'] = float(parser['Parameters']['t_0_u'])
    config_data['t_0_l'] = float(parser['Parameters']['t_0_l'])
    
    config_data['t_slab_u'] = float(parser['Parameters']['t_slab_u'])
    config_data['t_slab_l'] = float(parser['Parameters']['t_slab_l'])
    
    config_data['log_n_e_u'] = float(parser['Parameters']['log_n_e_u'])
    config_data['log_n_e_l'] = float(parser['Parameters']['log_n_e_l'])
    
    config_data['tau_u'] = float(parser['Parameters']['tau_u'])
    config_data['tau_l'] = float(parser['Parameters']['tau_l'])

    return config_data


def generate_initial_conditions(config_data,n_walkers):
    params = ['m', 'log_m_dot', 'b', 'inclination',  'log_n_e', 'r_star', 't_0', 't_slab', 'tau']
    initial_conditions = np.zeros((n_walkers, n_params))

    for i, param in enumerate(params):
        low = config_data[param + '_l']
        high = config_data[param + '_u']

        #this will generate the initial condition close to middle of the interval
        initial_conditions[:, i] = np.random.normal(loc = 0.5*(low+high), scale = (high-low)/3, size=n_walkers)
    
    return initial_conditions


def total_spec(theta,wavelength):
    """
    Generates the model spectra by running ysopy for the given parameters in theta array
    theta is the parameter array
    returns normalized flux evaluated at the passed wavelength array
    """

    t_start = time.time()
    # modify config file, to run model
    # params = ['m', 'log_m_dot', 'b', 'inclination',  'log_n_e', 'r_star', 't_0', 't_slab', 'tau']
    config = bf.config_read('config_file.cfg')
    config['m'] = theta[0] * const.M_sun
    config['m_dot'] = 10**theta[1] * const.M_sun / (1 * u.year).to(u.s) ## Ensure the 10** here
    config['b'] = theta[2] * u.kilogauss
    config['inclination'] = theta[3] * u.degree
    config['n_e'] = 10**13 * u.cm**-3  ## Ensure the 10** here  # HARD CODE
    config['r_star'] = theta[5] * const.R_sun
    config['t_0'] = theta[6] * u.K
    config['t_slab'] = 8000 * u.K    # HARD CODE
    config['tau'] = 1.0   # HARD CODE
    
    #run model
    dr, t_max, d, r_in, r_sub = bf.generate_temp_arr(config)
    wave, obs_viscous_disk_flux = bf.generate_visc_flux(config, d, t_max, dr)
    obs_mag_flux = bf.magnetospheric_component(config, r_in)
    obs_dust_flux = bf.generate_dusty_disk_flux(config, r_in, r_sub)
    obs_star_flux = bf.generate_photosphere_flux(config)
    total_flux = bf.dust_extinction_flux(config, wave, obs_viscous_disk_flux, obs_star_flux, obs_mag_flux, obs_dust_flux)

    #interpolate to required wavelength
    func = interp1d(wave,total_flux)## CHECK if this works, for units
    result_spec = func(wavelength)
    result_spec /= np.median(result_spec)

    print(f"model run .. time taken {t_start - time.time()} s")

    return result_spec


def log_prior(theta):
    """
    Define uniform priors, this can even be skipped
    """
    config_data = config_reader('mcmc_config.cfg')
    params = ['m', 'log_m_dot', 'b', 'inclination',  'log_n_e', 'r_star', 't_0', 't_slab', 'tau']
    condition = True

    for i, param in enumerate(params):
        low = config_data[param + '_l']
        high = config_data[param + '_u']
        condition = condition and (low < theta[i] < high)

    if condition:
        return 0.0
    return -np.inf


def log_likelihood(theta):
    #y is of the form (wavelength,normalized_flux,normalized_err), where normalization is by the median flux
    wavelength = data[0]*u.AA
    model = total_spec(theta,wavelength)
    sigma2 = data[2]**2
    return -0.5 *( np.sum((data[1] - model) ** 2 / sigma2 + np.log(sigma2)) )


def log_probability(theta): # gives the posterior probability
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta)


def main(p0,nwalkers,niter,ndim,lnprob):

    print("Running burn-in...")
    with Pool(processes=8) as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, pool=pool)
        start = time.time()
        sampler.run_mcmc(p0, 100, progress=True)
        end = time.time()
        multi_time = end - start
        print("Multiprocessing took {0:.1f} seconds".format(multi_time))

        p0 = sampler.get_last_sample(p0, 100)
        print(p0)    # test line
        sampler.reset()

        print("Running production...")

        sampler.run_mcmc(p0, niter, progress=True)
    
    # get the chain
    params = sampler.get_chain(flat=True)

    return params


#read the data
path_to_valid = "../../../../validation_files/"
data = ascii.read(path_to_valid+'HIRES_sci_42767_1/KOA_42767/HIRES/extracted/tbl/ccd0/flux/HI.20030211.26428_0_02_flux.tbl.gz')
data = [data['wave'],data['Flux']/np.median(data['Flux']),data['Error']/np.median(data['Flux'])]
#vac to air correction for given data
wavelengths_air = wave.vactoair(data[0]*u.AA)
data[0] = wavelengths_air

plt.plot(data[0],data[1])
plt.show()

n_params = 9 # number of parameters that are varying
nwalkers = 20
niter = 200

# generate initial conditions
config_data = config_reader('mcmc_config.cfg')
p0 = generate_initial_conditions(config_data, nwalkers)
#print(f"initial conditions {p0}")

params = main(p0,nwalkers,niter,n_params,log_probability)
print("completed")