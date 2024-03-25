import sys
from configparser import ConfigParser
import argparse
import os
from pathlib import Path
from . import base_funcs as bf
import time
import matplotlib.pyplot as plt
####### Console function

def main():
    parser = argparse.ArgumentParser(description="Console Caller Function", add_help=True)
    # Function
    parser.add_argument('function', choices=['t_visc', 'visc_disk', 'dust_disk', 'photo', 'mag_shock', 'total'],
                        help="Parameter to call specific functionality of YSOpy functions")
    parser.add_argument('-p', '--boolean_plot', action="store_true", dest="plot",
                        help="Boolean to plot, True if flag called else False")
    parser.add_argument('-mod', '--mag_shock_model', choices=['hslab', 'blackbody'],
                        default='hslab', const='hslab', nargs='?',dest="mag_comp",
                        help="Parameter controlling which model to be used for magnetospheric shock SED generation. "
                             "Default: hslab, Const: hslab")
    arguments = parser.parse_args()
    # print(arguments)
    dictt = vars(arguments)
    print(dictt)

    """If the function is "t_visc", the program will run the script for generating the
    radial temperature profile for the viscously heated disk. If the -p (optional) 
    flag is used, the plot is also generated."""
    if dictt['function'] == 't_visc':
        st = time.time()
        dict_config = bf.config_read('config_file.cfg')
        if dictt['plot']:
            dict_config['plot'] = True
        else:
            dict_config['plot'] = False
            print("Here in t_visc, plot")
        dr, t_max, d, r_in, r_sub = bf.generate_temp_arr(dict_config)
        et = time.time()
        print("Process completed in {:.2f} seconds".format(et-st))

    """
    If the function "visc_disk" is used, then SED for the viscously heated disk 
    is calculated for given parameters in config_file.cfg. If the -p (optional) 
    flag is used, the plot is also generated.
    """
    if dictt['function'] == 'visc_disk':
        st = time.time()
        dict_config = bf.config_read('config_file.cfg')
        dr, t_max, d, r_in, r_sub = bf.generate_temp_arr(dict_config)
        if dictt['plot']:
            dict_config['plot'] = True
        else:
            dict_config['plot'] = False
        wavelength, obs_viscous_disk_flux = bf.generate_visc_flux(dict_config, d, t_max, dr)
        print('Viscous disk done')
        et = time.time()
        print("Process completed in {:.2f} seconds".format(et - st))

    """
    If the function "photo" is used, then SED for the stellar photosphere 
    is calculated for given parameters in config_file.cfg. If the -p (optional) 
    flag is used, the plot is also generated.
    """
    if dictt['function'] == 'photo':
        st = time.time()
        dict_config = bf.config_read('config_file.cfg')
        if dictt['plot']:
            dict_config['plot'] = True
        else:
            dict_config['plot'] = False
        obs_star_flux = bf.generate_photosphere_flux(dict_config)
        print("Photospheric component done")
        et = time.time()
        print("Process completed in {:.2f} seconds".format(et - st))

    """
    If the function "mag_shock" is used then SED for the magnetospheric shock 
    region on the stellar photosphere is calculated for given parameters in 
    config_file.cfg. If -mod (optional) flag is associated with "hslab" then 
    hydrogen slab model is used else if "blackbody" then blackbody is used.
    If the -p (optional) flag is used, the plot is also generated.
    """

    if dictt['function'] == 'mag_shock':
        st = time.time()
        dict_config = bf.config_read('config_file.cfg')
        dr, t_max, d, r_in, r_sub = bf.generate_temp_arr(dict_config)
        if dictt['plot']:
            dict_config['plot'] = True
        else:
            dict_config['plot'] = False
        if dictt['mag_comp'] == 'hslab':
            dict_config['mag_comp'] = 'hslab'
        else:
            dict_config['mag_comp'] = 'blackbody'
        obs_mag_flux = bf.magnetospheric_component(dict_config, r_in)
        print("Magnetic component done")
        et = time.time()
        print("Process completed in {:.2f} seconds".format(et - st))


    """
    If the function "dust_disk" is used, then SED for the dust dominated disk 
    is calculated for given parameters in config_file.cfg. If the -p (optional) 
    flag is used, the plot is also generated.
    """
    if dictt['function'] == 'dust_disk':
        st = time.time()
        dict_config = bf.config_read('config_file.cfg')
        dr, t_max, d, r_in, r_sub = bf.generate_temp_arr(dict_config)
        if dictt['plot']:
            dict_config['plot'] = True
        else:
            dict_config['plot'] = False
        obs_dust_flux = bf.generate_dusty_disk_flux(dict_config, r_in, r_sub)
        print("Dust component done")
        et = time.time()
        print("Process completed in {:.2f} seconds".format(et - st))

    """
    If the function "total" is used, then SED for the total YSo system 
    is calculated for given parameters in config_file.cfg. If the -p (optional) 
    flag is used, the plot is also generated. For magnetospheric component, 
    if -mod (optional) flag is associated with "hslab" then hydrogen slab 
    model is used else if "blackbody" then blackbody is used.
    """
    if dictt['function'] == 'total':
        st = time.time()
        dict_config = bf.config_read("config_file.cfg")
        dr, t_max, d, r_in, r_sub = bf.generate_temp_arr(dict_config)

        wavelength, obs_viscous_disk_flux = bf.generate_visc_flux(dict_config, d, t_max, dr)
        print('Viscous disk done')

        if dictt['mag_comp'] == 'hslab':
            dict_config['mag_comp'] = 'hslab'
        else:
            dict_config['mag_comp'] = 'blackbody'
        obs_mag_flux = bf.magnetospheric_component(dict_config, r_in)
        print("Magnetic component done")

        obs_dust_flux = bf.generate_dusty_disk_flux(dict_config, r_in, r_sub)
        print("Dust component done")

        obs_star_flux = bf.generate_photosphere_flux(dict_config)
        print("Photospheric component done")

        total_flux = bf.dust_extinction_flux(dict_config, wavelength, obs_viscous_disk_flux, obs_star_flux, obs_mag_flux,
                                             obs_dust_flux)
        print("Total component done")

        if dictt['plot']:
            dict_config['plot'] = True
        else:
            dict_config['plot'] = False
        et = time.time()
        print("Process completed in {:.2f} seconds".format(et - st))

        if dict_config['plot']:
            plt.plot(wavelength, obs_star_flux, label="Stellar photosphere")
            plt.plot(wavelength, total_flux, label="Total")
            plt.plot(wavelength, obs_viscous_disk_flux, label="Viscous Disk")
            plt.plot(wavelength, obs_mag_flux, label="Magnetosphere")
            plt.plot(wavelength, obs_dust_flux, label="Dusty disk")
            plt.legend()
            plt.xlabel("Wavelength [$\AA$] ----->")
            plt.ylabel("Flux [erg / (cm^2 s $\AA$)] ----->")
            plt.title("Predicted SED from the YSO system")
            plt.show()
