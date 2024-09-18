from . import base_funcs as bf
from configparser import ConfigParser
import argparse
import os
from pathlib import Path
def intro():
    print("YSOpy package up and running\nCredits: Gautam Das and Archis Mukhopadhyay\nPI: Joe P. Ninan")

####### Config Generator function
def main():
    parser = argparse.ArgumentParser(description="Config File Generator Caller Function", add_help=True)
    # "b": 1
    parser.add_argument("-b", "--B_field", metavar="b", help="Magnetic field in units of kiloGauss",
                        type=float, default=1.0, const=1.0, nargs='?', dest="b")
    # "m": 1
    parser.add_argument("-m", "--stellar_mass", metavar="m", help="Stellar mass in units of SolarMass",
                        type=float, default=1.0, const=1.0, nargs='?', dest="m")
    # "m_dot": 4e-7
    parser.add_argument("-m_dot", "--accretion_rate", metavar="m_dot",
                        help="Mass accretion rate in units of SolarMass per year", type=float, default=4e-7,
                        const=4e-7, nargs='?', dest="m_dot")
    # "r_star": 1.7,
    parser.add_argument("-r_s", "--stellar_radius", metavar="r_star",
                        help="Radius of the star in units of SolarRadius", type=float, default=1.7,
                        const=1.7, nargs='?', dest="r_star")

    # "log_g_star": 3.5,
    parser.add_argument("-logg", "--logg", metavar="log_g_star",
                        help="Log g of the stellar surface. Assuming a red giant of low surface "
                             "gravity we have default value as 3.5", type=float, default=3.5,
                        const=3.5, nargs='?', dest="log_g_star")

    # "inclination": 38,
    parser.add_argument("-i", "--inclination_angle", metavar="i",
                        help="Inclination angle (in degrees) of the observer with respect to the YSO system"
                        , type=float, default=38, const=38, nargs='?', dest="inclination")

    # "d_star": 10,   # distance in parsec
    parser.add_argument("-d_star", "--distance_of_star", metavar="dist",
                        help="Distance of the YSO from earth in units of parsec."
                             "Not for convenience we have taken the distance at 10 pc.", type=float, default=10,
                        const=10, nargs='?', dest="d_star")

    # "t_star": 4600 # stellar photospheric temp
    parser.add_argument("-t_star", "--photospheric_temperature", metavar="t_photo",
                        help="Stellar photospheric temperature in units of Kelvin.", type=float, default=4600,
                        const=4600, nargs='?', dest="t_star")

    # "t_0": 4600  # effective temp of radiation # req for calc of dust
    parser.add_argument("-t_0", "--effective_temperature", metavar="t_eff",
                        help="Effective temperature of radiation reaching the "
                             "dusty disk in units of Kelvin.", type=float, default=4600,
                        const=4600, nargs='?', dest="t_0")

    # "av": 10
    parser.add_argument("-av", "--av", metavar="av",
                        help="Extinction", type=float, default=10,
                        const=10, nargs='?', dest="av")

    # "rv": 3.1
    parser.add_argument("-rv", "--rv", metavar="rv",
                        help="Extinction", type=float, default=3.1,
                        const=3.1, nargs='?', dest="rv")

    # "l_0": 3000,  # initial value of wavelength at which the kernel is calculated? Helps in calculating window of kernel
    parser.add_argument("-l_0", "--lambda_for_kernel_window", metavar="lkw",
                        help="initial value of wavelength at which the kernel is calculated? "
                             "Helps in calculating window of kernel Default:- 3000 A", type=float, default=3000,
                        const=3000, nargs='?', dest="l_0")

    # "l_l_slab": 3000 # ref wavelength used in calculating the length of the slab
    parser.add_argument("-lls", "--lambda_for_l_slab", metavar="lls",
                        help="The wavelength in units of angstrom at which the length"
                             "of the hydrogen slab is calculated. Default:- 3000 A", type=float, default=3000,
                        const=3000, nargs='?', dest="l_l_slab")
    # "mag_comp": "hslab"
    parser.add_argument('-mag', '--magnetospheric_component', choices=['hslab', 'blackbody'],
                        default='hslab', help="For the magnetospheric funnel accretion, "
                                              "we can model the accretion shock region using either a blackbody "
                                              "or a hydrogen slab. This parameter gives the option of implementing"
                                              " either. Default:- hslab", dest="mag_comp")
    # "t_slab": 8500
    parser.add_argument("-t_s", "--temperature_slab", metavar="ts",
                        help="Temperature of the hydrogen slab in units of K. Default:- 8500K", type=float, default=8500,
                        const=8500, nargs='?', dest="t_slab")

    # "n_e": 1e13
    parser.add_argument("-n_e", "--electron_density", metavar="ne",
                        help="Density of electrons in H slab. "
                             "Valid density range [10**11, 10**16], Default:- 1e13", type=float, default=1e13,
                        const=1e13, nargs='?', dest="n_e")
    # "tau": 1.0
    parser.add_argument("-tau", "--optical_depth_slab", metavar="t",
                        help="The optical depth of H-slab model. Valid range [0.1, 5]. Default:- 1", type=float,
                        default=1.0, const=1.0, nargs='?', dest="tau")

    # booleans manipulations
    # "save": False
    parser.add_argument('-s', '--boolean_save', action="store_true", dest="save")
    # "plot": True
    parser.add_argument('-p', '--boolean_plot', action="store_true", dest="plot")
    # "save_grid_data": False  # saving the grid for H slab
    parser.add_argument('-s_grid', '--save_grid_data', action="store_true", dest="save_grid_data")
    # "save_each": False  # for each annuli saving
    parser.add_argument('-s_e', '--save_each', action="store_true", dest="save_each")

    # # Wavelength domain parameters
    # "l_min": 1250,
    parser.add_argument("-l_min", "--lambda_minimum", metavar="l",
                        help="Minimum value of wavelength in angstrom. Default:- 1250", type=float,
                        default=1250, const=1250, nargs='?', dest="l_min")

    # "l_max": 5e4,
    parser.add_argument("-l_max", "--lambda_maximum", metavar="l",
                        help="Maximum value of wavelength in angstrom. Default:- 50000", type=float,
                        default=50000, const=50000, nargs='?', dest="l_max")

    # "n_data": 420000,  # number of points in the wavelength axis
    parser.add_argument("-n_data", "--number_of_data_points_disk", metavar="n",
                        help="Number of datapoints in wavelength axis. Default:- 420000", type=int,
                        default=420000, const=420000, nargs='?', dest="n_data")

    # # number of annuli in the disk
    # "n_disk": 500,  # viscous disk
    parser.add_argument("-n_disk", "--number_of_viscous_disk_annuli", metavar="n",
                        help="Number of viscous disk annuli. Default:- 500", type=int,
                        default=500, const=500, nargs='?', dest="n_disk")

    # "n_dust_disk": 2000,  # dust disk
    parser.add_argument("-n_dust", "--number_of_dusty_disk_annuli", metavar="n",
                        help="Number of dusty disk annuli. Default:- 2000", type=int,
                        default=2000, const=2000, nargs='?', dest="n_dust_disk")

    # "n_h_minus": 5000,  # wavelength axis size for h minus emission calc
    parser.add_argument("-n_h_m", "--number_of_data_points_h_min", metavar="n",
                        help="Number of datapoints in wavelength axis. Default:- 5000", type=int,
                        default=5000, const=5000, nargs='?', dest="n_h_minus")

    # "n_h": 150  # same for h emission. Small because of multiprocess happening and that variation is not much
    parser.add_argument("-n_h", "--number_of_data_points_h", metavar="n",
                        help="Number of datapoints in wavelength axis. Default:- 150", type=int,
                        default=150, const=150, nargs='?', dest="n_h")


    # "save_loc": r"/Users/tusharkantidas/NIUS/testing/Contribution/Planetesimal2",
    # "h_grid_path":  r"/Users/tusharkantidas/NIUS/refactoring/grid/test_details",#r"/Users/tusharkantidas/NIUS/refactoring/grid/h_emission/Sample",
    # "h_min_grid_path": r"/Users/tusharkantidas/NIUS/refactoring/grid/test_details", # r"/Users/tusharkantidas/NIUS/refactoring/grid/h_min_emission",
    # "h_grid_loc": r"/Users/tusharkantidas/NIUS/refactoring/grid/h_emission/Sample",
    # "h_min_grid_loc": r"/Users/tusharkantidas/NIUS/refactoring/grid/h_min_emission",

    my_file = Path("config_file.cfg")


    def dir_path(string):
        if os.path.isdir(string):
            return string
        else:
            raise NotADirectoryError(string)


    if my_file.is_file():
        config = bf.config_read("config_file.cfg")
        # bt_settl_path = /Users/tusharkantidas/NIUS/Temp
        parser.add_argument('-p_bt', '--path_to_bt_settl_grid', metavar="path", type=dir_path,
                            dest="bt_settl_path", default=config["bt_settl_path"], const=config["bt_settl_path"], nargs='?')
        # "save_loc": r"/Users/tusharkantidas/NIUS/testing/Contribution/Planetesimal2",
        parser.add_argument('-p_s', '--path_to_saving_loc', metavar="path", type=dir_path, dest="save_loc",
                            default=config["save_loc"], const=config["save_loc"], nargs='?')
        # "h_grid_path":  r"/Users/tusharkantidas/NIUS/refactoring/grid/test_details",#r"/Users/tusharkantidas/NIUS/refactoring/grid/h_emission/Sample",
        parser.add_argument('-p_h', '--path_to_h_grid', metavar="path", type=dir_path, dest="h_grid_path",
                            default=config["h_grid_path"], const=config["h_grid_path"], nargs='?')
        # "h_min_grid_path": r"/Users/tusharkantidas/NIUS/refactoring/grid/test_details", # r"/Users/tusharkantidas/NIUS/refactoring/grid/h_min_emission",
        parser.add_argument('-p_hm', '--path_to_h_minus_grid', metavar="path", type=dir_path,
                            dest="h_min_grid_path",default=config["h_min_grid_path"],
                            const=config["h_min_grid_path"], nargs='?')
    else:
        print("************** Instruction to generate the config file **************\n\n"
              "Since this is the first time config_file.cfg being created, You have to enter \nthe paths containing"
              " some required files in this particular order as described below:\n\n"
              "\n---> Bt Settl Path:\n\tPath to directory where the .xml files are stored.\n\tThese are to be "
              "downloaded from internet at\n\thttp://svo2.cab.inta-csic.es/theory/newov2/index.php?models=bt-settl-agss\n"
              "\n---> Saving location:\n\tPath to directory where all the output files\n\tlike "
              "plots, SEDs generated etc. should be stored\n"
              "\n---> H emission grid location:\n\tDirectory Path where the grid of H emission SEDs\n\tgenerated for "
              "different temperatures is stored.\n"
              "\n---> H minus emission grid location:\n\tDirectory Path where the grid of H minus emission SEDs\n\tgenerated for "
              "different temperatures (t_slab), different electron\n\tdensity (n_e), different optical depth "
              "(tau) is stored\n\n"
              "The console input should be like\n"
              "\t$ python config_generator.py <bt_s_path> <save_path> <h_path> <h_m_path>\n")
        # bt_settl_path = /Users/tusharkantidas/NIUS/Temp
        parser.add_argument('bt_settl_path', metavar="bt_s_path", type=dir_path)
        # "save_loc": r"/Users/tusharkantidas/NIUS/testing/Contribution/Planetesimal2",
        parser.add_argument('save_loc', metavar="save_path", type=dir_path)
        # "h_grid_path":  r"/Users/tusharkantidas/NIUS/refactoring/grid/test_details",#r"/Users/tusharkantidas/NIUS/refactoring/grid/h_emission/Sample",
        parser.add_argument('h_grid_path', metavar="h_path", type=dir_path)
        # "h_min_grid_path": r"/Users/tusharkantidas/NIUS/refactoring/grid/test_details", # r"/Users/tusharkantidas/NIUS/refactoring/grid/h_min_emission",
        parser.add_argument('h_min_grid_path', metavar="h_m_path", type=dir_path)

    arguments = parser.parse_args()
    print(arguments)
    dictt = vars(arguments)
    print(dictt)
    config = ConfigParser()
    config["Default"] = dictt
    with open("config_file.cfg", "w") as f:
        config.write(f)
