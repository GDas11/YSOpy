from configparser import ConfigParser

config = ConfigParser()

config['Default'] = {
    # stellar params
    "b": 1,
    "m": 1,
    "m_dot": 4e-7,
    "r_star": 1.7,
    "log_g_star": 3.5,
    "inclination": 38,
    "d_star": 10,   # distance in parsec
    "t_star": 4600, # stellar photospheric temp
    "t_0": 4600,    # effective temp of radiation # req for calc of dust
    "av": 10,
    "rv": 3.1,
    # for h slab part
    "l_0": 3000, #
    "l_l_slab": 3000, # ref wavelength used in calculating the length of the slab
    "mag_comp": "hslab",
    "t_slab": 8500,
    "n_e": 1e13,
    "tau": 1.0,
    # Saving, plotting booleans, Addresses of some resources
    "save": False,
    "plot": True,
    "bt_settl_path": r"/Users/tusharkantidas/NIUS/Temp",
    "save_loc": r"/Users/tusharkantidas/NIUS/testing/Contribution/Planetesimal2",
    "save_grid_data": False, # saving the grid for H slab
    "save_each": False, # for each annuli saving
    "h_grid_path":  r"/Users/tusharkantidas/NIUS/refactoring/grid/test_details",#r"/Users/tusharkantidas/NIUS/refactoring/grid/h_emission/Sample",
    "h_min_grid_path": r"/Users/tusharkantidas/NIUS/refactoring/grid/test_details", # r"/Users/tusharkantidas/NIUS/refactoring/grid/h_min_emission",
    # Wavelength domain parameters
    "l_min": 1250,
    "l_max": 5e4,
    "n_data": 420000,  # number of points in the wavelength axis
    # number of annuli in the disk
    "n_disk": 500,  # viscous disk
    "n_dust_disk": 2000,  # dust disk
    "n_h_minus": 5000,  # wavelength axis size for h minus emission calc
    "n_h": 150  # same for h emission. Small because of multiprocess happening and that variation is not much
}

with open("config_file.cfg", "w") as f:
    config.write(f)
