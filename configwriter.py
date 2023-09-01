from configparser import ConfigParser

config = ConfigParser()

config['Default'] = {
    "b": 1,
    "m": 0.3,
    "m_dot": 4e-7,
    "r_star": 1.7,
    "log_g_star": 3.5,
    "inclination": 38,
    "d_star": 10,   # distance in parsec
    "t_star": 4600, # stellar photospheric temp
    "t_0": 4600,    # effective temp of radiation # req for calc of dust
    "av": 10,
    "rv": 3.1,
    "l_0": 3000, #
    "l_l_slab": 3000, # using in H Slab
    "mag_comp": "hslab",
    "t_slab": 8000,
    "n_e": 1e13,
    "tau": 1.0,
    "save": True,
    "plot": True,
    "bt_settl_path": r"/Users/tusharkantidas/NIUS/Temp",
    "save_loc": r"/Users/tusharkantidas/NIUS/testing/yso/Data/test3",
    "save_grid_data": True, # saving the grid for H slab
    "h_grid_loc": r"/Users/tusharkantidas/NIUS/refactoring/grid/h_emission/Sample",
    "h_min_grid_loc": r"/Users/tusharkantidas/NIUS/refactoring/grid/h_min_emission",
    "l_min": 3e3,
    "l_max": 5e4,
    "n_data": 420000, # number of points in the wavelength axis
    "n_disk": 50,
    "n_h_minus": 5000,
    "n_h": 5000
}

with open("config_file.cfg", "w") as f:
    config.write(f)
