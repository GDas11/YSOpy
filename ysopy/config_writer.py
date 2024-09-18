from configparser import ConfigParser

config = ConfigParser()

config['Parameters'] = {
    "l_min": 3e3,
    "l_max": 5e4,
    "n_data": 420000,  # number of points in the wavelength axis
    "b": 1,
    "m": 0.3,
    "m_dot": 4e-7,
    "r_star": 1.7,
    "inclination": 38,
    "n_disk": 50,  # what does this do?
    "n_dust_disk": 50,
    "d_star": 10,   # distance in parsec
    "t_star": 4600,  # stellar photospheric temp
    "log_g_star": 3.5,
    "t_0": 4600,    # effective temp of radiation # req for calc of dust
    "av": 10,
    "rv": 3.1,
    "l_0": 3000,
    "mag_comp": "hslab",
    "t_slab": 8000,
    "n_e": 1e13,
    "tau": 1.0,
    "save": True,
    "plot": True,
    "bt_settl_path": r"home/arch/yso/trial_downloads",
    "save_loc": r"home/arch/yso/results/work_try",
    "save_grid_data": True, # saving the grid for H slab
    "h_grid_path": r"/home/arch/yso/",
    "h_min_grid_path": r"/home/arch/yso",

    "n_h_minus": 5000,
    "n_h": 5000,
    "l_l_slab": 3000  # using in H Slab
}

with open("config_file.cfg", "w") as f:
    config.write(f)