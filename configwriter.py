from configparser import ConfigParser

config = ConfigParser()

config['Default'] = {
    "b": 1,
    "m": 0.3,
    "m_dot": 4e-7,
    "r_star": 1.6,
    "inclination": 45,
    "d_star": 10,
    "t_star": 3400,
    "log_g_star": 3.5,
    "t_0": 3400,
    "av": 10,
    "rv": 3.1,
    "l_0": 3000,
    "mag_comp": "hslab",        # set the magnetosphere component to BLACKBODY or HSLAB
    "t_slab": 8000,
    "n_e": 1e15,
    "tau": 1.0,
    "save": True,
    "plot": True,
    "save_each": True,
    "bt_settl_path": r"/home/arch/yso/trial_downloads/",
    "h_data": r"/home/arch/yso/H_slab_data/GridHslab/temp_tau_vary",
    "h_min_data": r"/home/arch/yso/H_slab_data/GridH-slab/temp_tau_ne",
    "save_loc": r"/home/arch/yso/results/jul_26_run1",
    "l_min": 3000,
    "l_max": 50000,
    "n_data": 420000,
    "n_disk": 100,
    "r_cutoff": 5000
}

with open("config_file.das", "w") as f:
    config.write(f)
