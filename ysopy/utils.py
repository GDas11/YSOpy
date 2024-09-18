import numpy as np
from astropy.io import VOtable

def convert_to_npy(config):
    """
    Convert the stored BT Settl models from VOtable to npy format,
     for faster reading during the fitting process
    Parameters
    ----------
    path

    Returns
    -------

    """
