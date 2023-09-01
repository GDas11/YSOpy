# YSOSpectrum
*******************************
This Python package is currently in the DEVELOPMENTAL STAGE.
*******************************
We are currently working on bringing a new Python package to study and characterise the spectrum of young stellar objects (YSOs).
This pipeline currently looks into four aspects of the spectrum:-
- Viscously Heated Disk
- Magnetospheric Accretion
- Dusty Disk
- Stellar Photosphere

## Viscously heated disk:-
This is the most important part of the spectrum as it is the dominant component contributing to the flux. 
Most of the functions required for this component are in ```base_funcs.py```.
First we have to generate the temperature distribution with change in radius which is done by ```temp_visc()``` and ```generate_temp_arr()```.
Then we use ```read_bt_settl()``` to extract the flux data from the BT-Settl Model of Theoretical Spectra.
In our calculations the data generated is not evenly distributed across all wavelengths hence we have to interpolate them in certain ways which we accomplish by using three different interpolation functions namely ```unif_reinterpolate()```, ```interpolate_conv()``` and ```logspace_reinterp()```.
Now in order to capture the rotational broadening of the disk we have to convolve the flux values with a kernel. we define and implement the kernel in ```ker()``` and ```generate_kernel()``` respectively.
Finally we are having a function named ```generate_visc_flux()``` which is ultimately generating the convolved flux of the viscous disk.


## Magnetospheric accretion
For this component we are making use of 3 files ```base_funcs.py```, ```h_emission_refac.py``` and ```H-gen_file.py```.
As the names suggest we are generating the grids for H component and H- component of slab model using ```h_emission_refac.py``` and ```H-gen_file.py``` respectively.
Now using these grids we are using the function```cos_gamma_func()``` and ```magnetospheric_component()``` in ```base_funcs.py``` we are calculating the total flux due to this component.

## Dusty Disk
We are using two functions in ```base_funcs.py``` namely ```t_eff_dust()``` and ```generate_dusty_disk_flux()``` to get the dusty component of radiation.

## Stellar Photosphere
We are using ```generate_photosphere_flux()``` from ```base_funcs.py``` to generate the stellar photoshpheric flux.
