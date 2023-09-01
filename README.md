# YSOSpectrum
*******************************
This Python package is currently in the DEVELOPMENTAL STAGE.
*******************************
We are currently working on bringing a new Python package to study and characterise the spectrum of young stellar objects (YSOs).
This pipeline currently looks into four aspects of the spectrum:-
- Viscously Heated Disk
- Dusty Disk
- Magnetospheric Accretion
- Stellar Photosphere

## Viscously heated disk:-
This is the most important part of the spectrum as it is the dominant component contributing to the flux. 
Most of the functions required for this component are in ```base_funcs.py```.
First we have to generate the temperature distribution with change in radius which is done by ```temp_visc()``` and ```generate_temp_arr()```.
Then we use ```read_bt_settl()``` to extract the flux data from the BT-Settl Model of Theoretical Spectra.
In our calculations the data generated is not evenly distributed across all wavelengths hence we have to interpolate them in certain ways which we accomplish by using three different interpolation functions namely ```unif_reinterpolate()```, ```interpolate_conv()``` and ```logspace_reinterp()```
