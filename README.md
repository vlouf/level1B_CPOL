CPOL Level 1b
=============

Code for producing the CPOL level 1b. Note this code can easily be adapted to 
any other radar for processing as long as it can be read with Py-ART. The 
(mandatory) input radar fields are:
- DBZ (Reflectivity),
- VEL (Doppler velocity),
- WIDTH (Spectrum width),
- RHOHV (Cross correlation ratio),
- PHIDP (Differential phase).
 
Optionnal fields:
- SNR (signal to noise ratio)
- NCP (normalised coherent power)

All the other input fields will be copied and left untouched (except for the 
gridded data where only some selected fields are kept).

# How to use.

## Processing one file

To process one file at a time (ideal for testing and looking if everything 
works), use: onefile_process.py in the scripts/ directory:

`python onefile_process.py --input INPUT_FILE --output OUTPUT_DIRECTORY`

## Processing multiple files 

To process multiple file (using python multiprocessing capabilities), use 
raijin_multiproc_processing.py in the scripts/ directory. Change this 3 global 
variables in the if __main__ part of the script:

```
INPATH = # The root directory of your data (will recursively look into it)
OUTPATH = # The root directory for output data (will create directory structure)
SOUND_DIR = # Path to radiosounding data.
```

Then call:

`python raijin_multiproc_processing.py --start-date STARTING_DATE --end-date ENDING_DATE`

It will process all files from `INPATH` that have dates between `STARTING_DATE` 
and `ENDING_DATE`.

# The process:
The onefile_process.py and raijin_multiproc_processing.py only handles the 
directory navigation, creation, and looking for files part. When they find a 
file, they just send it to the `cpol_processing.py` that will do the actual 
processing work.

- Generate output file name. Check if output file already exists.
- Read input radar file.
- Check if radar file OK (no problem with azimuth and reflectivity).
- Get actual radar date.
- Check if NCP field exists (creating a fake one if it doesn't)
- Check if RHOHV field exists (creating a fake one if it doesn't)
- Compute SNR and temperature using radiosoundings.
- Correct RHOHV using Ryzhkov algorithm.
- Create gatefilter (remove noise and incorrect data).
- Correct ZDR using Ryzhkov algorithm.
- Process and unfold raw PHIDP using Giangrande's algorithm in Py-ART
- Unfold velocity using pyart.
- Compute attenuation for ZH.
- Compute attenuation for ZDR.
- Estimate Hydrometeors classification using csu toolbox.
- Estimate Rainfall rate using csu toolbox.
- Estimate DSD retrieval using csu toolbox.
- Removing fake/temporary fieds.
- Rename fields to pyart standard names.
- Plotting figure quicklooks.
- Hardcoding gatefilter.
- Writing output cf/radial file.
- Writing output gridded data.

# Output parameters

- specific_attenuation_reflectivity
- radar_estimated_rain_rate
- D0
- NW
- velocity
- region_dealias_velocity
- total_power
- corrected_reflectivity
- differential_reflectivity
- differential_phase
- spectrum_width
- temperature
- height
- unfolded_differential_phase
- specific_attenuation_differential_reflectivity
- cross_correlation_ratio
- corrected_differential_reflectivity
- corrected_differential_phase
- corrected_specific_differential_phase
- signal_to_noise_ratio

# Dependencies:

Works only with Python 3. Tested on Python 3.5 and 3.6.

## Libraries
- [Py-ART][1]
- [Numpy][2]
- [Pandas][3]
- [Crayons][4]
- [netCDF4][5]
- [matplotlib][6]

[1]: http://github.com/ARM-DOE/pyart
[2]: http://www.scipy.org/
[3]: http://pandas.pydata.org/
[4]: https://pypi.python.org/pypi/crayons
[5]: https://github.com/unidata/netcdf4-python/
[6]: https://matplotlib.org/
