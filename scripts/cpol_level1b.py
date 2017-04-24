"""
CPOL Level 1b main production line.

@title: CPOL_PROD_1b
@author: Valentin Louf <valentin.louf@monash.edu>
@institution: Bureau of Meteorology
@date: 22/04/2017
@version: 0.7

.. autosummary::
    :toctree: generated/

    timeout_handler
    chunks
    check_azimuth
    production_line
    production_line_manager
    main
"""
# Python Standard Library
import os
import sys
import glob
import time
import signal
import logging
import argparse
import datetime
import warnings
from multiprocessing import Pool

# Other Libraries -- Matplotlib must be imported first
import matplotlib
matplotlib.use('Agg')  # <- Reason why matplotlib is imported first.

import pyart
import netCDF4
import crayons  # For the welcoming message only.
import numpy as np
import pandas as pd

# Custom modules.
import plot_radar
import radar_codes
import raijin_tools
import gridding_codes


class TimeoutException(Exception):   # Custom exception class
    pass


def timeout_handler(signum, frame):   # Custom signal handler
    raise TimeoutException


def chunks(l, n):
    """
    Yield successive n-sized chunks from l.
    From http://stackoverflow.com/a/312464
    """
    for i in range(0, len(l), n):
        yield l[i:i + n]


def check_azimuth(radar, radar_file_name):
    """
    Checking if radar has a proper azimuth field.

    Parameters:
    ===========
        radar:
            Py-ART radar structure.
        radar_file_name: str
            Name of the input radar file.

    Return:
    =======
        is_good: bool
            True if radar has a proper azimuth field.
    """
    is_good = True
    azi = radar.azimuth['data']
    maxazi = np.max(azi)
    minazi = np.min(azi)

    if np.abs(maxazi - minazi) < 60:
        is_good = False
        # Keeping track of bad files:
        badfile = os.path.join(os.path.expanduser('~'), 'bad_radar_azimuth.txt')
        with open(badfile, 'a+') as fid:
            fid.write(radar_file_name + "\n")

    return is_good


def production_line(radar_file_name, outpath=None):
    """
    Production line for correcting and estimating CPOL data radar parameters.
    The naming convention for these parameters is assumed to be DBZ, ZDR, VEL,
    PHIDP, KDP, SNR, RHOHV, and NCP. KDP, NCP, and SNR are optional and can be
    recalculated.

    Parameters:
    ===========
        radar_file_name: str
            Name of the input radar file.
        outpath: str
            Path for saving output data.
    """
    # Create output file name and check if it already exists.
    outfilename = os.path.basename(radar_file_name)
    outfilename = outfilename.replace("level1a", "level1b")
    if outpath is None:
        outpath = os.path.expanduser('~')
    outfilename = os.path.join(outpath, outfilename)
    if os.path.isfile(outfilename):
        logger.error('Output file already exists for: %s.', outfilename)
        return None

    # Start chronometer.
    start_time = time.time()

    # Read the input radar file.
    try:
        radar = pyart.io.read(radar_file_name)
    except:
        logger.error("MAJOR ERROR: Can't read input file named {}".format(radar_file_name))
        return None

    # Check if radar is correct.
    if not check_azimuth(radar, radar_file_name):
        logger.error("MAJOR ERROR: %s does not have a proper azimuth.", radar_file_name)
        return None

    # Get radar's data date and time.
    radar_start_date = netCDF4.num2date(radar.time['data'][0], radar.time['units'])
    datestr = radar_start_date.strftime("%Y%m%d_%H%M")
    logger.info("%s read.", radar_file_name)

    # Check date, if velocity needs to be refolded.
    if radar_start_date.year > 2012:
        refold_velocity = True
        logger.info('PHIDP and VELOCITY will be refolded.')
    else:
        refold_velocity = False

    # Compute SNR
    height, temperature, snr = radar_codes.snr_and_sounding(radar, SOUND_DIR, 'DBZ')
    radar.add_field('sounding_temperature', temperature, replace_existing = True)
    radar.add_field('height', height, replace_existing = True)
    try:
        radar.fields['SNR']
        logger.info('SNR already exists.')
    except KeyError:
        radar.add_field('SNR', snr, replace_existing = True)
        logger.info('SNR calculated.')

    # Correct RHOHV
    rho_corr = radar_codes.correct_rhohv(radar)
    radar.add_field_like('RHOHV', 'RHOHV_CORR', rho_corr, replace_existing = True)
    logger.info('RHOHV corrected.')

    # Get filter
    gatefilter = radar_codes.do_gatefilter(radar)
    logger.info('Filter initialized.')

    # Correct ZDR
    corr_zdr = radar_codes.correct_zdr(radar)
    radar.add_field_like('ZDR', 'ZDR_CORR', corr_zdr, replace_existing=True)

    # Estimate KDP
    try:
        radar.fields['KDP']
        logger.info('KDP already exists')
    except KeyError:
        logger.info('We need to estimate KDP')
        kdp_con = radar_codes.estimate_kdp(radar, gatefilter)
        radar.add_field('KDP', kdp_con, replace_existing=True)
        logger.info('KDP estimated.')

    # Bringi PHIDP/KDP
    phidp_bringi, kdp_bringi = radar_codes.bringi_phidp_kdp(radar, gatefilter)
    radar.add_field_like('PHIDP', 'PHIDP_BRINGI', phidp_bringi, replace_existing=True)
    radar.add_field_like('KDP', 'KDP_BRINGI', kdp_bringi, replace_existing=True)
    # Correcting PHIDP and KDP Bringi's attributes.
    radar.fields['PHIDP_BRINGI']['long_name'] = "bringi_" + radar.fields['PHIDP_BRINGI']['long_name']
    radar.fields['KDP_BRINGI']['long_name'] = "bringi_" + radar.fields['KDP_BRINGI']['long_name']
    logger.info('KDP/PHIDP Bringi estimated.')

    # Unfold PHIDP, refold VELOCITY
    phidp_unfold, vdop_refolded = radar_codes.unfold_phidp_vdop(radar, unfold_vel=refold_velocity)
    if phidp_unfold is not None:
        logger.info('PHIDP has been unfolded.')
        radar.add_field_like('PHIDP', 'PHIDP_CORR', phidp_unfold, replace_existing=True)
    if vdop_refolded is not None:
        logger.info('Doppler velocity has been refolded.')
        radar.add_field_like('VEL', 'VEL_CORR', vdop_refolded, replace_existing=True)
        radar.fields['VEL_CORR']['long_name'] = radar.fields['VEL_CORR']['long_name'] + "_refolded"

    # Unfold VELOCITY
    # This function will check if a 'VEL_CORR' field exists anyway.
    vdop_unfold = radar_codes.unfold_velocity(radar, gatefilter)
    radar.add_field('VEL_UNFOLDED', vdop_unfold, replace_existing = True)
    logger.info('Doppler velocity unfolded.')

    # Correct Attenuation ZH
    atten_spec, zh_corr = radar_codes.correct_attenuation_zh(radar)
    radar.add_field_like('DBZ', 'DBZ_CORR', zh_corr, replace_existing=True)
    radar.add_field('specific_attenuation_zh', atten_spec, replace_existing=True)
    logger.info('Attenuation on reflectivity corrected.')

    # Correct Attenuation ZDR
    atten_spec_zdr, zdr_corr = radar_codes.correct_attenuation_zdr(radar)
    radar.add_field_like('ZDR', 'ZDR_CORR', zdr_corr, replace_existing=True)
    radar.add_field('specific_attenuation_zdr', atten_spec_zdr, replace_existing=True)
    logger.info('Attenuation on ZDR corrected.')

    # Hydrometeors classification
    hydro_class = radar_codes.hydrometeor_classification(radar)
    radar.add_field('HYDRO', hydro_class, replace_existing=True)
    logger.info('Hydrometeors classification estimated.')

    # Liquid/Ice Mass
    # We decided to not give these products.
    # liquid_water_mass, ice_mass = radar_codes.liquid_ice_mass(radar)
    # radar.add_field('LWC', liquid_water_mass)
    # radar.add_field('IWC', ice_mass)
    # logger.info('Liquid/Ice mass estimated.')

    # Treatment is finished!
    end_time = time.time()
    logger.info("Treatment for %s done in %0.2f seconds.", os.path.basename(outfilename), (end_time - start_time))

    # Plot check figure.
    logger.info('Plotting figure')
    plot_radar.plot_figure_check(radar, gatefilter, outfilename, radar_start_date)

    # Rename fields and remove unnecessary ones.
    radar.add_field('DBZ', radar.fields.pop('DBZ_CORR'), replace_existing=True)
    radar.add_field('RHOHV', radar.fields.pop('RHOHV_CORR'), replace_existing=True)
    radar.add_field('ZDR', radar.fields.pop('ZDR_CORR'), replace_existing=True)
    radar.add_field('VEL_RAW', radar.fields.pop('VEL'), replace_existing=True)
    radar.add_field('VEL', radar.fields.pop('VEL_UNFOLDED'), replace_existing=True)
    try:
        vdop_art = radar.fields['PHIDP_CORR']
        radar.add_field('PHIDP', radar.fields.pop('PHIDP_CORR'), replace_existing=True)
    except KeyError:
        pass

    # Hardcode mask
    for mykey in radar.fields:
        if mykey in ['sounding_temperature', 'height', 'SNR', 'NCP', 'HYDRO']:
            # Virgin fields that are left untouch.
            continue
        else:
            radar.fields[mykey]['data'] = radar_codes.filter_hardcoding(radar.fields[mykey]['data'], gatefilter)
    logger.info('Hardcoding gatefilter to Fields done.')

    # Write results
    pyart.io.write_cfradial(outfilename, radar, format='NETCDF4')
    save_time = time.time()
    logger.info('%s saved in %0.2f s.', os.path.basename(outfilename), (save_time - end_time))

    # Gridding (and saving)
    gridding_codes.gridding_radar_150km(radar, radar_start_date, outpath=OUTPATH_GRID)
    gridding_codes.gridding_radar_70km(radar, radar_start_date, outpath=OUTPATH_GRID)
    logger.info('Gridding done in %0.2f s.', (time.time() - save_time))

    # Processing finished!
    logger.info('%s processed in  %0.2f s.', os.path.basename(outfilename), (time.time() - start_time))

    return None


def production_line_manager(mydate):
    """
    The production line manager calls the production line and manages it ;-).
    It makes sure that input/output directories exist. This is where the
    multiprocessing is taken care of.

    INPATH and OUTPATH are global variables defined in __main__.

    Parameter:
    ==========
        mydate: datetime.datetime
            Radar data date for which we start the production.
    """
    year = str(mydate.year)
    datestr = mydate.strftime("%Y%m%d")
    indir = os.path.join(INPATH, year, datestr)
    outdir = os.path.join(OUTPATH, year, datestr)

    # Checking if input directory exists.
    if not os.path.exists(indir):
        logger.error("Input directory %s does not exist.", indir)
        return None

    # Checking if output directory exists. Creating them otherwise.
    if not os.path.isdir(os.path.join(OUTPATH, year)):
        os.mkdir(os.path.join(OUTPATH, year))
    if not os.path.isdir(outdir):
        os.mkdir(outdir)

    # List netcdf files in directory.
    flist = raijin_tools.get_files(indir)
    if len(flist) == 0:
        logger.error('%s empty.', indir)
        return None
    logger.info('%i files found for %s', len(flist), datestr)

    # Cutting the file list into smaller chunks. (The multiprocessing.Pool instance
    # is freed from memory, at each iteration of the main for loop).
    for flist_slice in chunks(flist, NCPU):
        # Because we use multiprocessing, we need to send a list of tuple as argument of Pool.starmap.
        args_list = [None]*len(flist_slice)  # yes, I like declaring empty array.
        for cnt, onefile in enumerate(flist_slice):
            args_list[cnt] = (onefile, outdir)

        # If we are stuck in the loop for more than TIME_BEFORE_DEATH seconds,
        # it will raise a TimeoutException that will kill the current process
        # and go to the next iteration.
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(TIME_BEFORE_DEATH)
        try:
            # Start multiprocessing.
            with Pool(NCPU) as pool:
                pool.starmap(production_line, args_list)
        except TimeoutException:
            # Treatment time was too long.
            print("TOO MUCH TIME SPENT FROM %s to %s " % (flist_slice[0], flist_slice[-1]))
            logger.error("TOO MUCH TIME SPENT FROM %s to %s " % (flist_slice[0], flist_slice[-1]))
            continue  # Go to next iteration.
        else:
            signal.alarm(0)

    return None


def main():
    """
    Just print a welcoming message and calls the production_line_manager.
    """
    # Start with a welcome message.
    print("#"*79)
    print("")
    print(" "*25 + crayons.red("CPOL Level 1b production line.", bold=True))
    print("")
    print("- Input data directory path is: " + crayons.yellow(INPATH))
    print("- Output data directory path is: " + crayons.yellow(OUTPATH))
    print("- Radiosounding directory path is: " + crayons.yellow(SOUND_DIR))
    print("- Figures will be saved in: " + crayons.yellow(FIGURE_CHECK_PATH))
    print("- Start date is: " + crayons.yellow(START_DATE))
    print("- End date is: " + crayons.yellow(END_DATE))
    print("- Log files can be found in: " + crayons.yellow(LOG_FILE_PATH))
    print("- Each subprocess has {}s of allowed time life before being killed.".format(TIME_BEFORE_DEATH))
    print("#"*79)
    print("")

    # Serious stuffs begin here.
    date_range = pd.date_range(START_DATE, END_DATE)
    # One date at a time.
    for thedate in date_range:
        try:
            production_line_manager(thedate)
        except Exception:
            # Keeping track of any exceptions that may happen.
            logger.exception("Received an error for %s", thedate.strftime("%Y%m%d"))

    return None


if __name__ == '__main__':
    """
    Global variables definition and logging file initialisation.
    """
    # Main global variables (Path directories).
    # Input radar data directory
    INPATH = "/g/data2/rr5/vhl548/CPOL_level_1/"
    # Output directory for CF/Radial PPIs
    OUTPATH = "/g/data2/rr5/vhl548/CPOL_PROD_1b/"
    # Output directory for GRIDDED netcdf data.
    OUTPATH_GRID = "/g/data2/rr5/vhl548/CPOL_PROD_1b/GRIDDED/"
    # Input directory for Radiosoundings (use my other script, named caprica to
    # download and format these datas).
    SOUND_DIR = "/g/data2/rr5/vhl548/soudings_netcdf/"
    # Output directory for verification figures.
    FIGURE_CHECK_PATH = "/g/data2/rr5/vhl548/CPOL_PROD_1b/FIGURE_CHECK/"
    # Output directory for log files.
    LOG_FILE_PATH = os.path.join(os.path.expanduser('~'), 'logfiles')
    # Time in seconds for which each subprocess is allowed to live.
    TIME_BEFORE_DEATH = 600 # seconds before killing process.

    # Check if paths exist.
    if not os.path.isdir(LOG_FILE_PATH):
        print("Creating log files directory: {}.".format(LOG_FILE_PATH))
        os.mkdir(LOG_FILE_PATH)
    if not os.path.isdir(OUTPATH_GRID):
        print("Creating output figures directory: {}.".format(OUTPATH_GRID))
        os.mkdir(OUTPATH_GRID)
    if not os.path.isdir(FIGURE_CHECK_PATH):
        print("Creating output figures directory: {}.".format(FIGURE_CHECK_PATH))
        os.mkdir(FIGURE_CHECK_PATH)
    if not os.path.isdir(SOUND_DIR):
        print("Radiosoundings directory does not exist (or invalid): {}.".format(SOUND_DIR))
        sys.exit()
    if not os.path.isdir(INPATH):
        print("Input data directory does not exist (or invalid): {}.".format(INPATH))
        sys.exit()

    # Parse arguments
    parser_description = "Leveling treatment of CPOL data from level 1a to level 1b."
    parser = argparse.ArgumentParser(description=parser_description)
    parser.add_argument(
        '-j',
        '--cpu',
        dest='ncpu',
        default=16,
        type=int,
        help='Number of process')
    parser.add_argument(
        '-s',
        '--start-date',
        dest='start_date',
        default=None,
        type=str,
        help='Starting date.')
    parser.add_argument(
        '-e',
        '--end-date',
        dest='end_date',
        default=None,
        type=str,
        help='Ending date.')

    args = parser.parse_args()
    NCPU = args.ncpu
    START_DATE = args.start_date
    END_DATE = args.end_date

    if not (START_DATE and END_DATE):
        parser.error("Starting and ending date required.")

    # Checking that dates are recognize.
    try:
        datetime.datetime.strptime(START_DATE, "%Y%m%d")
        datetime.datetime.strptime(END_DATE, "%Y%m%d")
    except:
        print("Did not understand the date format. Must be YYYYMMDD.")
        sys.exit()

    # Creating the general log file.
    logname = "cpol_level1b_from_{}_to_{}.log".format(START_DATE, END_DATE)
    log_file_name =  os.path.join(LOG_FILE_PATH, logname)
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        filename=log_file_name,
        filemode='a+')
    logger = logging.getLogger(__name__)

    with warnings.catch_warnings():
        # Just ignoring warning messages.
        warnings.simplefilter("ignore")
        main()
