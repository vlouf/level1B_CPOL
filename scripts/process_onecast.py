"""
CPOL Level 1b main production line.

@title: CPOL_PROD_1b
@author: Valentin Louf <valentin.louf@monash.edu>
@institution: Bureau of Meteorology
@date: 24/04/2017
@version: 0.8

.. autosummary::
    :toctree: generated/

    timeout_handler
    chunks
    production_line
    production_line_manager
    main
"""
# Python Standard Library
import os
import sys
import time
import signal
import argparse
import datetime
import warnings

# Other Libraries -- Matplotlib must be imported first
import matplotlib
matplotlib.use('Agg')  # <- Reason why matplotlib is imported first.
import matplotlib.colors as colors
import matplotlib.pyplot as pl

import pyart
import netCDF4
import crayons  # For the welcoming message only.
import numpy as np
import pandas as pd

# Custom modules.
from processing_codes import radar_codes
from processing_codes import atten_codes
from processing_codes import phase_codes
from processing_codes import raijin_tools
from processing_codes import gridding_codes


def plot_figure_check(radar, gatefilter, outfilename, radar_date):
    """
    Plot figure of old/new radar parameters for checking purpose.

    Parameters:
    ===========
        radar:
            Py-ART radar structure.
        gatefilter:
            The Gate filter.
        outfilename: str
            Name given to the output netcdf data file.
        radar_date: datetime
            Datetime stucture of the radar data.
    """
    def adjust_fhc_colorbar_for_pyart(cb):
        """
        adjust_fhc_colorbar_for_pyart
        """
        cb.set_ticks(np.arange(1.4, 10, 0.9))
        cb.ax.set_yticklabels(['Drizzle', 'Rain', 'Ice Crystals', 'Aggregates',
                               'Wet Snow', 'Vertical Ice', 'LD Graupel',
                               'HD Graupel', 'Hail', 'Big Drops'])
        cb.ax.set_ylabel('')
        cb.ax.tick_params(length=0)
        return cb

    # Extracting year and date.
    year = str(radar_date.year)
    datestr = radar_date.strftime("%Y%m%d")
    # Path for saving Figures.
    outfile_path = os.path.join(FIGURE_CHECK_PATH, year, datestr)

    # Checking if output directory exists. Creating them otherwise.
    if not os.path.isdir(os.path.join(FIGURE_CHECK_PATH, year)):
        os.mkdir(os.path.join(FIGURE_CHECK_PATH, year))
    if not os.path.isdir(outfile_path):
        os.mkdir(outfile_path)

    # Checking if figure already exists.
    outfile = os.path.basename(outfilename)
    outfile = outfile[:-2] + "png"
    outfile = os.path.join(outfile_path, outfile)
    # if os.path.isfile(outfile):
    #     return None

    # Initializing figure.
    gr = pyart.graph.RadarDisplay(radar)
    fig, the_ax = pl.subplots(6, 2, figsize=(12, 30), sharex=True, sharey=True)
    the_ax = the_ax.flatten()
    # Plotting reflectivity
    gr.plot_ppi('DBZ', ax = the_ax[0], vmin=-10, vmax=70)
    gr.plot_ppi('DBZ_CORR', ax = the_ax[1], gatefilter=gatefilter, cmap=pyart.graph.cm.NWSRef, vmin=-10, vmax=70)

    gr.plot_ppi('ZDR', ax = the_ax[2], vmin=-5, vmax=10, cmap='rainbow')  # ZDR
    gr.plot_ppi('ZDR_CORR', ax = the_ax[3], gatefilter=gatefilter, vmin=-5, vmax=10, cmap='rainbow')

    gr.plot_ppi('PHIDP', ax = the_ax[4], vmin=0, vmax=180, cmap='OrRd')
    try:
        gr.plot_ppi('PHIDP_CORR', ax = the_ax[5], gatefilter=gatefilter, vmin=0, vmax=180, cmap='OrRd')
    except KeyError:
        gr.plot_ppi('PHIDP', ax = the_ax[5], gatefilter=gatefilter, vmin=0, vmax=180, cmap='OrRd')

    gr.plot_ppi('VEL', ax = the_ax[6], cmap=pyart.graph.cm.NWSVel, vmin=-40, vmax=40)
    gr.plot_ppi('VEL_UNFOLDED', ax = the_ax[7], gatefilter=gatefilter, cmap=pyart.graph.cm.NWSVel, vmin=-40, vmax=40)

    gr.plot_ppi('RHOHV', ax = the_ax[8], vmin=0, vmax=1, norm=colors.LogNorm(vmin=0.4, vmax=1), cmap='rainbow')
    gr.plot_ppi('RHOHV_CORR', ax = the_ax[9], vmin=0, vmax=1, norm=colors.LogNorm(vmin=0.4, vmax=1), cmap='rainbow')

    gr.plot_ppi('SNR', ax = the_ax[10], cmap='OrRd', vmin=0, vmax=80)
    gr.plot_ppi('TEXTURE', ax = the_ax[11], vmin=0, vmax=10, cmap='jet')

    for ax_sl in the_ax:
        gr.plot_range_rings([50, 100, 150], ax=ax_sl)
        ax_sl.set_aspect(1)
        ax_sl.set_xlim(-150, 150)
        ax_sl.set_ylim(-150, 150)

    pl.tight_layout()
    pl.savefig(outfile)  # Saving figure.
    pl.close()

    # HYDRO CLASSIFICATION
    hid_colors = ['White', 'LightBlue', 'MediumBlue', 'DarkOrange', 'LightPink',
                  'Cyan', 'DarkGray', 'Lime', 'Yellow', 'Red', 'Fuchsia']
    cmaphid = colors.ListedColormap(hid_colors)

    fig, ax0 = pl.subplots(1, 1, figsize = (6, 5))
    gr.plot_ppi('HYDRO', vmin=0, vmax=10, cmap=cmaphid)
    gr.cbs[-1] = adjust_fhc_colorbar_for_pyart(gr.cbs[-1])
    pl.savefig(outfile.replace(".png", "_HYDROCLASS.png"))

    return None


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
    # Generate output file name.
    outfilename = os.path.basename(radar_file_name)
    outfilename = outfilename.replace("level1a", "level1b")
    if outpath is None:
        outpath = os.path.expanduser('~')
    outfilename = os.path.join(outpath, outfilename)

    # Check if output file already exists.
    if os.path.isfile(outfilename):
        logger.error('Output file already exists for: %s.', outfilename)
        return None

    # Start chronometer.
    start_time = time.time()

    # Read the input radar file.
    try:
        radar = pyart.io.read(radar_file_name)
    except:
        logger.error("MAJOR ERROR: unable to read input file {}".format(radar_file_name))
        return None

    # Check if radar scan is complete.
    if not radar_codes.check_azimuth(radar):
        logger.error("MAJOR ERROR: %s does not have a proper azimuth.", radar_file_name)
        return None
    # Check if radar reflecitivity field is correct.
    if not radar_codes.check_reflectivity(radar):
        logger.error("MAJOR ERROR: %s reflectivity field is empty.", radar_file_name)
        return None

    # Get radar's data date and time.
    radar_start_date = netCDF4.num2date(radar.time['data'][0], radar.time['units'])
    datestr = radar_start_date.strftime("%Y%m%d_%H%M")
    logger.info("%s read.", radar_file_name)

    # Compute SNR
    try:
        height, temperature, snr = radar_codes.snr_and_sounding(radar, SOUND_DIR)
    except ValueError:
        logger.error("Impossible to compute SNR")
        return None
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

    # Get velocity field texture and noise threshold. (~3min to execute this function)
    vel_texture, noise_threshold = radar_codes.get_texture(radar)
    radar.add_field_like('VEL', 'TEXTURE', vel_texture, replace_existing = True)
    logger.info('Texture computed.')

    # Get filter
    gatefilter = radar_codes.do_gatefilter(radar, noise_threshold, rhohv_name='RHOHV_CORR')
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
        kdp_con = phase_codes.estimate_kdp(radar, gatefilter)
        radar.add_field('KDP', kdp_con, replace_existing=True)
        logger.info('KDP estimated.')

    # Giangrande PHIDP/KDP
    phidp_gg, kdp_gg = phase_codes.phidp_giangrande(radar)
    radar.add_field('PHIDP_GG', phidp_gg, replace_existing=True)
    radar.add_field('KDP_GG', kdp_gg, replace_existing=True)
    radar.fields['PHIDP_GG']['long_name'] = "giangrande_" + radar.fields['PHIDP_GG']['long_name']
    radar.fields['KDP_GG']['long_name'] = "giangrande_" + radar.fields['KDP_GG']['long_name']
    logger.info('KDP/PHIDP Giangrande estimated.')

    # Bringi PHIDP/KDP
    phidp_bringi, kdp_bringi = phase_codes.bringi_phidp_kdp(radar, gatefilter)
    radar.add_field_like('PHIDP', 'PHIDP_BRINGI', phidp_bringi, replace_existing=True)
    radar.add_field_like('KDP', 'KDP_BRINGI', kdp_bringi, replace_existing=True)
    # Correcting PHIDP and KDP Bringi's attributes.
    radar.fields['PHIDP_BRINGI']['long_name'] = "bringi_" + radar.fields['PHIDP_BRINGI']['long_name']
    radar.fields['KDP_BRINGI']['long_name'] = "bringi_" + radar.fields['KDP_BRINGI']['long_name']
    logger.info('KDP/PHIDP Bringi estimated.')

    # Unfold PHIDP, refold VELOCITY
    phidp_unfold, vdop_refolded = phase_codes.unfold_phidp_vdop(radar)
    if phidp_unfold is not None:
        logger.info('PHIDP has been unfolded.')
        radar.add_field_like('PHIDP', 'PHIDP_CORR', phidp_unfold, replace_existing=True)
    # Check if velocity was refolded.
    if vdop_refolded is None:
        refold_velocity = False
    else:
        refold_velocity = True
        logger.info('Doppler velocity has been refolded.')
        radar.add_field_like('VEL', 'VEL_CORR', vdop_refolded, replace_existing=True)
        radar.fields['VEL_CORR']['long_name'] = radar.fields['VEL_CORR']['long_name'] + "_refolded"

    # Unfold VELOCITY
    # This function will check if a 'VEL_CORR' field exists anyway.
    vdop_unfold = radar_codes.unfold_velocity(radar, gatefilter, bobby_params=refold_velocity)
    radar.add_field('VEL_UNFOLDED', vdop_unfold, replace_existing = True)
    logger.info('Doppler velocity unfolded.')

    # Correct Attenuation ZH
    atten_spec, zh_corr = atten_codes.correct_attenuation_zh(radar)
    radar.add_field_like('DBZ', 'DBZ_CORR', zh_corr, replace_existing=True)
    radar.add_field('specific_attenuation_zh', atten_spec, replace_existing=True)
    logger.info('Attenuation on reflectivity corrected.')

    # Correct Attenuation ZDR
    atten_spec_zdr, zdr_corr = atten_codes.correct_attenuation_zdr(radar)
    radar.add_field_like('ZDR', 'ZDR_CORR', zdr_corr, replace_existing=True)
    radar.add_field('specific_attenuation_zdr', atten_spec_zdr, replace_existing=True)
    logger.info('Attenuation on ZDR corrected.')

    # Hydrometeors classification
    hydro_class = radar_codes.hydrometeor_classification(radar)
    radar.add_field('HYDRO', hydro_class, replace_existing=True)
    logger.info('Hydrometeors classification estimated.')
    # Check if Hail it found hail.
    if (hydro_class['data'] == 9).sum() != 0:
        print("WARNING: hail detection in Darwin. NOT POSSIBLE!", os.path.basename(outfilename))
        fout = os.path.join(os.path.expanduser('~'), "hail_detection.txt")
        with open(fout, "a+") as fid:
            fid.write(os.path.basename(outfilename) + "\n")

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
    plot_figure_check(radar, gatefilter, outfilename, radar_start_date)

    # Rename fields and remove unnecessary ones.
    radar.add_field('DBZ_RAW', radar.fields.pop('DBZ'), replace_existing=True)
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
        if mykey in ['sounding_temperature', 'height', 'SNR', 'NCP', 'HYDRO', 'DBZ_RAW']:
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
    print("#"*79)
    print("")


    return None


if __name__ == '__main__':
    """
    Global variables definition and logging file initialisation.
    """
    # Main global variables (Path directories).
    # Output directory for CF/Radial PPIs
    OUTPATH = "/g/data2/rr5/vhl548/v2CPOL_PROD_1b/"
    # Output directory for GRIDDED netcdf data.
    OUTPATH_GRID = os.path.join(OUTPATH, 'GRIDDED')
    # Input directory for Radiosoundings (use my other script, named caprica to
    # download and format these datas).
    SOUND_DIR = "/g/data2/rr5/vhl548/soudings_netcdf/"
    # Output directory for verification figures.

    # Parse arguments
    parser_description = "Leveling treatment of CPOL data from level 1a to level 1b."
    parser = argparse.ArgumentParser(description=parser_description)
    parser.add_argument(
        '-i',
        '--input',
        dest='infile',
        default=None,
        type=str,
        help='Input radar file.')

    args = parser.parse_args()
    INFILE = args.infile

    if INFILE is None:
        parser.error("Input file required")
        sys.exit()

    with warnings.catch_warnings():
        # Just ignoring warning messages.
        warnings.simplefilter("ignore")
        main()
