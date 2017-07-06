"""
CPOL Level 1b main production line.

@title: CPOL_PROD_1b
@author: Valentin Louf <valentin.louf@monash.edu>
@institution: Bureau of Meteorology
@date: 6/06/2017
@version: 0.99

.. autosummary::
    :toctree: generated/

    plot_figure_check
    rename_radar_fields
    production_line
"""
# Python Standard Library
import gc
import os
import time
import logging
import datetime

# Other Libraries -- Matplotlib must be imported first
import matplotlib
matplotlib.use('Agg')  # <- Reason why matplotlib is imported first.
import matplotlib.colors as colors
import matplotlib.pyplot as pl

import pyart
import netCDF4
import numpy as np

# Custom modules.
from processing_codes import radar_codes
from processing_codes import atten_codes
from processing_codes import phase_codes
from processing_codes import gridding_codes


def plot_figure_check(radar, gatefilter, outfilename, radar_date, figure_path):
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
    # Extracting year and date.
    year = str(radar_date.year)
    datestr = radar_date.strftime("%Y%m%d")
    # Path for saving Figures.
    outfile_path = os.path.join(figure_path, year, datestr)

    # Checking if output directory exists. Creating them otherwise.
    if not os.path.isdir(os.path.join(figure_path, year)):
        os.mkdir(os.path.join(figure_path, year))
    if not os.path.isdir(outfile_path):
        os.mkdir(outfile_path)

    # Checking if figure already exists.
    outfile = os.path.basename(outfilename)
    outfile = outfile[:-2] + "png"
    outfile = os.path.join(outfile_path, outfile)
    # if os.path.isfile(outfile):
    #     return None

    # Initializing figure.
    with pl.style.context('seaborn-paper'):
        gr = pyart.graph.RadarDisplay(radar)
        fig, the_ax = pl.subplots(4, 3, figsize=(12, 13.5), sharex=True, sharey=True)
        the_ax = the_ax.flatten()
        # Plotting reflectivity
        gr.plot_ppi('total_power', ax=the_ax[0])
        gr.plot_ppi('corrected_reflectivity', ax=the_ax[1], gatefilter=gatefilter)
        gr.plot_ppi('radar_echo_classification', ax=the_ax[2], gatefilter=gatefilter)

        gr.plot_ppi('differential_reflectivity', ax=the_ax[3])
        gr.plot_ppi('corrected_differential_reflectivity', ax=the_ax[4], gatefilter=gatefilter)
        gr.plot_ppi('cross_correlation_ratio', ax=the_ax[5], norm=colors.LogNorm(vmin=0.5, vmax=1.05))

        gr.plot_ppi('giangrande_differential_phase', ax=the_ax[6],
                    gatefilter=gatefilter, vmin=-360, vmax=360,
                    cmap=pyart.config.get_field_colormap('corrected_differential_phase'))
        gr.plot_ppi('giangrande_specific_differential_phase', ax=the_ax[7],
                    gatefilter=gatefilter, vmin=-5, vmax=10,
                    cmap=pyart.config.get_field_colormap('specific_differential_phase'))
        gr.plot_ppi('radar_estimated_rain_rate', ax=the_ax[8], gatefilter=gatefilter)

        gr.plot_ppi('corrected_velocity', ax=the_ax[9], cmap=pyart.graph.cm.NWSVel, vmin=-30, vmax=30)
        gr.plot_ppi('region_dealias_velocity', ax=the_ax[10], gatefilter=gatefilter, cmap=pyart.graph.cm.NWSVel, vmin=-30, vmax=30)
        gr.plot_ppi('D0', ax=the_ax[11], gatefilter=gatefilter, cmap='pyart_Wild25', vmin=0, vmax=20)

        for ax_sl in the_ax:
            gr.plot_range_rings([50, 100, 150], ax=ax_sl)
            ax_sl.set_aspect(1)
            ax_sl.set_xlim(-150, 150)
            ax_sl.set_ylim(-150, 150)

        pl.tight_layout()
        pl.savefig(outfile)  # Saving figure.
        fig.clf()  # Clear figure
        pl.close()  # Release memory
    del gr  # Releasing memory
    gc.collect()  # Collecting memory garbage ;-)

    return None


def get_field_names():
    """
    Fields name definition.

    Returns:
    ========
        fields_names: array
            Containing [(old key, new key), ...]
    """
    fields_names = [('VEL', 'velocity'),
                    ('VEL_CORR', 'corrected_velocity'),
                    ('VEL_UNFOLDED', 'region_dealias_velocity'),
                    ('VEL_UNWRAP', 'dealias_velocity'),
                    ('DBZ', 'total_power'),
                    ('DBZ_CORR', 'corrected_reflectivity'),
                    ('RHOHV_CORR', 'RHOHV'),
                    ('RHOHV', 'cross_correlation_ratio'),
                    ('ZDR', 'differential_reflectivity'),
                    ('ZDR_CORR', 'corrected_differential_reflectivity'),
                    ('PHIDP', 'differential_phase'),
                    ('PHIDP_BRINGI', 'bringi_differential_phase'),
                    ('PHIDP_GG', 'giangrande_differential_phase'),
                    ('PHIDP_SIM', 'simulated_differential_phase'),
                    ('KDP', 'specific_differential_phase'),
                    ('KDP_BRINGI', 'bringi_specific_differential_phase'),
                    ('KDP_GG', 'giangrande_specific_differential_phase'),
                    ('KDP_SIM', 'simulated_specific_differential_phase'),
                    ('WIDTH', 'spectrum_width'),
                    ('SNR', 'signal_to_noise_ratio'),
                    ('NCP', 'normalized_coherent_power')]

    return fields_names


def rename_radar_fields(radar):
    """
    Rename radar fields from their old name to the Py-ART default name.

    Parameter:
    ==========
        radar:
            Py-ART radar structure.

    Returns:
    ========
        radar:
            Py-ART radar structure.
    """
    fields_names = get_field_names()

    # Try to remove occasional fields.
    try:
        vdop_art = radar.fields['PHIDP_CORR']
        radar.add_field('PHIDP', radar.fields.pop('PHIDP_CORR'), replace_existing=True)
    except KeyError:
        pass

    # Parse array old_key, new_key
    for old_key, new_key in fields_names:
        try:
            radar.add_field(new_key, radar.fields.pop(old_key), replace_existing=True)
        except KeyError:
            continue

    return radar


def production_line(radar_file_name, outpath, outpath_grid, figure_path, sound_dir):
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
        outpath_grid: str
            Path for saving gridded data.
        figure_path: str
            Path for saving figures.
        sound_dir: str
            Path to radiosoundings directory.
    """
    # Get logger.
    logger = logging.getLogger()
    # Generate output file name.
    outfilename = os.path.basename(radar_file_name)
    outfilename = outfilename.replace("level1a", "level1b")
    # Correct an occasional mislabelling from RadX.
    if "SURV" in outfilename:
        outfilename = outfilename.replace("SURV", "PPI")
    if "SUR" in outfilename:
        outfilename = outfilename.replace("SUR", "PPI")
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
    except Exception:
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

    # Correct Doppler velocity units.
    radar.fields['VEL']['units'] = "m/s"
    radar.fields['VEL']['standard_name'] = "radial_velocity"

    # Looking for NCP field
    try:
        radar.fields['NCP']
        fake_ncp = False
    except KeyError:
        # Creating a fake NCP field.
        tmp = np.zeros_like(radar.fields['DBZ']['data']) + 1
        ncp_meta = pyart.config.get_metadata('normalized_coherent_power')
        ncp_meta['data'] = tmp
        ncp_meta['description'] = "THIS FIELD IS FAKE. DO NOT USE IT!"
        radar.add_field('NCP', ncp_meta)
        fake_ncp = True

    # Looking for RHOHV field
    # For CPOL, season 09/10, there are no RHOHV fields before March!!!!
    try:
        radar.fields['RHOHV']
        fake_rhohv = False
    except KeyError:
        # Creating a fake RHOHV field.
        tmp = np.zeros_like(radar.fields['DBZ']['data']) + 1
        rho_meta = pyart.config.get_metadata('cross_correlation_ratio')
        rho_meta['data'] = tmp
        rho_meta['description'] = "THIS FIELD IS FAKE. DO NOT USE IT!"
        radar.add_field('RHOHV', rho_meta)
        fake_rhohv = True

    if fake_rhohv:
        radar.metadata['debug_info'] = 'RHOHV field does not exist in RAW data. ' +\
                                       'A fake RHOHV field has been used to process this file. Be careful.'
        logger.critical("RHOHV field not found, creating a fake RHOHV")

    # Compute SNR
    try:
        height, temperature, snr = radar_codes.snr_and_sounding(radar, sound_dir)
    except ValueError:
        logger.error("Impossible to compute SNR")
        return None
    radar.add_field('temperature', temperature, replace_existing=True)
    radar.add_field('height', height, replace_existing=True)
    try:
        radar.fields['SNR']
        logger.info('SNR already exists.')
    except KeyError:
        radar.add_field('SNR', snr, replace_existing=True)
        logger.info('SNR calculated.')

    # Correct RHOHV
    rho_corr = radar_codes.correct_rhohv(radar)
    radar.add_field_like('RHOHV', 'RHOHV_CORR', rho_corr, replace_existing=True)
    logger.info('RHOHV corrected.')

    # Get velocity field texture and noise threshold. (~3min to execute this function)
    # vel_texture, noise_threshold = radar_codes.get_texture(radar)
    # radar.add_field_like('VEL', 'TEXTURE', vel_texture, replace_existing = True)
    # logger.info('Texture computed.')

    # Get filter
    gatefilter = radar_codes.do_gatefilter(radar, rhohv_name='RHOHV_CORR')
    logger.info('Filter initialized.')

    # Correct ZDR
    corr_zdr = radar_codes.correct_zdr(radar)
    radar.add_field_like('ZDR', 'ZDR_CORR', corr_zdr, replace_existing=True)

    # PHIDP refolded.
    phidp_ref = phase_codes.refold_phidp(radar)
    radar.add_field_like('PHIDP', 'PHIDP', phidp_ref, replace_existing=True)

    # KDP from disdrometer.
    kdp_simu, phidp_simu = phase_codes.kdp_phidp_disdro_darwin(radar, refl_field="DBZ", zdr_field="ZDR")
    radar.add_field('KDP_SIM', kdp_simu, replace_existing=True)
    radar.add_field('PHIDP_SIM', phidp_simu, replace_existing=True)

    # Estimate KDP
    try:
        radar.fields['KDP']
        logger.info('KDP already exists')
    except KeyError:
        logger.info('We need to estimate KDP')
        kdp_con = phase_codes.estimate_kdp(radar, gatefilter)
        radar.add_field('KDP', kdp_con, replace_existing=True)
        logger.info('KDP estimated.')

    # Bringi PHIDP/KDP
    # phidp_bringi, kdp_bringi = phase_codes.bringi_phidp_kdp(radar, gatefilter)
    # radar.add_field_like('PHIDP', 'PHIDP_BRINGI', phidp_bringi, replace_existing=True)
    # radar.add_field_like('KDP', 'KDP_BRINGI', kdp_bringi, replace_existing=True)
    # radar.fields['PHIDP_BRINGI']['long_name'] = "bringi_" + radar.fields['PHIDP_BRINGI']['long_name']
    # radar.fields['KDP_BRINGI']['long_name'] = "bringi_" + radar.fields['KDP_BRINGI']['long_name']
    # logger.info('KDP/PHIDP Bringi estimated.')

    # Giangrande PHIDP/KDP
    phidp_gg, kdp_gg = phase_codes.phidp_giangrande(radar, gatefilter, phidp_field='PHIDP', kdp_field='KDP_SIM')
    radar.add_field('PHIDP_GG', phidp_gg, replace_existing=True)
    radar.add_field('KDP_GG', kdp_gg, replace_existing=True)
    radar.fields['PHIDP_GG']['long_name'] = "giangrande_" + radar.fields['PHIDP_GG']['long_name']
    radar.fields['KDP_GG']['long_name'] = "giangrande_" + radar.fields['KDP_GG']['long_name']
    logger.info('KDP/PHIDP Giangrande estimated.')

    # Refold VELOCITY using unfolded PHIDP
    vel_refolded, is_refolded = radar_codes.refold_velocity(radar)
    # Check if velocity was refolded.
    if is_refolded:
        logger.info('Doppler velocity has been refolded.')
        radar.add_field_like('VEL', 'VEL_CORR', vel_refolded, replace_existing=True)
        radar.fields['VEL_CORR']['long_name'] = radar.fields['VEL_CORR']['long_name'] + "_refolded"

    # Unfold VELOCITY
    # This function will check if a 'VEL_CORR' field exists anyway.
    if is_refolded:
        vdop_unfold = radar_codes.unfold_velocity(radar, gatefilter, bobby_params=is_refolded, vel_name='VEL_CORR')
    else:
        vdop_unfold = radar_codes.unfold_velocity(radar, gatefilter, bobby_params=is_refolded, vel_name='VEL')
    radar.add_field('VEL_UNFOLDED', vdop_unfold, replace_existing=True)
    logger.info('Doppler velocity unfolded.')

    # This function will use the unwrap phase algorithm.
    # vdop_unfold = radar_codes.unfold_velocity_bis(radar, gatefilter)
    # radar.add_field('VEL_UNWRAP', vdop_unfold, replace_existing=True)
    # logger.info('Doppler velocity unwrapped.')

    # Correct Attenuation ZH
    # logger.info("Starting computation of attenuation")
    # atten_spec, zh_corr = atten_codes.correct_attenuation_zh(radar)
    # radar.add_field('DBZ_CORR', zh_corr, replace_existing=True)
    # radar.add_field('specific_attenuation_reflectivity', atten_spec, replace_existing=True)
    # logger.info('Attenuation on reflectivity corrected.')
    atten_spec, zh_corr = atten_codes.correct_attenuation_zh_pyart(radar)
    radar.add_field('DBZ_CORR', zh_corr, replace_existing=True)
    radar.add_field('specific_attenuation_reflectivity', atten_spec, replace_existing=True)
    logger.info('Attenuation on reflectivity corrected.')

    # Correct Attenuation ZDR
    atten_spec_zdr, zdr_corr = atten_codes.correct_attenuation_zdr(radar)
    radar.add_field_like('ZDR', 'ZDR_CORR', zdr_corr, replace_existing=True)
    radar.add_field('specific_attenuation_differential_reflectivity', atten_spec_zdr, replace_existing=True)
    logger.info('Attenuation on ZDR corrected.')

    # Hydrometeors classification
    hydro_class = radar_codes.hydrometeor_classification(radar)
    radar.add_field('radar_echo_classification', hydro_class, replace_existing=True)
    logger.info('Hydrometeors classification estimated.')
    # Check if Hail it found hail.
    # if (hydro_class['data'] == 9).sum() != 0:
    #     print("WARNING: hail detection in Darwin. NOT POSSIBLE!", os.path.basename(outfilename))

    # Rainfall rate
    rainfall = radar_codes.rainfall_rate(radar)
    radar.add_field("radar_estimated_rain_rate", rainfall)
    logger.info('Rainfall rate estimated.')

    # DSD retrieval
    nw_dict, d0_dict = radar_codes.dsd_retrieval(radar)
    radar.add_field("D0", d0_dict)
    radar.add_field("NW", nw_dict)
    logger.info('DSD estimated.')

    # Liquid/Ice Mass
    # We decided to not give these products.
    # liquid_water_mass, ice_mass = radar_codes.liquid_ice_mass(radar)
    # radar.add_field('LWC', liquid_water_mass)
    # radar.add_field('IWC', ice_mass)
    # logger.info('Liquid/Ice mass estimated.')

    # Check if NCP field is fake.
    if fake_ncp:
        radar.fields.pop('NCP')

    # Remove useless fields:
    for mykey in ["KDP_SIM", "PHIDP_SIM", "KDP_BRINGI", "PHIDP_BRINGI"]:
        try:
            radar.fields.pop(mykey)
        except KeyError:
            continue

    # Rename fields to pyart defaults.
    radar = rename_radar_fields(radar)

    # Treatment is finished!
    end_time = time.time()
    logger.info("Treatment for %s done in %0.2f seconds.", os.path.basename(outfilename), (end_time - start_time))

    # Plot check figure.
    logger.info('Plotting figure')
    try:
        plot_figure_check(radar, gatefilter, outfilename, radar_start_date, figure_path)
    except Exception:
        logger.exception("Problem while trying to plot figure.")
    figure_time = time.time()
    logger.info('Figure saved in %0.2fs.', (figure_time - end_time))

    # Hardcode mask
    for mykey in radar.fields:
        if mykey in ['temperature', 'height', 'signal_to_noise_ratio',
                     'normalized_coherent_power', 'spectrum_width', 'total_power',
                     'giangrande_differential_phase', 'giangrande_specific_differential_phase']:
            # Virgin fields that are left untouch.
            continue
        else:
            radar.fields[mykey]['data'] = radar_codes.filter_hardcoding(radar.fields[mykey]['data'], gatefilter)
    logger.info('Hardcoding gatefilter to Fields done.')

    # Write results
    pyart.io.write_cfradial(outfilename, radar, format='NETCDF4')
    save_time = time.time()
    logger.info('%s saved in %0.2f s.', os.path.basename(outfilename), (save_time - figure_time))

    # Free memory from everything useless before gridding
    gc.collect()

    # Deleting all unwanted keys for gridded product.
    logger.info("Gridding started.")
    unwanted_keys = []
    goodkeys = ['corrected_differential_reflectivity', 'cross_correlation_ratio',
                'temperature', 'giangrande_differential_phase',
                'radar_echo_classification', 'radar_estimated_rain_rate', 'D0',
                'NW', 'corrected_reflectivity', 'corrected_velocity', 'region_dealias_velocity']
    for mykey in radar.fields.keys():
        if mykey not in goodkeys:
            unwanted_keys.append(mykey)
    for mykey in unwanted_keys:
        radar.fields.pop(mykey)

    try:
        # Gridding (and saving)
        gridding_codes.gridding_radar_150km(radar, radar_start_date, outpath=outpath_grid)
        gridding_codes.gridding_radar_70km(radar, radar_start_date, outpath=outpath_grid)
        logger.info('Gridding done in %0.2f s.', (time.time() - save_time))
    except Exception:
        logging.error('Problem while gridding.')
        raise

    # Processing finished!
    logger.info('%s processed in  %0.2f s.', os.path.basename(outfilename), (time.time() - start_time))

    return None
