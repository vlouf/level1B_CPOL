# Python Standard Library
import os
import sys
import pickle
import logging
import argparse
import datetime
import subprocess
from multiprocessing import Pool

# Other Libraries
import pyart
import netCDF4
import numpy as np
# import pandas as pd

# Custom modules.
# import raijin_tools
import radar_codes


def do_gatefilter(radar, refl_name='DBZ', rhohv_name='RHOHV'):
    """
    Basic filtering

    Parameters:
    ===========
        radar:
            Py-ART radar structure.
        refl_name: str
            Reflectivity field name.
        rhohv_name: str
            Cross correlation ratio field name.

    Returns:
    ========
        gf_despeckeld: GateFilter
            Gate filter (excluding all bad data).
    """
    gf = pyart.filters.GateFilter(radar)
    gf.exclude_outside(refl_name, -20, 90)
    gf.exclude_below(RHOHV, 0.5)

    gf_despeckeld = pyart.correct.despeckle_field(radar, refl_name, gatefilter=gf)

    return gf_despeckeld


def production_line(radar_file_name):

    try:
        radar = pyart.io.read(radar_file_name)
    except:
        logger.error("MAJOR ERROR: Can't read input file named {}".format(radar_file_name))
        return None

    radar_start_date = netCDF4.num2date(radar.time['data'][0], radar.time['units'])

    # Compute SNR
    height, temperature, snr = radar_codes.snr_and_sounding(radar, sound_dir, 'DBZ')
    radar.add_field('sounding_temperature', temperature, replace_existing = True)
    radar.add_field('height', height, replace_existing = True)
    radar.add_field('SNR', snr, replace_existing = True)

    # Correct RHOHV
    rho_corr = radar_codes.correct_rhohv(radar)
    radar.add_field_like('RHOHV', 'RHOHV_CORR', rho_corr, replace_existing = True)

    # Get filter
    gatefilter = do_gatefilter(radar)

    # Correct ZDR
    corr_zdr = radar_codes.correct_zdr(radar)
    radar.add_field_like('ZDR', 'ZDR_CORR', corr_zdr, replace_existing=True)

    # Estimate KDP
    try:
        radar.fields['KDP']
    except KeyError:
        logger.info('We need to estimate KDP')
        kdp_con = radar_codes.estimate_kdp(radar, gatefilter)
        radar.add_field('KDP', kdp_con, replace_existing=True)

    # Bringi PHIDP/KDP
    phidp_bringi, kdp_bringi = radar_codes.bringi_phidp_kdp()
    radar.add_field_like('PHIDP', 'PHIDP_BRINGI', phidp_bringi, replace_existing=True)
    radar.add_field_like('KDP', 'KDP_BRINGI', kdp_bringi, replace_existing=True)

    # Unfold PHIDP, refold VELOCITY
    phidp_unfold, vdop_refolded = radar_codes.unfold_phidp_vdop(radar, unfold_vel=False)
    radar.add_field_like('PHIDP', 'PHIDP_CORR', rslt, replace_existing=True)
    if vdop_refolded is not None:
        radar.add_field_like('VEL', 'VEL_CORR', vdop_refolded, replace_existing=True)
        doppler_refold = True

    # Unfold VELOCITY
    if doppler_refold:
        vdop_unfold = radar_codes.unfold_velocity(radar, my_gatefilter, vel_name='VEL_CORR')
    else:
        vdop_unfold = radar_codes.unfold_velocity(radar, my_gatefilter, vel_name='VEL')
    radar.add_field('VEL_UNFOLDED', vdop_unfold, replace_existing = True)

    # Correct Attenuation ZH
    atten_spec, zh_corr = radar_codes.correct_attenuation_zh(radar)
    radar.add_field_like('DBZ', 'DBZ_CORR', zh_corr, replace_existing=True)
    radar.add_field('specific_attenuation_zh', atten_spec, replace_existing=True)

    # Correct Attenuation ZDR
    atten_spec_zdr, zdr_corr = radar_codes.correct_attenuation_zdr(radar)
    radar.add_field_like('ZDR', 'ZDR_CORR', zdr_corr, replace_existing=True)
    radar.add_field('specific_attenuation_zdr', atten_spec_zdr, replace_existing=True)

    # Hydrometeors classification
    hydro_class = radar_codes.hydrometeor_classification(radar)
    radar.add_field('HYDRO', hydro_class, replace_existing=True)

    # Liquid/Ice Mass
    liquid_water_mass, ice_mass = radar_codes.liquid_ice_mass(radar)
    radar.add_field('LWC', liquid_water_mass)
    radar.add_field('IWC', ice_mass)

    return None


def main():

    return None


if __name__ == '__main__':

    INPATH = "/g/data2/rr5/vhl548/CPOL_level_1/"
    OUTPATH = "/g/data2/rr5/vhl548/CPOL_PROD_1b/"

    log_file_name =  os.path.join(os.path.expanduser('~'), 'cpol_level1b.log')
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        filename=log_file_name,
        filemode='w')
    logger = logging.getLogger(__name__)

    main()
