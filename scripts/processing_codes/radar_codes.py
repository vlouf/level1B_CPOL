"""
Codes for correcting and estimating various radar and meteorological parameters.

@title: radar_codes
@author: Valentin Louf <valentin.louf@monash.edu>
@institutions: Monash University and the Australian Bureau of Meteorology
@creation: 04/04/2017
@date: 15/05/2017

.. autosummary::
    :toctree: generated/

    _unfold_phidp
    _refold_vdop
    bringi_phidp_kdp
    correct_rhohv
    correct_zdr
    do_gatefilter
    estimate_kdp
    hydrometeor_classification
    kdp_from_phidp_finitediff
    liquid_ice_mass
    nearest
    phidp_giangrande
    snr_and_sounding
    unfold_phidp_vdop
    unfold_velocity
"""
# Python Standard Library
import os
import glob
import time
import copy
import fnmatch
import datetime
from copy import deepcopy

# Other Libraries
import pyart
import scipy
import netCDF4
import numpy as np

from scipy import ndimage, signal, integrate, interpolate
from csu_radartools import csu_liquid_ice_mass, csu_fhc, csu_blended_rain, csu_dsd


def _my_snr_from_reflectivity(radar, refl_field='DBZ'):
    """
    Just in case pyart.retrieve.calculate_snr_from_reflectivity, I can calculate
    it 'by hand'.
    Parameter:
    ===========
        radar:
            Py-ART radar structure.
        refl_field_name: str
            Name of the reflectivity field.

    Return:
    =======
        snr: dict
            Signal to noise ratio.

    """
    range_grid, azi_grid = np.meshgrid(radar.range['data'], radar.azimuth['data'])
    range_grid += 1  # Cause of 0

    # remove range scale.. This is basically the radar constant scaled dBm
    pseudo_power = (radar.fields[refl_field]['data'] - 20.0*np.log10(range_grid / 1000.0))
    # The noise_floor_estimate can fail sometimes in pyart, that's the reason
    # why this whole function exists.
    noise_floor_estimate = -40

    snr_field = pyart.config.get_field_name('signal_to_noise_ratio')
    snr_dict = pyart.config.get_metadata(snr_field)
    snr_dict['data'] = pseudo_power - noise_floor_estimate

    return snr_dict


def _nearest(items, pivot):
    """
    Find the nearest item.

    Parameters:
    ===========
        items:
            List of item.
        pivot:
            Item we're looking for.

    Returns:
    ========
        item:
            Value of the nearest item found.
    """
    return min(items, key=lambda x: abs(x - pivot))


def _get_noise_threshold(filtered_data):
    """
    Compute the noise threshold.
    """
    n, bins = np.histogram(filtered_data, bins = 150)
    peaks = scipy.signal.find_peaks_cwt(n, np.array([10]))
    centers = bins[0:-1] + (bins[1] - bins[0])
    search_data = n[peaks[0]:peaks[1]]
    search_centers = centers[peaks[0]:peaks[1]]
    locs = search_data.argsort()
    noise_threshold = search_centers[locs[0]]

    return noise_threshold


def check_azimuth(radar):
    """
    Checking if radar has a proper azimuth field.  It's a minor problem
    concerning less than 7 days in the entire dataset.

    Parameter:
    ===========
        radar:
            Py-ART radar structure.

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

    return is_good


def check_reflectivity(radar, refl_field_name='DBZ'):
    """
    Checking if radar has a proper reflectivity field.  It's a minor problem
    concerning a few days in 2011 for CPOL.

    Parameters:
    ===========
        radar:
            Py-ART radar structure.
        refl_field_name: str
            Name of the reflectivity field.

    Return:
    =======
        is_good: bool
            True if radar has a proper azimuth field.
    """
    is_good = True
    dbz = radar.fields[refl_field_name]['data']

    if np.ma.isMaskedArray(dbz):
        if dbz.count() == 0:
            # Reflectivity field is empty.
            is_good = False

    return is_good


def correct_rhohv(radar, rhohv_name='RHOHV', snr_name='SNR'):
    """
    Correct cross correlation ratio (RHOHV) from noise. From the Schuur et al.
    2003 NOAA report (p7 eq 5)

    Parameters:
    ===========
        radar:
            Py-ART radar structure.
        rhohv_name: str
            Cross correlation field name.
        snr_name: str
            Signal to noise ratio field name.

    Returns:
    ========
        rho_corr: array
            Corrected cross correlation ratio.
    """
    rhohv = radar.fields[rhohv_name]['data']
    snr = radar.fields[snr_name]['data']
    natural_snr = 10**(0.1*snr)
    rho_corr = rhohv / (1 + 1/natural_snr)

    return rho_corr


def correct_zdr(radar, zdr_name='ZDR', snr_name='SNR'):
    """
    Correct differential reflectivity (ZDR) from noise. From the Schuur et al.
    2003 NOAA report (p7 eq 6)

    Parameters:
    ===========
        radar:
            Py-ART radar structure.
        zdr_name: str
            Differential reflectivity field name.
        snr_name: str
            Signal to noise ratio field name.

    Returns:
    ========
        corr_zdr: array
            Corrected differential reflectivity.
    """
    zdr = radar.fields[zdr_name]['data']
    snr = radar.fields[snr_name]['data']
    alpha = 1.48
    natural_zdr = 10**(0.1*zdr)
    natural_snr = 10**(0.1*snr)
    corr_zdr = 10*np.log10((alpha*natural_snr*natural_zdr) / (alpha*natural_snr + alpha - natural_zdr))

    return corr_zdr


def do_gatefilter(radar, noise_threshold=None, texture_name='TEXTURE',
                  refl_name='DBZ', rhohv_name='RHOHV_CORR', ncp_name='NCP'):
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
    gf.exclude_below(rhohv_name, 0.7)

    # if noise_threshold is not None:
    #     if not np.isnan(noise_threshold):
    #         try:
    #             gf.exclude_above(texture_name, noise_threshold)
    #         except KeyError:
    #             pass

    try:
        # NCP field is not present for older seasons.
        radar.fields[ncp_name]
        gf.exclude_below(ncp_name, 0.3)
    except KeyError:
        pass

    gf_despeckeld = pyart.correct.despeckle_field(radar, refl_name, gatefilter=gf)

    return gf_despeckeld


def dsd_retrieval(radar, refl_name='DBZ', zdr_name='ZDR', kdp_name='KDP_GG'):
    """
    Compute the DSD retrieval using the csu library.

    Parameters:
    ===========
        radar:
            Py-ART radar structure.
        refl_name: str
            Reflectivity field name.
        zdr_name: str
            ZDR field name.
        kdp_name: str
            KDP field name.

    Returns:
    ========
        nw_dict: dict
            Normalized Intercept Parameter.
        d0_dict: dict
            Median Volume Diameter.
    """
    dbz = radar.fields[refl_name]['data'].filled(np.NaN)
    zdr = radar.fields[zdr_name]['data'].filled(np.NaN)
    try:
        kdp = radar.fields[kdp_name]['data'].filled(np.NaN)
    except AttributeError:
        kdp = radar.fields[kdp_name]['data']

    d0, Nw, mu = csu_dsd.calc_dsd(dz=dbz, zdr=zdr, kdp=kdp, band='C')

    Nw = np.log10(Nw)
    Nw = np.ma.masked_where(np.isnan(Nw), Nw)
    d0 = np.ma.masked_where(np.isnan(d0), d0)

    nw_dict = {'data': Nw,
               'units': 'unitless (log10)', 'long_name': 'Log10 of the Normalized Intercept Parameter',
               'standard_name': 'Log10 of the Normalized Intercept Parameter',
               'description': "NW retrieval based on Bringi et al. (2009). Mu can not be retrieved alongside NW and D0."}

    d0_dict = {'data': d0,
               'units': 'mm', 'long_name': 'Median Volume Diameter',
               'standard_name': 'Median Volume Diameter',
               'description': "D0 retrieval based on Bringi et al. (2009). Mu can not be retrieved alongside NW and D0."}

    return nw_dict, d0_dict


def get_texture(radar, vel_field_name='VEL'):
    """
    Compute the texture of the velocity field and its noise threshold.

    Parameters:
    ===========
        radar:
            Py-ART radar structure.
        vel_field_name: str
            Doppler velocity field name.

    Returns:
    ========
        filtered_data: array
            Velocity texture.
        noise_threshold: float
            The noise threshold estimated.
    """
    nyq = radar.instrument_parameters['nyquist_velocity']['data'][0]
    vel = radar.fields[vel_field_name]['data']
    data = ndimage.filters.generic_filter(vel, pyart.util.interval_std, size = (4,4), extra_arguments = (-nyq, nyq))
    filtered_data = ndimage.filters.median_filter(data, size = (4,4))

    try:
        noise_threshold = _get_noise_threshold(filtered_data)
    except:
        noise_threshold = np.NaN
        print("Could not determine the noise threshold")

    return filtered_data, noise_threshold


def filter_hardcoding(my_array, nuke_filter, bad=-9999):
    """
    Harcoding GateFilter into an array.

    Parameters:
    ===========
        my_array: array
            Array we want to clean out.
        nuke_filter: gatefilter
            Filter we want to apply to the data.
        bad: float
            Fill value.

    Returns:
    ========
        to_return: masked array
            Same as my_array but with all data corresponding to a gate filter
            excluded.
    """
    filt_array = np.ma.masked_where(nuke_filter.gate_excluded, my_array)
    filt_array.set_fill_value(bad)
    filt_array = filt_array.filled(fill_value=bad)
    to_return = np.ma.masked_where(filt_array == bad, filt_array)
    return to_return


def hydrometeor_classification(radar, refl_name='DBZ_CORR', zdr_name='ZDR_CORR',
                               kdp_name='KDP_GG', rhohv_name='RHOHV_CORR',
                               temperature_name='temperature',
                               height_name='height'):
    """
    Compute hydrometeo classification.

    Parameters:
    ===========
        radar:
            Py-ART radar structure.
        refl_name: str
            Reflectivity field name.
        zdr_name: str
            ZDR field name.
        kdp_name: str
            KDP field name.
        rhohv_name: str
            RHOHV field name.
        temperature_name: str
            Sounding temperature field name.
        height: str
            Gate height field name.

    Returns:
    ========
        hydro_meta: dict
            Hydrometeor classification.
    """
    refl = radar.fields[refl_name]['data']
    zdr = radar.fields[zdr_name]['data']
    kdp = radar.fields[kdp_name]['data']
    rhohv = radar.fields[rhohv_name]['data']
    radar_T = radar.fields[temperature_name]['data']
    radar_z = radar.fields[height_name]['data']

    scores = csu_fhc.csu_fhc_summer(dz=refl, zdr=zdr, rho=rhohv, kdp=kdp,
        use_temp=True, band='C', T=radar_T)

    hydro = np.argmax(scores, axis=0) + 1
    fill_value = -32768
    hydro_data = np.ma.masked_where(hydro == fill_value, hydro)

    the_comments = "1: Drizzle; 2: Rain; 3: Ice Crystals; 4: Aggregates; " +\
                   "5: Wet Snow; 6: Vertical Ice; 7: LD Graupel; 8: HD Graupel; 9: Hail; 10: Big Drops"

    hydro_meta = {'data': hydro_data, 'units': ' ', 'long_name': 'Hydrometeor classification',
                  'standard_name': 'Hydrometeor_ID', 'comments': the_comments }

    return hydro_meta


def liquid_ice_mass(radar, refl_name='DBZ_CORR', zdr_name='ZDR_CORR',
                    temperature_name='temperature', height_name='height'):
    """
    Compute the liquid/ice water content using the csu library.

    Parameters:
    ===========
        radar:
            Py-ART radar structure.
        refl_name: str
            Reflectivity field name.
        zdr_name: str
            ZDR field name.
        temperature_name: str
            Sounding temperature field name.
        height: str
            Gate height field name.

    Returns:
    ========
        liquid_water_mass: dict
            Liquid water content.
        ice_mass: dict
            Ice water content.
    """
    refl = radar.fields[refl_name]['data']
    zdr = radar.fields[zdr_name]['data']
    radar_T = radar.fields[temperature_name]['data']
    radar_z = radar.fields[height_name]['data']

    liquid_water_mass, ice_mass = csu_liquid_ice_mass.calc_liquid_ice_mass(refl,
        zdr, radar_z/1000, T=radar_T)

    liquid_water_mass = {'data': liquid_water_mass, 'units': 'g m-3',
                        'long_name': 'Liquid Water Content',
                        'standard_name': 'liquid_water_content',
                        'description': "Liquid Water Content using Carey and Rutledge (2000) algorithm."}
    ice_mass = {'data': ice_mass, 'units': 'g m-3', 'long_name': 'Ice Water Content',
                'standard_name': 'ice_water_content',
                'description': "Ice Water Content using Carey and Rutledge (2000) algorithm."}

    return liquid_water_mass, ice_mass


def rainfall_rate(radar, refl_name='DBZ_CORR', zdr_name='ZDR_CORR', kdp_name='KDP_GG', hydro_name='radar_echo_classification'):
    """
    Rainfall rate algorithm from csu_radartools.

    Parameters:
    ===========
        radar:
            Py-ART radar structure.
        refl_name: str
            Reflectivity field name.
        zdr_name: str
            ZDR field name.
        kdp_name: str
            KDP field name.
        hydro_name: str
            Hydrometeor classification field name.

    Returns:
    ========
        rainrate: dict
            Rainfall rate.
    """
    dbz = radar.fields[refl_name]['data'].filled(np.NaN)
    zdr = radar.fields[zdr_name]['data'].filled(np.NaN)
    fhc = radar.fields[hydro_name]['data']
    try:
        kdp = radar.fields[kdp_name]['data'].filled(np.NaN)
    except AttributeError:
        kdp = radar.fields[kdp_name]['data']

    rain, method = csu_blended_rain.calc_blended_rain_tropical(dz=dbz, zdr=zdr, kdp=kdp, fhc=fhc, band='C')
    rain[rain == 0] = np.NaN
    rain = np.ma.masked_where(np.isnan(rain), rain)

    rainrate = {"long_name": 'Blended Rainfall Rate',
           "units": "mm h-1",
           "standard_name": "Rainfall Rate",
           "description": "Rainfall rate algorithm based on Thompson et al. 2016.",
           "data": rain}

    return rainrate


def snr_and_sounding(radar, soundings_dir=None, refl_field_name='DBZ'):
    """
    Compute the signal-to-noise ratio as well as interpolating the radiosounding
    temperature on to the radar grid. The function looks for the radiosoundings
    that happened at the closest time from the radar. There is no time
    difference limit.

    Parameters:
    ===========
        radar:
        soundings_dir: str
            Path to the radiosoundings directory.
        refl_field_name: str
            Name of the reflectivity field.

    Returns:
    ========
        z_dict: dict
            Altitude in m, interpolated at each radar gates.
        temp_info_dict: dict
            Temperature in Celsius, interpolated at each radar gates.
        snr: dict
            Signal to noise ratio.
    """

    if soundings_dir is None:
        soundings_dir = "/g/data2/rr5/vhl548/soudings_netcdf/"

    # Getting radar date.
    radar_start_date = netCDF4.num2date(radar.time['data'][0], radar.time['units'])

    # Listing radiosounding files.
    sonde_pattern = datetime.datetime.strftime(radar_start_date, 'YPDN_%Y%m%d*')
    all_sonde_files = sorted(os.listdir(soundings_dir))

    try:
        # The radiosoundings for the exact date exists.
        sonde_name = fnmatch.filter(all_sonde_files, sonde_pattern)[0]
    except IndexError:
        # The radiosoundings for the exact date does not exist, looking for the
        # closest date.
        print("Sounding file not found, looking for the nearest date.")
        dtime = [datetime.datetime.strptime(dt, 'YPDN_%Y%m%d_%H.nc') for dt in all_sonde_files]
        closest_date = _nearest(dtime, radar_start_date)
        sonde_pattern = datetime.datetime.strftime(closest_date, 'YPDN_%Y%m%d*')
        radar_start_date = closest_date
        sonde_name = fnmatch.filter(all_sonde_files, sonde_pattern)[0]

    interp_sonde = netCDF4.Dataset(os.path.join(soundings_dir, sonde_name))
    temperatures = interp_sonde.variables['temp'][:]
    times = interp_sonde.variables['time'][:]
    heights = interp_sonde.variables['height'][:]

    # Height and temperature profiles.
    my_profile = pyart.retrieve.fetch_radar_time_profile(interp_sonde, radar)
    z_dict, temp_dict = pyart.retrieve.map_profile_to_gates(temperatures, my_profile['height'], radar)
    temp_info_dict = {'data': temp_dict['data'],
                 'long_name': 'Sounding temperature at gate',
                 'standard_name' : 'temperature',
                 'valid_min' : -100, 'valid_max' : 100,
                 'units' : 'degrees Celsius',
                 'comment': 'Radiosounding date: %s' % (radar_start_date.strftime("%Y/%m/%d"))}

    # Calculate SNR
    snr = pyart.retrieve.calculate_snr_from_reflectivity(radar, refl_field=refl_field_name)
    # Sometimes the SNR is an empty array, this is due to the toa parameter.
    # Here we try to recalculate the SNR with a lower value for toa (top of atm).
    if snr['data'].count() == 0:
        snr = pyart.retrieve.calculate_snr_from_reflectivity(radar, refl_field=refl_field_name, toa=20000)

    if snr['data'].count() == 0:
        # If it fails again, then we compute the SNR with the noise value
        # given by the CPOL radar manufacturer.
        snr = _my_snr_from_reflectivity(radar, refl_field=refl_field_name)

    return z_dict, temp_info_dict, snr


def unfold_velocity(radar, my_gatefilter, bobby_params=True, vel_name='VEL'):
    """
    Unfold Doppler velocity using Py-ART region based algorithm. Automatically
    searches for a folding-corrected velocity field.

    Parameters:
    ===========
        radar:
            Py-ART radar structure.
        my_gatefilter:
            GateFilter
        bobby_params: bool
            Using dealiasing parameters from Bobby Jackson. Otherwise using
            defaults configuration.
        vel_name: str
            Name of the (original) Doppler velocity field.

    Returns:
    ========
        vdop_vel: dict
            Unfolded Doppler velocity.
    """
    gf = deepcopy(my_gatefilter)
    try:
        # Looking for a folding-corrected velocity field.
        vdop_art = radar.fields['VEL_CORR']['data']
        # Because 'VEL_CORR' field is based upon 'PHIDP_BRINGI', we need to
        # exclude the gates has been dropped by the Bringi algo.
        gf.exclude_masked('PHIDP_BRINGI')
        vel_name = 'VEL_CORR'
    except KeyError:
        # Standard velocity field. No correction has been applied to it.
        vdop_art = radar.fields[vel_name]['data']

    # Trying to determine Nyquist velocity
    try:
        v_nyq_vel = radar.instrument_parameters['nyquist_velocity']['data'][0]
    except:
        v_nyq_vel = np.max(np.abs(vdop_art))

    # Cf. mail from Bobby Jackson for skip_between_rays parameters.
    # if bobby_params:
    #     vdop_vel = pyart.correct.dealias_region_based(radar,
    #                                                   vel_field=vel_name,
    #                                                   gatefilter=gf,
    #                                                   nyquist_vel=v_nyq_vel,
    #                                                   skip_between_rays=2000)
    # else:
    #
    vdop_vel = pyart.correct.dealias_region_based(radar, vel_field=vel_name, gatefilter=gf, nyquist_vel=v_nyq_vel)

    vdop_vel['units'] = "m/s"
    vdop_vel['description'] = "Velocity unfolded using Py-ART region based dealiasing algorithm."

    return vdop_vel


def unfold_velocity_bis(radar, my_gatefilter, vel_name='VEL'):
    """
    Unfold Doppler velocity using Py-ART the unwrap phase algorithm.

    Parameters:
    ===========
        radar:
            Py-ART radar structure.
        my_gatefilter:
            GateFilter
        vel_name: str
            Name of the (original) Doppler velocity field.

    Returns:
    ========
        vel_dealias: dict
            Unfolded Doppler velocity.
    """
    vel_dealias = pyart.correct.dealias_unwrap_phase(radar,
                                           unwrap_unit='sweep',
                                           keep_original=True,
                                           skip_checks=True,
                                           gatefilter=my_gatefilter,
                                           vel_field=vel_name)

    vel_dealias['units'] = "m/s"
    vdop_vel['description'] = "Velocity unfolded using Py-ART dealias phase unwraping algorithm."

    return vel_dealias
