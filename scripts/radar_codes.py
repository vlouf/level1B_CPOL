"""
Codes for correcting and estimating various radar and meteorological parameters.

@title: radar_codes
@author: Valentin Louf <valentin.louf@monash.edu>
@institution: Bureau of Meteorology
@date: 04/04/2017

.. autosummary::
    :toctree: generated/

    bringi_phidp_kdp
    compute_attenuation
    correct_attenuation_zdr
    correct_attenuation_zh
    correct_rhohv
    correct_zdr
    estimate_kdp
    hydrometeor_classification
    kdp_from_phidp_finitediff
    liquid_ice_mass
    nearest
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
import netCDF4
import numpy as np

from numba import jit, int32, float32
from csu_radartools import csu_kdp, csu_liquid_ice_mass, csu_fhc


@jit(nopython=True, cache=True)
def _unfold_phidp(the_phidp, rpos, azipos):
    """
    Internally called by unfold_phidp_vdop()
    """
    tmp = the_phidp
    for i, j in zip(azipos, rpos):
        tmp[i, j:] += 180
    return tmp


@jit(cache=True)
def _refold_vdop(vdop_art, v_nyq_vel, rpos, azipos):
    """
    Internally called by unfold_phidp_vdop()
    """
    tmp = vdop_art
    for i, j in zip(azipos, rpos):
        tmp[i, j:] += v_nyq_vel

    pos = (vdop_art > v_nyq_vel)
    tmp[pos] = tmp[pos] - 2*v_nyq_vel

    return tmp


def bringi_phidp_kdp(radar, gatefilter, refl_name='DBZ', phidp_name='PHIDP'):
    """
    Compute PHIDP and KDP using Bringi's algorithm.

    Parameters:
    ===========
        radar:
            Py-ART radar structure
        gatefilter:
            Radar GateFilter (excluding bad data).
        refl_name: str
            Reflectivity field name
        phidp_name: str
            PHIDP field name

    Returns:
    ========
        fdN: array
            PhiDP Bringi
        kdN: array
            KDP Bringi
    """
    refl = radar.fields[refl_name]['data']
    phidp = radar.fields[phidp_name]['data']
    refl = np.ma.masked_where(gatefilter.gate_excluded, refl).filled(-9999)
    phidp = np.ma.masked_where(gatefilter.gate_excluded, phidp).filled(-9999)
    r = radar.range['data']

    rng2d, az2d = np.meshgrid(radar.range['data'], radar.azimuth['data'])
    dr = (r[1] - r[0])  # m
    window_size = dr/1000*4 # in km!!!!!

    kdN, fdN, sdN = csu_kdp.calc_kdp_bringi(phidp, refl, rng2d/1000.0, gs=dr, window=window_size, bad=-9999)

    fdN = np.ma.masked_where(fdN == -9999, fdN)
    kdN = np.ma.masked_where(kdN == -9999, kdN)

    return fdN, kdN


def compute_attenuation(kdp, alpha = 0.08, dr = 0.25):
    """
    Alpha is defined by Ah=alpha*Kdp, beta is defined by Ah-Av=beta*Kdp.
    From Bringi et al. (2003)

    Parameters:
    ===========
        kdp: array
            Specific phase.
        alpha: float
            Parameter being defined by Ah = alpha*Kdp
        dr: float
            Gate range in km.

    Returns:
    ========
        atten_specific: array
            Specfific attenuation (dB/km)
        atten: array
            Cumulated attenuation (dB)
    """
    kdp = kdp.filled(0)  # 0 is the neutral value for a sum
    kdp[kdp < 0] = 0
    atten_specific = alpha*kdp
    atten_specific[np.isnan(atten_specific)] = 0
    atten = 2 * np.cumsum(atten_specific, axis=1) * dr

    return atten_specific, atten


def correct_attenuation_zdr(radar, zdr_name='ZDR', kdp_name='KDP_BRINGI'):
    """
    Correct attenuation on differential reflectivity. KDP_BRINGI has been
    cleaned of noise, that's why we use it.

    Parameters:
    ===========
        radar:
            Py-ART radar structure.
        zdr_name: str
            Differential reflectivity field name.
        kdp_name: str
            KDP field name.

    Returns:
    ========
        atten_meta: dict
            Specific attenuation.
        zdr_corr: array
            Attenuation corrected differential reflectivity.
    """
    r = radar.range['data']
    zdr = radar.fields[zdr_name]['data']
    kdp = radar.fields[kdp_name]['data']

    dr = (r[1] - r[0]) / 1000  # km

    atten_spec, atten = compute_attenuation(kdp, alpha=0.016, dr=dr)
    zdr_corr = zdr + atten

    atten_meta = {'data': atten_spec, 'units': 'dB/km', 'standard_name': 'specific_attenuation_zdr',
                  'long_name': 'Differential reflectivity specific attenuation'}

    return atten_meta, zdr_corr


def correct_attenuation_zh(radar, refl_name='DBZ', kdp_name='KDP_BRINGI'):
    """
    Correct attenuation on reflectivity. KDP_BRINGI has been
    cleaned of noise, that's why we use it.

    Parameters:
    ===========
        radar:
            Py-ART radar structure.
        refl_name: str
            Reflectivity field name.
        kdp_name: str
            KDP field name.

    Returns:
    ========
        atten_meta: dict
            Specific attenuation.
        zh_corr: array
            Attenuation corrected reflectivity.
    """
    r = radar.range['data']
    refl = radar.fields[refl_name]['data']
    kdp = radar.fields[kdp_name]['data']

    dr = (r[1] - r[0]) / 1000  # km

    atten_spec, atten = compute_attenuation(kdp, alpha=0.08, dr=dr)
    zh_corr = refl + atten

    atten_meta = {'data': atten_spec, 'units': 'dB/km', 'standard_name': 'specific_attenuation_zh',
                  'long_name': 'Reflectivity specific attenuation'}

    return atten_meta, zh_corr


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


def do_gatefilter(radar, refl_name='DBZ', rhohv_name='RHOHV', ncp_name='NCP'):
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
    gf.exclude_below(rhohv_name, 0.5)

    try:
        # NCP field is not present for older seasons.
        radar.fields[ncp_name]
        gf.exclude_below(ncp_name, 0.3)
    except KeyError:
        pass

    gf_despeckeld = pyart.correct.despeckle_field(radar, refl_name, gatefilter=gf)

    return gf_despeckeld


def estimate_kdp(radar, gatefilter, phidp_name='PHIDP'):
    """
    Estimate KDP.

    Parameters:
    ===========
        radar:
            Py-ART radar structure.
        gatefilter:
            Radar GateFilter (excluding bad data).
        phidp_name: str
            PHIDP field name.

    Returns:
    ========
        kdp_field: dict
            KDP.
    """
    r = radar.range['data']
    dr = (r[1] - r[0]) / 1000  # km

    # The next two lines are 3 step:
    # - Extracting PHIDP (it is a masked array)
    # - Masking gates in PHIDP that are excluded by the gatefilter
    # - Turning PHIDP into a normal array and filling all masked value to NaN.
    phidp = radar.fields[phidp_name]['data']
    phidp = np.ma.masked_where(gatefilter.gate_excluded, phidp).filled(np.NaN)

    kdp_data = kdp_from_phidp_finitediff(phidp, dr=dr)
    kdp_field = {'data': kdp_data, 'units': 'degrees/km', 'standard_name': 'specific_differential_phase_hv',
                 'long_name': 'Specific differential phase (KDP)'}

    return kdp_field


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
                               kdp_name='KDP', rhohv_name='RHOHV_CORR',
                               temperature_name='sounding_temperature',
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

    scores = csu_fhc.csu_fhc_summer(dz=refl, zdr=zdr, rho=rhohv, kdp=kdp, use_temp=True, band='C', T=radar_T)

    hydro = np.argmax(scores, axis=0) + 1
    fill_value = -32768
    hydro_data = np.ma.asanyarray(hydro)
    hydro_data.mask = hydro_data == fill_value

    the_comments = "1: Drizzle; 2: Rain; 3: Ice Crystals; 4: Aggregates; " +\
                   "5: Wet Snow; 6: Vertical Ice; 7: LD Graupel; 8: HD Graupel; 9: Hail; 10: Big Drops"

    hydro_meta = {'data': hydro_data, 'units': ' ', 'long_name': 'Hydrometeor classification',
                  'standard_name': 'Hydrometeor_ID', 'comments': the_comments }

    return hydro_meta


def kdp_from_phidp_finitediff(phidp, L=7, dr=1.):
    """
    Retrieves KDP from PHIDP by applying a moving window range finite
    difference derivative. Function from wradlib.

    Parameters
    ----------
    phidp : multi-dimensional array
        Note that the range dimension must be the last dimension of
        the input array.
    L : integer
        Width of the window (as number of range gates)
    dr : gate length in km
    """

    assert (L % 2) == 1, \
        "Window size N for function kdp_from_phidp must be an odd number."
    # Make really sure L is an integer
    L = int(L)
    kdp = np.zeros(phidp.shape)
    for r in range(int(L / 2), phidp.shape[-1] - int(L / 2)):
        kdp[..., r] = (phidp[..., r + int(L / 2)] -
                       phidp[..., r - int(L / 2)]) / (L - 1)
    return kdp / 2. / dr


def liquid_ice_mass(radar, refl_name='DBZ_CORR', zdr_name='ZDR_CORR',
                    temperature_name='sounding_temperature', height_name='height'):
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

    liquid_water_mass, ice_mass = csu_liquid_ice_mass.calc_liquid_ice_mass(refl, zdr, radar_z/1000, T=radar_T, method='cifelli')

    liquid_water_mass = {'data': liquid_water_mass, 'units': 'g m-3', 'long_name': \
                         'Liquid Water Content', 'standard_name': 'liquid_water_content'}
    ice_mass = {'data': ice_mass, 'units': 'g m-3', 'long_name': 'Ice Water Content',
                'standard_name': 'ice_water_content'}

    return liquid_water_mass, ice_mass


def nearest(items, pivot):
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
        closest_date = nearest(dtime, radar_start_date)
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
    cnt = 1
    snr = pyart.retrieve.calculate_snr_from_reflectivity(radar, refl_field=refl_field_name)
    # Sometimes the SNR is an empty array, this is due to the toa parameter.
    # Here we try to recalculate the SNR with a lower toa value.
    while snr['data'].count == 0:
        snr = pyart.retrieve.calculate_snr_from_reflectivity(radar, refl_field=refl_field_name, toa=25000-1000*cnt)
        cnt += 1
        # Break after the fifth attempt.
        if cnt > 5:
            break

    return z_dict, temp_info_dict, snr


def unfold_phidp_vdop(radar, phidp_name='PHIDP', phidp_bringi_name='PHIDP_BRINGI',
                      vel_name='VEL', unfold_vel=False):
    """
    Unfold PHIDP and refold Doppler velocity.

    Parameters:
    ===========
        radar:
            Py-ART radar structure.
        kdp_name:
            KDP field name.
        phidp_name: str
            PHIDP field name.
        vel_name: str
            VEL field name.
        unfold_vel: bool
            Switch the Doppler velocity refolding

    Returns:
    ========
        phidp_unfold: dict
            Unfolded PHIDP.
        vdop_refolded: dict
            Refolded Doppler velocity.
    """
    # Initialize returns
    phidp_unfold = None
    vdop_refolded = None

    # Extract data
    phidp = radar.fields[phidp_name]['data']
    phidp_bringi = radar.fields[phidp_bringi_name]['data'].filled(np.NaN)
    vdop = radar.fields[vel_name]['data'].filled(np.NaN)
    try:
        v_nyq_vel = radar.instrument_parameters['nyquist_velocity']['data'][0]
    except:
        v_nyq_vel = np.max(np.abs(vdop))

    # Create gatefilter on PHIDP Bringi (the unfolding is based upon PHIDP Bringi)
    gf = pyart.filters.GateFilter(radar)
    gf.exclude_masked(phidp_bringi_name)

    # Looking for folded area of PHIDP
    [beam, ray] = np.where(phidp_bringi < 0)
    print("Found {} negative values".format(len(beam)))
    apos = np.unique(beam)
    # Excluding the first 20 km.
    ray[ray <= 80] = 9999
    # Initializing empty array.
    posr = []
    posazi = []
    for onebeam in apos:
        # We exclude "noise" value by only taking into account beams that have a
        # significant amount of negative values (e.g. 5).
        if len(beam[beam == onebeam]) < 5:
            continue
        else:
            posr.append(np.nanmin(ray[beam == onebeam]))
            posazi.append(onebeam)

    # If there is no folding, Doppler does not have to be corrected.
    if len(posr) == 0:
        print("No posr found unfolding phidp")
        unfold_vel = False
    else:
        phidp = _unfold_phidp(deepcopy(phidp), posr, posazi)
        # Calculating the offset.
        tmp = deepcopy(phidp)
        tmp[tmp < 0]  = np.NaN
        phidp_offset = np.nanmean(np.nanmin(tmp, axis=1))
        if phidp_offset < 0 or phidp_offset > 90:
            # Offset too big or too low to be true, therefore it is not applied.
            phidp_unfold = phidp
        else:
            phidp_unfold = phidp - phidp_offset

    # Refold Doppler.
    if unfold_vel:
        vdop_refolded = _refold_vdop(deepcopy(vdop), v_nyq_vel, posr, posazi)

    return phidp_unfold, vdop_refolded


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
    if bobby_params:
        vdop_vel = pyart.correct.dealias_region_based(radar,
                                                      vel_field=vel_name,
                                                      gatefilter=gf,
                                                      nyquist_vel=v_nyq_vel,
                                                      skip_between_rays=2000)
    else:
        vdop_vel = pyart.correct.dealias_region_based(radar,
                                                      vel_field=vel_name,
                                                      gatefilter=gf,
                                                      nyquist_vel=v_nyq_vel)

    return vdop_vel
