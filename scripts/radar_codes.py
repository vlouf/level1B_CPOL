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
import skfuzzy as fuzz

from numba import jit, int32, float32

from scipy import ndimage, signal, integrate, interpolate
from scipy.stats import linregress
from scipy.ndimage.filters import convolve1d
from scipy.integrate import cumtrapz

from csu_radartools import csu_kdp


def compute_attenuation(kdp, alpha = 0.08, dr = 0.25):
    """
    Note: iflag_loc gives location flag
    c       1: Darwin data  2: SCSMEX data
    c Note: thrcrr is the rhohv threshold
    c       htL/htU are the lower and upper heights
    c       defining the melting layer , in km
    c       alpha is defined by Ah=alpha*Kdp
    c       beta is defined by  Ah-Av=beta*Kdp
    c       The settings below will override the
    c       default values set in plt.f main menu.
    """
    # alpha = 0.08
    # beta = 0.016
    kdp[kdp < 0] = 0
    atten_specific = alpha*kdp
    atten_specific[np.isnan(atten_specific)] = 0
    atten = 2 * np.cumsum(atten_specific, axis=1) * dr

    return atten_specific, atten


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
    snr_and_sounding
    TODO: Find Nearest date !!!!
    """

    if soundings_dir is None:
        soundings_dir = "/g/data2/rr5/vhl548/soudings_netcdf/"

    radar_start_date = netCDF4.num2date(radar.time['data'][0], radar.time['units'])
    sonde_pattern = datetime.datetime.strftime(radar_start_date, 'YPDN_%Y%m%d*')

    all_sonde_files = os.listdir(soundings_dir)
    sonde_name = fnmatch.filter(all_sonde_files, sonde_pattern)[0]

    interp_sonde = netCDF4.Dataset(os.path.join(soundings_dir, sonde_name))
    temperatures = interp_sonde.variables['temp'][:]
    times = interp_sonde.variables['time'][:]
    heights = interp_sonde.variables['height'][:]

    my_profile = pyart.retrieve.fetch_radar_time_profile(interp_sonde, radar)
    info_dict = {'long_name': 'Sounding temperature at gate',
                 'standard_name' : 'temperature',
                 'valid_min' : -100,
                 'valid_max' : 100,
                 'units' : 'degrees Celsius'}
    z_dict, temp_dict = pyart.retrieve.map_profile_to_gates(temperatures,
                                             my_profile['height'],
                                             radar)
    snr = pyart.retrieve.calculate_snr_from_reflectivity(radar, refl_field=refl_field_name)

    return z_dict, temp_dict, snr


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


@jit(cache=True)
def refold_vdop(vdop_art, v_nyq_vel, rth_position):
    tmp = vdop_art
    for j in range(len(rth_position)):
        i = rth_position[j]
        if i == 0:
            continue
        else:
            tmp[j, i:] += v_nyq_vel

    pos = (vdop_art > v_nyq_vel)
    tmp[pos] = tmp[pos] - 2*v_nyq_vel

    return tmp


def unfold_phi(phidp, kdp):
    """Alternative phase unfolding which completely relies on Kdp.
    This unfolding should be used in oder to iteratively reconstruct
    phidp and Kdp.
    Parameters
    ----------
    phidp : array of floats
    kdp : array of floats
    """
    # unfold phidp
    shape = phidp.shape
    phidp = phidp.reshape((-1, shape[-1]))
    kdp = kdp.reshape((-1, shape[-1]))

    rth_pos = np.zeros((phidp.shape[0]), dtype=np.int32)

    for beam in range(phidp.shape[0]):
        below_th3 = kdp[beam] < -20
        try:
            idx1 = np.where(below_th3)[0][2]
            phidp[beam, idx1:] += 360
            rth_pos[beam] = idx1
        except Exception:
            pass

    if len(rth_pos[rth_pos != 0]) == 0:
        rth_pos = None

    return phidp.reshape(shape), rth_pos

###############################################################################

def bringi_phidp_kdp(radar, refl_name='DBZ', phidp_name='PHIDP'):

    refl = radar.fields[refl_name]['data'].filled(fill_value = np.NaN)
    phidp = radar.fields[phidp_name]['data'].filled(fill_value = np.NaN)
    r = radar.range['data']

    rng2d, az2d = np.meshgrid(radar.range['data'], radar.azimuth['data'])
    dr = (r[1] - r[0])  # m

    kdN, fdN, sdN = csu_kdp.calc_kdp_bringi(dp=phidp, dz=refl, rng=rng2d/1000.0, thsd=6, gs=dr, window=3, bad=np.NaN)

    fdN = np.ma.masked_where(np.isnan(fdN), fdN)
    kdN = np.ma.masked_where(np.isnan(kdN), kdN)

    return fdN, kdN


def correct_attenuation_zh(radar, refl_name='DBZ', kdp_name='KDP'):

    r = radar.range['data']
    refl = radar.fields[refl_name]['data']
    kdp = radar.fields[kdp_name]['data']

    dr = (r[1] - r[0]) / 1000  # km

    atten_spec, atten = compute_attenuation(kdp, alpha=0.08, dr=dr)
    zh_corr = refl + atten

    atten_meta = {'data': atten_spec, 'units': 'dB/km', 'standard_name': 'specific_attenuation_zh',
                  'long_name': 'Reflectivity specific attenuation'}

    return atten_meta, zh_corr


def correct_attenuation_zdr(radar, zdr_name='ZDR', kdp_name='KDP'):

    r = radar.range['data']
    zdr = radar.fields[zdr_name]['data']
    kdp = radar.fields[kdp_name]['data']

    dr = (r[1] - r[0]) / 1000  # km

    atten_spec, atten = compute_attenuation(kdp, alpha=0.016, dr=dr)
    zdr_corr = zdr + atten

    atten_meta = {'data': atten_spec, 'units': 'dB/km', 'standard_name': 'specific_attenuation_zdr',
                  'long_name': 'Differential reflectivity specific attenuation'}

    return atten_meta, zdr_corr


def correct_rhohv(radar, rhohv_name='RHOHV', snr_name='SNR'):

    rhohv = radar.fields[rhohv_name]['data']
    snr = radar.fields[snr_name]['data']
    natural_snr = 10**(0.1*snr)
    rho_corr = rhohv / (1 + 1/natural_snr)

    return rho_corr


def correct_zdr(radar, zdr_name='ZDR_CORR', snr_name='SNR'):

    zdr = radar.fields[zdr_name]['data']
    snr = radar.fields[snr_name]['data']
    alpha = 1.48
    natural_zdr = 10**(0.1*zdr)
    natural_snr = 10**(0.1*snr)
    corr_zdr = 10*np.log10((alpha*natural_snr*natural_zdr) / (alpha*natural_snr + alpha - natural_zdr))

    return corr_zdr


def estimate_kdp(radar, gatefilter, phidp_name='PHIDP'):

    phidp = radar.fields[phidp_name]['data'].data
    r = radar.range['data']

    phidp[gatefilter.gate_excluded] = np.NaN
    dr = (r[1] - r[0]) / 1000  # km

    kdp_data = kdp_from_phidp_finitediff(phidp, dr=dr)
    kdp_field = {'data': kdp_data, 'units': 'degrees/km', 'standard_name': 'specific_differential_phase_hv',
                 'long_name': 'Specific differential phase (KDP)'}

    return kdp_field


def hydrometeor_classification(radar, refl_name='DBZ_CORR', zdr_name='ZDR_CORR',
                               kdp_name='KDP', rhohv_name='RHOHV',
                               temperature_name='sounding_temperature',
                               height_name='height'):

    refl = radar.fields['zcorr']['data']
    zdr = radar.fields['zdr_corr']['data']
    kdp = radar.fields['KDP']['data']
    rhohv = radar.fields['RHOHV']['data']
    radar_T = radar.fields['sounding_temperature']['data']
    radar_z = radar.fields['height']['data']

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


def liquid_ice_mass(radar, refl_name='DBZ_CORR', zdr_name='ZDR_CORR',
                    temperature_name='sounding_temperature', height_name='height'):

    refl = radar.fields['zcorr']['data']
    zdr = radar.fields['zdr_corr']['data']
    radar_T = radar.fields['sounding_temperature']['data']
    radar_z = radar.fields['height']['data']

    liquid_water_mass, ice_mass = csu_liquid_ice_mass.calc_liquid_ice_mass(refl, zdr, radar_z/1000, T=radar_T, method='cifelli')

    liquid_water_mass = {'data': liquid_water_mass, 'units': 'g m-3', 'long_name': 'Liquid Water Mass', 'standard_name': 'liquid_water_content'}
    ice_mass = {'data': ice_mass, 'units': 'g m-3', 'long_name': 'Ice Water Mass', 'standard_name': 'ice_water_content'}

    return liquid_water_mass, ice_mass


def unfold_phidp_vdop(radar, phidp_name='PHIDP', kdp_name='KDP', vel_name='VEL', unfold_vel=False):

    fdN = radar.fields[phidp_name]['data']
    kdN = radar.fields[kdp_name]['data']
    vdop_art = radar.fields[vel_name]['data']

    try:
        v_nyq_vel = radar.instrument_parameters['nyquist_velocity']['data'][0]
    except:
        v_nyq_vel = np.max(np.abs(vdop_art))

    phidp_unfold, pos_unfold = unfold_phi(fdN, kdN)

    vdop_refolded = None
    if unfold_vel:
        if pos_unfold is not None:
            vdop_refolded = refold_vdop(vdop_art, v_nyq_vel, pos_unfold)

    return phidp_unfold, vdop_refolded


def unfold_velocity(radar, my_gatefilter, vel_name=None):

    try:
        v_nyq_vel = radar.instrument_parameters['nyquist_velocity']['data'][0]
    except:
        vdop_art = radar.fields[vel_name]['data']
        v_nyq_vel = np.max(np.abs(vdop_art))

    vdop_vel = pyart.correct.dealias_region_based(radar, vel_field=vel_name, gatefilter=my_gatefilter, nyquist_vel=v_nyq_vel)
    vdop_vel['standard_name'] = radar.fields[vel_name]['standard_name']

    return vdop_vel
