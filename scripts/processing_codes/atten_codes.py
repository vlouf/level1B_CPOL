# Python Standard Library
import os
import copy
import datetime
from copy import deepcopy

# Other Libraries
import pyart
import numpy as np


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
    # Check if KDP is a masked array.
    if np.ma.isMaskedArray(kdp):
        kdp = kdp.filled(0)  # 0 is the neutral value for a sum
    else:
        kdp[np.isnan(kdp)] = 0

    kdp[:, :-40] = 0  # Removing the last gates because of artifacts created by the SOBEL window.
    kdp[kdp < 0] = 0
    kdp[kdp > 5] = 0

    atten_specific = alpha*kdp  # Bringi relationship
    atten_specific[np.isnan(atten_specific)] = 0
    # Path integrated attenuation
    atten = 2 * np.cumsum(atten_specific, axis=1) * dr

    if (atten > 10).sum() != 0:
        print("WARNING: be carfull, risk of overestimating attenuation.")
        print("Atten max is %f dB." % (atten.max()))

    return atten_specific, atten


def correct_attenuation_zdr(radar, zdr_name='ZDR', kdp_name='KDP_GG'):
    """
    Correct attenuation on differential reflectivity. KDP_GG has been
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

    atten_meta = {'data': atten_spec, 'units': 'dB/km',
                  'standard_name': 'specific_attenuation_zdr',
                  'long_name': 'Differential reflectivity specific attenuation'}

    return atten_meta, zdr_corr


def correct_attenuation_zh(radar, refl_name='DBZ', kdp_name='KDP_GG'):
    """
    Correct attenuation on reflectivity. KDP_GG has been
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
