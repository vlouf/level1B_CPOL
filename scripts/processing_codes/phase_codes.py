# Python Standard Library
from copy import deepcopy

# Other Libraries
import pyart
import wradlib
import numpy as np

from numba import jit, int32, float32
from csu_radartools import csu_kdp


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
    tmp[pos] = tmp[pos] - 2 * v_nyq_vel

    return tmp


def _kdp_from_phidp_finitediff(phidp, L=7, dr=1.):
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


# adapted smooth and trim function to work with 2dimensional arrays
def _smooth_and_trim_scan(x, window_len=35, window='hanning'):
    """
    Smooth data using a window with requested size.
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    Parameters
    ----------
    x : ndarray
        The input signal
    window_len: int
        The dimension of the smoothing window; should be an odd integer.
    window : str
        The type of window from 'flat', 'hanning', 'hamming', 'bartlett',
        'blackman' or 'sg_smooth'. A flat window will produce a moving
        average smoothing.
    Returns
    -------
    y : ndarray
        The smoothed signal with length equal to the input signal.
    """
    from scipy.ndimage.filters import convolve1d

    if x.ndim != 2:
        raise ValueError("smooth only accepts 2 dimension arrays.")
    if x.shape[1] < window_len:
        mess = "Input dimension 1 needs to be bigger than window size."
        raise ValueError(mess)
    if window_len < 3:
        return x
    valid_windows = ['flat', 'hanning', 'hamming', 'bartlett', 'blackman',
                     'sg_smooth']
    if window not in valid_windows:
        raise ValueError("Window is on of " + ' '.join(valid_windows))

    if window == 'flat':  # moving average
        w = np.ones(int(window_len), 'd')
    elif window == 'sg_smooth':
        w = np.array([0.1, .25, .3, .25, .1])
    else:
        w = eval('np.' + window + '(window_len)')

    y = convolve1d(x, w / w.sum(), axis=1)

    return y


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
    window_size = dr / 1000 * 4  # in km!!!

    kdN, fdN, sdN = csu_kdp.calc_kdp_bringi(phidp, refl, rng2d / 1000.0, gs=dr, window=window_size, bad=-9999)

    fdN = np.ma.masked_where(fdN == -9999, fdN)
    kdN = np.ma.masked_where(kdN == -9999, kdN)

    return fdN, kdN


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

    kdp_data = _kdp_from_phidp_finitediff(phidp, dr=dr)
    kdp_field = {'data': kdp_data, 'units': 'degrees/km', 'standard_name': 'specific_differential_phase_hv',
                 'long_name': 'Specific differential phase (KDP)'}

    return kdp_field


def kdp_phidp_disdro_darwin(radar, refl_field="DBZ", zdr_field="ZDR"):
    """
    Estimating PHIDP and KDP using the self-consistency relationship.

    Parameters:
    ===========
        radar:
            Py-ART radar structure.
        refl_field: str
            Reflectivity field name.
        zdr_field: str
            Differential reflectivity field name.

    Returns:
    ========
        kdp_field:
            KDP estimation.
        phidp_field:
            PHIDP estimation
    """
    # Disdro relationship.
    myfit = np.poly1d([-1.55277347e-06, 1.34659900e-05, -5.13632902e-05, 9.95614998e-05])

    # Extract data
    dbz = deepcopy(radar.fields[refl_field]['data'].filled(np.NaN))
    zdr = deepcopy(radar.fields[zdr_field]['data'].filled(np.NaN))
    r = radar.range['data']
    dr = r[1] - r[0]

    # Compute KDP
    kdp = 10**(dbz / 10.0) * myfit(zdr)
    kdp[dbz > 50] = np.NaN  # Clutter
    kdp[kdp < -2] = np.NaN
    kdp = np.ma.masked_where(np.isnan(kdp), kdp)

    phidp = dr / 1000 * np.nancumsum(kdp, axis=1)

    phidp_field = pyart.config.get_metadata('differential_phase')
    phidp_field['data'] = phidp
    phidp_field["description"] = "Simulation using Darwin disdrometer."

    kdp_field = {'data': kdp, 'units': 'degrees/km', 'standard_name': 'simulated_specific_differential_phase_hv',
                 'long_name': 'Specific differential phase (KDP) simulated',
                 "description": "Simulation using Darwin disdrometer."}

    return kdp_field, phidp_field


def phidp_giangrande(radar, gatefilter, refl_field='DBZ', ncp_field='NCP',
                     rhv_field='RHOHV', phidp_field='PHIDP', kdp_field='KDP'):
    """
    Phase processing using the LP method in Py-ART. A LP solver is required,
    I only have pyglpk and cvxopt, that's why I'm specifying it while calling
    pyart.correct.phase_proc_lp.

    Parameters:
    ===========
        radar:
            Py-ART radar structure.
        refl_field: str
            Reflectivity field name.
        ncp_field: str
            Normalised coherent power field name.
        rhv_field: str
            Cross correlation ration field name.
        phidp_field: str
            Phase field name.
        kdp_field: str
            Specific phase field name.

    Returns:
    ========
        phidp_gg: dict
            Field dictionary containing processed differential phase shifts.
        kdp_gg: dict
            Field dictionary containing recalculated differential phases.
    """
    phidp_gg, kdp_gg = pyart.correct.phase_proc_lp(radar, 0.0,
                                                   LP_solver='cylp',
                                                   refl_field=refl_field,
                                                   ncp_field=ncp_field,
                                                   rhv_field=rhv_field,
                                                   phidp_field=phidp_field,
                                                   kdp_field=kdp_field)
    phi = phidp_gg['data']
    phidp_gg['data'] = _smooth_and_trim_scan(phi.T, window_len=5).T

    return phidp_gg, kdp_gg


def refold_phidp(radar, phidp_name='PHIDP'):
    """
    Refold PHIDP.

    Parameters:
    ===========
        radar:
            Py-ART radar structure.
        phidp_name: str
            PHIDP field name.

    Returns:
    ========
        phi: array
            Refolded PHIDP.
    """
    phi = deepcopy(radar.fields[phidp_name]['data'])
    pos = (phi > 0)

    if np.nanmin(phi) < 100:
        phi[pos] -= 180
        phi[~pos] += 180
    else:
        phi[pos] -= 90
        phi[~pos] += 90

    return phi


def smooth_phidp(radar, phidp_name='PHIDP'):
    """
    PHIDP estimation by smoothing its derivative.

    Parameters:
    ===========
        radar:
            Py-ART radar structure.
        phidp_name: str
            PHIDP field name.

    Returns:
    ========
        nphi: array
            Refolded PHIDP.
    """
    phi = deepcopy(radar.fields[phidp_name]['data'])
    nphi = _smooth_and_trim_scan(phi, window_len=10)
    kdp = np.gradient(nphi, axis=1)
    kdp[kdp < 0] = 0
    nphi = np.cumsum(kdp, axis=1) * .25
    nphi = _smooth_and_trim_scan(nphi.T, window_len=15).T

    return nphi


def wradlib_unfold_phidp(radar, phidp_name="PHIDP"):
    """
    Processing raw PHIDP using Vulpiani algorithm in Wradlib.

    Parameters:
    ===========
    radar:
        Py-ART radar structure.
    phidp_name: str
        PHIDP field name.

    Returns:
    ========
        newphi: array
            Refolded PHIDP.
        newkdp: arrau
            KDP estimation.
    """
    phi = deepcopy(radar.fields[phidp_name]['data'])
    r_range = radar.range['data'] / 1000.0
    dr = r_range[2] - r_range[1]
    newphi, newkdp = wradlib.dp.process_raw_phidp_vulpiani(phi, dr, L=15, niter=2,)

    return newphi, newkdp
