"""
Codes for correcting and unfolding PHIDP as well as estimating KDP.

@title: phase_codes
@author: Valentin Louf <valentin.louf@monash.edu>
@institutions: Monash University and the Australian Bureau of Meteorology
@date: 15/08/2017

.. autosummary::
    :toctree: generated/

    phidp_giangrande
    wradlib_unfold_phidp
"""

# Python Standard Library
from copy import deepcopy

# Other Libraries
import pyart
import scipy
import numpy as np


def _kdp_sobel(phidp, dr=0.25, L=7):
    """
    Compute KDP from PHIDP using a Sobel filter. Inspired from a bit of PyART
    code.

    Parameters:
    ===========
    phidp: ndarray
        Differential phase.
    dr: float
        Gate spacing in km.
    L: int
        Window length.

    Returns:
    ========
    kdp: ndarray
        Specific differential phase.
    """
    window_len = L
    gate_spacing = dr

    # Create SOBEL window.
    sobel = 2. * np.arange(window_len) / (window_len - 1.0) - 1.0
    sobel = sobel / (abs(sobel).sum())
    sobel = sobel[::-1]

    # Compute KDP.
    kdp = (scipy.ndimage.filters.convolve1d(phidp, sobel, axis=1) / ((window_len / 3.0) * 2.0 * gate_spacing))
    # Smooth KDP alongside the azimuth.
    # kdp = pyart.correct.phase_proc.smooth_and_trim_scan(kdp.T, window_len=3).T
    return kdp


def _linear_despeckle(data, N=3):
    """
    Remove floating pixels in between NaNs in a multi-dimensional array. From
    wradlib.

    Parameters:
    ===========
    data: ndarray
        Note that the range dimension must be the last dimension of the input array.
    N: int
        Width of the window in which we check for speckle

    Returns:
    ========
    data: ndarray
        Input array despeckled
    """
    assert N in (3, 5), "Window size N for function linear_despeckle must be 3 or 5."

    axis = data.ndim - 1
    arr = np.ones(data.shape, dtype="i4")
    arr[np.isnan(data)] = 0
    arr_plus1 = np.roll(arr, shift=1, axis=axis)
    arr_minus1 = np.roll(arr, shift=-1, axis=axis)
    if N == 3:
        # for a window of size 3
        test = arr + arr_plus1 + arr_minus1
        data[np.logical_and(np.logical_not(np.isnan(data)), test < 2)] = np.nan
    else:
        # for a window of size 5
        arr_plus2 = np.roll(arr, shift=2, axis=axis)
        arr_minus2 = np.roll(arr, shift=-2, axis=axis)
        test = arr + arr_plus1 + arr_minus1 + arr_plus2 + arr_minus2
        data[np.logical_and(np.logical_not(np.isnan(data)), test < 3)] = np.nan
    # remove isolated pixels at the first gate
    secondgate = np.squeeze(np.take(data, range(1, 2), data.ndim - 1))
    data[..., 0][np.isnan(secondgate)] = np.nan
    return data


def _process_raw_phidp_vulpiani(phidp, dr, N_despeckle=5, L=7, niter=2):
    """
    Establish consistent PHIDP profiles from raw data. This approach is based
    on Vulpiani et al. (2012) and involves a two step procedure of PHIDP
    reconstruction. Processing of raw PHIDP data contains the following steps:
        - Despeckle
        - Initial KDP estimation
        - Removal of artifacts
        - Phase unfolding
        - PHIDP reconstruction using iterative estimation
          of KDP
    Original function from wradlib (modified). This one is much more faster
    than the original.

    Parameters:
    ===========
    phidp: ndarray
        array of shape (n azimuth angles, n range gates)
    dr: float
        Gate length in km
    N_despeckle: int
        N parameter of function dp.linear_despeckle
    L: int
        Window length
    niter: int
        Number of iterations in which phidp is retrieved from kdp
        and vice versa

    Returns:
    ========
    phidp : array of shape (n azimuth angles, n range gates)
        reconstructed phidp
    kdp : array of shape (n azimuth angles, n range gates)
        kdp estimate corresponding to phidp output
    """

    # despeckle
    phidp = _linear_despeckle(phidp, N_despeckle)
    # kdp retrieval first guess
    kdp = _kdp_sobel(phidp, dr=dr, L=L)
    # remove extreme values
    kdp[kdp > 20] = 0
    kdp[np.logical_and(kdp < -2, kdp > -20)] = 0

    # unfold phidp
    phidp = _unfold_phi_vulpiani(phidp, kdp)

    # clean up unfolded PhiDP
    phidp[phidp > 360] = np.nan

    # kdp retrieval second guess
    kdp = _kdp_sobel(phidp, dr=dr, L=L)
    kdp = np.nan_to_num(kdp)

    # remove remaining extreme values
    kdp[kdp > 20] = 0
    kdp[kdp < -2] = 0

    # start the actual phidp/kdp iteration
    for i in range(niter):
        # phidp from kdp through integration
        phidp = 2 * np.cumsum(kdp, axis=-1) * dr
        # kdp from phidp by convolution
        kdp = _kdp_sobel(phidp, dr=dr, L=L)
        # convert all NaNs to zeros (normally, this line can be assumed to be redundant)
        kdp = np.nan_to_num(kdp)

    return phidp, kdp


def _unfold_phi_vulpiani(phidp, kdp):
    """
    Alternative phase unfolding which completely relies on Kdp.
    This unfolding should be used in oder to iteratively reconstruct
    phidp and Kdp (From wradlib).

    Parameters:
    ===========
    phidp: ndarray
        Initial differential phase.
    kdp: ndarray
        Specific differential phase.

    Returns:
    ========
    phidp: ndarray
        Unfolded differential phase.
    """
    # unfold phidp
    shape = phidp.shape
    phidp = phidp.reshape((-1, shape[-1]))
    kdp = kdp.reshape((-1, shape[-1]))

    for beam in range(len(phidp)):
        below_th3 = kdp[beam] < -20
        try:
            idx1 = np.where(below_th3)[0][2]
            phidp[beam, idx1:] += 360
        except Exception:
            pass

    return phidp.reshape(shape)


def estimate_kdp_sobel(radar, window_len=7, phidp_name="PHIDP"):
    """
    Compute KDP from PHIDP using a Sobel filter. Inspired from a bit of PyART
    code.

    Parameters:
    ===========
    radar: dict
        Py-ART radar structure.
    window_len: int
        Window length.
    phidp_name: str
        Default PHIDP name.

    Returns:
    ========
    kdp_meta: dict
        Specific differential phase dictionary.
    """
    # Extract data
    phidp = radar.fields[phidp_name]['data'].copy()
    r = radar.range['data']
    dr = (r[1] - r[0]) / 1000  # km
    gate_spacing = dr

    # Smoothing PHIDP
    phidp = pyart.correct.phase_proc.smooth_and_trim_scan(phidp)

    # Create SOBEL window.
    sobel = 2. * np.arange(window_len) / (window_len - 1.0) - 1.0
    sobel = sobel / (abs(sobel).sum())
    sobel = sobel[::-1]

    # Compute KDP.
    kdp = (scipy.ndimage.filters.convolve1d(phidp, sobel, axis=1) / ((window_len / 3.0) * 2.0 * gate_spacing))

    # Removing aberrant values.
    kdp[kdp < -2] = 0
    kdp[kdp > 15] = 0
    kdp_meta = pyart.config.get_metadata('specific_differential_phase')
    kdp_meta['data'] = kdp

    return kdp_meta


def phidp_giangrande(radar, gatefilter, refl_field='DBZ', ncp_field='NCP',
                     rhv_field='RHOHV', phidp_field='PHIDP', kdp_field='KDP'):
    """
    Phase processing using the LP method in Py-ART. A LP solver is required,

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
    return phidp_gg, kdp_gg


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
    newkdp: array
        KDP estimation.
    """
    phi = deepcopy(radar.fields[phidp_name]['data'])
    r_range = radar.range['data'] / 1000.0
    dr = r_range[2] - r_range[1]
    newphi, newkdp = _process_raw_phidp_vulpiani(phi, dr)

    return newphi, newkdp
