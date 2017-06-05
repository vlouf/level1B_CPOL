# Python Standard Library
from copy import deepcopy

# Other Libraries
import pyart
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
    tmp[pos] = tmp[pos] - 2*v_nyq_vel

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
    window_size = dr/1000*4  # in km!!!

    kdN, fdN, sdN = csu_kdp.calc_kdp_bringi(phidp, refl, rng2d/1000.0, gs=dr, window=window_size, bad=-9999)

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
    kdp =  10**(dbz/10.0) * myfit(zdr)
    kdp[dbz > 50] = np.NaN  # Clutter
    kdp[kdp < -2] = np.NaN
    kdp = np.ma.masked_where(np.isnan(kdp), kdp)

    phidp = dr/1000*np.nancumsum(kdp, axis=1)

    phidp_field = pyart.config.get_metadata('differential_phase')
    phidp_field['data'] = phidp
    phidp_field["description"] = "Simulation using Darwin disdrometer."

    kdp_field = {'data': kdp, 'units': 'degrees/km', 'standard_name': 'simulated_specific_differential_phase_hv',
                 'long_name': 'Specific differential phase (KDP) simulated',
                 "description": "Simulation using Darwin disdrometer."}

    return kdp_field, phidp_field


def phidp_giangrande(myradar, gatefilter, refl_field='DBZ', ncp_field='NCP',
                     rhv_field='RHOHV', phidp_field='PHIDP', kdp_field='KDP'):
    """
    Phase processing using the LP method in Py-ART. A LP solver is required,
    I only have pyglpk and cvxopt, that's why I'm specifying it while calling
    pyart.correct.phase_proc_lp.

    Parameters:
    ===========
        myradar:
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
    def _filter_hardcoding(my_array, nuke_filter, bad=-9999):
        # Hardcoding filter sub-function.
        filt_array = np.ma.masked_where(nuke_filter.gate_excluded, my_array)
        filt_array.set_fill_value(bad)
        filt_array = filt_array.filled(fill_value=bad)
        to_return = np.ma.masked_where(filt_array == bad, filt_array)
        return to_return
    # Deepcopy the radar structure.
    radar = deepcopy(myradar)
    try:
        # Looking for an NCP field.
        radar.fields[ncp_field]
    except KeyError:
        # Create NCP field. The radar=deepcopy(myradar) is here so that the
        # "fake" NCP field we're adding is temporary, i.e. it is not added to
        # the 'main' radar stucture but just a copy of it that will disappear
        # when this function returns.
        tmp = np.zeros_like(radar.fields[rhv_field]['data']) + 1
        radar.add_field_like(rhv_field, ncp_field, tmp)  # Adding a fake NCP field.

    for mykey in [refl_field, ncp_field, rhv_field, phidp_field, kdp_field]:
        radar.fields[mykey]['data'] = _filter_hardcoding(radar.fields[mykey]['data'], gatefilter)

    phidp_gg, kdp_gg = pyart.correct.phase_proc_lp(radar, 0.0,
                                                   LP_solver='cylp',
                                                   refl_field=refl_field,
                                                   ncp_field=ncp_field,
                                                   rhv_field=rhv_field,
                                                   phidp_field=phidp_field,
                                                   kdp_field=kdp_field)

    return phidp_gg, kdp_gg


def unfold_phidp_vdop(radar, phidp_name='PHIDP', phidp_bringi_name='PHIDP_BRINGI', vel_name='VEL'):
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
    except (KeyError, IndexError):
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
        unfold_vel = True
        phidp = _unfold_phidp(deepcopy(phidp), posr, posazi)
        # Calculating the offset.
        tmp = deepcopy(phidp)
        tmp[tmp < 0] = np.NaN
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
