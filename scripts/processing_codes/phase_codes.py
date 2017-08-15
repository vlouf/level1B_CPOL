# Python Standard Library
from copy import deepcopy

# Other Libraries
import pyart
import wradlib


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
    newphi, newkdp = wradlib.dp.process_raw_phidp_vulpiani(phi, dr)

    return newphi, newkdp
