import numpy as np
from numba import jit, int32, float32


@jit(nopython=True, cache=True)
def unfold_phidp(the_phidp, rpos, azipos):
    tmp = the_phidp
    for i, j in zip(azipos, rpos):
        tmp[i, j:] += 180
    return tmp


@jit(cache=True)
def refold_vdop(vdop_art, v_nyq_vel, rpos, azipos):
    tmp = vdop_art
    for i, j in zip(azipos, rpos):
        tmp[i, j:] += v_nyq_vel

    pos = (vdop_art > v_nyq_vel)
    tmp[pos] = tmp[pos] - 2*v_nyq_vel

    return tmp
