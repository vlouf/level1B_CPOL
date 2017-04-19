import numpy as np
from numba import jit, int32, float32


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


@jit(nopython=True, cache=True)
def smooth_data_unfolding(mydata, window=10):
    rslt = mydata
    for ray in range(mydata.shape[1]):
        for beam in range(1, mydata.shape[0] - 1):
            rslt[beam, ray] = np.nanmean(rslt[beam-1:beam+1, ray])
    for beam in range(mydata.shape[0]):
        for ray in range(window, mydata.shape[1] - window):
            rslt[beam, ray] = np.nanmean(rslt[beam, ray-window:ray+window])

    return rslt


@jit
def smooth_data(mydata, window=10):
    rslt = np.zeros(mydata.shape)
    for cnt, beam in enumerate(mydata):
        rslt[cnt, :] = np.convolve(np.ones((window,)), beam, 'same') / window
    for cnt, ray in enumerate(mydata.T):
        rslt[:, cnt] = np.convolve([0.5,1,0.5], ray, 'same') / 2

    return rslt


@jit(nopython=True, cache=True)
def redress_data(mydata, thrld):
    rslt = mydata

    for beam in range(mydata.shape[0]):
        for ray in range(50, mydata.shape[1]):
            px0 = mydata[beam, ray-1]
            px1 = mydata[beam, ray]

            if np.isnan(px0) or np.isnan(px1):
                continue

            if (px1 < px0) & (px1 < thrld):
                rslt[beam, ray] = px0

    return rslt


@jit(nopython=True, cache=True)
def get_fold_position(mydata):
    window = 5

    tmp = mydata
    rth_pos = np.zeros((tmp.shape[0]), dtype=np.int32)

    for beam in range(mydata.shape[0]):
        for ray in range(50, mydata.shape[1]):
#             phidp_mean0 = np.nanmean(mydata[beam, ray-window:ray])
#             phidp_mean1 = np.nanmean(mydata[beam, ray:ray+window])
#             if (phidp_mean0 > phidp_mean1) & (phidp_mean1 < 0):
#                 rth_pos[beam] = ray
#                 break
#            phidp_mean = np.nanmean(mydata[beam, ray-1:ray+1])
            phidp_mean = mydata[beam, ray]
            if phidp_mean < -10:
                rth_pos[beam] = ray-1
                break

    return rth_pos


@jit(nopython=True, cache=True)
def unfold_phidp(the_phidp, rth_position):
    tmp = the_phidp
    for j in range(len(rth_position)):
        i = rth_position[j]
        if i == 0:
            continue
        else:
            tmp[j, i:] += 180
    return tmp
