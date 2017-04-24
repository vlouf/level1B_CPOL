import os
import datetime
import matplotlib
matplotlib.use('Agg')  # <- Reason why matplotlib is imported first.

import pyart
import matplotlib.colors as colors
import matplotlib.pyplot as pl

import numpy as np


def plot_figure_check(radar, gatefilter, outfilename, radar_date):
    """
    Plot figure of old/new radar parameters for checking purpose.

    Parameters:
    ===========
        radar:
            Py-ART radar structure.
        gatefilter:
            The Gate filter.
        outfilename: str
            Name given to the output netcdf data file.
        radar_date: datetime
            Datetime stucture of the radar data.
    """
    # Extracting year and date.
    year = str(radar_date.year)
    datestr = radar_date.strftime("%Y%m%d")
    # Path for saving Figures.
    outfile_path = os.path.join(FIGURE_CHECK_PATH, year, datestr)

    # Checking if output directory exists. Creating them otherwise.
    if not os.path.isdir(os.path.join(FIGURE_CHECK_PATH, year)):
        os.mkdir(os.path.join(FIGURE_CHECK_PATH, year))
    if not os.path.isdir(outfile_path):
        os.mkdir(outfile_path)

    # Checking if figure already exists.
    outfile = os.path.basename(outfilename)
    outfile = outfile[:-2] + "png"
    outfile = os.path.join(outfile_path, outfile)
    # if os.path.isfile(outfile):
    #     return None

    # Initializing figure.
    gr = pyart.graph.RadarDisplay(radar)
    fig, the_ax = pl.subplots(6, 2, figsize=(10, 30), sharex=True, sharey=True)
    the_ax = the_ax.flatten()
    # Plotting reflectivity
    gr.plot_ppi('DBZ', ax = the_ax[0], vmin=-10, vmax=70)
    gr.plot_ppi('DBZ_CORR', ax = the_ax[1], gatefilter=gatefilter, cmap=pyart.graph.cm.NWSRef, vmin=-10, vmax=70)

    gr.plot_ppi('ZDR', ax = the_ax[2], vmin=-5, vmax=10)  # ZDR
    gr.plot_ppi('ZDR_CORR', ax = the_ax[3], gatefilter=gatefilter, vmin=-5, vmax=10)

    gr.plot_ppi('PHIDP', ax = the_ax[4], vmin=0, vmax=180, cmap='jet')
    try:
        gr.plot_ppi('PHIDP_CORR', ax = the_ax[5], gatefilter=gatefilter, vmin=0, vmax=180, cmap='jet')
    except KeyError:
        gr.plot_ppi('PHIDP', ax = the_ax[5], gatefilter=gatefilter, vmin=0, vmax=180, cmap='jet')

    gr.plot_ppi('VEL', ax = the_ax[6], cmap=pyart.graph.cm.NWSVel, vmin=-40, vmax=40)
    gr.plot_ppi('VEL_UNFOLDED', ax = the_ax[7], gatefilter=gatefilter, cmap=pyart.graph.cm.NWSVel, vmin=-40, vmax=40)

    gr.plot_ppi('SNR', ax = the_ax[8])
    gr.plot_ppi('RHOHV', ax = the_ax[9], vmin=0, vmax=1)

    gr.plot_ppi('sounding_temperature', ax = the_ax[10], cmap='YlOrRd', vmin=-10, vmax=30)
    gr.plot_ppi('KDP', ax = the_ax[11], vmin=-1, vmax=1)
    # gr.plot_ppi('LWC', ax = the_ax[11], norm=colors.LogNorm(vmin=0.01, vmax=10), gatefilter=gatefilter, cmap='YlOrRd')

    for ax_sl in the_ax:
        gr.plot_range_rings([50, 100, 150], ax=ax_sl)
        ax_sl.axis((-150, 150, -150, 150))

    pl.tight_layout()
    pl.savefig(outfile)  # Saving figure.

    return None
