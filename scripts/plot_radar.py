import os
import datetime
import matplotlib
matplotlib.use('Agg')  # <- Reason why matplotlib is imported first.

import pyart
import matplotlib.colors as colors
import matplotlib.pyplot as pl

import numpy as np


def adjust_fhc_colorbar_for_pyart(cb):
    """
    adjust_fhc_colorbar_for_pyart
    """
    cb.set_ticks(np.arange(1.4, 10, 0.9))
    cb.ax.set_yticklabels(['Drizzle', 'Rain', 'Ice Crystals', 'Aggregates',
                           'Wet Snow', 'Vertical Ice', 'LD Graupel',
                           'HD Graupel', 'Hail', 'Big Drops'])
    cb.ax.set_ylabel('')
    cb.ax.tick_params(length=0)
    return cb


def plot_figure_check(radar, gatefilter, outfilename, radar_date, path_save_figure=None):
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
    if path_save_figure is None:
        path_save_figure = "/g/data2/rr5/vhl548/CPOL_PROD_1b/FIGURE_CHECK/"
    # Extracting year and date.
    year = str(radar_date.year)
    datestr = radar_date.strftime("%Y%m%d")
    # Path for saving Figures.
    outfile_path = os.path.join(path_save_figure, year, datestr)

    # Checking if output directory exists. Creating them otherwise.
    if not os.path.isdir(os.path.join(path_save_figure, year)):
        os.mkdir(os.path.join(path_save_figure, year))
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
    fig, the_ax = pl.subplots(6, 2, figsize=(12, 30), sharex=True, sharey=True)
    the_ax = the_ax.flatten()
    # Plotting reflectivity
    gr.plot_ppi('DBZ', ax = the_ax[0], vmin=-10, vmax=70)
    gr.plot_ppi('DBZ_CORR', ax = the_ax[1], gatefilter=gatefilter, cmap=pyart.graph.cm.NWSRef, vmin=-10, vmax=70)

    gr.plot_ppi('ZDR', ax = the_ax[2], vmin=-5, vmax=10, cmap='rainbow')  # ZDR
    gr.plot_ppi('ZDR_CORR', ax = the_ax[3], gatefilter=gatefilter, vmin=-5, vmax=10, cmap='rainbow')

    gr.plot_ppi('PHIDP', ax = the_ax[4], vmin=0, vmax=180, cmap='OrRd')
    try:
        gr.plot_ppi('PHIDP_CORR', ax = the_ax[5], gatefilter=gatefilter, vmin=0, vmax=180, cmap='OrRd')
    except KeyError:
        gr.plot_ppi('PHIDP', ax = the_ax[5], gatefilter=gatefilter, vmin=0, vmax=180, cmap='OrRd')

    gr.plot_ppi('VEL', ax = the_ax[6], cmap=pyart.graph.cm.NWSVel, vmin=-40, vmax=40)
    gr.plot_ppi('VEL_UNFOLDED', ax = the_ax[7], gatefilter=gatefilter, cmap=pyart.graph.cm.NWSVel, vmin=-40, vmax=40)

    gr.plot_ppi('SNR', ax = the_ax[8], cmap='OrRd', vmin=0, vmax=80)
    gr.plot_ppi('RHOHV', ax = the_ax[9], vmin=0, vmax=1, norm=colors.LogNorm(vmin=0.4, vmax=1), cmap='rainbow')

    gr.plot_ppi('sounding_temperature', ax = the_ax[10], cmap='OrRd', vmin=-10, vmax=30)
    gr.plot_ppi('KDP', ax = the_ax[11], vmin=0, vmax=1, cmap='OrRd')

    for ax_sl in the_ax:
        gr.plot_range_rings([50, 100, 150], ax=ax_sl)
        ax_sl.set_aspect(1)
        ax_sl.set_xlim(-150, 150)
        ax_sl.set_ylim(-150, 150)

    pl.tight_layout()
    pl.savefig(outfile)  # Saving figure.
    pl.close()

    # HYDRO CLASSIFICATION
    hid_colors = ['White', 'LightBlue', 'MediumBlue', 'DarkOrange', 'LightPink',
                  'Cyan', 'DarkGray', 'Lime', 'Yellow', 'Red', 'Fuchsia']
    cmaphid = colors.ListedColormap(hid_colors)

    fig, ax0 = pl.subplots(1, 1, figsize = (6, 5))
    gr.plot_ppi('HYDRO', vmin=0, vmax=10, cmap=cmaphid)
    gr.cbs[-1] = adjust_fhc_colorbar_for_pyart(gr.cbs[-1])
    pl.savefig(outfile.replace(".png", "_HYDROCLASS.png"))

    return None
