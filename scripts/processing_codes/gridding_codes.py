import os
import pyart
import datetime


def gridding_radar_70km(radar, radar_date, outpath):
    """
    Map a single radar to a Cartesian grid of 70 km range and 1 km resolution.

    Parameters:
    ===========
        radar:
            Py-ART radar structure.
        radar_date: datetime
            Datetime stucture of the radar data.
    """
    # Extracting year, date, and datetime.
    year = str(radar_date.year)
    datestr = radar_date.strftime("%Y%m%d")
    datetimestr = radar_date.strftime("%Y%m%d_%H%M")
    fname = "CPOL_{}_GRIDS_2500m.nc".format(datetimestr)

    # Output directory
    outdir_150km = os.path.join(outpath, "GRID_150km_2500m")
    if not os.path.isdir(outdir_150km):
        os.mkdir(outdir_150km)

    outdir_150km = os.path.join(outdir_150km, year)
    if not os.path.isdir(outdir_150km):
        os.mkdir(outdir_150km)

    outdir_150km = os.path.join(outdir_150km, datestr)
    if not os.path.isdir(outdir_150km):
        os.mkdir(outdir_150km)

    # Output file name
    outfilename = os.path.join(outdir_150km, fname)

    # exclude masked gates from the gridding
    my_gatefilter = pyart.filters.GateFilter(radar)
    my_gatefilter.exclude_transition()
    my_gatefilter.exclude_masked('DBZ')

    # Gridding
    grid_150km = pyart.map.grid_from_radars(
        radar, gatefilters=my_gatefilter,
        grid_shape=(41, 117, 117),
        grid_limits=((0, 20000), (-145000.0, 145000.0), (-145000.0, 145000.0)))

    # Saving data.
    grid_150km.write(outfilename, arm_time_variables=True)

    return None


def gridding_radar_150km(radar, radar_date, outpath):
    """
    Map a single radar to a Cartesian grid of 150 km range and 2.5km resolution.

    Parameters:
    ===========
        radar:
            Py-ART radar structure.
        my_gatefilter:
            The Gate filter.
        radar_date: datetime
            Datetime stucture of the radar data.
    """
    # Extracting year, date, and datetime.
    year = str(radar_date.year)
    datestr = radar_date.strftime("%Y%m%d")
    datetimestr = radar_date.strftime("%Y%m%d_%H%M")
    fname = "CPOL_{}_GRIDS_1000m.nc".format(datetimestr)

    # Output directory
    outdir_70km = os.path.join(outpath, "GRID_70km_1000m")
    if not os.path.isdir(outdir_70km):
        os.mkdir(outdir_70km)

    outdir_70km = os.path.join(outdir_70km, year)
    if not os.path.isdir(outdir_70km):
        os.mkdir(outdir_70km)

    outdir_70km = os.path.join(outdir_70km, datestr)
    if not os.path.isdir(outdir_70km):
        os.mkdir(outdir_70km)

    # Output file name
    outfilename = os.path.join(outdir_70km, fname)

    # exclude masked gates from the gridding
    my_gatefilter = pyart.filters.GateFilter(radar)
    my_gatefilter.exclude_transition()
    my_gatefilter.exclude_masked('DBZ')

    # Gridding
    grid_70km = pyart.map.grid_from_radars(
        radar, gatefilters=my_gatefilter,
        grid_shape=(41, 141, 141),
        grid_limits=((0, 20000), (-70000.0, 70000.0), (-70000.0, 70000.0)))

    # Saving data.
    grid_70km.write(outfilename, arm_time_variables=True)

    return None