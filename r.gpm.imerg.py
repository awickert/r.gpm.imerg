#!/usr/bin/python3
############################################################################
#
# MODULE:       r.gpm.imerg
#
# AUTHOR(S):    Andrew Wickert
#
# PURPOSE:      Import NASA GPM IMERG precipitation data into GRASS GIS
#
# COPYRIGHT:    (c) 2026 Andrew Wickert
#
#               This program is free software under the GNU General Public
#               License (>=v2). Read the file COPYING that comes with GRASS
#               for details.
#
#############################################################################

#%module
#% description: Import NASA GPM IMERG global precipitation data
#% keyword: raster
#% keyword: import
#% keyword: precipitation
#% keyword: GPM
#% keyword: IMERG
#% keyword: climate
#%end

#%option G_OPT_R_BASENAME_OUTPUT
#%  key: output
#%  label: Base name for output raster maps
#%  required: yes
#%end

#%option
#%  key: variable
#%  type: string
#%  label: IMERG variable to import
#%  options: precipitationCal,precipitationUncal,HQprecipitation,IRprecipitation,randomError,probabilityLiquidPrecipitation
#%  description: precipitationCal=calibrated precip(mm/hr), precipitationUncal=uncalibrated precip(mm/hr), HQprecipitation=high-quality microwave precip(mm/hr), IRprecipitation=IR-only precip(mm/hr), randomError=random error estimate(mm/hr), probabilityLiquidPrecipitation=liquid precip probability(%)
#%  answer: precipitationCal
#%  required: no
#%end

#%option
#%  key: product
#%  type: string
#%  label: Temporal product
#%  options: HH,D,M
#%  description: HH=half-hourly (30 min), D=daily, M=monthly
#%  answer: D
#%  required: no
#%end

#%option
#%  key: run
#%  type: string
#%  label: Processing run
#%  options: E,L,F
#%  description: E=Early (~4 hr lag), L=Late (~12 hr lag), F=Final (~3.5 month lag, most accurate). Monthly only available as F.
#%  answer: F
#%  required: no
#%end

#%option
#%  key: start
#%  type: string
#%  label: Start date (YYYY-MM-DD)
#%  required: yes
#%end

#%option
#%  key: end
#%  type: string
#%  label: End date (YYYY-MM-DD; default: start date)
#%  required: no
#%end

#%option
#%  key: resample
#%  type: string
#%  label: Resampling method for r.import
#%  options: nearest,bilinear,bicubic,lanczos,bilinear_f,bicubic_f,lanczos_f
#%  answer: bilinear
#%  required: no
#%end

#%option
#%  key: username
#%  type: string
#%  label: NASA Earthdata username (password prompted at runtime)
#%  description: If omitted, credentials are read from EARTHDATA_USERNAME/EARTHDATA_PASSWORD environment variables or ~/.netrc
#%  required: no
#%end

#%flag
#%  key: t
#%  description: Register output maps as a space-time raster dataset (strds)
#%end

import os
import re
import subprocess
import tempfile
import atexit
from datetime import date, datetime, timedelta

import grass.script as gs

if os.path.exists('/usr/share/proj/proj.db'):
    os.environ['PROJ_DATA'] = '/usr/share/proj'

TMPFILES = []
TMPDIRS  = []

# earthaccess short names keyed by (product, run)
_SHORT_NAME = {
    ('HH', 'E'): 'GPM_3IMERGHHE',
    ('HH', 'L'): 'GPM_3IMERGHHL',
    ('HH', 'F'): 'GPM_3IMERGHHF',
    ('D',  'E'): 'GPM_3IMERGDE',
    ('D',  'L'): 'GPM_3IMERGDL',
    ('D',  'F'): 'GPM_3IMERGDF',
    ('M',  'F'): 'GPM_3IMERGMF',
}

_DATASET_START = date(2000, 6, 1)   # IMERG V07 starts 2000-06-01 for HH/D

# IMERG global grid parameters (0.1° WGS84)
_NLON    = 3600
_NLAT    = 1800
_LON_MIN = -180.0
_LAT_MAX =   90.0
_RES     =    0.1

_FILL_VALUE = -9999.9


def cleanup():
    import shutil
    for f in TMPFILES:
        try:
            os.remove(f)
        except OSError:
            pass
    for d in TMPDIRS:
        try:
            shutil.rmtree(d)
        except OSError:
            pass


def parse_imerg_filename(path):
    """Return (start_dt, end_dt) parsed from an IMERG HDF5 filename."""
    bn = os.path.basename(path)
    m = re.search(r'(\d{8})-S(\d{6})-E(\d{6})', bn)
    if not m:
        raise ValueError("Cannot parse IMERG filename: {}".format(bn))
    date_str, s_str, e_str = m.group(1), m.group(2), m.group(3)
    d = date(int(date_str[:4]), int(date_str[4:6]), int(date_str[6:]))
    start_dt = datetime(d.year, d.month, d.day,
                        int(s_str[:2]), int(s_str[2:4]), int(s_str[4:]))
    end_dt   = datetime(d.year, d.month, d.day,
                        int(e_str[:2]), int(e_str[2:4]), int(e_str[4:]))
    if end_dt < start_dt:
        end_dt += timedelta(days=1)
    return start_dt, end_dt


def hdf5_to_geotiff(hdf5_path, variable, out_tif):
    """Extract variable from IMERG HDF5, write a north-up WGS84 GeoTiff."""
    import h5py
    import numpy as np

    with h5py.File(hdf5_path, 'r') as f:
        grid = f['/Grid']
        lat_arr = grid['lat'][:]   # 1-D
        lon_arr = grid['lon'][:]   # 1-D
        ds  = grid[variable]
        raw = ds[0, :, :]          # squeeze time dim → 2-D
        fv  = ds.attrs.get('_FillValue', np.float32(_FILL_VALUE))

    nlat = len(lat_arr)  # expected 1800
    nlon = len(lon_arr)  # expected 3600

    # Auto-detect dimension order: raw.shape is either (nlon, nlat) or (nlat, nlon)
    if raw.shape == (nlon, nlat):
        data = raw.T                  # → (nlat, nlon) = (lat, lon)
    elif raw.shape == (nlat, nlon):
        data = raw
    else:
        gs.fatal("Unexpected variable shape {} for lat={} lon={}".format(
            raw.shape, nlat, nlon))

    # Flip to north-up: lat_arr[0] is southernmost; GeoTiff row 0 must be northernmost
    if lat_arr[0] < lat_arr[-1]:
        data = data[::-1, :]

    data = np.ascontiguousarray(data.astype('<f4'))  # little-endian float32

    # Replace fill values
    fill = float(fv)
    data[np.abs(data - fill) < 1.0] = _FILL_VALUE

    # Write raw binary
    fd, raw_path = tempfile.mkstemp(suffix='.raw')
    os.close(fd)
    TMPFILES.append(raw_path)
    data.tofile(raw_path)

    # Build VRT that describes the raw file with correct georeferencing
    # GT: (lon_min, x_res, 0, lat_max, 0, -y_res)
    lon_min = float(lon_arr.min()) - _RES / 2.0
    lat_max = float(lat_arr.max()) + _RES / 2.0
    vrt_path = raw_path.replace('.raw', '.vrt')
    TMPFILES.append(vrt_path)
    line_offset = data.shape[1] * 4   # ncols * bytes_per_float32

    with open(vrt_path, 'w') as vf:
        vf.write(
            '<VRTDataset rasterXSize="{ncol}" rasterYSize="{nrow}">\n'
            '  <SRS>EPSG:4326</SRS>\n'
            '  <GeoTransform>{x0}, {dx}, 0, {y0}, 0, {neg_dy}</GeoTransform>\n'
            '  <VRTRasterBand dataType="Float32" band="1"'
            ' subClass="VRTRawRasterBand">\n'
            '    <SourceFilename relativeToVRT="0">{raw}</SourceFilename>\n'
            '    <ImageOffset>0</ImageOffset>\n'
            '    <PixelOffset>4</PixelOffset>\n'
            '    <LineOffset>{lo}</LineOffset>\n'
            '    <ByteOrder>LSB</ByteOrder>\n'
            '    <NoDataValue>{fv}</NoDataValue>\n'
            '  </VRTRasterBand>\n'
            '</VRTDataset>\n'.format(
                ncol=data.shape[1], nrow=data.shape[0],
                x0=lon_min, dx=_RES,
                y0=lat_max, neg_dy=-_RES,
                raw=raw_path, lo=line_offset,
                fv=_FILL_VALUE,
            )
        )

    r = subprocess.run(
        ['gdal_translate', '-q', '-of', 'GTiff', vrt_path, out_tif],
        capture_output=True,
    )

    os.remove(raw_path)
    TMPFILES.remove(raw_path)
    os.remove(vrt_path)
    TMPFILES.remove(vrt_path)

    if r.returncode != 0:
        gs.warning("gdal_translate failed: {}".format(r.stderr.decode().strip()))
        return False
    return True


def register_strds(output, map_entries, product):
    """Create a strds and register the supplied maps."""
    gs.run_command(
        't.create',
        output=output,
        type='strds',
        temporaltype='absolute',
        title=output,
        description='GPM IMERG data imported by r.gpm.imerg',
        overwrite=gs.overwrite(),
        quiet=True,
    )

    fd, reg_file = tempfile.mkstemp(suffix='.txt')
    os.close(fd)
    TMPFILES.append(reg_file)

    with open(reg_file, 'w') as f:
        for map_name, start_dt, end_dt in map_entries:
            if product == 'HH':
                f.write('{}|{}|{}\n'.format(
                    map_name,
                    start_dt.strftime('%Y-%m-%d %H:%M:%S'),
                    end_dt.strftime('%Y-%m-%d %H:%M:%S'),
                ))
            else:
                f.write('{}|{}|{}\n'.format(
                    map_name,
                    start_dt.strftime('%Y-%m-%d'),
                    end_dt.strftime('%Y-%m-%d'),
                ))

    gs.run_command(
        't.register',
        input=output,
        file=reg_file,
        overwrite=gs.overwrite(),
        quiet=True,
    )


def main():
    options, flags = gs.parser()
    atexit.register(cleanup)

    output   = options['output']
    variable = options['variable']
    product  = options['product']
    run      = options['run']
    start    = options['start']
    end      = options['end'] or start
    resample = options['resample']
    username = options['username']
    flag_t   = flags['t']

    try:
        import earthaccess
        import h5py
    except ImportError as e:
        gs.fatal(
            "Required package not found: {}. "
            "Install with: sudo apt install python3-h5py; "
            "pip3 install --break-system-packages earthaccess".format(e)
        )

    # Validate product/run combination
    if (product, run) not in _SHORT_NAME:
        gs.fatal(
            "Run '{}' is not available for product '{}'. "
            "Monthly data is only available as Final (F).".format(run, product)
        )

    short_name = _SHORT_NAME[(product, run)]

    start_date = date.fromisoformat(start)
    end_date   = date.fromisoformat(end)

    if start_date > end_date:
        gs.fatal("start= must be on or before end=.")
    if start_date < _DATASET_START:
        gs.warning(
            "start= {} is before the IMERG V07 dataset start ({}).".format(
                start_date, _DATASET_START)
        )
    if end_date > date.today():
        gs.warning(
            "end= {} is in the future; data may not yet be available.".format(end_date)
        )

    if product == 'HH':
        n_days = (end_date - start_date).days + 1
        gs.message(
            "Half-hourly product: up to {} granules expected ({} days × 48).".format(
                n_days * 48, n_days)
        )

    # NASA Earthdata login
    gs.message("Logging in to NASA Earthdata...")
    _AUTH_HELP = (
        "Provide username= (password prompted at runtime), set "
        "EARTHDATA_USERNAME and EARTHDATA_PASSWORD environment variables, "
        "or add urs.earthdata.nasa.gov to ~/.netrc."
    )
    try:
        if username:
            import getpass
            password = getpass.getpass(
                "NASA Earthdata password for {}: ".format(username)
            )
            os.environ['EARTHDATA_USERNAME'] = username
            os.environ['EARTHDATA_PASSWORD'] = password
            earthaccess.login(strategy="environment")
        elif os.environ.get('EARTHDATA_USERNAME') and os.environ.get('EARTHDATA_PASSWORD'):
            earthaccess.login(strategy="environment")
        else:
            try:
                earthaccess.login(strategy="netrc")
            except Exception:
                gs.fatal("NASA Earthdata login failed. " + _AUTH_HELP)
    except Exception as e:
        gs.fatal("NASA Earthdata login failed: {}. {}".format(e, _AUTH_HELP))

    # Search for granules
    gs.message("Searching {} ({} to {})...".format(short_name, start_date, end_date))
    results = earthaccess.search_data(
        short_name=short_name,
        temporal=(start, end),
    )

    if not results:
        gs.warning("No granules found for the requested date range.")
        return

    gs.message("Found {} granule(s).".format(len(results)))

    # Download to a single temp directory (earthaccess manages the filenames)
    tmpdir = tempfile.mkdtemp(prefix='grass_imerg_')
    TMPDIRS.append(tmpdir)

    gs.message("Downloading to {}...".format(tmpdir))
    try:
        downloaded = earthaccess.download(results, local_path=tmpdir)
    except Exception as e:
        gs.fatal("Download failed: {}".format(e))

    if not downloaded:
        gs.warning("No files downloaded.")
        return

    map_entries = []

    for hdf5_path in sorted(downloaded):
        try:
            start_dt, end_dt = parse_imerg_filename(hdf5_path)
        except ValueError as e:
            gs.warning(str(e))
            continue

        # Build map name
        if product == 'HH':
            map_name = '{}_{}'.format(output, start_dt.strftime('%Y%m%d_%H%M'))
        elif product == 'D':
            map_name = '{}_{}'.format(output, start_dt.strftime('%Y%m%d'))
        else:
            map_name = '{}_{}'.format(output, start_dt.strftime('%Y%m'))

        gs.message("Processing {}...".format(map_name))

        fd, day_tif = tempfile.mkstemp(suffix='.tif')
        os.close(fd)
        TMPFILES.append(day_tif)

        if not hdf5_to_geotiff(hdf5_path, variable, day_tif):
            os.remove(day_tif)
            TMPFILES.remove(day_tif)
            continue

        gs.run_command(
            'r.import',
            input=day_tif,
            output=map_name,
            resample=resample,
            overwrite=gs.overwrite(),
            quiet=True,
        )

        os.remove(day_tif)
        TMPFILES.remove(day_tif)

        # Build strds entry bounds
        if product == 'HH':
            entry_end = end_dt
        elif product == 'D':
            entry_end = (start_dt + timedelta(days=1)).date()
            start_dt  = start_dt.date()
        else:  # M
            m, y = start_dt.month, start_dt.year
            entry_end = date(y + (1 if m == 12 else 0), (m % 12) + 1, 1)
            start_dt  = start_dt.date()

        map_entries.append((map_name, start_dt, entry_end))

    gs.message("Imported {} map(s).".format(len(map_entries)))

    if flag_t and map_entries:
        gs.message("Registering {} maps in strds '{}'...".format(
            len(map_entries), output))
        register_strds(output, map_entries, product)
        gs.message("strds '{}' created.".format(output))


if __name__ == '__main__':
    main()
