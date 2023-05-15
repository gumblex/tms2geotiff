#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import io
import os
import re
import math
import time
import argparse
import itertools
import concurrent.futures

from PIL import Image
from PIL import TiffImagePlugin
Image.MAX_IMAGE_PIXELS = None

try:
    import httpx
    SESSION = httpx.Client()
except ImportError:
    import requests
    SESSION = requests.Session()


SESSION.headers.update({
    "Accept": "*/*",
    "Accept-Encoding": "gzip, deflate",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; rv:91.0) Gecko/20100101 Firefox/91.0",
})

re_coords_split = re.compile('[ ,;]+')


EARTH_EQUATORIAL_RADIUS = 6378137.0

DEFAULT_TMS = 'https://tile.openstreetmap.org/{z}/{x}/{y}.png'


WKT_3857 = 'PROJCS["WGS 84 / Pseudo-Mercator",GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","4326"]],PROJECTION["Mercator_1SP"],PARAMETER["central_meridian",0],PARAMETER["scale_factor",1],PARAMETER["false_easting",0],PARAMETER["false_northing",0],UNIT["metre",1,AUTHORITY["EPSG","9001"]],AXIS["X",EAST],AXIS["Y",NORTH],EXTENSION["PROJ4","+proj=merc +a=6378137 +b=6378137 +lat_ts=0.0 +lon_0=0.0 +x_0=0.0 +y_0=0 +k=1.0 +units=m +nadgrids=@null +wktext +no_defs"],AUTHORITY["EPSG","3857"]]'


def from4326_to3857(lat, lon):
    xtile = math.radians(lon) * EARTH_EQUATORIAL_RADIUS
    ytile = math.log(math.tan(math.radians(45 + lat / 2.0))) * EARTH_EQUATORIAL_RADIUS
    return (xtile, ytile)


def deg2num(lat, lon, zoom):
    n = 2 ** zoom
    xtile = ((lon + 180) / 360 * n)
    ytile = (1 - math.asinh(math.tan(math.radians(lat))) / math.pi) * n / 2
    return (xtile, ytile)


def is_empty(im):
    extrema = im.getextrema()
    if len(extrema) >= 3:
        if len(extrema) > 3 and extrema[-1] == (0, 0):
            return True
        for ext in extrema[:3]:
            if ext != (0, 0):
                return False
        return True
    else:
        return extrema[0] == (0, 0)


def paste_tile(bigim, base_size, tile, corner_xy, bbox):
    if tile is None:
        return bigim
    im = Image.open(io.BytesIO(tile))
    mode = 'RGB' if im.mode == 'RGB' else 'RGBA'
    size = im.size
    if bigim is None:
        base_size[0] = size[0]
        base_size[1] = size[1]
        newim = Image.new(mode, (
            size[0]*(bbox[2]-bbox[0]), size[1]*(bbox[3]-bbox[1])))
    else:
        newim = bigim

    dx = abs(corner_xy[0] - bbox[0])
    dy = abs(corner_xy[1] - bbox[1])
    xy0 = (size[0]*dx, size[1]*dy)
    if mode == 'RGB':
        newim.paste(im, xy0)
    else:
        if im.mode != mode:
            im = im.convert(mode)
        if not is_empty(im):
            newim.paste(im, xy0)
    im.close()
    return newim


def get_tile(url):
    retry = 3
    while 1:
        try:
            r = SESSION.get(url, timeout=60)
            break
        except Exception:
            retry -= 1
            if not retry:
                raise
    if r.status_code == 404:
        return None
    elif not r.content:
        return None
    r.raise_for_status()
    return r.content


def print_progress(progress, total, done=False):
    if done:
        print('Downloaded image %d/%d' % (progress, total))


def download_extent(
    source, lat0, lon0, lat1, lon1, zoom,
    progress_callback=print_progress,
    callback_interval=0.05
):
    x0, y0 = deg2num(lat0, lon0, zoom)
    x1, y1 = deg2num(lat1, lon1, zoom)
    if x0 > x1:
        x0, x1 = x1, x0
    if y0 > y1:
        y0, y1 = y1, y0
    corners = tuple(itertools.product(
        range(math.floor(x0), math.ceil(x1)),
        range(math.floor(y0), math.ceil(y1))))
    totalnum = len(corners)
    futures = {}
    done_num = 0
    progress_callback(done_num, totalnum, False)
    last_done_num = 0
    last_callback = time.monotonic()
    cancelled = False
    with concurrent.futures.ThreadPoolExecutor(5) as executor:
        for x, y in corners:
            future = executor.submit(get_tile, source.format(z=zoom, x=x, y=y))
            futures[future] = (x, y) 
        bbox = (math.floor(x0), math.floor(y0), math.ceil(x1), math.ceil(y1))
        bigim = None
        base_size = [256, 256]
        while futures:
            done, not_done = concurrent.futures.wait(
                futures.keys(), timeout=callback_interval,
                return_when=concurrent.futures.FIRST_COMPLETED
            )
            for fut in done:
                bigim = paste_tile(bigim, base_size, fut.result(), futures[fut], bbox)
                del futures[fut]
                done_num += 1
            if time.monotonic() > last_callback + callback_interval:
                try:
                    progress_callback(done_num, totalnum, (done_num > last_done_num))
                except TaskCancelled:
                    for fut in futures.keys():
                        fut.cancel()
                    futures.clear()
                    cancelled = True
                    break
                last_callback = time.monotonic()
                last_done_num = done_num
    if cancelled:
        raise TaskCancelled()
    progress_callback(done_num, totalnum, True)

    xfrac = x0 - bbox[0]
    yfrac = y0 - bbox[1]
    x2 = round(base_size[0]*xfrac)
    y2 = round(base_size[1]*yfrac)
    imgw = round(base_size[0]*(x1-x0))
    imgh = round(base_size[1]*(y1-y0))
    retim = bigim.crop((x2, y2, x2+imgw, y2+imgh))
    if retim.mode == 'RGBA' and retim.getextrema()[3] == (255, 255):
        retim = retim.convert('RGB')
    bigim.close()
    xp0, yp0 = from4326_to3857(lat0, lon0)
    xp1, yp1 = from4326_to3857(lat1, lon1)
    pwidth = abs(xp1 - xp0) / retim.size[0]
    pheight = abs(yp1 - yp0) / retim.size[1]
    matrix = (min(xp0, xp1), pwidth, 0, max(yp0, yp1), 0, -pheight)
    return retim, matrix


def generate_tiffinfo(matrix):
    ifd = TiffImagePlugin.ImageFileDirectory_v2()
    # GeoKeyDirectoryTag
    gkdt = [
        1, 1,
        0,  # GeoTIFF 1.0
        0,  # NumberOfKeys
    ]
    # KeyID, TIFFTagLocation, KeyCount, ValueOffset
    geokeys = [
        # GTModelTypeGeoKey
        (1024, 0, 1, 1),  # 2D projected coordinate reference system
        # GTRasterTypeGeoKey
        (1025, 0, 1, 1),  # PixelIsArea
        # GTCitationGeoKey
        (1026, 34737, 25, 0),
        # GeodeticCitationGeoKey
        (2049, 34737, 7, 25),
        # GeogAngularUnitsGeoKey
        (2054, 0, 1, 9102),  # degree
        # ProjectedCRSGeoKey
        (3072, 0, 1, 3857),
        # ProjLinearUnitsGeoKey
        (3076, 0, 1, 9001),  # metre
    ]
    gkdt[3] = len(geokeys)
    ifd.tagtype[34735] = 3  # short
    ifd[34735] = tuple(itertools.chain(gkdt, *geokeys))
    # GeoDoubleParamsTag
    ifd.tagtype[34736] = 12  # double
    # GeoAsciiParamsTag
    ifd.tagtype[34737] = 1  # byte
    ifd[34737] = b'WGS 84 / Pseudo-Mercator|WGS 84|\x00'
    a, b, c, d, e, f = matrix
    # ModelPixelScaleTag
    ifd.tagtype[33550] = 12  # double
    # ModelTiepointTag
    ifd.tagtype[33922] = 12  # double
    # ModelTransformationTag
    ifd.tagtype[34264] = 12  # double
    # This matrix tag should not be used
    # if the ModelTiepointTag and the ModelPixelScaleTag are already defined
    if c == 0 and e == 0:
        ifd[33550] = (b, -f, 0.0)
        ifd[33922] = (0.0, 0.0, 0.0, a, d, 0.0)
    else:
        ifd[34264] = (
            b, c, 0.0, a,
            e, f, 0.0, d,
            0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 1.0
        )
    return ifd


def save_image(img, filename, matrix, **params):
    wld_ext = {
        '.gif': '.gfw',
        '.jpg': '.jgw',
        '.jpeg': '.jgw',
        '.jp2': '.j2w',
        '.png': '.pgw',
        '.tif': '.tfw',
        '.tiff': '.tfw',
    }
    basename, ext = os.path.splitext(filename)
    ext = ext.lower()
    wld_name = basename + wld_ext.get(ext, '.wld')
    img_params = params.copy()
    if ext == '.jpg':
        img_params['quality'] = 92
        img_params['optimize'] = True
    elif ext == '.png':
        img_params['optimize'] = True
    elif ext.startswith('.tif'):
        img_params['compression'] = 'tiff_adobe_deflate'
        img_params['tiffinfo'] = generate_tiffinfo(matrix)
    img.save(filename, **img_params)
    if not ext.startswith('.tif'):
        with open(wld_name, 'w', encoding='utf-8') as f_wld:
            a, b, c, d, e, f = matrix
            f_wld.write('\n'.join(map(str, (b, e, c, f, a, d, ''))))
    return img


def save_geotiff_gdal(img, filename, matrix):
    if 'GDAL_DATA' in os.environ:
        del os.environ['GDAL_DATA']
    if 'PROJ_LIB' in os.environ:
        del os.environ['PROJ_LIB']

    import numpy
    from osgeo import gdal
    gdal.UseExceptions()

    imgbands = len(img.getbands())
    driver = gdal.GetDriverByName('GTiff')
    gtiff = driver.Create(filename, img.size[0], img.size[1],
        imgbands, gdal.GDT_Byte,
        options=['COMPRESS=DEFLATE', 'PREDICTOR=2', 'ZLEVEL=9', 'TILED=YES'])
    gtiff.SetGeoTransform(matrix)
    gtiff.SetProjection(WKT_3857)
    for band in range(imgbands):
        array = numpy.array(img.getdata(band), dtype='u8')
        array = array.reshape((img.size[1], img.size[0]))
        band = gtiff.GetRasterBand(band + 1)
        band.WriteArray(array)
    gtiff.FlushCache()
    return img


def save_image_auto(img, filename, matrix, use_gdal=False, **params):
    ext = os.path.splitext(filename)[1].lower()
    if ext in ('.tif', '.tiff') and use_gdal:
        return save_geotiff_gdal(img, filename, matrix)
    else:
        return save_image(img, filename, matrix, **params)


class TaskCancelled(RuntimeError):
    pass


def parse_extent(s):
    coords_text = re_coords_split.split(s)
    return (float(coords_text[1]), float(coords_text[0]),
            float(coords_text[3]), float(coords_text[2]))


def gui():
    import tkinter as tk
    import tkinter.ttk as ttk
    import tkinter.messagebox

    root_tk = tk.Tk()

    def cmd_get_save_file():
        result = root_tk.tk.eval("""tk_getSaveFile -filetypes {
            {{GeoTIFF} {.tiff}}
            {{JPG} {.jpg}}
            {{PNG} {.png}}
            {{All Files} *}
        } -defaultextension .tiff""")
        if result:
            v_output.set(result)

    frame = ttk.Frame(root_tk, padding=8)
    frame.grid(column=0, row=0, sticky='nsew')
    frame.master.title('Download TMS image')
    frame.master.resizable(0, 0)
    l_url = ttk.Label(frame, width=50, text="URL: (with {x}, {y}, {z})")
    l_url.grid(column=0, row=0, columnspan=3, sticky='w', pady=(0, 2))
    v_url = tk.StringVar()
    e_url = ttk.Entry(frame, textvariable=v_url)
    e_url.grid(column=0, row=1, columnspan=3, sticky='we', pady=(0, 5))
    l_extent = ttk.Label(frame, text="Extent: (min_lon,min_lat,max_lon,max_lat)")
    l_extent.grid(column=0, row=2, columnspan=3, sticky='w', pady=(0, 2))
    v_extent = tk.StringVar()
    e_extent = ttk.Entry(frame, width=50, textvariable=v_extent)
    e_extent.grid(column=0, row=3, columnspan=3, sticky='we', pady=(0, 5))
    l_zoom = ttk.Label(frame, width=5, text="Zoom:")
    l_zoom.grid(column=0, row=4, sticky='w')
    v_zoom = tk.StringVar()
    v_zoom.set('13')
    e_zoom = ttk.Spinbox(frame, width=10, textvariable=v_zoom, **{
        'from': 1, 'to': 19, 'increment': 1
    })
    e_zoom.grid(column=1, row=4, sticky='w')
    l_output = ttk.Label(frame, width=10, text="Output:")
    l_output.grid(column=0, row=5, sticky='w')
    v_output = tk.StringVar()
    e_output = ttk.Entry(frame, width=30, textvariable=v_output)
    e_output.grid(column=1, row=5, sticky='we')
    b_output = ttk.Button(frame, text='...', width=3, command=cmd_get_save_file)
    b_output.grid(column=2, row=5, sticky='we')
    p_progress = ttk.Progressbar(frame, mode='determinate')
    p_progress.grid(column=0, row=6, columnspan=3, sticky='we', pady=(5, 2))

    started = False
    stop_download = False

    def reset():
        b_download.configure(
            text='Download', state='normal', command=cmd_download)
        root_tk.update()

    def update_progress(progress, total, done):
        nonlocal started, stop_download
        if not started:
            if done:
                p_progress.configure(maximum=total, value=progress)
            else:
                p_progress.configure(maximum=total)
            started = True
        elif done:
            p_progress.configure(value=progress)
        root_tk.update()
        if stop_download:
            raise TaskCancelled()

    def cmd_download():
        nonlocal started, stop_download
        started = False
        stop_download = False
        b_download.configure(text='Cancel', command=cmd_cancel)
        root_tk.update()
        try:
            url = v_url.get().strip()
            args = [url]
            args.extend(parse_extent(v_extent.get()))
            args.append(int(v_zoom.get()))
            filename = v_output.get()
            if not all(args) or not filename:
                raise ValueError("Empty input")
        except (TypeError, ValueError, IndexError) as ex:
            reset()
            tkinter.messagebox.showerror(
                title='tms2geotiff',
                message="Invalid input: %s: %s" % (type(ex).__name__, ex),
                master=frame
            )
            return
        root_tk.update()
        try:
            img, matrix = download_extent(
                *args, progress_callback=update_progress)
            b_download.configure(text='Saving...', state='disabled')
            root_tk.update()
            save_image_auto(img, filename, matrix)
            reset()
        except TaskCancelled:
            reset()
            tkinter.messagebox.showwarning(
                title='tms2geotiff',
                message="Download cancelled.",
                master=frame
            )
            return
        except Exception as ex:
            reset()
            tkinter.messagebox.showerror(
                title='tms2geotiff',
                message="%s: %s" % (type(ex).__name__, ex),
                master=frame
            )
            return
        tkinter.messagebox.showinfo(
            title='tms2geotiff',
            message="Download complete.",
            master=frame
        )

    def cmd_cancel():
        nonlocal started, stop_download
        started = False
        stop_download = True
        reset()

    b_download = ttk.Button(
        width=15, text='Download', default='active', command=cmd_download)
    b_download.grid(column=0, row=6, columnspan=3, pady=2)

    root_tk.mainloop()


def main():
    parser = argparse.ArgumentParser(
        description="Merge TMS tiles to a big image.",
        epilog="If no parameters are specified, it will open the GUI.")
    parser.add_argument(
        "-s", "--source", metavar='URL', default=DEFAULT_TMS,
        help="TMS server url (default is OpenStreetMap: %s)" % DEFAULT_TMS)
    parser.add_argument("-f", "--from", metavar='LAT,LON', help="one corner")
    parser.add_argument("-t", "--to", metavar='LAT,LON', help="the other corner")
    parser.add_argument("-e", "--extent",
        metavar='min_lon,min_lat,max_lon,max_lat',
        help="extent in one string (use either -e, or -f and -t)")
    parser.add_argument("-z", "--zoom", type=int, help="zoom level")
    parser.add_argument("-g", "--gui", action='store_true', help="show GUI")
    parser.add_argument("output", nargs='?', help="output file")
    args = parser.parse_args()
    if args.gui or not all(getattr(args, opt, None) for opt in
        ('zoom', 'output')):
        gui()
        # parser.print_help()
        return 1

    download_args = [args.source]
    try:
        if args.extent:
            download_args.extend(parse_extent(args.extent))
        else:
            coords0 = tuple(map(float, getattr(args, 'from').split(',')))
            coords1 = tuple(map(float, getattr(args, 'to').split(',')))
            download_args.extend((coords0[0], coords0[1], coords1[0], coords1[1]))
    except Exception:
        parser.print_help()
        return 1
    download_args.append(args.zoom)
    download_args.append(print_progress)

    img, matrix = download_extent(*download_args)
    save_image_auto(img, args.output, matrix)
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
