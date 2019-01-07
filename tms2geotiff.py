#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import io
import math
import argparse
import itertools
import concurrent.futures

import numpy
import requests
from PIL import Image, ImageDraw
from osgeo import gdal

EARTH_EQUATORIAL_RADIUS = 6378137.0

DEFAULT_TMS = 'https://c.tile.openstreetmap.org/{z}/{x}/{y}.png'

gdal.UseExceptions()

WKT_3857 = 'PROJCS["WGS 84 / Pseudo-Mercator",GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","4326"]],PROJECTION["Mercator_1SP"],PARAMETER["central_meridian",0],PARAMETER["scale_factor",1],PARAMETER["false_easting",0],PARAMETER["false_northing",0],UNIT["metre",1,AUTHORITY["EPSG","9001"]],AXIS["X",EAST],AXIS["Y",NORTH],EXTENSION["PROJ4","+proj=merc +a=6378137 +b=6378137 +lat_ts=0.0 +lon_0=0.0 +x_0=0.0 +y_0=0 +k=1.0 +units=m +nadgrids=@null +wktext +no_defs"],AUTHORITY["EPSG","3857"]]'

def from4326_to3857(lat, lon):
    xtile = math.radians(lon) * EARTH_EQUATORIAL_RADIUS
    ytile = math.log(math.tan(math.radians(45 + lat / 2.0))) * EARTH_EQUATORIAL_RADIUS
    return (xtile, ytile)

def deg2num(lat, lon, zoom):
    lat_r = math.radians(lat)
    n = 2 ** zoom
    xtile = ((lon + 180) / 360 * n)
    ytile = ((1 - math.log(math.tan(lat_r) + 1/math.cos(lat_r)) / math.pi) / 2 * n)
    return (xtile, ytile)

def is_black(im):
    if len(im.mode) >= 3:
        bandn = 3
    else:
        bandn = 1
    for band in range(bandn):
        if any(im.getdata(band)):
            return False
    return True

def stitch_tiles(tiles, corners, x0, y0, x1, y1):
    bbox = (math.floor(x0), math.floor(y0), math.ceil(x1), math.ceil(y1))
    xfrac = x0 - bbox[0]
    yfrac = y0 - bbox[1]
    ims = [Image.open(io.BytesIO(b)) for b in tiles]
    size = ims[0].size
    mode = 'RGB' if ims[0].mode == 'RGB' else 'RGBA'
    newim = Image.new(mode, (
        size[0]*(bbox[2]-bbox[0]), size[1]*(bbox[3]-bbox[1])))
    for i, xy in enumerate(corners):
        dx = abs(xy[0] - bbox[0])
        dy = abs(xy[1] - bbox[1])
        xy0 = (size[0]*dx, size[1]*dy)
        if mode == 'RGB':
            newim.paste(ims[i], xy0)
        else:
            im = ims[i].convert(mode)
            if is_black(im):
                newimdraw = ImageDraw.Draw(newim)
                newimdraw.rectangle(
                    (xy0, (xy0[0]+size[0], xy0[1]+size[1])), (0,0,0,0), None)
                del newimdraw
            else:
                newim.paste(im, xy0)
    x2 = round(size[0]*xfrac)
    y2 = round(size[1]*yfrac)
    imgw = round(size[0]*(x1-x0))
    imgh = round(size[1]*(y1-y0))
    retim = newim.crop((x2, y2, x2+imgw, y2+imgh))
    if mode == 'RGBA':
        if all(x == 255 for x in im.getdata(3)):
            retim = retim.convert('RGB')
    newim.close()
    [i.close() for i in ims]
    return retim

def get_tile(url):
    retry = 3
    while 1:
        try:
            r = requests.get(url, timeout=60)
            break
        except Exception:
            retry -= 1
            if not retry:
                raise
    r.raise_for_status()
    return r.content

def draw_tile(source, lat0, lon0, lat1, lon1, zoom, filename):
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
    futures = []
    tiles = []
    with concurrent.futures.ThreadPoolExecutor(5) as executor:
        for x, y in corners:
            futures.append(executor.submit(get_tile,
                source.format(z=zoom, x=x, y=y)))
        for k, fut in enumerate(concurrent.futures.as_completed(futures), 1):
            print('Downloaded image %d/%d' % (k, totalnum))
        tiles = [fut.result() for fut in futures]
    img = stitch_tiles(tiles, corners, x0, y0, x1, y1)
    imgbands = len(img.getbands())
    driver = gdal.GetDriverByName('GTiff')
    gtiff = driver.Create(filename, img.size[0], img.size[1],
        imgbands, gdal.GDT_Byte,
        options=['COMPRESS=DEFLATE', 'PREDICTOR=2', 'ZLEVEL=9', 'TILED=YES'])
    xp0, yp0 = from4326_to3857(lat0, lon0)
    xp1, yp1 = from4326_to3857(lat1, lon1)
    pwidth = abs(xp1 - xp0) / img.size[0]
    pheight = abs(yp1 - yp0) / img.size[1]
    gtiff.SetGeoTransform((
        min(xp0, xp1), pwidth, 0, max(yp0, yp1), 0, -pheight))
    gtiff.SetProjection(WKT_3857)
    for band in range(imgbands):
        array = numpy.array(img.getdata(band), dtype='u8')
        array = array.reshape((img.size[1], img.size[0]))
        band = gtiff.GetRasterBand(band + 1)
        band.WriteArray(array)
    gtiff.FlushCache()
    return img

def main():
    parser = argparse.ArgumentParser(
        description="Merge TMS tiles to a big GeoTIFF image.",
        epilog="The -f, -t, -z arguments are required")
    parser.add_argument(
        "-s", "--source", metavar='URL', default=DEFAULT_TMS,
        help="TMS server url (default is OpenStreetMap: %s)" % DEFAULT_TMS)
    parser.add_argument("-f", "--from", metavar='LAT,LON', help="one corner")
    parser.add_argument("-t", "--to", metavar='LAT,LON', help="the other corner")
    parser.add_argument("-z", "--zoom", type=int, help="zoom level")
    parser.add_argument("output", help="output file")
    args = parser.parse_args()
    if not all(getattr(args, opt, None) for opt in
        ('from', 'to', 'zoom', 'output')):
        parser.print_help()
        return 1
    try:
        coords0 = tuple(map(float, getattr(args, 'from').split(',')))
        coords1 = tuple(map(float, getattr(args, 'to').split(',')))
    except Exception:
        parser.print_help()
        return 1
    draw_tile(args.source, coords0[0], coords0[1], coords1[0], coords1[1],
        args.zoom, args.output)
    return 0

if __name__ == '__main__':
    import sys
    sys.exit(main())