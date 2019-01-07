# tms2geotiff
Download tiles from Tile Map Server (online maps) and make a large GeoTIFF image.

Dependencies: GDAL, Pillow, numpy, requests

    usage: tms2geotiff.py [-h] [-s URL] [-f LAT,LON] [-t LAT,LON] [-z ZOOM] output

    Merge TMS tiles to a big GeoTIFF image.

    positional arguments:
      output                output file

    optional arguments:
      -h, --help            show this help message and exit
      -s URL, --source URL  TMS server url (default is OpenStreetMap:
                            https://app.gumble.pw/osm/{z}/{x}/{y}.png)
      -f LAT,LON, --from LAT,LON
                            one corner
      -t LAT,LON, --to LAT,LON
                            the other corner
      -z ZOOM, --zoom ZOOM  zoom level

    The -f, -t, -z arguments are required

For example,

    python3 tms2geotiff.py -s https://b.tile.openstreetmap.org/{z}/{x}/{y}.png -f 45.699,127 -t 30,148.492 -z 6 output.tiff

downloads a map of Japan.
