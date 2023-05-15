# tms2geotiff
Download tiles from [Tile Map Server](https://wiki.openstreetmap.org/wiki/Slippy_map_tilenames) (online maps) and make a large image.

If output is TIFF, it can write to a GeoTIFF image.
Otherwise, it will save to a normal image with a World File for georeferencing (in EPSG:3857).

Dependencies: Pillow, requests/httpx.

The GDAL and numpy are no longer needed for writing GeoTIFF images.

**GUI**: Directly run `python3 tms2geotiff.py` to open a GUI window.

    usage: tms2geotiff.py [-h] [-s URL] [-f LAT,LON] [-t LAT,LON]
                          [-e min_lon,min_lat,max_lon,max_lat] [-z ZOOM] [-g]
                          [output]

    Merge TMS tiles to a big image.

    positional arguments:
      output                output file

    options:
      -h, --help            show this help message and exit
      -s URL, --source URL  TMS server url (default is OpenStreetMap:
                            https://tile.openstreetmap.org/{z}/{x}/{y}.png)
      -f LAT,LON, --from LAT,LON
                            one corner
      -t LAT,LON, --to LAT,LON
                            the other corner
      -e min_lon,min_lat,max_lon,max_lat, --extent min_lon,min_lat,max_lon,max_lat
                            extent in one string (use either -e, or -f and -t)
      -z ZOOM, --zoom ZOOM  zoom level
      -g, --gui             show GUI

    If no parameters are specified, it will open the GUI.

For example,

    python3 tms2geotiff.py -s https://tile.openstreetmap.org/{z}/{x}/{y}.png -f 45.699,127 -t 30,148.492 -z 6 output.tiff

downloads a map of Japan.

If the coordinates are negative, use `--from=-12.34,56.78 --to=-13.45,57.89`


# tmssplit
Split a large GeoTIFF image into tiles for a Tile Map Server.

Dependencies: GDAL, Pillow, numpy, scipy, pyproj

    usage: tmssplit.py [-h] [-z ZOOM] [-n NAME] [-s SIZE] [-p PROJ] [-t THREADS]
                       inputfile outputdir

    Split a big GeoTIFF image to TMS tiles.

    positional arguments:
      inputfile             input GeoTIFF file
      outputdir             output directory

    optional arguments:
      -h, --help            show this help message and exit
      -z ZOOM, --zoom ZOOM  zoom level(s), eg. 15 or 14-17
      -n NAME, --name NAME  image file name format, default {z}_{x}_{y}.png
      -s SIZE, --size SIZE  image size in px, default 256px
      -p PROJ, --proj PROJ  set projection id
      -t THREADS, --threads THREADS
                            set thread number

    -z is required

