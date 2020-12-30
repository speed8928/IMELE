#!/usr/bin/env python3

import os, gdal
import re
import sys
import glob
from  gdalconst import *
import math

class GeoTiffSplitter:
  def __init__( self, directoryName, xTileSize, yTileSize, doDryRun, xlimit=-1, ylimit=-1):
    self.xTileSize = int( xTileSize )
    self.yTileSize = int( yTileSize )

    if os.path.exists( directoryName ):
      print( "Directory exists: " + directoryName )
      geoTiffFileList = glob.glob( directoryName + "/*" )
      



      for geoTiffFileName in geoTiffFileList:
        print( "Processing File Directory exists: " + geoTiffFileName )
        # Open the file:
        raster = gdal.Open(geoTiffFileName)
        absolutePath = os.path.abspath(os.path.splitext(geoTiffFileName)[0])
        basename = os.path.basename(geoTiffFileName)
        geoTiffPartFileName = os.path.join(absolutePath,os.path.splitext(basename)[0])

        # Check type of the variable 'raster'
        width  = raster.RasterXSize
        height = raster.RasterYSize
        xmax = math.ceil(width / xTileSize ) 
        ymax = math.ceil(height / yTileSize)


        print( "Tiles of size (%d,%d) amounts (x-dir, y-dir) = (%d, %d)" % ( self.xTileSize,
                                                      self.yTileSize, xmax, ymax ) )
        for it in range( xmax ):
     
          if doDryRun == False:
            os.system("mkdir -p " + absolutePath )
          for jt in range( ymax ):

            if it == 0:
              init_x = 0
           
            if jt == 0:
              init_y = 0
            

            xx= init_x+it*xTileSize
            yy= init_y+jt*yTileSize

            #if it == xmax:
            #  init_x = 83440-500
            #  xx = init_x
         
            #if jt == ymax:
            #  init_y = 24040-500
            #  yy = init_y





              



            gdaltranString = "gdal_translate -of GTIFF -srcwin " + str(it*self.xTileSize) + ", "+str(jt*self.yTileSize) \
                       +", "+str(self.xTileSize)+", " +str(self.yTileSize) \
                      + " " + geoTiffFileName + " " + geoTiffPartFileName + "_" + str(it) + "_" + str(jt)+".tif"

            if raster.RasterCount == 3 or raster.RasterCount == 4:
              gdaltranNoIR = "gdal_translate -b 1 -b 2 -b 3 -of GTIFF -srcwin " \
                                      + str(xx) + ", " + str(yy) \
                                      +", "+str(self.xTileSize)+", " +str(self.yTileSize) \
                                      + " " + geoTiffFileName + " " + geoTiffPartFileName \
                                      + "_" + str(it) + "_" + str(jt)+".tif"
            else:
              gdaltranNoIR = "gdal_translate -b 1 -of GTIFF -srcwin " \
                                      + str(xx) + ", " + str(yy) \
                                      +", "+str(self.xTileSize)+", " +str(self.yTileSize) \
                                      + " " + geoTiffFileName + " " + geoTiffPartFileName \
                                      + "_" + str(it) + "_" + str(jt)+".tif"
          
            if doDryRun == False:
              #os.system(gdaltranString)
              os.system(gdaltranNoIR)
    else:
      print( "Don't have the directory: " + directoryName )
      sys.exit(1)


def main():
  xStep=500
  yStep=500
  xlimit = ylimit = -1
  doDryRun = False

   
  fileNames = 'Vaihingen'


  GeoTiffSplitter( fileNames, xStep, yStep, doDryRun, xlimit, ylimit )

def usage( argv ):
  if len(argv) <= 1:
    print( "Usage:     " + argv[0] + " --directory=path/*.tif --x=xStep --y=yStep\n" )
    print( "                         directory=NAME Reads in a bunch of named tiff files with wildcards" )
    print( "                         dryrun         demonstrates the tasks that would be performed" )
    print( "                         x              x-step size for each image")
    print( "                         y              y-step size for each image")
    print( "                         xlimit         limits the number of iterations through the image" )
    print( "                         ylimit         limits the number of iterations through the image" )

    print( "e.g:       " + argv[0] + " --directory=../OSi-GeoTiffs --x=1024 --y=1024" )
    print( "           " + argv[0] + " --directory=../OSi-GeoTiffs --x=13334 --y=600 --ylimit=1" )
    sys.exit(1)

if __name__ == "__main__":
  #usage( sys.argv )
  main()
