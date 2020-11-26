import numpy
import scipy.special
import ctypes
import wave
import defopt
from pathlib import Path
import pandas
import os


def run(*, contour_csv_file : Path='', lmbda : float=0.5, nx : int=10, ny : int=10):
    """Run simulation

    :param contour_csv_file: CSV file holding obstacle contour data
    :param lmbda: wave length
    :param nx: number of x grid cells
    :param ny: number of y grid cells
    """

    twoPi = 2. * numpy.pi

    # incident wavenumber
    knum = twoPi / lmbda
    kvec = numpy.array([knum, 0.,], numpy.float64)
    # get the pointers from the numpy arrays
    kvecPtr = kvec.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

    # contour points of the obstacle
    df = pandas.read_csv(contour_csv_file)
    xc = numpy.array(df['x'].values, numpy.float64)
    yc = numpy.array(df['y'].values, numpy.float64)
    # rescale to wave length
    xscale = xc.max() - xc.min()
    yscale = yc.max() - yc.min()
    # want each obstacle to be about 5 wavelengths
    factor = (5*lmbda) / xscale
    xc *= factor
    yc *= factor
    nc1 = len(xc)
    xcPtr = xc.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    ycPtr = yc.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

    # create grid 
    xmin, xmax = xc.min() - 5*lmbda, xc.max() + 3*lmbda
    ymin, ymax = yc.min() - 3*lmbda, yc.max() + 4*lmbda
    xg = numpy.linspace(xmin, xmax, nx + 1)
    yg = numpy.linspace(ymin, ymax, ny + 1)

    # find the library under the build directory
    waveLibFile = ''
    for root, dirs, files in os.walk('build/'):
        for file in files:
            if file[-3:] == '.so':
                waveLibFile = os.path.join(root, file)

    # open the shared library 
    wavelib = ctypes.CDLL(waveLibFile)

    # create some types for calling C++
    doubleStarType = ctypes.POINTER(ctypes.c_double) 

    # containers to receive the output values of the C function
    realVal, imagVal = ctypes.c_double(0.), ctypes.c_double(0.)

    # returns void
    wavelib.cincident.restype = None
    # double*, double*, double*, double*
    wavelib.cincident.argtypes = [doubleStarType, doubleStarType, 
                                  doubleStarType, doubleStarType]

    # returns void
    wavelib.computeScatteredWave.restype = None
    # double*, int, double*, double*, double*, double*, double* 
    wavelib.computeScatteredWave.argtypes = [doubleStarType,
                                             ctypes.c_int, 
                                             doubleStarType,
                                             doubleStarType,
                                             doubleStarType,
                                             doubleStarType,
                                             doubleStarType]

    # compute the field
    scat = numpy.zeros((ny + 1, nx + 1), numpy.complex64)
    inci = numpy.zeros((ny + 1, nx + 1), numpy.complex64)
    for j in range(ny + 1):
        y = yg[j]
        for i in range(nx + 1):
            x = xg[i]

            # need to check that x,y are outside contour
            # otherwise continue
            pPtr = (ctypes.c_double * 2)(x, y)

            # skip if point is inside closed contour
            if wavelib.isInsideContour(pPtr, nc1, xcPtr, ycPtr) == 1:
                continue

            wavelib.cincident(kvecPtr, pPtr, ctypes.byref(realVal), ctypes.byref(imagVal))
            inci[j, i] = realVal.value + 1j*imagVal.value

            wavelib.computeScatteredWave(kvecPtr, nc1, xcPtr, ycPtr, pPtr, 
                                         ctypes.byref(realVal), ctypes.byref(imagVal))
            scat[j, i] = realVal.value + 1j*imagVal.value


if __name__ == '__main__':
    defopt.run(run)
