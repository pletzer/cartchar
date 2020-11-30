import numpy
import scipy.special
import ctypes
import wave
import defopt
from pathlib import Path
import pandas
import os
import random


def run(*, contour_csv_dir : Path='', output_csv_file : Path='', nexp : int=100, lmbda : float=0.5, nx : int=1, ny : int=2):
    """Run simulation

    :param contour_csv_dir: directory holding the countour CSV files
    :param output_csv_file: output CSV file
    :param nexp: number of experiments
    :param lmbda: wave length
    :param nx: number of x grid cells
    :param ny: number of y grid cells
    """

    if not str(contour_csv_dir):
        raise RuntimeError('ERROR must provide directory containing contour CSV files')
    else:
        if not contour_csv_dir.exists():
            raise RuntimeError('ERROR must input directory containing CSV files does not exist!')

    if str(output_csv_file) == '.':
        raise RuntimeError('ERROR must provide output CSV file name')

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

    # constants
    twoPi = 2. * numpy.pi

    # incident wavenumber
    knum = twoPi / lmbda
    kvec = numpy.array([knum, 0.,], numpy.float64)
    # get the pointers from the numpy arrays
    kvecPtr = kvec.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

    # create grid 
    xmin, xmax = -10*lmbda, 0.0
    ymin, ymax = -3.5*lmbda, 4.2*lmbda
    xg = numpy.linspace(xmin, xmax, nx + 1)
    yg = numpy.linspace(ymin, ymax, ny + 1)

    # contour points of the obstacle
    csvFiles = list(contour_csv_dir.glob('*.csv'))

    outputDf = {'iexp': numpy.zeros(nexp, numpy.int), 'character': []}
    count = 0
    for j in range(ny + 1):
        for i in range(nx + 1):
            outputDf[f'scattered_re_{count}'] = numpy.zeros(nexp, numpy.float32)
            outputDf[f'scattered_im_{count}'] = numpy.zeros(nexp, numpy.float32)
            count += 1

    for iexp in range(nexp):

        csvFile = random.choice(csvFiles)
        outputDf['character'].append(csvFile.name.split('.')[0])

        df = pandas.read_csv(csvFile)
        xc = numpy.array(df['x'].values, numpy.float64)
        yc = numpy.array(df['y'].values, numpy.float64)
        # rescale to wave length
        xscale = xc.max() - xc.min()
        yscale = yc.max() - yc.min()
        # want each obstacle to be about 5 wavelengths
        factor = (5*lmbda) / yscale
        xc *= factor
        yc *= factor

        # random rotation
        theta = -numpy.pi/2. + numpy.pi*random.random()

        cosTheta = numpy.cos(theta)
        sinTheta = numpy.sin(theta)
        xc2 =  cosTheta*xc + sinTheta*yc
        yc2 = -sinTheta*xc + cosTheta*yc
        xc[:] = xc2
        yc[:] = yc2

        # shift to the right location
        xc -= xc.min()
        xc += 3.5*lmbda
        yc -= yc.min()
        yc -= 2.1*lmbda
        # number of contour points
        nc1 = len(xc)

        xcPtr = xc.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        ycPtr = yc.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

        # compute the field
        count = 0
        for j in range(ny + 1):
            y = yg[j]
            for i in range(nx + 1):
                x = xg[i]
                pPtr = (ctypes.c_double * 2)(x, y)

                wavelib.computeScatteredWave(kvecPtr, nc1, xcPtr, ycPtr, pPtr, 
                                             ctypes.byref(realVal), ctypes.byref(imagVal))

                outputDf[f'scattered_re_{count}'][iexp] = realVal.value
                outputDf[f'scattered_im_{count}'][iexp] = imagVal.value
                count += 1

        outputDf['iexp'] = iexp

    # write the results
    print(f'writing the results to {output_csv_file}')
    pandas.DataFrame(outputDf).to_csv(output_csv_file)

if __name__ == '__main__':
    defopt.run(run)
