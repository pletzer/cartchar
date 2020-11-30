import numpy
import scipy.special
import ctypes
import wave
import defopt
from pathlib import Path
import pandas
import os
import random


def run(*, contour_csv_dir : Path='', output_csv_file : Path='', nexp : int=100, lmbda : float=0.2, nt : int=16):
    """Run simulation

    :param contour_csv_dir: directory holding the countour CSV files
    :param output_csv_file: output CSV file
    :param nexp: number of experiments
    :param lmbda: wave length
    :param nt: number of observation points
    """

    if not str(contour_csv_dir):
        raise RuntimeError('ERROR must provide directory containing input contour CSV files')
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

    # generate the observation points (grid)
    rObs = 10 * lmbda
    dt = twoPi / float(nt)
    ts = numpy.linspace(0. + 0.5*dt, twoPi	- 0.5*dt)
    xg = rObs * numpy.cos(ts)
    yg = rObs * numpy.sin(ts)

    # contour points of the obstacle
    csvFiles = list(contour_csv_dir.glob('*.csv'))

    # allocate
    outputData = {'iexp': numpy.zeros(nexp, numpy.int), 'object': []}
    for count in range(nt):
        outputData[f'scattered_re_{count}'] = numpy.zeros(nexp, numpy.float32)
        outputData[f'scattered_im_{count}'] = numpy.zeros(nexp, numpy.float32)

    # generate the data
    for iexp in range(nexp):

    	# randomly choose an object (aka CSV file with contour points)
        csvFile = random.choice(csvFiles)
        objectType = csvFile.name.split('.')[0]
        outputData['object'].append(objectType)

        contourDf = pandas.read_csv(csvFile)
        xc = numpy.array(contourDf['x'].values, numpy.float64)
        yc = numpy.array(contourDf['y'].values, numpy.float64)

        # rescale the object
        xcmin, xcmax = xc.min(), xc.max()
        ycmin, ycmax = yc.min(), yc.max()
        xcmid = 0.5*(xcmin + xcmax)
        ycmid = 0.5*(ycmin + ycmax)
        # centre the contour
        xc -= xcmid
        yc -= ycmid
        # rescale so the object is of size 5 * wavelength
        xscale = xcmax - xcmin
        yscale = ycmax - ycmin
        factor = (5*lmbda) / max(yscale, xscale)
        xc *= factor
        yc *= factor

        # apply a random rotation to the contour
        theta = dt * numpy.pi*random.random()
        cosTheta = numpy.cos(theta)
        sinTheta = numpy.sin(theta)
        xc2 =  cosTheta*xc + sinTheta*yc
        yc2 = -sinTheta*xc + cosTheta*yc
        xc[:] = xc2
        yc[:] = yc2

        # apply a small random shift
        xc += lmbda * (random.random() - 0.5)
        yc += lmbda * (random.random() - 0.5)

        # number of contour points
        nc1 = len(xc)

        xcPtr = xc.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        ycPtr = yc.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

        # compute the field
        for count in range(nt):
            x, y = xg[count], yg[count]
            pPtr = (ctypes.c_double * 2)(x, y)

            # compute the scattered wave
            wavelib.computeScatteredWave(kvecPtr, nc1, xcPtr, ycPtr, pPtr, 
                                         ctypes.byref(realVal), ctypes.byref(imagVal))

            # separately store the real/imag parts of the scattered wave
            outputData[f'scattered_re_{count}'][iexp] = realVal.value
            outputData[f'scattered_im_{count}'][iexp] = imagVal.value

        outputData['iexp'] = iexp

    # write the results
    print(f'writing the results to {output_csv_file}')
    pandas.DataFrame(outputData).to_csv(output_csv_file)

if __name__ == '__main__':
    defopt.run(run)
