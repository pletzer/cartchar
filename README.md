# cartchar
A program to generate machine learning data suitable for detecting objects

## Problem setting

An incident, plane wave running from left to right is reflected on an object, which is represented as a closed
contour. Around the object, there are detectors (observation points) which measure the amplitude of the reflected
wave (real and imginary parts). The reflection pattern depends on the position, size, shape and orientation 
of the object. Position, size and shape are randomly varied for each set of measurements.

## How to add new objects

Add a file IMAGE_FILE.jpg to `pictures/`. To generate a new contour, run 
```
python generate_point_coords.py select -i pictures/IMAGE_FILE.jpg -z ZOOM_FACTOR
```
Use mouse clicks to record the coordinate points of the contour. When done type ESCAPE. The result will be saved in `objects/IMAGE_FILE.csv`. 

## How to build and run the scatter code

The scatter code computes the amplitudes of the reflected wave. Instructions to build scatter are provided in `scatter/README.md`. Type 
```
python scatter.py -h
```
to get the list of command line options. You can specify the number of observation point (`--nt=NT`), the wavelength of the wave (the smaller the more intricate the reflection patterns), the number of experiments (`--num_exp=NUM_EXP`). You will need to specify the output CSV file 
to store the results (`-o OUPUT_FILE.csv`).


 
