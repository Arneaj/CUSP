# MagCUSPS — Magnetopause Continuous Unsupervised Simulation Profiling & Synthesis

## Map of content

## Introduction

This library has been created in the context of a Master's Thesis in Computer science at Imperial College London in collaboration with the Space Plasma science research team. 

The first goal was to provide tools to extract certain topological elements from 3D voxel grids of simulation data extracted from ICL's Gorgon model. 
It provides optimized C++ tools to read and process data, obtain stream and field lines, and extract the magnetopause with less than a second of computation even with high precision.
Some other python tools are present for plotting and graphing purposes, as well as to provide extra tools to extract features like the current sheet and the X-line.

The main application of this library for computational models of the Earth magnetosphere is to determine in real time, through the extraction of the mentioned features, if the model is performing correctly or if it has failed in some way. 
Through statistical testing, it has been determined that the real time analysis is performant enough to be able to evaluate the output of the model with reasonable certainty.

Though this library has been created with the Gorgon model in mind, it should be model agnostic if the user provides their own implementation of the ReaderWriter interface present to read the output of their model into the provided Matrix class.    

## Installation

### Dependencies

The library provides an example implementation of a `.pvtr` to `.bin` ReaderWriter used for all of the tests. 
In the case that the user wants to test their installation with the provided tests or wants to use the `.pvtr` reader provided, they will need to have a installation of the C++ library: 
- **VTK**: can be obtained from [gitlab](https://gitlab.kitware.com/vtk/vtk) (only for the C++ library)

For the least squares fitting to the analytical models, the following dependencies will be installed:
- **Eigen** from [gitlab](https://gitlab.com/libeigen/eigen)
- **Abseil** from [github](https://github.com/abseil/abseil-cpp)
- **GoogleTest** from [github](https://github.com/google/googletest)
- **Ceres** from [github](https://github.com/ceres-solver/ceres-solver)

### C++ library

```bash
mkdir build && cd build
cmake ..
make
```

### Python library

```bash
pip install mag_cusps
```

## How to use

### Reading data

The library provides a ReaderWriter interface in `./headers_cpp/reader_writer.h` explaining how to implement the proper reader for your data. As explained in the header, the indexing of the data should be in Fortran style indexing, i.e.

```cpp
M(ix,iy,iz,ic) = M.mat[ ix + iy*shape.x + iz*shape.x*shape.y + ic*shape.x*shape.y*shape.z ];
```

If the indexing of your data is different, a Shape strides object will need to be passed when constructing the matrix, which will then be indexed like so:

```cpp
M(ix,iy,iz,ic) = M.mat[ ix*stride.x + iy*strides.y + iz*strides.z + ic*strides.i ];
```

### Preprocessing data

The library provides functions to extrapolate data from different grid types. 
It can modify the shape of the grid to increase of decrease resolution through interpolation, but also create a uniform grid from non-uniform data if provided with the X, Y and Z matrices describing the value that corresponds to each cell. 

### Extracting features

The library provides functions to extract features of the simulation:
- **get_interest_points**: will return a C-style array of `InterestPoint`, containing, for some point on the magnetopause, its angle theta in `InterestPoint::theta`, its angle phi in `InterestPoint::phi`, its radius away from the Earth in `InterestPoint::radius` and finally its *weight*, a number from 0 to 1 in `InterestPoint::weight`, being to the certainty the algorithm gives to that particular point being in the correct place. Do note a weight of 1 doesn't guarantee that the point is indeed 100% correct but simply that the algorithm did not find an alternative.
- **get_bowshock**: will return a `std::vector<Point>` 
