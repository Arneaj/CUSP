"""Module to import vtk files into numpy arrays."""

import os

import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy


def VTK_import_files(filedir, ctime, varlist, space_import=True):
    """Import VTK files from a given directory and time step for a list of variables.

    Args:
    ----
        filedir (str): The directory containing the VTK files.
        ctime (str): The time step of the VTK files to import.
        varlist (list): A list of variable names to import.
        space_import (bool, optional): Whether to import the spatial coordinates
        as well. Defaults to True.

    Returns:
    -------
        If space_import is True, returns a tuple containing the spatial coordinates
        (x, y, z) and a dictionary
        containing the imported variables. If space_import is False, returns only the
        dictionary of imported variables.

    """
    if space_import:
        x, y, z = VTK_import_space(filedir + varlist[0] + "-" + ctime + ".vti")

    D = {
        s: VTK_import_array(filedir + s + "-" + ctime + ".vti", "x44_" + s)
        for s in varlist
    }

    if space_import:
        return x, y, z, D
    else:
        return D


def VTK_import_file(filename, varname):
    """Import a VTK file and returns the x, y, z coordinates and the variable data.

    Args:
    ----
        filename (str): The path to the VTK file.
        varname (str): The name of the variable to import.

    Returns:
    -------
        tuple: A tuple containing the x, y, z coordinates and the variable data.

    """
    if os.path.isfile(filename):
        print("File", filename, varname)
    else:
        print("File", filename + " not found")
        return

    # Load file

    # varname = filename[:4] + varname

    reader = vtk.vtkXMLImageDataReader()
    reader.SetFileName(filename)
    reader.Update()
    reader.GetNumberOfCells()

    data = reader.GetOutput()

    """ Import space """

    dim = np.asarray(data.GetDimensions())
    c = np.asarray(data.GetOrigin())
    d = np.asarray(data.GetSpacing())

    x = np.arange(dim[0]) * d[0] + c[0]
    y = np.arange(dim[1]) * d[1] + c[1]
    z = np.arange(dim[2]) * d[2] + c[2]

    x = 0.5 * (x[1:] + x[:-1])
    y = 0.5 * (y[1:] + y[:-1])
    z = 0.5 * (z[1:] + z[:-1])

    # Import array

    cdata = data.GetCellData()
    N_comps = cdata.GetNumberOfComponents()

    v = vtk_to_numpy(data.GetCellData().GetArray(varname))

    vec = [int(i - 1) for i in dim]
    if N_comps > 1:
        vec.append(N_comps)
    v = v.reshape(vec, order="F")

    return x, y, z, v


def VTK_import_space(filename):
    """Import space from a VTK file.

    Args:
    ----
        filename (str): The path to the VTK file.

    Returns:
    -------
        tuple: A tuple containing the x, y, and z coordinates of the space.

    """
    if os.path.isfile(filename):
        print("Space", filename)
    else:
        print("Space", filename + " not found")
        return

    reader = vtk.vtkXMLImageDataReader()
    reader.SetFileName(filename)
    reader.Update()
    reader.GetNumberOfCells()

    data = reader.GetOutput()

    """ Import space """

    dim = np.asarray(data.GetDimensions())
    c = np.asarray(data.GetOrigin())
    d = np.asarray(data.GetSpacing())

    x = np.arange(dim[0]) * d[0] + c[0]
    y = np.arange(dim[1]) * d[1] + c[1]
    z = np.arange(dim[2]) * d[2] + c[2]

    x = 0.5 * (x[1:] + x[:-1])
    y = 0.5 * (y[1:] + y[:-1])
    z = 0.5 * (z[1:] + z[:-1])

    return x, y, z


def VTK_import_array(filename, varname):
    """Read a VTK file and returns a numpy array of the specified variable.

    Args:
    ----
        filename (str): The path to the VTK file.
        varname (str): The name of the variable to extract.

    Returns:
    -------
        numpy.ndarray: A numpy array of the specified variable.

    """
    if os.path.isfile(filename):
        print("File", filename, varname)
    else:
        print("File", filename + " not found")
        return

    # varname = filename[:4] + varname

    reader = vtk.vtkXMLImageDataReader()
    reader.SetFileName(filename)
    reader.Update()
    reader.GetNumberOfCells()

    data = reader.GetOutput()
    dim = data.GetDimensions()

    cdata = data.GetCellData()
    N_comps = cdata.GetNumberOfComponents()

    v = vtk_to_numpy(data.GetCellData().GetArray(varname))

    vec = [int(i - 1) for i in dim]
    if N_comps > 1:
        vec.append(N_comps)
    v = v.reshape(vec, order="F")

    return v


def VTK_import_scalar_file(filename, varname):
    """Import a scalar file in VTK format and returns a numpy array.

    Read a VTK XML image data file and returns a numpy array of the scalar data
    associated with the given variable name.

    Args:
    ----
        filename (str): The path to the VTK XML image data file.
        varname (str): The name of the scalar variable to extract.

    Returns:
    -------
        numpy.ndarray: A numpy array of the scalar data associated with the given
        variable name.

    """
    print("s", filename, varname)

    reader = vtk.vtkXMLImageDataReader()
    reader.SetFileName(filename)
    reader.Update()
    reader.GetNumberOfCells()

    data = reader.GetOutput()
    dim = data.GetDimensions()

    cdata = data.GetCellData()

    v = vtk_to_numpy(cdata.GetArray(varname))

    vec = [int(i - 1) for i in dim]
    v = v.reshape(vec, order="F")

    return v


def VTK_import_vector_file(filename, varname):
    """Import a vector file in VTK format and returns a numpy array.

    Args:
    ----
        filename (str): The path to the VTK file.
        varname (str): The name of the vector variable to import.

    Returns:
    -------
        numpy.ndarray: A numpy array containing the vector data.

    """
    print("v", filename, varname)
    reader = vtk.vtkXMLImageDataReader()
    reader.SetFileName(filename)
    reader.Update()
    reader.GetNumberOfCells()

    data = reader.GetOutput()
    dim = data.GetDimensions()

    cdata = data.GetCellData()
    N_comps = cdata.GetNumberOfComponents()

    v = vtk_to_numpy(data.GetCellData().GetArray(varname))

    vec = [int(i - 1) for i in dim]
    vec.append(N_comps)
    v = v.reshape(vec, order="F")

    return v
