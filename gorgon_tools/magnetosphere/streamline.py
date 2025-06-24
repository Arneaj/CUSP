"""Module for the streamline class."""
import matplotlib.pyplot as plt
import numpy as np
import vtk
from scipy.interpolate import RegularGridInterpolator as interpolate
from vtk.util import numpy_support as vtk_np

from ._fortran import streamtracer


# %% Single Streamline
class streamline:
    """For calculating a single streamline.

    Calculating a streamline from a given seed point in a 3D uniform grid. Has
    capability for setting the number of step, setting the direction of the
    streamline, stitching forward and backward directions together if you want
    to calculate both. It's a wrapper around the F2py streamtracer module,
    written in fortran.

    Args:
    ----
        n_steps (int): maximum number of steps for the streamtracer
        step_size (float): step size of streamline calculation
        direction (int): direction streamline: 1 for forward, -1 for backward
            and 0 for both

    """

    def __init__(self, n_steps, step_size, direction=0):
        """Initialise the streamline object."""
        self.ns = n_steps  # Number of steps
        self.ns0 = n_steps  # Save original number
        self.ds = step_size  # Integration step size
        self.dir = direction  # Integration direction of streamline

        self.ROT = 0  # Reason of termination

        self._ROT_reasons = ["Uncalculated", "Out of steps", "Out of domain", "Isnan"]
        self._dir_str = {-1: "Reverse", 0: "Both", 1: "Forward"}

        self.var = {}
        self.var_names = []
        self.cell_data = {}

        # Preallocate some arrays
        if direction == 1 or direction == -1:
            self.xs = np.zeros([n_steps, 3])
            self.s = np.zeros(n_steps)
        else:  # If direction is both (0), need twice the space
            self.xs = np.zeros([2 * n_steps, 3])
            self.s = np.zeros(2 * n_steps)

    def __str__(self):
        """Return a string representation of the streamline object."""
        if isinstance(self.ROT, int):
            ROT = self._ROT_reasons[self.ROT]
        else:
            ROT = [self._ROT_reasons[i] for i in self.ROT]

        direction = self._dir_str[self.dir]

        var_list = [s + ": " + str(self.var[s].shape) for s in self.var]

        return (
            "Streamline object:\n No. steps = {}\n Step Size = {}\n "
            "Direction = {}\n Reason of Termination = {}\n Variable list: {}"
        ).format(self.ns, self.ds, direction, ROT, var_list)

    # Calculate the streamline from a vector array
    def calc(self, x0, v, d, xc):
        """Calculate the streamline.

        Args:
        ----
            x0 (np.array): streamline start position (np.array)
            v (np.array): velocity array
            d (np.array): grid spacing in each direction
            xc (np.array): position of the [0, 0, 0] cell

        """
        self.x0 = x0
        streamtracer.ds = self.ds
        streamtracer.xc = xc

        x0 += xc

        if self.dir == 1 or self.dir == -1:
            self.xs, ROT, self.ns = streamtracer.streamline(x0, v, d, self.dir, self.ns)

            self.xs = self.xs[: self.ns, :]

        elif self.dir == 0:
            xs_f, ROT_f, ns_f = streamtracer.streamline(x0, v, d, 1, self.ns)
            xs_r, ROT_r, ns_r = streamtracer.streamline(x0, v, d, -1, self.ns)

            self.ROT = np.array([ROT_f, ROT_r])

            self.xs = np.vstack([xs_r[ns_r:0:-1, :], xs_f[:ns_f, :]])
            self.ns = self.xs.shape[0]

            self.ROT = np.array([ROT_f, ROT_r])

        self.xs = np.array([xi - xc for xi in self.xs])

        self.s = np.arange(self.ns) * self.ds

    def reset(self, ns=None, ds=None):
        """Reset the streamline object with new values for step size and no. of steps.

        Args:
        ----
            ns (int, optional): Number of steps. Defaults to None.
            ds (float, optional): Step size. Defaults to None.

        """
        if ns is None:
            ns = self.ns
        if ds is None:
            ds = self.ds
        self.__init__(ns, ds)

    # Interpolate for other quantities

    def __interp_scalar(self, x, y, z, v):
        interpolated_values = interpolate((x, y, z), v, bounds_error=False)
        return interpolated_values(self.xs)

    def _interp_vector(self, x, y, z, v):
        x_interp = interpolate((x, y, z), v[:, :, :, 0], bounds_error=False)
        y_interp = interpolate((x, y, z), v[:, :, :, 1], bounds_error=False)
        z_interp = interpolate((x, y, z), v[:, :, :, 2], bounds_error=False)

        return np.array([x_interp(self.xs), y_interp(self.xs), z_interp(self.xs)]).T

    def interp(self, x, y, z, v, varname):
        """Interpolate scalar or vector data onto the streamline points.

        Args:
        ----
            x (ndarray): x-coordinates of the streamline points.
            y (ndarray): y-coordinates of the streamline points.
            z (ndarray): z-coordinates of the streamline points.
            v (ndarray): Scalar or vector data to interpolate.
            varname (str): Name of the interpolated variable.

        Returns:
        -------
            None

        """
        if len(v.shape) == 3:
            vi = self._interp_scalar(x, y, z, v)
        elif len(v.shape) == 4:
            vi = self._interp_vector(x, y, z, v)

        self.var[varname] = vi

    # Plotting functions
    def plot(self, ax=None):
        """Plot the streamline on the given axis.

        Args:
        ----
            ax (matplotlib.axes.Axes, optional): The axis on which to plot the
            streamline. If None, a new axis is created.

        Returns:
        -------
            matplotlib.figure.Figure: The figure object.
            matplotlib.axes.Axes: The axis object.

        """
        i = 0
        if ax is None:
            fig, ax = plt.subplots()
            i = 1

        ax.plot(self.s, self.xs)

        if i == 1:
            return fig, ax

    def plot3D(self, ax=None):
        """Plot the 3D streamlines.

        Args:
        ----
            ax (matplotlib.axes.Axes, optional): The axis on which to plot the
            streamline. If None, a new axis is created.

        Returns:
        -------
            matplotlib.figure.Figure: The figure object.
            matplotlib.axes.Axes: The axis object.

        """
        i = 0
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")
            i = 1

        ax.plot(self.xs[:, 0], self.xs[:, 1], self.xs[:, 2])

        if i == 1:
            return fig, ax


# %% Streamline array


class streamline_array(streamline):
    """Class for storing and manipulating streamline arrays."""

    def __init__(self, n_steps, step_size, direction=0, inner_boundary=True, r_IB=1.0):
        """Initialise the streamline_array object."""
        self.ns = n_steps  # Number of steps
        self.ns0 = n_steps  # Save original number
        self.ds = step_size  # Integration step size
        self.dir = direction  # Integration direction of streamline

        streamtracer.inner_boundary = inner_boundary
        streamtracer.r_IB = 1.0

        self._ROT_reasons = ["Uncalculated", "Out of steps", "Out of domain", "Isnan"]
        self._dir_str = {-1: "Reverse", 0: "Both", 1: "Forward"}

        self.var = {}
        self.var_names = []
        self.cell_data = {}

    def reset(self, ns=None, ds=None):
        """Reset the streamline object with new starting point and step size.

        Args:
        ----
            ns (float, optional): New starting point. Defaults to None.
            ds (float, optional): New step size. Defaults to None.

        """
        del self.xs
        del self.ROT

        if ns is None:
            ns = self.ns0
        if ds is None:
            ds = self.ds

        self.__init__(ns, ds)

    # Calculate the streamline from a vector array
    def calc(self, x0, v, d, xc, v_name="v"):
        """Calculate the streamline array.

        Args:
        ----
            x0 (np.array): streamline start position (np.array)
            v (np.array): velocity array
            d (np.array): grid spacing in each direction
            xc (np.array): position of the [0, 0, 0] cell
            v_name (str): Name of the variable to store the velocity field.

        Returns:
        -------
            None

        """
        self.x0 = x0.copy()
        self.n_lines = x0.shape[0]
        streamtracer.ds = self.ds
        streamtracer.xc = xc.copy()

        self.x0 = np.array([xi + xc for xi in self.x0])

        if self.dir == 1 or self.dir == -1:
            # Calculate streamlines
            self.xs, vs, ROT, self.ns = streamtracer.streamline_array(
                self.x0, v, d, self.dir, self.ns
            )

            # Reduce the size of the array
            self.xs = np.array(
                [xi[:ni, :] for xi, ni in zip(self.xs, self.ns)], dtype=object
            )
            vs = np.array([vi[:ni, :] for vi, ni in zip(vs, self.ns)], dtype=object)

            # Save the Reason of Termination
            self.ROT = ROT

        elif self.dir == 0:
            # Calculate forward streamline
            xs_f, vs_f, ROT_f, ns_f = streamtracer.streamline_array(
                self.x0, v, d, 1, self.ns
            )
            # Calculate backward streamline
            xs_r, vs_r, ROT_r, ns_r = streamtracer.streamline_array(
                self.x0, v, d, -1, self.ns
            )

            # Reduce the size of the arrays, and flip the reverse streamlines
            xs_f = np.array([xi[:ni, :] for xi, ni in zip(xs_f, ns_f)], dtype=object)
            vs_f = np.array([vi[:ni, :] for vi, ni in zip(vs_f, ns_f)], dtype=object)

            xs_r = np.array(
                [xi[ni - 1 : 0 : -1, :] for xi, ni in zip(xs_r, ns_r)], dtype=object
            )
            vs_r = np.array(
                [vi[ni - 1 : 0 : -1, :] for vi, ni in zip(vs_r, ns_r)], dtype=object
            )

            self.xs_f = np.array([xi - xc for xi in xs_f], dtype=object)
            self.xs_r = np.array([xi - xc for xi in xs_r], dtype=object)

            # Stack the forward and reverse arrays
            self.xs = np.array(
                [np.vstack([xri, xfi]) for xri, xfi in zip(xs_r, xs_f)], dtype=object
            )
            vs = np.array(
                [np.vstack([vri, vfi]) for vri, vfi in zip(vs_r, vs_f)], dtype=object
            )
            self.ns = np.fromiter([len(xsi) for xsi in self.xs], int)

            self.ROT = np.vstack([ROT_f, ROT_r]).T

        # Remove streamlines with zero size
        el = self.ns > 1
        self.ROT = self.ROT[el]  # , :
        self.ns = self.ns[el]

        self.xs = np.array([xi - xc for xi in self.xs], dtype=object)

        self.var[v_name] = vs.copy()
        self.var_names = np.array([s for s in self.var])
        del vs

        for s in self.cell_data:
            self.cell_data[s] = self.cell_data[s][el]

    # Interpolate for other quantities

    def interp(self, x, y, z, v, var_name):
        """Interpolate scalar or vector data onto the streamline points.

        Args:
        ----
            x (ndarray): x-coordinates of the streamline points.
            y (ndarray): y-coordinates of the streamline points.
            z (ndarray): z-coordinates of the streamline points.
            v (ndarray): scalar or vector data to be interpolated.
            var_name (str): name of the interpolated variable.

        Returns:
        -------
            None

        """
        if len(v.shape) == 3:
            vi = self._interp_scalar(x, y, z, v)
        elif len(v.shape) == 4:
            vi = self._interp_vector(x, y, z, v)

        self.var[var_name] = vi

        self.var_names = np.array([s for s in self.var])

    def _interp_scalar(self, x, y, z, f):
        """Interpolates scalar values onto the streamline.

        Args:
        ----
            x (ndarray): x-coordinates of the scalar values.
            y (ndarray): y-coordinates of the scalar values.
            z (ndarray): z-coordinates of the scalar values.
            f (ndarray): Scalar values to interpolate.

        Returns:
        -------
            ndarray: Interpolated scalar values.

        """
        f_interp = interpolate((x, y, z), f, bounds_error=False)

        xI = np.vstack(self.xs)
        fI = f_interp(xI)

        fI = np.array(np.split(fI, np.cumsum(self.ns)), dtype=object)[:-1]

        return fI

    def _interp_vector(self, x, y, z, v):
        """Interpolates a vector field `v` at the given coordinates `(x, y, z)`.

        Args:
        ----
            x (ndarray): x-coordinates of the vector field.
            y (ndarray): y-coordinates of the vector field.
            z (ndarray): z-coordinates of the vector field.
            v (ndarray): Vector field to interpolate.

        Returns:
        -------
            ndarray: Interpolated vector field.

        """
        Ix = interpolate((x, y, z), v[:, :, :, 0], bounds_error=False, fill_value=None)
        Iy = interpolate((x, y, z), v[:, :, :, 1], bounds_error=False, fill_value=None)
        Iz = interpolate((x, y, z), v[:, :, :, 2], bounds_error=False, fill_value=None)

        xI = np.vstack(self.xs)
        vI = np.array([Ix(xI), Iy(xI), Iz(xI)]).T

        vI = np.array(np.vsplit(vI, np.cumsum(self.ns)), dtype=object)[:-1]

        return vI

    # Plotting methods

    def plot_line(self, i, ax=None, plot_names=None):
        """Plot a single streamline and its variables.

        Args:
        ----
            i (int): Index of the streamline to plot.
            ax (matplotlib.axes.Axes, optional): The axis on which to plot the
            streamline. If None, a new axis is created.
            plot_names (list, optional): List of variables to plot. If None, all
            variables are plotted.

        Returns:
        -------
            matplotlib.figure.Figure: The figure object.
            matplotlib.axes.Axes: The axis object.

        """
        if plot_names is None:
            plot_names = self.var_names

        n_plots = len(plot_names) + 1

        fig = plt.figure(figsize=(12, 6))

        ax_line = [
            plt.subplot2grid((n_plots, 5), (i, 0), colspan=3) for i in range(n_plots)
        ]

        ax_3D = plt.subplot2grid(
            (n_plots, 5), (0, 3), rowspan=n_plots, colspan=2, projection="3d"
        )

        self.plot3D(ax_3D, i)
        ax_3D.plot([self.xs[i][0, 0]], [self.xs[i][0, 1]], [self.xs[i][0, 2]])

        self.plot_linevars(i, ax=ax_line)

        for axi in ax_line[-2::-1]:
            axi.set(xlim=ax_line[-1].get_xlim(), xticklabels=[])

        fig.tight_layout()

        return fig, ax_line, ax_3D

    def plot_linevars(self, i, ax=None, plot_names=None):
        """Plot the variables of a single streamline.

        Args:
        ----
            i (int): The index of the streamline to plot.
            ax (matplotlib.axes.Axes, optional): The axes to plot on. If not provided,
            a new figure is created.
            plot_names (list of str, optional): The names of the variables to plot.
            If not provided, all variables are plotted.

        Returns:
        -------
            If ax is not provided, returns a tuple (fig, ax) containing the created
            figure and axes.

        """
        if plot_names is None:
            plot_names = self.var_names

        ret = False
        if ax is None:
            fig, ax = plt.subplots(len(plot_names) + 1, 1, sharex=True)
            ret = True

        s = np.linspace(0, 1, self.ns[i]) * self.ns[i] * self.ds

        ax[0].plot(s, self.xs[i])
        mag = np.sqrt(np.sum(self.xs[i] ** 2, axis=1))
        ax[0].plot(s, mag)

        for axi, n in zip(ax[1:], plot_names):
            axi.plot(s, self.var[n][i])
            if self.var[n][i].shape[-1] == 3:
                mag = np.sqrt(np.sum(self.var[n][i] ** 2, axis=1))
                axi.plot(s, mag)
            axi.set_ylabel(n)

        if ret:
            return fig, ax

    def plot3D(self, ax=None, i=None, seed=False, max_lines=200):
        """Plot the 3D streamlines of the magnetic field.

        Args:
        ----
            ax (matplotlib.axes.Axes, optional): The axis on which to plot the
            streamlines. If None, a new axis is created.
            i (int, optional): The index of the streamline to plot. If None, all
            streamlines are plotted.
            seed (bool, optional): Whether to plot the seed points. Defaults to False.
            max_lines (int, optional): The maximum number of streamlines to plot.
            Defaults to 200.

        Returns:
        -------
            matplotlib.figure.Figure: The figure object.
            matplotlib.axes.Axes: The axis object.

        """
        ret = False
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")
            ret = True

        step = int(np.ceil(self.n_lines / max_lines))

        if i is None:
            if seed:
                ax.plot(self.x0[:, 0], self.x0[:, 1], self.x0[:, 2], ".")
            for xi in self.xs[::step]:
                ax.plot(xi[:, 0], xi[:, 1], xi[:, 2], linewidth=1)

        elif isinstance(i, int):
            if seed:
                ax.plot(self.x0[i, 0], self.x0[i, 1], self.x0[i, 2], ".")
            ax.plot(self.xs[i][:, 0], self.xs[i][:, 1], self.xs[i][:, 2], linewidth=1)

        else:
            if seed:
                ax.plot(self.x0[i, 0], self.x0[i, 1], self.x0[i, 2], ".")
            for xi in self.xs[i]:
                ax.plot(xi[:, 0], xi[:, 1], xi[:, 2], linewidth=1)

        ax.set(xlabel="x", ylabel="y", zlabel="z")

        if ret:
            return fig, ax

    def plot_seeds3D(self, ax=None):
        """Plot the 3D seed points of the streamline.

        Args:
        ----
            ax (matplotlib.axes.Axes, optional): The axis on which to plot the
            streamlines. If None, a new axis is created.

        Returns:
        -------
            matplotlib.figure.Figure: The figure object.
            matplotlib.axes.Axes: The axis object.

        """
        ret = False
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")
            ret = True

        ax.plot(self.x0[:, 0], self.x0[:, 1], self.x0[:, 2], ".")
        ax.set(xlabel="x", ylabel="y", zlabel="z")

        if ret:
            return fig, ax

        # %% Write to vtp

    def write_vtp(self, fname, pts_step=10):
        """Write the streamline data to a VTK XML PolyData file format (.vtp).

        Args:
        ----
            fname (str): The name of the output file.
            pts_step (int, optional): The step size for the points. Defaults to 10.

        Returns:
        -------
            None

        """
        if pts_step is None:
            pts_step = int(max(1, 1.0 / self.ds))

        # Points
        pts = np.vstack([xi[::pts_step] for xi in self.xs]).ravel()
        doubleArray = vtk_np.numpy_to_vtk(pts)
        doubleArray.SetNumberOfComponents(3)

        points = vtk.vtkPoints()
        points.SetData(doubleArray)

        # Cells

        n_pts_in_cell = np.array([len(xi[::pts_step]) for xi in self.xs])

        i = np.arange(np.sum(n_pts_in_cell), dtype=np.int64)
        i = np.array(np.split(i, n_pts_in_cell.cumsum())[:-1], dtype=object)

        id_array = np.array(
            [np.hstack([ni, ii]) for ni, ii in zip(n_pts_in_cell, i)], dtype=object
        )

        id_array = np.hstack([ii for ii in id_array])

        cellArray = vtk.vtkCellArray()
        idArray = vtk_np.numpy_to_vtkIdTypeArray(id_array)
        cellArray.SetCells(len(self.ns), idArray)

        # Pointdata

        point_arrays = self._vtk_pointData_arrays(pts_step)

        # Cell Data

        cell_arrays = self._vtk_cellData_arrays(pts_step)

        # Polydata

        polyData = vtk.vtkPolyData()
        polyData.SetPoints(points)
        polyData.SetLines(cellArray)

        for Arr in point_arrays:
            polyData.GetPointData().AddArray(Arr)

        for Arr in cell_arrays:
            polyData.GetCellData().AddArray(Arr)

        # Write to file

        writer = vtk.vtkXMLPolyDataWriter()
        writer.SetFileName(fname)
        writer.SetInputData(polyData)
        writer.Write()

    def _vtk_pointData_arrays(self, pts_step):
        def create_pointDataArrays(arr, name):
            if len(arr[0].shape) == 1:
                data = np.hstack([fi[::pts_step] for fi in arr])
            else:
                data = np.vstack([fi[::pts_step] for fi in arr]).ravel()

            data = data.astype(np.float64)

            doubleArray = vtk_np.numpy_to_vtk(data, deep=1)
            doubleArray.SetName(name)

            if len(arr[0].shape) > 1:
                doubleArray.SetNumberOfComponents(arr[0].shape[1])

            return doubleArray

        return [create_pointDataArrays(self.var[name], name) for name in self.var]

    def _vtk_cellData_arrays(self, pts_step):
        def create_cellDataArrays(arr, name):
            data = arr.ravel()

            data = data.astype(np.float64)

            doubleArray = vtk_np.numpy_to_vtk(data, deep=1)
            doubleArray.SetName(name)

            if len(arr.shape) > 1:
                doubleArray.SetNumberOfComponents(arr.shape[1])

            return doubleArray

        cell_arrays = [self.ns, self.ROT]
        cell_names = ["ns", "ROT"]

        for s in self.cell_data:
            cell_arrays.append(self.cell_data[s])
            cell_names.append(s)

        return [
            create_cellDataArrays(arr, name)
            for arr, name in zip(cell_arrays, cell_names)
        ]
