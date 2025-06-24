"""Vortex.

Created on Tue Aug 06 11:42:00 2024

@author: Harley Kelly
@email: h.kelly21@imperial.ac.uk
"""

import numpy as np
from scipy.interpolate import griddata

from gorgon_tools.magnetosphere import decomp

# Define the Earth radii
R_E = 6.3781e6  # m
# Define the magnetic permeability of free space
mu0 = 4 * np.pi * 1e-7  # Tm/A

"This function solves the eigenvalue problem for the lambda criteria"


def lambda_criterion(lambda_matrix):
    """Lambda criterion.

    Parameters
    ----------
    lambda_matrix : np.ndarray
        The matrix of the lambda criteria.

    Returns
    -------
    np.ndarray : The lambda2 criterion for the vortex.
    """
    # Flatten the array and transpose it to prepare for eigvals function
    sp = lambda_matrix.shape
    lambda_matrix_flat = lambda_matrix.reshape(sp[0], sp[1], sp[2] * sp[3] * sp[4])
    lambda_matrix_flat = lambda_matrix_flat.transpose(2, 0, 1)

    # Find the eigenvalues and then only keep the lambda2 value
    eigs_mat_flat = np.linalg.eigvals(lambda_matrix_flat).transpose(1, 0)
    eigs_mat = eigs_mat_flat.reshape(3, sp[2], sp[3], sp[4])

    return np.sort(eigs_mat, axis=0)[1, :, :, :]


"""
Calculate the incompressible Q criterion for a vortex using the velocity
vector field of shape (v,x,y,z) where v is the number of components of
the velocity vector field. This method will take about 40 seconds to run
on a 480x320x320 grid.
"""


def calc_Q_criterion(V_arr, x, y, z):
    """Calculate the incompressible Q criterion for a vortex.

    Parameters
    ----------
    V_arr : np.ndarray
        The velocity field in the form of a numpy array with shape (3, x, y, z).
    x : np.ndarray
        The x coordinates of the grid.
    y : np.ndarray
        The y coordinates of the grid.
    z : np.ndarray
        The z coordinates of the grid.

    Returns
    -------
    np.ndarray : The Q criterion for

    """
    # Define the gradient tensor
    V_grad = np.gradient(
        V_arr, x * R_E, y * R_E, z * R_E, axis=(1, 2, 3)
    )  # units of s-1

    # Define the Velocity Strain rate tensor
    S_V = 0.5 * (V_grad + np.transpose(V_grad, (1, 0, 2, 3, 4)))  # units of s-1

    # Define the Velocity Rotation rate tensor
    Omega_V = 0.5 * (V_grad - np.transpose(V_grad, (1, 0, 2, 3, 4)))  # units of s-1

    # Q = 1/2 ( ||Omega|| ^ 2 - ||S||^2 )
    Q_incomp = 0.5 * (
        np.square(np.linalg.norm(Omega_V, axis=(0, 1), ord="fro"))
        - np.square(np.linalg.norm(S_V, axis=(0, 1), ord="fro"))
    )

    return Q_incomp


"""
Calculate the lambda2 criterion for a vortex
"""


def calc_lambda2(V_arr, x, y, z):
    """Calculate the lambda2 criterion for a vortex.

    Parameters
    ----------
    V_arr : np.ndarray
        The velocity field in the form of a numpy array with shape (3, x, y, z).
    x : np.ndarray
        The x coordinates of the grid.
    y : np.ndarray
        The y coordinates of the grid.
    z : np.ndarray
        The z coordinates of the grid.

    Returns
    -------
    np.ndarray : The lambda2 criterion for the vortex.

    """
    # Define the gradient tensor
    V_grad = np.gradient(
        V_arr, x * R_E, y * R_E, z * R_E, axis=(1, 2, 3)
    )  # units of s-1

    # Define the Velocity Strain rate tensor
    S_V = 0.5 * (V_grad + np.transpose(V_grad, (1, 0, 2, 3, 4)))  # units of s-1

    # Define the Velocity Rotation rate tensor
    Omega_V = 0.5 * (V_grad - np.transpose(V_grad, (1, 0, 2, 3, 4)))  # units of s-1

    # Calculate the S^2 and Omega^2 tensors
    S2_O2 = np.einsum("ik...,kj...->ij...", S_V, S_V) + np.einsum(
        "ik...,kj...->ij...", Omega_V, Omega_V
    )  # units of s-2

    return lambda_criterion(S2_O2)


"""
Calculate the weighted lambda2 criterion for a vortex
"""


def calc_weighted_lambda2(V_arr, Rho_arr, x, y, z):
    """Calculate the weighted lambda2 criterion for a vortex.

    Parameters
    ----------
    V_arr : np.ndarray
        The velocity field in the form of a numpy array with shape (3, x, y, z).
    Rho_arr : np.ndarray
        The density field in the form of a numpy array with shape (x, y, z).
    x : np.ndarray
        The x coordinates of the grid.
    y : np.ndarray
        The y coordinates of the grid.
    z : np.ndarray
        The z coordinates of the grid.

    Returns
    -------
    np.ndarray : The weighted lambda2 criterion for the vortex.

    """
    # Define the gradient tensor
    V_grad = np.gradient(
        V_arr, x * R_E, y * R_E, z * R_E, axis=(1, 2, 3)
    )  # units of s-1

    # Define the Velocity Strain rate tensor
    S_V = 0.5 * (V_grad + np.transpose(V_grad, (1, 0, 2, 3, 4)))  # units of s-1

    # Define the Velocity Rotation rate tensor
    Omega_V = 0.5 * (V_grad - np.transpose(V_grad, (1, 0, 2, 3, 4)))  # units of s-1

    # Calculate the S^2 and Omega^2 tensors
    S2_O2 = np.einsum("ik...,kj...->ij...", S_V, S_V) + np.einsum(
        "ik...,kj...->ij...", Omega_V, Omega_V
    )  # units of s-2

    # Weight the S2_O2 tensor by the density field
    WL2 = np.einsum("xyz,ijxyz->ijxyz", Rho_arr, S2_O2)

    return lambda_criterion(WL2)


"""
Calculate the lambda rho criterion for a vortex
"""


def calc_lambda_rho(V_arr, Rho_arr, x, y, z):
    """Calculate the lambda rho criterion for a vortex.

    Parameters
    ----------
    V_arr : np.ndarray
        The velocity field in the form of a numpy array with shape (3, x, y, z).
    Rho_arr : np.ndarray
        The density field in the form of a numpy array with shape (x, y, z).
    x : np.ndarray
        The x coordinates of the grid.
    y : np.ndarray
        The y coordinates of the grid.
    z : np.ndarray
        The z coordinates of the grid.

    Returns
    -------
    np.ndarray : The lambda rho criterion for the vortex.

    """
    # Define the gradient tensor
    V_grad = np.gradient(
        V_arr, x * R_E, y * R_E, z * R_E, axis=(1, 2, 3)
    )  # units of s-1

    # Define the Velocity Strain rate tensor
    S_V = 0.5 * (V_grad + np.transpose(V_grad, (1, 0, 2, 3, 4)))  # units of s-1

    # Define the Velocity Rotation rate tensor
    Omega_V = 0.5 * (V_grad - np.transpose(V_grad, (1, 0, 2, 3, 4)))  # units of s-1

    # Calculate the S^2 and Omega^2 tensors
    S2_O2 = np.einsum("ik...,kj...->ij...", S_V, S_V) + np.einsum(
        "ik...,kj...->ij...", Omega_V, Omega_V
    )  # units of s-2

    # Weight the S2_O2 tensor by the density field
    WL2 = np.einsum("xyz,ijxyz->ijxyz", Rho_arr, S2_O2)

    # Calculate Term 1
    grad_Rho = np.gradient(
        Rho_arr, x * R_E, y * R_E, z * R_E, axis=(0, 1, 2)
    )  # units of kg m-4

    A = np.einsum("i...,jk...->ij...", V_arr, V_grad)  # units of m s-2

    Inhom = 0.5 * np.einsum(
        "k...,ij...->ij...", grad_Rho, (A + np.transpose(A, (1, 0, 2, 3, 4)))
    )

    term_1 = WL2 + Inhom  # units of kg m-3 s-2

    # Calculate Term 2
    theta = np.einsum("kkxyz->xyz", V_grad)  # units of s-1

    A = np.einsum("xyz,ixyz->ixyz", Rho_arr, V_arr)  # units of kg m-2 s-1

    B = np.einsum("xyz,ixyz->ixyz", theta, A)  # units of kg m-2 s-2

    C = np.gradient(B, x * R_E, y * R_E, z * R_E, axis=(1, 2, 3))  # units of kg m-3 s-2

    term_2 = 0.5 * (C + np.transpose(C, (1, 0, 2, 3, 4)))  # units of kg m-3 s-2

    return lambda_criterion(term_1 + term_2)


"""
Calculate the lambda MHD criterion for a vortex
"""


def calc_lambda_MHD(V_arr, Rho_arr, J_arr, B_arr, x, y, z, R=6):
    """Lambda MHD criterion.

    This function calculates the lambda criterion for a MHD vortex. The
    function takes in the velocity field, density field, current field and
    magnetic field and calculates the lambda criterion for the MHD vortex.

    Parameters
    ----------
    V_arr : np.ndarray
        The velocity field in the form of a numpy array with shape (3, x, y, z).
    Rho_arr : np.ndarray
        The density field in the form of a numpy array with shape (x, y, z).
    J_arr : np.ndarray
        The current field in the form of a numpy array with shape (3, x, y, z).
    B_arr : np.ndarray
        The magnetic field in the form of a numpy array with shape (3, x, y, z).
    x : np.ndarray
        The x coordinates of the grid.
    y : np.ndarray
        The y coordinates of the grid.
    z : np.ndarray
        The z coordinates of the grid.
    R : float
        The radius of the inner boundary in Earth radii.

    Returns
    -------
    lambda_criterion : np.ndarray
        The lambda criterion for the MHD vortex.
    """
    # Define the gradient tensor
    V_grad = np.gradient(
        V_arr, x * R_E, y * R_E, z * R_E, axis=(1, 2, 3)
    )  # units of s-1

    # Define the Velocity Strain rate tensor
    S_V = 0.5 * (V_grad + np.transpose(V_grad, (1, 0, 2, 3, 4)))  # units of s-1

    # Define the Velocity Rotation rate tensor
    Omega_V = 0.5 * (V_grad - np.transpose(V_grad, (1, 0, 2, 3, 4)))  # units of s-1

    # Calculate the S^2 and Omega^2 tensors
    S2_O2 = np.einsum("ik...,kj...->ij...", S_V, S_V) + np.einsum(
        "ik...,kj...->ij...", Omega_V, Omega_V
    )  # units of s-2

    # Weight the S2_O2 tensor by the density field
    WL2 = np.einsum("xyz,ijxyz->ijxyz", Rho_arr, S2_O2)

    # Calculate Term 1
    grad_Rho = np.gradient(
        Rho_arr, x * R_E, y * R_E, z * R_E, axis=(0, 1, 2)
    )  # units of kg m-4

    A = np.einsum("i...,jk...->ij...", V_arr, V_grad)  # units of m s-2

    Inhom = 0.5 * np.einsum(
        "k...,ij...->ij...", grad_Rho, (A + np.transpose(A, (1, 0, 2, 3, 4)))
    )

    term_1 = WL2 + Inhom  # units of kg m-3 s-2

    # Calculate Term 2
    theta = np.einsum("kkxyz->xyz", V_grad)  # units of s-1

    A = np.einsum("xyz,ixyz->ixyz", Rho_arr, V_arr)  # units of kg m-2 s-1

    B = np.einsum("xyz,ixyz->ixyz", theta, A)  # units of kg m-2 s-2

    C = np.gradient(B, x * R_E, y * R_E, z * R_E, axis=(1, 2, 3))  # units of kg m-3 s-2

    term_2 = 0.5 * (C + np.transpose(C, (1, 0, 2, 3, 4)))  # units of kg m-3 s-2

    # Calculate Term 3
    Lorentz_force = np.cross(J_arr, B_arr, axis=0)  # units of kg m-1 s-2

    # Delete the region around the inner boundary
    xb, yb, zb = np.meshgrid(x, y, z, indexing="ij")

    # Find r at every point in the box
    r = np.sqrt(np.square(xb) + np.square(yb) + np.square(zb))

    # This is the coordinates of the inner boundary
    ib_x, ib_y, ib_z = np.asarray(np.argwhere(r < R)).T

    Lorentz_force[:, ib_x, ib_y, ib_z] = 0.0

    Lorentz_rotational = decomp.Helmholtz_Decomp_Poisson_rotational_ib(
        Lorentz_force, 0.25 * R_E, R, x, y, z
    )

    grad_rot = np.gradient(
        Lorentz_rotational, x * R_E, y * R_E, z * R_E, axis=(1, 2, 3)
    )  # units of kg m-3 s-2

    term_3 = 0.5 * (
        grad_rot + np.transpose(grad_rot, (1, 0, 2, 3, 4))
    )  # units of kg m-3 s-2

    return lambda_criterion(term_1 + term_2 - term_3)


def calc_lambda_MHD_projections(V_arr, Rho_arr, J_arr, B_arr, x, y, z, R=6):
    """Calculate the lambda MHD projections for a vortex.

    Parameters
    ----------
    V_arr : np.ndarray
        The velocity field in the form of a numpy array with shape (3, x, y, z).
    Rho_arr : np.ndarray
        The density field in the form of a numpy array with shape (x, y, z).
    J_arr : np.ndarray
        The current field in the form of a numpy array with shape (3, x, y, z).
    B_arr : np.ndarray
        The magnetic field in the form of a numpy array with shape (3, x, y, z).
    x : np.ndarray
        The x coordinates of the grid.
    y : np.ndarray
        The y coordinates of the grid.
    z : np.ndarray
        The z coordinates of the grid.
    R : float
        The radius of the inner boundary in Earth radii. Defaults to 6.

    Returns
    -------
    tuple : A tuple containing the eigenvalues and the projections of the
    weighted lambda2 tensor.
    """
    # Define the gradient tensor
    V_grad = np.gradient(
        V_arr, x * R_E, y * R_E, z * R_E, axis=(1, 2, 3)
    )  # units of s-1

    # Define the Velocity Strain rate tensor
    S_V = 0.5 * (V_grad + np.transpose(V_grad, (1, 0, 2, 3, 4)))  # units of s-1

    # Define the Velocity Rotation rate tensor
    Omega_V = 0.5 * (V_grad - np.transpose(V_grad, (1, 0, 2, 3, 4)))  # units of s-1

    # Calculate the S^2 and Omega^2 tensors
    S2_O2 = np.einsum("ik...,kj...->ij...", S_V, S_V) + np.einsum(
        "ik...,kj...->ij...", Omega_V, Omega_V
    )  # units of s-2

    # Weight the S2_O2 tensor by the density field
    WL2 = np.einsum("xyz,ijxyz->ijxyz", Rho_arr, S2_O2)

    # Calculate Term 1
    grad_Rho = np.gradient(
        Rho_arr, x * R_E, y * R_E, z * R_E, axis=(0, 1, 2)
    )  # units of kg m-4

    A = np.einsum("i...,jk...->ij...", V_arr, V_grad)  # units of m s-2

    Inhom = 0.5 * np.einsum(
        "k...,ij...->ij...", grad_Rho, (A + np.transpose(A, (1, 0, 2, 3, 4)))
    )

    term_1 = WL2 + Inhom  # units of kg m-3 s-2

    # Calculate Term 2
    theta = np.einsum("kkxyz->xyz", V_grad)  # units of s-1

    A = np.einsum("xyz,ixyz->ixyz", Rho_arr, V_arr)  # units of kg m-2 s-1

    B = np.einsum("xyz,ixyz->ixyz", theta, A)  # units of kg m-2 s-2

    C = np.gradient(B, x * R_E, y * R_E, z * R_E, axis=(1, 2, 3))  # units of kg m-3 s-2

    term_2 = 0.5 * (C + np.transpose(C, (1, 0, 2, 3, 4)))  # units of kg m-3 s-2

    # Calculate Term 3
    Lorentz_force = np.cross(J_arr, B_arr, axis=0)  # units of kg m-1 s-2

    # Delete the region around the inner boundary
    xb, yb, zb = np.meshgrid(x, y, z, indexing="ij")

    # Find r at every point in the box
    r = np.sqrt(np.square(xb) + np.square(yb) + np.square(zb))

    # This is the coordinates of the inner boundary
    ib_x, ib_y, ib_z = np.asarray(np.argwhere(r < R)).T

    Lorentz_force[:, ib_x, ib_y, ib_z] = 0.0

    Lorentz_rotational = decomp.Helmholtz_Decomp_Poisson_rotational_ib(
        Lorentz_force, 0.25 * R_E, R, x, y, z
    )

    grad_rot = np.gradient(
        Lorentz_rotational, x * R_E, y * R_E, z * R_E, axis=(1, 2, 3)
    )  # units of kg m-3 s-2

    term_3 = 0.5 * (
        grad_rot + np.transpose(grad_rot, (1, 0, 2, 3, 4))
    )  # units of kg m-3 s-2

    lambda_matrix = term_1 + term_2 - term_3

    # Flatten the array and transpose it to prepare for eigvals function

    sp = lambda_matrix.shape

    lambda_matrix_flat = lambda_matrix.reshape(sp[0], sp[1], sp[2] * sp[3] * sp[4])

    lambda_matrix_flat = lambda_matrix_flat.transpose(2, 0, 1)

    # Find eigenvalues and eigenvectors and then only keen the lambda2 value

    eval_mat_flat, evec_mat_flat = np.linalg.eig(lambda_matrix_flat)

    sort = np.asarray(np.argsort(eval_mat_flat))[:, 1]

    A = np.asarray([np.arange(len(sort)), sort])

    eval_mat = eval_mat_flat[A[0], A[1]].reshape(sp[2], sp[3], sp[4])

    evec_mat = evec_mat_flat[A[0], :, A[1]].reshape(sp[2], sp[3], sp[4], 3)

    # Project the tensors onto the eigenvectors

    W_L2_proj = np.einsum(
        "ixyz , xyzi -> xyz",
        np.einsum("xyzj , ijxyz -> ixyz", evec_mat, WL2, dtype=float),
        evec_mat,
        dtype=float,
    )

    inhom_proj = np.einsum(
        "ixyz , xyzi -> xyz",
        np.einsum("xyzj , ijxyz -> ixyz", evec_mat, Inhom, dtype=float),
        evec_mat,
        dtype=float,
    )

    comp_proj = np.einsum(
        "ixyz , xyzi -> xyz",
        np.einsum("xyzj , ijxyz -> ixyz", evec_mat, term_2, dtype=float),
        evec_mat,
        dtype=float,
    )

    tens_proj = np.einsum(
        "ixyz , xyzi -> xyz",
        np.einsum("xyzj , ijxyz -> ixyz", evec_mat, -1.0 * term_3, dtype=float),
        evec_mat,
        dtype=float,
    )

    return eval_mat, W_L2_proj, inhom_proj, comp_proj, tens_proj


def int_ib(arr, R, X, Y, Z):
    """Interpolation over the inner boundary.

    This function interpolates over the inner boundary of the
    vector field arr up to a sphere of radius R.
    """
    if R > 9:
        raise ValueError("The radius interpolating over must be less than 9.")

    # Define the grid
    x, y, z = np.meshgrid(X, Y, Z, indexing="ij")
    pos = np.asarray([x, y, z])

    # Find r
    r = np.sqrt(np.square(x) + np.square(y) + np.square(z))

    # This is the coordinates of the inner boundary
    ib_x, ib_y, ib_z = np.asarray(np.argwhere(r < R)).T

    # Delete contribution from inner boundary for each component of v field
    arr[:, ib_x, ib_y, ib_z] = 0.0

    # 'Delete' inner boundary points so they can be interpolated over instead
    x[ib_x, ib_y, ib_z] = 0
    y[ib_x, ib_y, ib_z] = 0
    z[ib_x, ib_y, ib_z] = 0

    # Define new grid without inner boundary points
    pos_red = np.asarray([x, y, z])

    x_bound_l = int(120 - R / 0.25)
    x_bound_u = int(120 + R / 0.25)

    y_bound_l = int(160 - R / 0.25)
    y_bound_u = int(160 + R / 0.25)

    z_bound_l = int(160 - R / 0.25)
    z_bound_u = int(160 + R / 0.25)

    # Perform interpolation in 3-D using the scipy.interpolate.griddata
    points = (
        pos_red[:, x_bound_l:x_bound_u, y_bound_l:y_bound_u, z_bound_l:z_bound_u]
        .reshape(3, -1)
        .transpose(1, 0)
    )
    values_x = arr[0][
        x_bound_l:x_bound_u, y_bound_l:y_bound_u, z_bound_l:z_bound_u
    ].reshape(-1)
    values_y = arr[1][
        x_bound_l:x_bound_u, y_bound_l:y_bound_u, z_bound_l:z_bound_u
    ].reshape(-1)
    values_z = arr[2][
        x_bound_l:x_bound_u, y_bound_l:y_bound_u, z_bound_l:z_bound_u
    ].reshape(-1)
    xi = (
        pos[:, x_bound_l:x_bound_u, y_bound_l:y_bound_u, z_bound_l:z_bound_u]
        .reshape(3, -1)
        .transpose(1, 0)
    )

    interpolated_values_x = griddata(
        points=points,
        values=values_x,
        xi=xi,
        method="linear",
        fill_value=np.nan,
        rescale=False,
    ).reshape(int(2 * R / 0.25), int(2 * R / 0.25), int(2 * R / 0.25))
    interpolated_values_y = griddata(
        points=points,
        values=values_y,
        xi=xi,
        method="linear",
        fill_value=np.nan,
        rescale=False,
    ).reshape(int(2 * R / 0.25), int(2 * R / 0.25), int(2 * R / 0.25))
    interpolated_values_z = griddata(
        points=points,
        values=values_z,
        xi=xi,
        method="linear",
        fill_value=np.nan,
        rescale=False,
    ).reshape(int(2 * R / 0.25), int(2 * R / 0.25), int(2 * R / 0.25))

    # Output the interpolated data and insert it into the lorentz force array
    Interpolated_arr = arr.copy()

    Interpolated_arr[
        0, x_bound_l:x_bound_u, y_bound_l:y_bound_u, z_bound_l:z_bound_u
    ] = interpolated_values_x[:, :, :]
    Interpolated_arr[
        1, x_bound_l:x_bound_u, y_bound_l:y_bound_u, z_bound_l:z_bound_u
    ] = interpolated_values_y[:, :, :]
    Interpolated_arr[
        2, x_bound_l:x_bound_u, y_bound_l:y_bound_u, z_bound_l:z_bound_u
    ] = interpolated_values_z[:, :, :]

    # Decay the power near the inner boundary
    ib_x, ib_y, ib_z = np.asarray(np.argwhere(r < (7.5))).T

    Interpolated_arr[:, ib_x, ib_y, ib_z] -= Interpolated_arr[:, ib_x, ib_y, ib_z] * (
        np.exp(-0.2 * r[ib_x, ib_y, ib_z])
    )
    return Interpolated_arr
