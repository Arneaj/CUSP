#ifndef FIT_TO_ANALYTICAL_H
#define FIT_TO_ANALYTICAL_H

#include <array>

#include "matrix.h"
#include "points.h"
#include "streamlines.h"


#include <Eigen/Dense>
#include <ceres/ceres.h>




/// @brief analytical approximation of the Magnetopause topology
/// @param theta rotation around the \hat{y} axis in [0; \pi]
/// @param phi rotation around the \hat{x} axis in [-\pi; \pi)
/// @param params [r_0, alpha_0, alpha_1, alpha_2, d_n, l_n, s_n, d_s, l_s, s_s, e]
/// @return the radius at angle (theta, phi)
float Rolland25( float theta, float phi, const float* params );


class SphericalResidual 
{
private:
    const float m_theta, m_phi, m_weight, m_observed_radius;

public:
    SphericalResidual(float theta, float phi, float weight, float observed_radius) 
        : m_theta(theta), m_phi(phi), m_weight(weight), m_observed_radius(observed_radius) {;}


    bool operator()(const float* params, float* residual) const;
};








#endif
