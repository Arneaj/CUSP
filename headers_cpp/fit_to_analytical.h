#ifndef FIT_TO_ANALYTICAL_H
#define FIT_TO_ANALYTICAL_H

#include <array>

#include "matrix.h"
#include "points.h"
#include "streamlines.h"


#include <Eigen/Dense>
#include <ceres/ceres.h>


const float PI = 3.1415926535;



template <typename T>
T is_pos( T v ) { return T( v>=T(0) ); }


template <typename T>
T abs_approx( T v ) { return v*(v>=T(0)) - v*(v<T(0)); }


/// @brief analytical approximation of the Magnetopause topology
/// @param theta rotation around the \hat{y} axis in [0; \pi]
/// @param phi rotation around the \hat{x} axis in [-\pi; \pi)
/// @param params [r_0, alpha_0, alpha_1, alpha_2, d_n, l_n, s_n, d_s, l_s, s_s, e]
/// @return the radius at angle (theta, phi)
template <typename T>
T Rolland25( const T* const params, T theta, T phi )
{
    // if (theta<0 || theta>PI) { std::cout << "theta should be in [0; pi]\n"; exit(1); }
    // if (phi<-PI || phi>PI) { std::cout << "phi should be in [-pi; pi)\n"; exit(1); }

    T cos_theta = ceres::cos(theta);
    T cos_phi = ceres::cos(phi);

    return params[0] * (
        (T(1.0)+params[10]) / (T(1.0)+params[10]*cos_theta)
    ) * ceres::pow(
        T(2.0) / (T(1.0)+cos_theta), 
        params[1] + params[2]*cos_phi + params[3]*cos_phi*cos_phi
    ) - (
        params[4] * ceres::exp( -ceres::abs(theta - params[5]) / params[6] ) * is_pos<T>( cos_phi ) +
        params[7] * ceres::exp( -ceres::abs(theta - params[8]) / params[9] ) * is_pos<T>( -cos_phi )
    ) * cos_phi*cos_phi;
}


class SphericalResidual 
{
private:
    const double m_theta, m_phi, m_weight, m_observed_radius;

public:
    SphericalResidual(double theta, double phi, double weight, double observed_radius) 
        : m_theta(theta), m_phi(phi), m_weight(weight), m_observed_radius(observed_radius) {;}

    template <typename T>
    bool operator()(const T* const params, T* residual) const 
    {
        T predicted_radius = Rolland25(params, T(m_theta), T(m_phi));
        residual[0] = (m_observed_radius - predicted_radius)*m_weight;
        return true;
    }
};








#endif
