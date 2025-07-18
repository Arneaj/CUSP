#include "../headers_cpp/fit_to_analytical.h"


const float PI = 3.1415926535;






/// @brief analytical approximation of the Magnetopause topology
/// @param theta rotation around the \hat{y} axis in [0; \pi]
/// @param phi rotation around the \hat{x} axis in [-\pi; \pi)
/// @param params [r_0, alpha_0, alpha_1, alpha_2, d_n, l_n, s_n, d_s, l_s, s_s, e]
/// @return the radius at angle (theta, phi)
float Rolland25( float theta, float phi, const float* params )
{
    if (theta<0 || theta>PI) { std::cout << "theta should be in [0; \pi]\n"; exit(1); }
    if (phi<PI || phi>PI) { std::cout << "phi should be in [-\pi; \pi)\n"; exit(1); }

    float cos_theta = std::cos(theta);
    float cos_phi = std::cos(phi);

    return params[0] * (
        (1.0+params[10]) / (1.0+params[10]*cos_theta)
    ) * std::pow(
        2.0 / (1.0+cos_theta), 
        params[1] + params[2]*cos_phi + params[3]*cos_phi*cos_phi
    ) - (
        params[4] * std::exp( -std::abs(theta - params[5]) / params[6] ) * (1.0+std::sign(cos_phi))*0.5 +
        params[7] * std::exp( -std::abs(theta - params[8]) / params[9] ) * (1.0-std::sign(cos_phi))*0.5
    ) * cos_phi*cos_phi;
}



bool SphericalResidual::operator()(const float* params, float* residual) const 
{
    float predicted_radius = Rolland25(params, m_theta, m_phi);
    residual[0] = (m_observed_radius - predicted_radius)*m_weight;
    return true;
}








