#ifndef ANALYSIS_H
#define ANALYSIS_H

#include "points.h"
#include "matrix.h"
#include "raycast.h"





enum class DerivativeAccuracy { normal, high };

Point local_grad_of_normed_matrix(const Matrix& M_norm, const Point& p, DerivativeAccuracy accuracy, float dx=1.0f, float dy=1.0f, float dz=1.0f);


float get_grad_J_fit_over_interest_points( double (*fn)(const double* const, double, double), const std::vector<double>& params, 
                            const InterestPoint* const interest_points, int nb_interest_points,
                            const Matrix& J_norm,
                            const Point& earth_pos,
                            float dx=0.5f, float dy=0.5f, float dz=0.5f  );


/// @brief returns l_n + l_s
float get_delta_l( float l_n, float l_s );

/// @brief returns the number of parameters in params that have reached their lower or upper bounds after fitting
/// @param params parameters after fitting with fit_MP in fit_to_analytical.h
int get_params_at_boundaries( double* params, double* lowerbound, double* upperbound, int nb_params );

/// @brief analysis function giving information on the value of `theta` after which the interest points reach a  
///                 `P.x < max(P.x) - threshold` where `max(P.x)` is the maximum x value of all the interest points 
///                 (over a certain weight of 0.7). This is done for the interest points in and around the `(x,z)`
///                 plane, which contains the cusps.
/// @param interest_points 
/// @param nb_theta 
/// @param nb_phi 
/// @param p_is_concave optional parameter which will be true if the `P.x` values around `theta=0` are lower than `max(P.x)`
/// @param threshold 
/// @param phi_radius the `phi` value inside which the interest points will be considered for the check
/// @return 
float interest_point_flatness_checker( const InterestPoint* const interest_points, int nb_theta, int nb_phi, bool* p_is_concave=nullptr, float threshold=2.0f, float phi_radius=0.3f );






#endif