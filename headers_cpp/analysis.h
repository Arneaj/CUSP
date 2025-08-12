#ifndef ANALYSIS_H
#define ANALYSIS_H

#include "points.h"
#include "matrix.h"
#include "raycast.h"





enum class DerivativeAccuracy { normal, high };

Point local_grad_of_normed_matrix(const Matrix& M_norm, const Point& p, DerivativeAccuracy accuracy, double dx=1.0, double dy=1.0, double dz=1.0);


double get_grad_J_fit_over_interest_points( double (*fn)(const double* const, double, double), const std::vector<double>& params, 
                            const InterestPoint* const interest_points, int nb_interest_points,
                            const Matrix& J_norm,
                            const Point& earth_pos,
                            double dx=0.5, double dy=0.5, double dz=0.5  );


/// @brief returns l_n + l_s
double get_delta_l( double l_n, double l_s );


/// @brief returns `r_0 - avg_r` of the interest points where `theta < theta_used`
/// @param r_0 
/// @param interest_points 
/// @param nb_theta 
/// @param nb_phi 
/// @param theta_used 
double get_delta_r_0( double r_0, const InterestPoint* const interest_points, int nb_theta, int nb_phi, double theta_used=0.2 );


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
double interest_point_flatness_checker( const InterestPoint* const interest_points, int nb_theta, int nb_phi, bool* p_is_concave=nullptr, double threshold=2.0, double phi_radius=0.3 );


void save_analysis_csv( std::string filepath, 
                        const std::vector<double>& inputs, const std::vector<std::string>& inputs_names,
                        const std::vector<double>& params, const std::vector<std::string>& params_names, 
                        const std::vector<double>& metrics, const std::vector<std::string>& metrics_names );



#endif