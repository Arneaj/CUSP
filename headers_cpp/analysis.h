#ifndef ANALYSIS_H
#define ANALYSIS_H

#include "points.h"
#include "matrix.h"
#include "raycast.h"





enum class DerivativeAccuracy { normal, high };

Point local_grad_of_normed_matrix(const Matrix& M_norm, const Point& p, DerivativeAccuracy accuracy, float dx=1.0f, float dy=1.0f, float dz=1.0f);


float get_avg_grad_of_func( double (*fn)(const double* const, double, double), const std::vector<double>& params, 
                            const Matrix& J_norm,
                            int nb_theta, int nb_phi,
                            const Point& earth_pos,
                            float dx=0.5f, float dy=0.5f, float dz=0.5f  );


float get_delta_l( float l_n, float l_s );


int get_params_at_boundaries( double* params, double* lowerbound, double* upperbound, int nb_params );


float interest_point_flatness_checker( const InterestPoint* const interest_points, int nb_theta, int nb_phi, bool* p_is_concave=nullptr, float threshold=2.0f, float phi_radius=0.3f );






#endif