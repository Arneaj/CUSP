#ifndef ANALYSIS_H
#define ANALYSIS_H

#include "points.h"
#include "matrix.h"





enum class DerivativeAccuracy { normal, high };

Point local_grad_of_normed_matrix(const Matrix& M_norm, const Point& p, DerivativeAccuracy accuracy, float dx=1.0f, float dy=1.0f, float dz=1.0f);


float get_avg_grad_of_func( double (*fn)(const double* const, double, double), const std::vector<double>& params, 
                            const Matrix& J_norm,
                            int nb_theta, int nb_phi,
                            const Point& earth_pos,
                            float dx=0.5f, float dy=0.5f, float dz=0.5f  );


#endif