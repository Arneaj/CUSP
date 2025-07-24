#ifndef ANALYSIS_H
#define ANALYSIS_H

#include "points.h"
#include "matrix.h"


const float PI = 3.141592653589793238462643383279502884f;


enum class DerivativeAccuracy { normal, high };

Point local_grad_of_normed_matrix(const Matrix& M_norm, const Point& p, DerivativeAccuracy accuracy, float dx, float dy, float dz);


#endif