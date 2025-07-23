#ifndef ANALYSIS_H
#define ANALYSIS_H

#include "points.h"
#include "matrix.h"


enum class DerivativeAccuracy { normal, high };

Point local_grad_of_normed_matrix(Matrix M_norm, Point p, DerivativeAccuracy accuracy=DerivativeAccuracy::normal, float dx=1.0, float dy=1.0, float dz=1.0);



#endif