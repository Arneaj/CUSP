#include "../headers_cpp/analysis.h"



Point local_grad_of_normed_matrix(Matrix M_norm, Point p, DerivativeAccuracy accuracy, float dx, float dy, float dz)
{
    if (M_norm.get_shape().i > 1) { std::cout << "ERROR: please provide a matrix of component dim shape.i = 1\n"; exit(1); }

    Point p_dx = Point(dx, 0.0, 0.0);
    Point p_dy = Point(0.0, dy, 0.0);
    Point p_dz = Point(0.0, 0.0, dz);

    if (accuracy == DerivativeAccuracy::normal)
    {
        float grad_x = ( M_norm(p + p_dx, 0) - M_norm(p - p_dx, 0) ) / (2.0*dx);
        float grad_y = ( M_norm(p + p_dy, 0) - M_norm(p - p_dy, 0) ) / (2.0*dy);
        float grad_z = ( M_norm(p + p_dz, 0) - M_norm(p - p_dz, 0) ) / (2.0*dz);

        return Point( grad_x, grad_y, grad_z );
    }

    float grad_x = ( -M_norm(p + 2.0*p_dx, 0) + 8.0*M_norm(p + p_dx, 0) - 8.0*M_norm(p - p_dx, 0) + M_norm(p - 2.0*p_dx, 0) ) / (12.0*dx);
    float grad_y = ( -M_norm(p + 2.0*p_dy, 0) + 8.0*M_norm(p + p_dy, 0) - 8.0*M_norm(p - p_dy, 0) + M_norm(p - 2.0*p_dy, 0) ) / (12.0*dy);
    float grad_z = ( -M_norm(p + 2.0*p_dz, 0) + 8.0*M_norm(p + p_dz, 0) - 8.0*M_norm(p - p_dz, 0) + M_norm(p - 2.0*p_dz, 0) ) / (12.0*dz);

    return Point( grad_x, grad_y, grad_z );
}






