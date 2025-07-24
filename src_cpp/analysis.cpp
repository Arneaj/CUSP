#include "../headers_cpp/analysis.h"



Point local_grad_of_normed_matrix(const Matrix& M_norm, const Point& p, DerivativeAccuracy accuracy, float dx, float dy, float dz)
{
    if (M_norm.get_shape().i > 1) { std::cout << "ERROR: please provide a matrix of component dim shape.i = 1\n"; exit(1); }

    Point p_dx = Point(dx, 0.0f, 0.0f);
    Point p_dy = Point(0.0f, dy, 0.0f);
    Point p_dz = Point(0.0f, 0.0f, dz);

    if (accuracy == DerivativeAccuracy::normal)
    {
        float grad_x = ( M_norm(p + p_dx, 0) - M_norm(p - p_dx, 0) ) / (2.0f*dx);
        float grad_y = ( M_norm(p + p_dy, 0) - M_norm(p - p_dy, 0) ) / (2.0f*dy);
        float grad_z = ( M_norm(p + p_dz, 0) - M_norm(p - p_dz, 0) ) / (2.0f*dz);

        return Point( grad_x, grad_y, grad_z );
    }

    float grad_x = ( -M_norm(p + 2.0f*p_dx, 0) + 8.0f*M_norm(p + p_dx, 0) - 8.0f*M_norm(p - p_dx, 0) + M_norm(p - 2.0f*p_dx, 0) ) / (12.0f*dx);
    float grad_y = ( -M_norm(p + 2.0f*p_dy, 0) + 8.0f*M_norm(p + p_dy, 0) - 8.0f*M_norm(p - p_dy, 0) + M_norm(p - 2.0f*p_dy, 0) ) / (12.0f*dy);
    float grad_z = ( -M_norm(p + 2.0f*p_dz, 0) + 8.0f*M_norm(p + p_dz, 0) - 8.0f*M_norm(p - p_dz, 0) + M_norm(p - 2.0f*p_dz, 0) ) / (12.0f*dz);

    return Point( grad_x, grad_y, grad_z );
}


float get_avg_grad_of_func( double (*fn)(const double* const, double, double), const double* const params, 
                            const Matrix& J_norm,
                            int nb_params, int nb_theta, int nb_phi,
                            const Point& earth_pos,
                            float dx, float dy, float dz  )
{
    float total_grad_norm = 0.0f;

    float dtheta = PI / nb_theta;
    float dphi = 2.0f*PI / nb_phi;

    for (float theta=0.0f; theta<PI; theta+=dtheta)
    {
        float cos_theta = std::cos(theta);
        float sin_theta = std::sin(theta);

        for (float phi=-PI; phi<PI; phi+=dphi)
        {
            float radius = fn(params, theta, phi);

            Point proj = Point(
                -cos_theta,
                sin_theta*std::sin(phi),
                sin_theta*std::cos(phi)
            );

            Point p = radius * proj + earth_pos;

            Point grad_J = local_grad_of_normed_matrix(J_norm, p, DerivativeAccuracy::high, dx, dy, dz);
            
            total_grad_norm += grad_J.norm();
        }
    }

    return total_grad_norm / (nb_phi*nb_theta);
}




