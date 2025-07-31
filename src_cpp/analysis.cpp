#include "../headers_cpp/analysis.h"


const float PI = 3.141592653589793238462643383279502884f;


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


float get_avg_grad_of_func( double (*fn)(const double* const, double, double), const std::vector<double>& params, 
                            const Matrix& J_norm,
                            int nb_theta, int nb_phi,
                            const Point& earth_pos,
                            float dx, float dy, float dz  )
{
    float total_grad_norm = 0.0f;

    float dtheta = PI / nb_theta;
    float dphi = 2.0f*PI / nb_phi;

    Point dp = Point(dx, dy, dz);

    int valid_points = 0;

    for (float theta=0.0f; theta<PI; theta+=dtheta)
    {
        float cos_theta = std::cos(theta);
        float sin_theta = std::sin(theta);

        for (float phi=-PI; phi<PI; phi+=dphi)
        {
            float radius = fn(params.data(), theta, phi);

            Point proj = Point(
                -cos_theta,
                sin_theta*std::sin(phi),
                sin_theta*std::cos(phi)
            );

            Point p = radius * proj + earth_pos;


            if ( J_norm.is_point_OOB(p + 2.0f*dp) ) continue;
            if ( J_norm.is_point_OOB(p - 2.0f*dp) ) continue;

            Point grad_J = local_grad_of_normed_matrix(J_norm, p, DerivativeAccuracy::high, dx, dy, dz);
            
            // std::cout << "grad norm: " << grad_J.norm() << std::endl;
            total_grad_norm += grad_J.norm();
            valid_points++;
        }
    }

    // std::cout << "total grad norm: " << total_grad_norm << std::endl;
    // std::cout << "avg grad norm: " << total_grad_norm / valid_points << std::endl;
    return total_grad_norm / valid_points;
}




float get_delta_l( float l_n, float l_s ) { return l_n + l_s; }



int get_params_at_boundaries( double* params, double* lowerbound, double* upperbound, int nb_params )
{
    int count = 0;

    for (int i=0; i<nb_params; i++) 
        if (
            std::abs(params[i]-lowerbound[i]) < 1e-3 || 
            std::abs(params[i]-upperbound[i]) < 1e-3 
        )
            count++;

    return count;
}



float interest_point_flatness_checker( const InterestPoint* const interest_points, int nb_theta, int nb_phi, bool* p_is_concave, float threshold, float phi_radius )
{
    float avg_X[nb_theta];
    float max_X = 0.0f;

    for (int itheta=0; itheta<nb_theta; itheta++) 
    {
        avg_X[itheta] = 0.0f;
        float sum_weights = 0.0f;

        for (int iphi=0; iphi<nb_phi; iphi++) 
        {
            const InterestPoint& ip = interest_points[itheta*nb_phi + iphi];

            if ( 
                std::abs(ip.phi) < phi_radius ||
                PI + ip.phi < phi_radius ||
                PI - ip.phi < phi_radius
            ) continue;

            sum_weights += ip.weight;
            avg_X[itheta] += ip.radius * std::cos(ip.theta) * ip.weight;
        }
        
        avg_X[itheta] /= sum_weights;
        if (sum_weights/nb_theta > 0.7 && avg_X[itheta] > max_X) max_X = avg_X[itheta];
    }

    float max_theta_in_threshold = 0.0f;

    if (p_is_concave) *p_is_concave = (avg_X[0] < max_X) && (avg_X[1] < max_X) && (avg_X[2] < max_X);

    for (int itheta=0; itheta<nb_theta; itheta++) 
        if ( max_X - avg_X[itheta] < threshold ) max_theta_in_threshold = interest_points[itheta*nb_phi].theta;

    return max_theta_in_threshold;
}





