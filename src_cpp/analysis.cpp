#include "../headers_cpp/analysis.h"


#ifndef CUSTOM_PI
#define CUSTOM_PI
const double PI = 3.141592653589793238462643383279502884;
#endif


Point local_grad_of_normed_matrix(const Matrix& M_norm, const Point& p, DerivativeAccuracy accuracy, double dx, double dy, double dz)
{
    if (M_norm.get_shape().i > 1) { std::cout << "ERROR: please provide a matrix of component dim shape.i = 1\n"; exit(1); }

    Point p_dx = Point(dx, 0.0, 0.0);
    Point p_dy = Point(0.0, dy, 0.0);
    Point p_dz = Point(0.0, 0.0, dz);

    if (accuracy == DerivativeAccuracy::normal)
    {
        double grad_x = ( M_norm(p + p_dx, 0) - M_norm(p - p_dx, 0) ) / (2.0*dx);
        double grad_y = ( M_norm(p + p_dy, 0) - M_norm(p - p_dy, 0) ) / (2.0*dy);
        double grad_z = ( M_norm(p + p_dz, 0) - M_norm(p - p_dz, 0) ) / (2.0*dz);

        return Point( grad_x, grad_y, grad_z );
    }

    double grad_x = ( -M_norm(p + 2.0*p_dx, 0) + 8.0*M_norm(p + p_dx, 0) - 8.0*M_norm(p - p_dx, 0) + M_norm(p - 2.0*p_dx, 0) ) / (12.0*dx);
    double grad_y = ( -M_norm(p + 2.0*p_dy, 0) + 8.0*M_norm(p + p_dy, 0) - 8.0*M_norm(p - p_dy, 0) + M_norm(p - 2.0*p_dy, 0) ) / (12.0*dy);
    double grad_z = ( -M_norm(p + 2.0*p_dz, 0) + 8.0*M_norm(p + p_dz, 0) - 8.0*M_norm(p - p_dz, 0) + M_norm(p - 2.0*p_dz, 0) ) / (12.0*dz);

    return Point( grad_x, grad_y, grad_z );
}





double get_grad_J_fit_over_interest_points( double (*fn)(const double* const, double, double), const std::vector<double>& params, 
                            const InterestPoint* const interest_points, int nb_interest_points,
                            const Matrix& J_norm,
                            const Point& earth_pos,
                            double dx, double dy, double dz )
{
    double total_grad_norm = 0.0;
    double total_grad_norm_ip = 0.0;

    Point dp(dx, dy, dz);

    for (int i=0; i<nb_interest_points; i++)
    {
        const InterestPoint& ip = interest_points[i];

        double sin_theta = std::sin(ip.theta);
        double radius = fn(params.data(), ip.theta, ip.phi);
        Point proj = Point(
            -std::cos(ip.theta),
            sin_theta*std::sin(ip.phi),
            sin_theta*std::cos(ip.phi)
        );

        Point p_ip = ip.radius * proj + earth_pos;
        Point p = radius * proj + earth_pos;

        if ( J_norm.is_point_OOB(p+2.0*dp) || J_norm.is_point_OOB(p-2.0*dp) ||
             J_norm.is_point_OOB(p_ip+2.0*dp) || J_norm.is_point_OOB(p_ip-2.0*dp) ) continue;

        

        Point grad_J_ip = local_grad_of_normed_matrix(J_norm, p_ip, DerivativeAccuracy::high, dx, dy, dz);
        Point grad_J = local_grad_of_normed_matrix(J_norm, p, DerivativeAccuracy::high, dx, dy, dz);

        total_grad_norm_ip += grad_J_ip.norm() * ip.weight;
        total_grad_norm += grad_J.norm() * ip.weight;
    }

    return total_grad_norm / total_grad_norm_ip;
}




double get_delta_l( double l_n, double l_s ) { return l_n + l_s; }



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



double interest_point_flatness_checker( const InterestPoint* const interest_points, int nb_theta, int nb_phi, bool* p_is_concave, double threshold, double phi_radius )
{
    double avg_X[nb_theta] = {0.0};
    double max_X = 0.0;

    for (int itheta=0; itheta<nb_theta; itheta++) 
    {
        // avg_X[itheta] = 0.0;
        double sum_weights = 0.0;

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

    double max_theta_in_threshold = 0.0;

    if (p_is_concave) *p_is_concave = (avg_X[0] < max_X) && (avg_X[1] < max_X) && (avg_X[2] < max_X);

    for (int itheta=0; itheta<nb_theta; itheta++) 
        if ( avg_X[itheta] > max_X - threshold ) max_theta_in_threshold = interest_points[itheta*nb_phi].theta;

    return max_theta_in_threshold;
}



double get_delta_r_0( double r_0, const InterestPoint* const interest_points, int nb_theta, int nb_phi, double theta_used )
{
    double avg_r = 0.0;
    double sum_weights = 0.0;

    for (int iphi=0; iphi<nb_phi; iphi++) for (int itheta=0; itheta<nb_theta; itheta++)
    {
        const InterestPoint& ip = interest_points[itheta*nb_phi + iphi];

        if ( ip.theta > theta_used ) break;

        sum_weights += ip.weight;
        avg_r += ip.radius * ip.weight;
    }

    avg_r /= sum_weights;

    return r_0 - avg_r;
}




void save_analysis_csv( std::string filepath, 
                        const std::vector<double>& inputs, const std::vector<std::string>& inputs_names,
                        const std::vector<double>& params, const std::vector<std::string>& params_names, 
                        const std::vector<double>& metrics, const std::vector<std::string>& metrics_names )
{
    std::ofstream fs;
    fs.open(filepath);

    for (const std::string& input_name: inputs_names) 
    {
        if (input_name == inputs_names.front()) fs << input_name;
        else fs << ',' << input_name;
    }
    for (const std::string& param_name: params_names) fs << ',' << param_name;
    for (const std::string& metric_name: metrics_names) fs << ',' << metric_name;

    fs << '\n';

    for (double input: inputs) 
    {
        if (input == inputs.front()) fs << input;
        else fs << ',' << input;
    }
    for (double param: params) fs << ',' << param;
    for (double metric: metrics) fs << ',' << metric;

    fs.close();
}

