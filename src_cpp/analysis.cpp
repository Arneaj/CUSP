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





float get_grad_J_fit_over_interest_points( double (*fn)(const double* const, double, double), const std::vector<double>& params, 
                            const InterestPoint* const interest_points, int nb_interest_points,
                            const Matrix& J_norm,
                            const Point& earth_pos,
                            float dx, float dy, float dz )
{
    float total_grad_norm = 0.0f;
    float total_grad_norm_ip = 0.0f;

    Point dp(dx, dy, dz);

    for (int i=0; i<nb_interest_points; i++)
    {
        const InterestPoint& ip = interest_points[i];

        float sin_theta = std::sin(ip.theta);
        float radius = fn(params.data(), ip.theta, ip.phi);
        Point proj = Point(
            -std::cos(ip.theta),
            sin_theta*std::sin(ip.phi),
            sin_theta*std::cos(ip.phi)
        );

        Point p_ip = ip.radius * proj + earth_pos;
        Point p = radius * proj + earth_pos;

        if ( J_norm.is_point_OOB(p+2.0f*dp) || J_norm.is_point_OOB(p-2.0f*dp) ||
             J_norm.is_point_OOB(p_ip+2.0f*dp) || J_norm.is_point_OOB(p_ip-2.0f*dp) ) continue;

        

        Point grad_J_ip = local_grad_of_normed_matrix(J_norm, p_ip, DerivativeAccuracy::high, dx, dy, dz);
        Point grad_J = local_grad_of_normed_matrix(J_norm, p, DerivativeAccuracy::high, dx, dy, dz);

        total_grad_norm_ip += grad_J_ip.norm() * ip.weight;
        total_grad_norm += grad_J.norm() * ip.weight;
    }

    return total_grad_norm / total_grad_norm_ip;
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
    float avg_X[nb_theta] = {0.0f};
    float max_X = 0.0f;

    for (int itheta=0; itheta<nb_theta; itheta++) 
    {
        // avg_X[itheta] = 0.0f;
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
        if ( avg_X[itheta] > max_X - threshold ) max_theta_in_threshold = interest_points[itheta*nb_phi].theta;

    return max_theta_in_threshold;
}





void save_analysis_csv( std::string filepath, 
                        const std::vector<float>& inputs, const std::vector<std::string>& inputs_names,
                        const std::vector<double>& params, const std::vector<std::string>& params_names, 
                        const std::vector<float>& metrics, const std::vector<std::string>& metrics_names )
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

    for (float input: inputs) 
    {
        if (input == inputs.front()) fs << input;
        else fs << ',' << input;
    }
    for (double param: params) fs << ',' << param;
    for (float metric: metrics) fs << ',' << metric;

    fs.close();
}

