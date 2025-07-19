#include "../headers_cpp/fit_to_analytical.h"

#include <fstream>


int main(int argc, char* argv[])
{
    if (argc < 2)
    {
        std::cout << "No interest points path given!\n";
        exit(1);
    }

    std::string filepath(argv[1]);

    std::vector<double> theta;
    std::vector<double> phi;
    std::vector<double> radius;
    std::vector<double> weight;

    std::ifstream fs;
    fs.open(filepath);
    char* s = new char[64];

    while ( !fs.fail() )
    {
        try
        {
            fs.getline( s, 64, ',' );   theta.push_back( std::stod(s) );
            fs.getline( s, 64, ',' );   phi.push_back( std::stod(s) );
            fs.getline( s, 64, ',' );   radius.push_back( std::stod(s) );
            fs.getline( s, 64 );        weight.push_back( std::stod(s) );
        }
        catch(const std::exception& e)
        {
            break;
        }
    }

    fs.close();
    delete[] s;

    double* parameters = new double[11];
    parameters[0] = 9;      // r_0
    parameters[1] = 0.5;    // alpha_0
    parameters[2] = 0;      // alpha_1
    parameters[3] = 0;      // alpha_2
    parameters[4] = 2;      // d_n
    parameters[5] = 0.55;   // l_n
    parameters[6] = 0.55;   // s_n
    parameters[7] = 2;      // d_s
    parameters[8] = 0.55;   // l_s
    parameters[9] = 0.55;   // s_s
    parameters[10] = 0;     // e

    std::cout << "Initial values:\n";
    for (int i=0; i<11; i++) std::cout << "param[" << i << "] = " << parameters[i] << '\n';
    std::cout << std::endl;

    ceres::Problem problem;

    for (int i=0; i<theta.size(); i++) 
    {
        ceres::CostFunction* cost_function = new ceres::AutoDiffCostFunction<SphericalResidual, 1, 11>(
            new SphericalResidual(theta[i], phi[i], radius[i], weight[i])
        );
        problem.AddResidualBlock( cost_function, nullptr, parameters );
    }

    std::vector<double> lower_bounds = {5.0, 0.3, -1.0, -1.0, 0.0, 0.1, 0.1, 0.0, 0.1, 0.1, -0.5};
    std::vector<double> upper_bounds = {15.0, 0.8, 1.0, 1.0, 4.0, 2, 1.0, 4.0, 2, 1.0, 0.5};

    for (int i = 0; i < 11; i++) {
        problem.SetParameterLowerBound(parameters, i, lower_bounds[i]);
        problem.SetParameterUpperBound(parameters, i, upper_bounds[i]);
    }

    ceres::Solver::Options options;
    options.max_num_iterations = 50;
    // options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = true;

    // Try different solver strategies
    options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
    options.linear_solver_type = ceres::DENSE_QR;
    
    // // More aggressive convergence criteria
    // options.function_tolerance = 1e-12;
    // options.gradient_tolerance = 1e-12;
    // options.parameter_tolerance = 1e-12;
    
    // Allow more iterations for line search
    options.max_line_search_step_contraction = 1e-3;
    options.min_line_search_step_size = 1e-9;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    std::cout << summary.BriefReport() << "\n";


    std::cout << "Final values:\n";
    for (int i=0; i<11; i++) std::cout << "param[" << i << "] = " << parameters[i] << '\n';
    std::cout << std::endl;


    delete[] parameters;

    return 0;
}