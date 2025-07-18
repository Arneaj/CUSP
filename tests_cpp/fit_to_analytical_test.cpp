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

    std::vector<float> theta;
    std::vector<float> phi;
    std::vector<float> radius;
    std::vector<float> weight;

    std::ifstream fs;
    fs.open(filepath);
    char* s = new char[64];

    while (!fs.eof)
    {
        fs.getline( s, 64, ',' );   theta = std::stof(s);
        fs.getline( s, 64, ',' );   phi = std::stof(s);
        fs.getline( s, 64, ',' );   radius = std::stof(s);
        fs.getline( s, 64 );        weight = std::stof(s);
    }

    fs.close();
    delete[] s;

    float* parameters = new float[11];
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

    ceres::Solver::Options options;
    options.max_num_iterations = 100;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = true;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    std::cout << summary.BriefReport() << "\n";


    std::cout << "Final values:\n";
    for (int i=0; i<11; i++) std::cout << "param[" << i << "] = " << parameters[i] << '\n';
    std::cout << std::endl;


    return 0;
}