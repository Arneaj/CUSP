#include "../headers_cpp/fit_to_analytical.h"

#include <fstream>
#include <chrono>

typedef std::chrono::high_resolution_clock Time;
typedef std::chrono::duration<float> fsec;


int main(int argc, char* argv[])
{
    if (argc < 2)
    {
        std::cout << "No interest points path given!\n";
        exit(1);
    }

    auto t0 = Time::now();

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

    int nb_interest_points = theta.size();

    InterestPoint interest_points[nb_interest_points];

    for (int i=0; i<nb_interest_points; i++)
    {
        interest_points[i].theta = theta[i];
        interest_points[i].phi = phi[i];
        interest_points[i].radius = radius[i];
        interest_points[i].weight = weight[i];
    }

    auto t1 = Time::now();
    std::cout << "Interest point reading done. Time taken: " << fsec((t1-t0)).count() << 's' << std::endl;

    // TODO: in reality, I need to transform these interest points to the real coordinate system rather than sim, but this will be done in full_process

    t0 = Time::now();

    double initial_params[11];
    initial_params[0] = 9;      // r_0
    initial_params[1] = 0.5;    // alpha_0
    initial_params[2] = 0;      // alpha_1
    initial_params[3] = 0;      // alpha_2
    initial_params[4] = 2;      // d_n
    initial_params[5] = 0.55;   // l_n
    initial_params[6] = 0.55;   // s_n
    initial_params[7] = 2;      // d_s
    initial_params[8] = 0.55;   // l_s
    initial_params[9] = 0.55;   // s_s
    initial_params[10] = 0;     // e

    double lowerbound[11] = {  5.0,    0.3,    -1.0,   -1.0,   0.0,    0.1,    0.1,    0.0,    0.1,    0.1,    -0.5};
    double upperbound[11] = {  15.0,   0.8,    1.0,    1.0,    4.0,    2.0,    1.0,    4.0,    2.0,    1.0,    0.5};
    double radii[11] =      {  3.0,    0.1,    0.5,    0.5,    1.0,    0.05,   0.25,   1.0,    0.05,   0.25,   0.15};


    int nb_runs = 10;


    OptiResult result = fit_MP<SphericalResidual, 11>( 
        interest_points, nb_interest_points, 
        initial_params, 
        lowerbound, upperbound, radii, 
        nb_runs
    );

    t1 = Time::now();
    std::cout << "Fitting with " << 11 << " initial parameters done. Time taken: " << fsec((t1-t0)).count() << 's' << std::endl;


    return 0;
}