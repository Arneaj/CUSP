#include "../headers_cpp/raycast.h"
#include "../headers_cpp/read_file.h"
#include "../headers_cpp/magnetopause.h"
#include "../headers_cpp/preprocessing.h"

#include <iostream>


#include <chrono>

typedef std::chrono::high_resolution_clock Time;
typedef std::chrono::duration<double> fsec;


int main(int argc, char* argv[])
{
    if (argc < 2)
    {
        std::cout << "No Run path given!\n";
        exit(1);
    }

    std::string filepath(argv[1]);


    auto t0 = Time::now();
    Matrix J_processed_sim = read_file(filepath + std::string("/J_processed_sim.txt"));
    Matrix B_processed_sim = read_file(filepath + std::string("/B_processed_sim.txt"));
    auto t1 = Time::now();
    std::cout << "File reading done. Time taken: " << fsec((t1-t0)).count() << 's' << std::endl;

    t0 = Time::now();
    Matrix J_norm_sim = J_processed_sim.norm();

    Point earth_pos = find_earth_pos( B_processed_sim );

    int nb_theta = 100;
    int nb_phi = 50;

    std::array<double, 4>* interest_points = get_interest_points(
        J_norm_sim, earth_pos,
        nb_theta, nb_phi,
        0.2, 0.1,
        0.6, 0.7, 4,
        // 1.15, 1.8, 20
        1.15, 1.8, 20
    );


    t1 = Time::now();
    std::cout << "Interest point search done. Time taken: " << fsec((t1-t0)).count() << 's' << std::endl;



    t0 = Time::now();
    save_interest_points( filepath + std::string("/interest_points_cpp.txt"), interest_points, nb_theta, nb_phi );
    t1 = Time::now();
    std::cout << "Interest point and file saving done. Time taken: " << fsec((t1-t0)).count() << 's' << std::endl;


    B_processed_sim.del(); J_processed_sim.del(); J_norm_sim.del();
}
