#include "../headers_cpp/raycast.h"
#include "../headers_cpp/read_file.h"
#include "../headers_cpp/magnetopause.h"
#include <iostream>


#include <chrono>

typedef std::chrono::high_resolution_clock Time;
typedef std::chrono::duration<float> fsec;


int main()
{
    auto t0 = Time::now();
    Matrix J = read_file("../data/Run1_28800/J_processed.txt");
    Matrix B = read_file("../data/Run1_28800/B_processed.txt");
    auto t1 = Time::now();
    std::cout << "File reading done. Time taken: " << fsec((t1-t0)).count() << 's' << std::endl;

    t0 = Time::now();
    Matrix J_norm = J.norm();
    t1 = Time::now();
    std::cout << "Matrix normalisation done. Time taken: " << fsec((t1-t0)).count() << 's' << std::endl;

    Point earth_pos = find_earth_pos( B );

    int nb_theta = 100;
    int nb_phi = 50;

    t0 = Time::now();
    std::array<float, 4>* interest_points = get_interest_points( 
        J_norm, earth_pos, 
        nb_theta, nb_phi, 
        0.2, 0.1, 
        0.5, 0.7, 10, 
        1.2, 1.8, 10 
    );
    t1 = Time::now();
    std::cout << "Interest point search done. Time taken: " << fsec((t1-t0)).count() << 's' << std::endl;

    t0 = Time::now();
    save_interest_points( "../data/Run1_28800/interest_points_cpp.txt", interest_points, nb_theta, nb_phi );
    t1 = Time::now();
    std::cout << "Interest point saving done. Time taken: " << fsec((t1-t0)).count() << 's' << std::endl;
}