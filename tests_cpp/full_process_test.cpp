#include "../headers_cpp/raycast.h"
#include "../headers_cpp/read_file.h"
#include "../headers_cpp/magnetopause.h"
#include "../headers_cpp/preprocessing.h"

#include <iostream>


#include <chrono>

typedef std::chrono::high_resolution_clock Time;
typedef std::chrono::duration<float> fsec;


int main(int argc, char* argv[])
{
    if (argc < 2)
    {
        std::cout << "No Run path given!\n";
        exit(1);
    }

    std::string filepath(argv[1]);


    auto t0 = Time::now();
    Matrix J = read_file(filepath + std::string("/J.txt"));
    Matrix B = read_file(filepath + std::string("/B.txt"));
    Matrix V = read_file(filepath + std::string("/V.txt"));

    Matrix X = read_file(filepath + std::string("/X.txt"));
    Matrix Y = read_file(filepath + std::string("/Y.txt"));
    Matrix Z = read_file(filepath + std::string("/Z.txt"));
    auto t1 = Time::now();
    std::cout << "File reading done. Time taken: " << fsec((t1-t0)).count() << 's' << std::endl;



    t0 = Time::now();
    Point p_min( X[0], Y[0], Z[0] );
    Point p_max( X[ X.get_shape().x-1 ], Y[ Y.get_shape().x-1 ], Z[ Z.get_shape().x-1 ] );
    Point p_range = p_max - p_min;

    float hyper_sampling = 1.5;

    Shape new_shape_real(std::round(p_range.x), std::round(p_range.y), std::round(p_range.z), V.get_shape().i);
    Shape new_shape_sim(V.get_shape().x*hyper_sampling, V.get_shape().y*hyper_sampling, V.get_shape().z*hyper_sampling, V.get_shape().i);

    Matrix B_processed_sim = orthonormalise(B, X, Y, Z, &new_shape_sim);
    Matrix J_processed_sim = orthonormalise(J, X, Y, Z, &new_shape_sim);
    Matrix V_processed_sim = orthonormalise(V, X, Y, Z, &new_shape_sim);

    Matrix B_processed_real = orthonormalise(B, X, Y, Z, &new_shape_real);
    Matrix J_processed_real = orthonormalise(J, X, Y, Z, &new_shape_real);
    Matrix V_processed_real = orthonormalise(V, X, Y, Z, &new_shape_real);
    t1 = Time::now();
//    std::cout << "Preprocessing files done. Time taken: " << fsec((t1-t0)).count() << 's' << std::endl;



    t0 = Time::now();
    Matrix J_norm_sim = J_processed_sim.norm();
    Matrix J_norm_real = J_processed_real.norm();

    Point earth_pos = find_earth_pos( B_processed_sim );

    int nb_theta = 75;
    int nb_phi = 50;

    std::array<float, 4>* interest_points = get_interest_points(
        J_norm_sim, earth_pos,
        nb_theta, nb_phi,
        0.1, 0.1,
        0.6, 0.7, 2,
        1.15, 1.8, 20
    );


    t1 = Time::now();
   std::cout << "Interest point search done. Time taken: " << fsec((t1-t0)).count() << 's' << std::endl;



    t0 = Time::now();
    save_file( filepath + std::string("/J_norm_processed_sim.txt"), J_norm_sim );
    save_file( filepath + std::string("/B_processed_sim.txt"), B_processed_sim );
    save_file( filepath + std::string("/V_processed_sim.txt"), V_processed_sim );

    save_file( filepath + std::string("/J_norm_processed_real.txt"), J_norm_real );
    save_file( filepath + std::string("/B_processed_real.txt"), B_processed_real );
    save_file( filepath + std::string("/V_processed_real.txt"), V_processed_real );

    save_interest_points( filepath + std::string("/interest_points_cpp.txt"), interest_points, nb_theta, nb_phi );
    t1 = Time::now();
    std::cout << "Interest point and file saving done. Time taken: " << fsec((t1-t0)).count() << 's' << std::endl;


    V.del(); J.del(); B.del();
    X.del(); Y.del(); Z.del();
    V_processed_sim.del(); B_processed_sim.del(); J_processed_sim.del(); J_norm_sim.del();
    V_processed_real.del(); B_processed_real.del(); J_processed_real.del(); J_norm_real.del();
}
