#include "../headers_cpp/raycast.h"
#include "../headers_cpp/reader_writer.h"
#include "../headers_cpp/preprocessing.h"
#include "../headers_cpp/magnetopause.h"
#include <iostream>


#include <chrono>

const float PI = 3.141592653589793238462643383279502884f;

typedef std::chrono::high_resolution_clock Time;
typedef std::chrono::duration<float> fsec;


int main()
{
    auto t0 = Time::now();

    PVTRReaderBinWriter reader_writer;   
    std::string filepath("../test_data"); 

    Matrix J;
    reader_writer.read(filepath + std::string("/x00_jvec-21000.pvtr"), J);

    Matrix X;
    Matrix Y;
    Matrix Z;

    reader_writer.get_coordinates(filepath + std::string("/x00_jvec-21000.pvtr"), X, Y, Z);
    
    auto t1 = Time::now();
    std::cout << "File reading done. Time taken: " << fsec((t1-t0)).count() << 's' << std::endl;

    t0 = Time::now();

    Point p_min( X[0], Y[0], Z[0] );
    Point p_max( X[ X.get_shape().x-1 ], Y[ Y.get_shape().x-1 ], Z[ Z.get_shape().x-1 ] );
    Point p_range = p_max - p_min;

    float hyper_sampling = 1.2;

    Shape new_shape_real(std::round(p_range.x), std::round(p_range.y), std::round(p_range.z), J.get_shape().i);
    Shape new_shape_sim(J.get_shape().x*hyper_sampling, J.get_shape().y*hyper_sampling, J.get_shape().z*hyper_sampling, J.get_shape().i);

    Point earth_pos_real = find_real_earth_pos( X, Y, Z );
    Point earth_pos_sim = find_sim_earth_pos( earth_pos_real, new_shape_real, new_shape_sim );

    Matrix J_processed_sim = orthonormalise(J, X, Y, Z, &new_shape_sim);
    Matrix J_norm_sim = J_processed_sim.norm();
    
    t1 = Time::now();
    std::cout << "Preprocessing files done. Time taken: " << fsec((t1-t0)).count() << 's' << std::endl;


    t0 = Time::now();

    int nb_theta = 40;
    int nb_phi = 90;
    float theta_min = 0.0f;
    float theta_max = PI*0.85f;

    float dx = 0.1f;
    float dr = 0.1f;

    float alpha_0_min = 0.3f;
    float alpha_0_max = 0.6f;
    int nb_alpha_0 = 4;

    float r_0_mult_min = 1.15f;
    float r_0_mult_max = 2.1f;
    int nb_r_0 = 20;

    float avg_std_dev;

    InterestPoint* interest_points = get_interest_points(
        J_norm_sim, earth_pos_sim,
        theta_min, theta_max,
        nb_theta, nb_phi, 
        dx, dr,
        alpha_0_min, alpha_0_max, nb_alpha_0,
        r_0_mult_min, r_0_mult_max, nb_r_0,
        &avg_std_dev
    );

    t1 = Time::now();
    std::cout << "Interest point search done. Time taken: " << fsec((t1-t0)).count() << 's' << std::endl;

    t0 = Time::now();
    save_interest_points( filepath + std::string("/interest_points_cpp.csv"), interest_points, nb_theta, nb_phi );
    t1 = Time::now();
    std::cout << "Interest point saving done. Time taken: " << fsec((t1-t0)).count() << 's' << std::endl;

    J.del(); J_processed_sim.del(); J_norm_sim.del();
    X.del(); Y.del(); Z.del();
}