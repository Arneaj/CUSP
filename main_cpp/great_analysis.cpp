#include "../headers_cpp/raycast.h"
#include "../headers_cpp/read_file.h"
#include "../headers_cpp/read_pvtr.h"
#include "../headers_cpp/reader_writer.h"
#include "../headers_cpp/magnetopause.h"
#include "../headers_cpp/preprocessing.h"
#include "../headers_cpp/fit_to_analytical.h"
#include "../headers_cpp/analysis.h"

#include <iostream>
#include <fstream>
#include <chrono>

typedef std::chrono::high_resolution_clock Time;
typedef std::chrono::duration<double> fsec;

#ifndef CUSTOM_PI
#define CUSTOM_PI
const double PI = 3.141592653589793238462643383279502884;
#endif


int main(int argc, char* argv[])
{
    std::string output_path(".");
    if (argc > 1) output_path = argv[1];
    std::ofstream output(output_path);

    output  << "run,timestep,"

            << "Shue97_r_0,Shue97_a_0,"
            << "Shue97_fit_loss,Shue97_grad_J_fit_over_ip,Shue97_delta_r_0,Shue97_time_taken_s,"

            << "Liu12_r_0,Liu12_a_0,Liu12_a_1,Liu12_a_2,Liu12_d_n,Liu12_l_n,Liu12_s_n,Liu12_d_s,Liu12_l_s,Liu12_s_s,"
            << "Liu12_fit_loss,Liu12_grad_J_fit_over_ip,Liu12_delta_r_0,Liu12_time_taken_s,"

            << "Rolland25_r_0,Rolland25_a_0,Rolland25_a_1,Rolland25_a_2,Rolland25_d_n,Rolland25_l_n,Rolland25_s_n,Rolland25_d_s,Rolland25_l_s,Rolland25_s_s,Rolland25_e,"
            << "Rolland25_fit_loss,Rolland25_grad_J_fit_over_ip,Rolland25_delta_r_0,Rolland25_time_taken_s,"

            << "max_theta_in_threshold, is_concave"
            << std::endl;

    std::string J_format("jvec-");
    std::string B_format("Bvec_c-");
    std::string V_format("vvec-");              // TODO: not sure how best to do this
    std::string Rho_format("rho-");
    std::string T_format("Te-");
    std::string file_format("pvtr");

    std::string base_path = "/rds/general/user/avr24/projects/swimmr-sage/live/mheyns/benchmarking/runs/Run";
    std::string sub_path = "/MS/x00_";

    std::vector<std::string> timesteps; 
    for (int t=18000; t<=28800; t+=300) timesteps.push_back( std::to_string(t) );
    int nb_timesteps = timesteps.size();

    PVTRReaderBinWriter reader_writer;

    auto t_beg = Time::now();

    for (int run=1; run<20; run++) for (const std::string& t: timesteps)
    {
        Matrix J;
        reader_writer.read(base_path + std::to_string(run) + sub_path + J_format + t + std::string(".") + file_format, J);
        Matrix Rho;
        reader_writer.read(base_path + std::to_string(run) + sub_path + Rho_format + t + std::string(".") + file_format, Rho);

        Matrix X;
        Matrix Y;
        Matrix Z;

        reader_writer.get_coordinates(base_path + std::to_string(run) + sub_path + J_format + t + std::string(".") + file_format, X, Y, Z);

        Point p_min( X[0], Y[0], Z[0] );
        Point p_max( X[ X.get_shape().x-1 ], Y[ Y.get_shape().x-1 ], Z[ Z.get_shape().x-1 ] );
        Point p_range = p_max - p_min;

        double hyper_sampling = 1.2;
        double extra_precision = 1.0;

        Shape new_shape_real(std::round(extra_precision*p_range.x), std::round(extra_precision*p_range.y), std::round(extra_precision*p_range.z), J.get_shape().i);
        Shape new_shape_sim(J.get_shape().x*hyper_sampling, J.get_shape().y*hyper_sampling, J.get_shape().z*hyper_sampling, J.get_shape().i);

        Point earth_pos_real = extra_precision * find_real_earth_pos( X, Y, Z );
        Point earth_pos_sim = find_sim_earth_pos( earth_pos_real, new_shape_real, new_shape_sim );

        Matrix J_processed_sim = orthonormalise(J, X, Y, Z, &new_shape_sim);
        Matrix J_processed_real = orthonormalise(J, X, Y, Z, &new_shape_real);
        Matrix Rho_processed_sim = orthonormalise(Rho, X, Y, Z, &new_shape_sim);

        Matrix J_norm_sim = J_processed_sim.norm();
        Matrix J_norm_real = J_processed_real.norm();

        int nb_theta = 40;
        int nb_phi = 90;
        double theta_min = 0.0f;
        double theta_max = PI*0.85f;

        double dx = 0.1f;
        double dr = 0.1f;

        double alpha_0_min = 0.3f;
        double alpha_0_max = 0.6f;
        int nb_alpha_0 = 4;

        double r_0_mult_min = 1.15f;
        double r_0_mult_max = 2.1f;
        int nb_r_0 = 20;

        double avg_std_dev;

        std::vector<Point> bow_shock = get_bowshock( Rho_processed_sim, earth_pos_sim, dr, nb_phi, nb_theta, false );

        InterestPoint* interest_points = get_interest_points(
            J_norm_sim, earth_pos_sim,
            bow_shock.data(),
            theta_min, theta_max,
            nb_theta, nb_phi, 
            dx, dr,
            alpha_0_min, alpha_0_max, nb_alpha_0,
            r_0_mult_min, r_0_mult_max, nb_r_0,
            &avg_std_dev
        );

        process_points( bow_shock, new_shape_sim, new_shape_real, earth_pos_sim, earth_pos_real );    
        process_interest_points( interest_points, nb_theta, nb_phi, new_shape_sim, new_shape_real, earth_pos_sim, earth_pos_real );
        
        output << run << ',' << t << ',';

        {   // Shue97
        //     index                    0       1       
        //     param                    r_0,    a_0        
        double initial_params[2] = {   10.0,   0.5 };
        double lowerbound[2] =     {   5.0,    0.2 };
        double upperbound[2] =     {   15.0,   0.8 };
        double radii[2] =          {   3.0,    0.1 };

        int nb_runs = 50;
        int nb_interest_points = nb_theta * nb_phi;
        const int nb_params_shue = 2;

        auto t0 = Time::now();
        OptiResult result = fit_MP<Shue97Residual, nb_params_shue>( 
            interest_points, nb_interest_points, 
            initial_params, 
            lowerbound, upperbound, radii, 
            nb_runs
        );
        auto t1 = Time::now();

        double grad_J_fit_over_ip = get_grad_J_fit_over_interest_points( Shue97, result.params, interest_points, nb_interest_points, J_norm_real, earth_pos_real );
        double delta_r_0 = get_delta_r_0(result.params[0], interest_points, nb_theta, nb_theta);

        for (int i=0; i<nb_params_shue; i++) output << result.params[i] << ',';
        output  << result.cost / nb_interest_points << ','
                << grad_J_fit_over_ip << ','
                << delta_r_0 << ','
                << fsec((t1-t0)).count() << ',';
        }
        
        {   // Liu12
        //     index                    0       1       2       3       4       5       6       7       8       9
        //     param                    r_0,    a_0,    a_1,    a_2,    d_n,    l_n,    s_n,    d_s,    l_s,    s_s
        double initial_params[10] = {   10.0,   0.5,    0.0,    0.0,    3.0,    0.55,   5.0,    3.0,    0.55,   5.0     };
        double lowerbound[10] =     {   5.0,    0.2,    -1.0,   -1.0,   0.0,    0.1,    0.1,    0.0,    0.1,    0.1     };
        double upperbound[10] =     {   15.0,   0.8,    1.0,    1.0,    6.0,    2.0,    10.0,   6.0,    2.0,    10.0    };
        double radii[10] =          {   3.0,    0.1,    0.5,    0.5,    1.0,    0.05,   1.5,    1.0,    0.05,   1.5     };

        int nb_runs = 50;
        int nb_interest_points = nb_theta * nb_phi;
        const int nb_params_liu = 10;

        auto t0 = Time::now();
        OptiResult result = fit_MP<Liu12Residual, nb_params_liu>( 
            interest_points, nb_interest_points, 
            initial_params, 
            lowerbound, upperbound, radii, 
            nb_runs
        );
        auto t1 = Time::now();

        double grad_J_fit_over_ip = get_grad_J_fit_over_interest_points( Liu12, result.params, interest_points, nb_interest_points, J_norm_real, earth_pos_real );
        double delta_r_0 = get_delta_r_0(result.params[0], interest_points, nb_theta, nb_theta);

        for (int i=0; i<nb_params_liu; i++) output << result.params[i] << ',';
        output  << result.cost / nb_interest_points << ','
                << grad_J_fit_over_ip << ','
                << delta_r_0 << ','
                << fsec((t1-t0)).count() << ',';
        }
        
        {   // Rolland25
        //     index                    0       1       2       3       4       5       6       7       8       9       10
        //     param                    r_0,    a_0,    a_1,    a_2,    d_n,    l_n,    s_n,    d_s,    l_s,    s_s,    e    
        double initial_params[11] = {   10.0,   0.5,    0.0,    0.0,    3.0,    0.55,   5.0,    3.0,    0.55,   5.0,    0.0     };
        double lowerbound[11] =     {   5.0,    0.2,    -1.0,   -1.0,   0.0,    0.1,    0.1,    0.0,    0.1,    0.1,    -0.8    };
        double upperbound[11] =     {   15.0,   0.8,    1.0,    1.0,    6.0,    2.0,    10.0,   6.0,    2.0,    10.0,   0.8     };
        double radii[11] =          {   3.0,    0.1,    0.5,    0.5,    1.0,    0.05,   1.5,    1.0,    0.05,   1.5,    0.2     };

        int nb_runs = 50;
        int nb_interest_points = nb_theta * nb_phi;
        const int nb_params_rolland = 11;

        auto t0 = Time::now();
        OptiResult result = fit_MP<EllipsisPolyResidual, nb_params_rolland>( 
            interest_points, nb_interest_points, 
            initial_params, 
            lowerbound, upperbound, radii, 
            nb_runs
        );
        auto t1 = Time::now();

        double grad_J_fit_over_ip = get_grad_J_fit_over_interest_points( EllipsisPoly, result.params, interest_points, nb_interest_points, J_norm_real, earth_pos_real );
        double delta_r_0 = get_delta_r_0(result.params[0], interest_points, nb_theta, nb_theta);

        for (int i=0; i<nb_params_rolland; i++) output << result.params[i] << ',';
        output  << result.cost / nb_interest_points << ','
                << grad_J_fit_over_ip << ','
                << delta_r_0 << ','
                << fsec((t1-t0)).count() << ',';
        }

        const double threshold = 2.0f;
        bool is_concave;

        double max_theta_in_threshold = interest_point_flatness_checker( interest_points, nb_theta, nb_phi, &is_concave, threshold );

        output  << max_theta_in_threshold << ','
                << 1 * is_concave << std::endl;

        J.del();
        Rho.del();

        X.del(); Y.del(); Z.del();

        J_processed_sim.del(); 
        J_norm_sim.del();

        J_processed_real.del(); 
        J_norm_real.del(); 
        
        Rho_processed_sim.del();

        delete[] interest_points;
    }

    auto t_end = Time::now();
    std::cout   << "\nFinished! Overall average time per timestep: " 
                << fsec((t_end-t_beg)).count() / (19.0 * nb_timesteps) << 's' 
                << std::endl;
}
