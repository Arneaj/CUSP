#include "../headers_cpp/raycast.h"
#include "../headers_cpp/read_file.h"
#include "../headers_cpp/read_pvtr.h"
#include "../headers_cpp/magnetopause.h"
#include "../headers_cpp/preprocessing.h"
#include "../headers_cpp/fit_to_analytical.h"
#include "../headers_cpp/analysis.h"

#include <iostream>


#include <chrono>

typedef std::chrono::high_resolution_clock Time;
typedef std::chrono::duration<float> fsec;


int main(int argc, char* argv[])
{
    // if (argc < 2)
    // {
    //     std::cout << "No Run path given!\n";
    //     exit(1);
    // }

    // if (argc < 3)
    // {
    //     std::cout << "No timestep given!\n";
    //     exit(1);
    // }

    // if (argc < 4)
    // {
    //     std::cout << "No save path given!\n";
    //     exit(1);
    // }

    // std::string filepath(argv[1]);
    // std::string timestep(argv[2]);
    // std::string savepath(argv[3]);

    bool save_J(false);
    bool save_B(false);
    bool save_V(false);

    bool save_J_norm(true);

    bool save_X(true);
    bool save_Y(true);
    bool save_Z(true);

    bool save_ip(true);
    bool save_params(true);

    bool logging(true);
    bool timing(true);
    bool warnings(true);

    std::string filepath(".");
    std::string timestep("");                       // TODO: could add support for multiple timesteps at a time?
    std::string savepath(".");

    std::string J_format("x00_jvec-");
    std::string B_format("x00_Bvec_c-");
    std::string V_format("x00_vvec-");              // TODO: not sure how best to do this

    std::string file_format("pvtr");                // TODO: would be interesting to support mutiple file formats (pvti, ...)

    std::string analytical_model("Rolland25");      // TODO: would be interesting to add support for multiple models, maybe multiple at once?


    for (int i=1; i<argc; i+=2)
    {
        if (i+1>=argc || argv[i+1][0]=='-') { std::cout << "ERROR: no parameter provided for flag: " << argv[i] << std::endl; exit(1); }

        if( std::string(argv[i]) == "--input_dir" || std::string(argv[i]) == "-i" ) filepath = argv[i+1];
        else if( std::string(argv[i]) == "--output_dir" || std::string(argv[i]) == "-o" ) savepath = argv[i+1];
        else if( std::string(argv[i]) == "--output_dir" || std::string(argv[i]) == "-o" ) savepath = argv[i+1];

        else if( std::string(argv[i]) == "--timestep" || std::string(argv[i]) == "-t" ) timestep = argv[i+1];

        // else if( std::string(argv[i]) == "--file_format" ) file_format = argv[i+1];              // TODO: can't do that yet

        else if( std::string(argv[i]) == "--J_format" ) J_format = argv[i+1];
        else if( std::string(argv[i]) == "--B_format" ) B_format = argv[i+1];
        else if( std::string(argv[i]) == "--V_format" ) V_format = argv[i+1];

        // else if( std::string(argv[i]) == "--analytical_model" ) analytical_model = argv[i+1];    // TODO: can't do that yet

        else if( std::string(argv[i]) == "--save_J" || std::string(argv[i]) == "-J" )
        {
            if (std::string(argv[i+1]) == "true") save_J = true;
            else if (std::string(argv[i+1]) == "false") save_J = false;
            else { std::cout << "ERROR: unknown parameter for flag --save_J\n"; exit(1); }
            break;
        }
        else if( std::string(argv[i]) == "--save_B" || std::string(argv[i]) == "-B" )
        {
            if (std::string(argv[i+1]) == "true") save_B = true;
            else if (std::string(argv[i+1]) == "false") save_B = false;
            else { std::cout << "ERROR: unknown parameter for flag --save_B\n"; exit(1); }
            break;
        }
        else if( std::string(argv[i]) == "--save_V" || std::string(argv[i]) == "-V" )
        {
            if (std::string(argv[i+1]) == "true") save_V = true;
            else if (std::string(argv[i+1]) == "false") save_V = false;
            else { std::cout << "ERROR: unknown parameter for flag --save_V\n"; exit(1); }
            break;
        } 
        
        else if( std::string(argv[i]) == "--save_J_norm" || std::string(argv[i]) == "-J" )
        {
            if (std::string(argv[i+1]) == "true") save_J_norm = true;
            else if (std::string(argv[i+1]) == "false") save_J_norm = false;
            else { std::cout << "ERROR: unknown parameter for flag --save_J_norm\n"; exit(1); }
            break;
        }

        else if( std::string(argv[i]) == "--save_X" || std::string(argv[i]) == "-X" )
        {
            if (std::string(argv[i+1]) == "true") save_X = true;
            else if (std::string(argv[i+1]) == "false") save_X = false;
            else { std::cout << "ERROR: unknown parameter for flag --save_X\n"; exit(1); }
            break;
        }
        else if( std::string(argv[i]) == "--save_Y" || std::string(argv[i]) == "-Y" )
        {
            if (std::string(argv[i+1]) == "true") save_Y = true;
            else if (std::string(argv[i+1]) == "false") save_Y = false;
            else { std::cout << "ERROR: unknown parameter for flag --save_Y\n"; exit(1); }
            break;
        }
        else if( std::string(argv[i]) == "--save_Z" || std::string(argv[i]) == "-Z" )
        {
            if (std::string(argv[i+1]) == "true") save_Z = true;
            else if (std::string(argv[i+1]) == "false") save_Z = false;
            else { std::cout << "ERROR: unknown parameter for flag --save_Z\n"; exit(1); }
            break;
        }

        else if( std::string(argv[i]) == "--save_interest_points" )
        {
            if (std::string(argv[i+1]) == "true") save_ip = true;
            else if (std::string(argv[i+1]) == "false") save_ip = false;
            else { std::cout << "ERROR: unknown parameter for flag --save_interest_points\n"; exit(1); }
            break;
        }
        else if( std::string(argv[i]) == "--save_params" )
        {
            if (std::string(argv[i+1]) == "true") save_params = true;
            else if (std::string(argv[i+1]) == "false") save_params = false;
            else { std::cout << "ERROR: unknown parameter for flag --save_params\n"; exit(1); }
            break;
        }

        else if( std::string(argv[i]) == "--logging" )
        {
            if (std::string(argv[i+1]) == "true") logging = true;
            else if (std::string(argv[i+1]) == "false") logging = false;
            else { std::cout << "ERROR: unknown parameter for flag --logging\n"; exit(1); }
            break;
        }

        else if( std::string(argv[i]) == "--timing" )
        {
            if (std::string(argv[i+1]) == "true") timing = true;
            else if (std::string(argv[i+1]) == "false") timing = false;
            else { std::cout << "ERROR: unknown parameter for flag --timing\n"; exit(1); }
            break;
        }

        else if( std::string(argv[i]) == "--warnings" )
        {
            if (std::string(argv[i+1]) == "true") warnings = true;
            else if (std::string(argv[i+1]) == "false") warnings = false;
            else { std::cout << "ERROR: unknown parameter for flag --warnings\n"; exit(1); }
            break;
        }

        else { std::cout << "ERROR: unknown command line argument: " << argv[i] << std::endl; exit(1); }
    }


    // "/rds/general/user/avr24/projects/swimmr-sage/live/mheyns/benchmarking/runs/Run1/MS/x00_Bvec_c-21000.pvtr"

    // *********************************************************************************************
    auto t0 = Time::now();

    Matrix J = read_pvtr(filepath + std::string("/") + J_format + timestep + std::string(".") + file_format);
    Matrix B = read_pvtr(filepath + std::string("/") + B_format + timestep + std::string(".") + file_format);
    Matrix V = read_pvtr(filepath + std::string("/") + V_format + timestep + std::string(".") + file_format);

    Matrix X;
    Matrix Y;
    Matrix Z;

    get_coord(X, Y, Z, filepath + std::string("/") + B_format + timestep + std::string(".") + file_format);

    if (save_X) save_file( savepath + std::string("/X.txt"), X );
    if (save_Y) save_file( savepath + std::string("/Y.txt"), Y );
    if (save_Z) save_file( savepath + std::string("/Z.txt"), Z );

    auto t1 = Time::now();
    if (timing) std::cout << "File reading done. Time taken: " << fsec((t1-t0)).count() << 's' << std::endl;


    // *********************************************************************************************
    t0 = Time::now();

    Point p_min( X[0], Y[0], Z[0] );
    Point p_max( X[ X.get_shape().x-1 ], Y[ Y.get_shape().x-1 ], Z[ Z.get_shape().x-1 ] );
    Point p_range = p_max - p_min;

    float hyper_sampling = 1.2;

    Shape new_shape_real(std::round(p_range.x), std::round(p_range.y), std::round(p_range.z), V.get_shape().i);
    Shape new_shape_sim(V.get_shape().x*hyper_sampling, V.get_shape().y*hyper_sampling, V.get_shape().z*hyper_sampling, V.get_shape().i);

    Point earth_pos_real = find_real_earth_pos( X, Y, Z );
    Point earth_pos_sim = find_sim_earth_pos( earth_pos_real, new_shape_real, new_shape_sim );

    Matrix B_processed_sim = orthonormalise(B, X, Y, Z, &new_shape_sim);
    Matrix J_processed_sim = orthonormalise(J, X, Y, Z, &new_shape_sim);
    Matrix V_processed_sim = orthonormalise(V, X, Y, Z, &new_shape_sim);

    Matrix B_processed_real = orthonormalise(B, X, Y, Z, &new_shape_real);
    Matrix J_processed_real = orthonormalise(J, X, Y, Z, &new_shape_real);
    Matrix V_processed_real = orthonormalise(V, X, Y, Z, &new_shape_real);

    Matrix J_norm_sim = J_processed_sim.norm();
    Matrix J_norm_real = J_processed_real.norm();
    
    t1 = Time::now();
    if (timing) std::cout << "Preprocessing files done. Time taken: " << fsec((t1-t0)).count() << 's' << std::endl;


    // *********************************************************************************************
    t0 = Time::now();

    int nb_theta = 40;
    int nb_phi = 90;

    float avg_std_dev;

    InterestPoint* interest_points = get_interest_points(
        J_norm_sim, earth_pos_sim,
        nb_theta, nb_phi,
        0.1, 0.1,
        0.6, 0.7, 2,
        1.15, 1.8, 20,
        &avg_std_dev
    );

    process_interest_points( interest_points, nb_theta, nb_phi, new_shape_sim, new_shape_real, earth_pos_sim, earth_pos_real );

    t1 = Time::now();
    if (timing) std::cout << "Interest point search done. Time taken: " << fsec((t1-t0)).count() << 's' << std::endl;


    // *********************************************************************************************
    t0 = Time::now();

    //                              r_0,    a_0,    a_1,    a_2,    d_n,    l_n,    s_n,    d_s,    l_s,    s_s,    e    
    double initial_params[11] = {   10.0,   0.5,    0.0,    0.0,    2.0,    0.55,   0.55,   2.0,    0.55,   0.55,   0.0     };
    double lowerbound[11] =     {   5.0,    0.3,    -1.0,   -1.0,   0.0,    0.1,    0.1,    0.0,    0.1,    0.1,    -0.5    };
    double upperbound[11] =     {   15.0,   0.8,    1.0,    1.0,    4.0,    2.0,    1.0,    4.0,    2.0,    1.0,    0.5     };
    double radii[11] =          {   3.0,    0.1,    0.5,    0.5,    1.0,    0.05,   0.25,   1.0,    0.05,   0.25,   0.15    };


    int nb_runs = 50;
    int nb_interest_points = nb_theta * nb_phi;
    const int nb_params = 11;

    OptiResult result = fit_MP<SphericalResidual, nb_params>( 
        interest_points, nb_interest_points, 
        initial_params, 
        lowerbound, upperbound, radii, 
        nb_runs
    );

    t1 = Time::now();
    if (timing) std::cout << "Fitting done. Time taken: " << fsec((t1-t0)).count() << 's' << std::endl;



    // *********************************************************************************************
    t0 = Time::now();

    float avg_J_norm_grad = get_avg_grad_of_func( Rolland25, result.params, J_norm_real, nb_theta, nb_phi, earth_pos_real );

    if (logging)
    {
        std::cout << "Average standard deviation of the interest points is " << avg_std_dev << std::endl;

        std::cout << "Average cost of best fit to the analytical function is " << result.cost / nb_interest_points << std::endl;
        std::cout << "Final parameters are { ";
        std::cout << result.params[0];
        for (int i=1; i<nb_params; i++) std::cout << ", " << result.params[i];
        std::cout << " }" << std::endl;

        std::cout << "Average ||grad(||J||)|| is " << avg_J_norm_grad << std::endl;
    }


    t1 = Time::now();
    if (timing) std::cout << "Analysis done. Time taken: " << fsec((t1-t0)).count() << 's' << std::endl;



    // *********************************************************************************************
    t0 = Time::now();

    if (save_J) save_file_bin( savepath + std::string("/J_processed_real.bin"), J_processed_real );
    if (save_B) save_file_bin( savepath + std::string("/B_processed_real.bin"), B_processed_real );
    if (save_V) save_file_bin( savepath + std::string("/V_processed_real.bin"), V_processed_real );

    if (save_J_norm) save_file_bin( savepath + std::string("/J_norm_processed_real.bin"), J_norm_real );

    if (save_ip) save_interest_points( savepath + std::string("/interest_points_cpp.txt"), interest_points, nb_theta, nb_phi );

    if (save_params) save_parameters( savepath + std::string("/params_cpp.txt"), result.params );

    t1 = Time::now();
    if (timing) std::cout << "Interest points, parameters and file saving done. Time taken: " << fsec((t1-t0)).count() << 's' << std::endl;


    // *********************************************************************************************
    V.del(); J.del(); B.del();
    X.del(); Y.del(); Z.del();
    V_processed_sim.del(); B_processed_sim.del(); J_processed_sim.del(); J_norm_sim.del();
    V_processed_real.del(); B_processed_real.del(); J_processed_real.del(); J_norm_real.del();
}
