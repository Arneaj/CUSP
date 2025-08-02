#include "../headers_cpp/raycast.h"
#include "../headers_cpp/read_file.h"
#include "../headers_cpp/read_pvtr.h"
#include "../headers_cpp/reader_writer.h"
#include "../headers_cpp/magnetopause.h"
#include "../headers_cpp/preprocessing.h"
#include "../headers_cpp/fit_to_analytical.h"
#include "../headers_cpp/analysis.h"

#include <iostream>


#include <chrono>

typedef std::chrono::high_resolution_clock Time;
typedef std::chrono::duration<float> fsec;

// const float PI = 3.141592653589793238462643383279502884f;


int main(int argc, char* argv[])
{
    // *** flags ***********************************************************************************

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
    std::string file_save_format("bin");

    std::string analytical_model("Rolland25");      // TODO: would be interesting to add support for multiple models, maybe multiple at once?


    for (int i=1; i<argc; i+=2)
    {
        if (i+1>=argc || argv[i+1][0]=='-') { std::cout << "ERROR: no parameter provided for flag: " << argv[i] << std::endl; exit(1); }

        if( std::string(argv[i]) == "--input_dir" || std::string(argv[i]) == "-i" ) filepath = argv[i+1];
        else if( std::string(argv[i]) == "--output_dir" || std::string(argv[i]) == "-o" ) savepath = argv[i+1];
        else if( std::string(argv[i]) == "--output_dir" || std::string(argv[i]) == "-o" ) savepath = argv[i+1];

        else if( std::string(argv[i]) == "--timestep" || std::string(argv[i]) == "-t" ) timestep = argv[i+1];

        // else if( std::string(argv[i]) == "--analytical_model" ) analytical_model = argv[i+1];    // TODO: can't do that yet

        else if( std::string(argv[i]) == "--save_J" || std::string(argv[i]) == "-J" )
        {
            if (std::string(argv[i+1]) == "true") save_J = true;
            else if (std::string(argv[i+1]) == "false") save_J = false;
            else { std::cout << "ERROR: unknown parameter for flag --save_J\n"; exit(1); }
        }
        else if( std::string(argv[i]) == "--save_B" || std::string(argv[i]) == "-B" )
        {
            if (std::string(argv[i+1]) == "true") save_B = true;
            else if (std::string(argv[i+1]) == "false") save_B = false;
            else { std::cout << "ERROR: unknown parameter for flag --save_B\n"; exit(1); }
        }
        else if( std::string(argv[i]) == "--save_V" || std::string(argv[i]) == "-V" )
        {
            if (std::string(argv[i+1]) == "true") save_V = true;
            else if (std::string(argv[i+1]) == "false") save_V = false;
            else { std::cout << "ERROR: unknown parameter for flag --save_V\n"; exit(1); }
        } 
        
        else if( std::string(argv[i]) == "--save_J_norm" )
        {
            if (std::string(argv[i+1]) == "true") save_J_norm = true;
            else if (std::string(argv[i+1]) == "false") save_J_norm = false;
            else { std::cout << "ERROR: unknown parameter for flag --save_J_norm\n"; exit(1); }
        }

        else if( std::string(argv[i]) == "--save_X" || std::string(argv[i]) == "-X" )
        {
            if (std::string(argv[i+1]) == "true") save_X = true;
            else if (std::string(argv[i+1]) == "false") save_X = false;
            else { std::cout << "ERROR: unknown parameter for flag --save_X\n"; exit(1); }
        }
        else if( std::string(argv[i]) == "--save_Y" || std::string(argv[i]) == "-Y" )
        {
            if (std::string(argv[i+1]) == "true") save_Y = true;
            else if (std::string(argv[i+1]) == "false") save_Y = false;
            else { std::cout << "ERROR: unknown parameter for flag --save_Y\n"; exit(1); }
        }
        else if( std::string(argv[i]) == "--save_Z" || std::string(argv[i]) == "-Z" )
        {
            if (std::string(argv[i+1]) == "true") save_Z = true;
            else if (std::string(argv[i+1]) == "false") save_Z = false;
            else { std::cout << "ERROR: unknown parameter for flag --save_Z\n"; exit(1); }
        }

        else if( std::string(argv[i]) == "--save_interest_points" )
        {
            if (std::string(argv[i+1]) == "true") save_ip = true;
            else if (std::string(argv[i+1]) == "false") save_ip = false;
            else { std::cout << "ERROR: unknown parameter for flag --save_interest_points\n"; exit(1); }
        }
        else if( std::string(argv[i]) == "--save_params" )
        {
            if (std::string(argv[i+1]) == "true") save_params = true;
            else if (std::string(argv[i+1]) == "false") save_params = false;
            else { std::cout << "ERROR: unknown parameter for flag --save_params\n"; exit(1); }
        }

        else if( std::string(argv[i]) == "--logging" )
        {
            if (std::string(argv[i+1]) == "true") logging = true;
            else if (std::string(argv[i+1]) == "false") logging = false;
            else { std::cout << "ERROR: unknown parameter for flag --logging\n"; exit(1); }
        }
        else if( std::string(argv[i]) == "--timing" )
        {
            if (std::string(argv[i+1]) == "true") timing = true;
            else if (std::string(argv[i+1]) == "false") timing = false;
            else { std::cout << "ERROR: unknown parameter for flag --timing\n"; exit(1); }
        }
        else if( std::string(argv[i]) == "--warnings" )
        {
            if (std::string(argv[i+1]) == "true") warnings = true;
            else if (std::string(argv[i+1]) == "false") warnings = false;
            else { std::cout << "ERROR: unknown parameter for flag --warnings\n"; exit(1); }
        }

        else { std::cout << "ERROR: unknown command line argument: " << argv[i] << std::endl; exit(1); }
    }


    // "/rds/general/user/avr24/projects/swimmr-sage/live/mheyns/benchmarking/runs/Run1/MS/x00_Bvec_c-21000.pvtr"

    // *** file reading ****************************************************************************
    
    auto t0 = Time::now();

    PVTRReaderBinWriter reader_writer;    

    Matrix J;
    reader_writer.read(filepath + std::string("/") + J_format + timestep + std::string(".") + file_format, J);
    Matrix B;
    if (save_B) reader_writer.read(filepath + std::string("/") + B_format + timestep + std::string(".") + file_format, B);

    Matrix V;
    if (save_V) reader_writer.read(filepath + std::string("/") + V_format + timestep + std::string(".") + file_format, V);

    Matrix X;
    Matrix Y;
    Matrix Z;

    reader_writer.get_coordinates(filepath + std::string("/") + B_format + timestep + std::string(".") + file_format, X, Y, Z);

    if (save_X) reader_writer.write( savepath + std::string("/X.txt"), X );
    if (save_Y) reader_writer.write( savepath + std::string("/Y.txt"), Y );
    if (save_Z) reader_writer.write( savepath + std::string("/Z.txt"), Z );

    auto t1 = Time::now();
    if (timing) std::cout << "File reading done. Time taken: " << fsec((t1-t0)).count() << 's' << std::endl;


    // *** preprocessing ***************************************************************************
    
    t0 = Time::now();

    Point p_min( X[0], Y[0], Z[0] );
    Point p_max( X[ X.get_shape().x-1 ], Y[ Y.get_shape().x-1 ], Z[ Z.get_shape().x-1 ] );
    Point p_range = p_max - p_min;

    float hyper_sampling = 1.2;

    Shape new_shape_real(std::round(p_range.x), std::round(p_range.y), std::round(p_range.z), J.get_shape().i);
    Shape new_shape_sim(J.get_shape().x*hyper_sampling, J.get_shape().y*hyper_sampling, J.get_shape().z*hyper_sampling, J.get_shape().i);

    Point earth_pos_real = find_real_earth_pos( X, Y, Z );
    Point earth_pos_sim = find_sim_earth_pos( earth_pos_real, new_shape_real, new_shape_sim );

    Matrix B_processed_sim;
    if (save_B) B_processed_sim = orthonormalise(B, X, Y, Z, &new_shape_sim);
    Matrix J_processed_sim = orthonormalise(J, X, Y, Z, &new_shape_sim);

    Matrix B_processed_real;
    if (save_B) B_processed_real = orthonormalise(B, X, Y, Z, &new_shape_real);
    Matrix J_processed_real = orthonormalise(J, X, Y, Z, &new_shape_real);
    Matrix V_processed_real;
    if (save_V) V_processed_real = orthonormalise(V, X, Y, Z, &new_shape_real);

    Matrix J_norm_sim = J_processed_sim.norm();
    Matrix J_norm_real = J_processed_real.norm();
    
    t1 = Time::now();
    if (timing) std::cout << "Preprocessing files done. Time taken: " << fsec((t1-t0)).count() << 's' << std::endl;


    // *** interest points *************************************************************************
    
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

    process_interest_points( interest_points, nb_theta, nb_phi, new_shape_sim, new_shape_real, earth_pos_sim, earth_pos_real );

    t1 = Time::now();
    if (timing) std::cout << "Interest point search done. Time taken: " << fsec((t1-t0)).count() << 's' << std::endl;


    // *** fitting *********************************************************************************
    
    t0 = Time::now();

    //     index                    0       1       2       3       4       5       6       7       8       9       10
    //     param                    r_0,    a_0,    a_1,    a_2,    d_n,    l_n,    s_n,    d_s,    l_s,    s_s,    e    
    double initial_params[11] = {   10.0,   0.5,    0.0,    0.0,    3.0,    0.55,   5.0,    3.0,    0.55,   5.0,    0.0     };
    double lowerbound[11] =     {   5.0,    0.2,    -1.0,   -1.0,   0.0,    0.1,    0.1,    0.0,    0.1,    0.1,    -0.8    };
    double upperbound[11] =     {   15.0,   0.8,    1.0,    1.0,    6.0,    2.0,    10.0,   6.0,    2.0,    10.0,   0.8     };
    double radii[11] =          {   3.0,    0.1,    0.5,    0.5,    1.0,    0.05,   1.5,    1.0,    0.05,   1.5,    0.2     };


    int nb_runs = 50;
    int nb_interest_points = nb_theta * nb_phi;
    const int nb_params = 11;

    OptiResult result = fit_MP<EllipsisPolyResidual, nb_params>( 
        interest_points, nb_interest_points, 
        initial_params, 
        lowerbound, upperbound, radii, 
        nb_runs
    );

    t1 = Time::now();
    if (timing) std::cout << "Fitting done. Time taken: " << fsec((t1-t0)).count() << 's' << std::endl;



    // *** analysis ********************************************************************************
    
    t0 = Time::now();

    const float threshold = 2.0f;

    bool is_concave;

    float grad_J_fit_over_ip = get_grad_J_fit_over_interest_points( EllipsisPoly, result.params, interest_points, nb_interest_points, J_norm_real, earth_pos_real );
    float delta_l = get_delta_l( result.params[5], result.params[8] );
    int nb_params_at_boundaries = get_params_at_boundaries( result.params.data(), lowerbound, upperbound, nb_params );
    float max_theta_in_threshold = interest_point_flatness_checker( interest_points, nb_theta, nb_phi, &is_concave, threshold );

    const float delta_l_lowerbound = 1.0;
    const float r_0_lowerbound = 8.0;
    const float avg_std_dev_upperbound = 1.5;

    if (logging) std::cout << "Average standard deviation of the interest points is " << avg_std_dev << std::endl;
    if (warnings && avg_std_dev > avg_std_dev_upperbound) std::cout << "\t--> WARNING: high average interest point standard deviation\n";

    if (logging)
    {
        std::cout << "Average cost of best fit to the analytical function is " << result.cost / nb_interest_points << std::endl;
        std::cout << "Final parameters are { ";
        std::cout << result.params[0];
        for (int i=1; i<nb_params; i++) std::cout << ", " << result.params[i];
        std::cout << " }" << std::endl;
    }
    if (warnings && result.cost / nb_interest_points > 1.5) std::cout << "\t--> WARNING: high average fitting cost\n";


    if (logging) std::cout << "||grad(||J||)||_{fit} / ||grad(||J||)||_{ip} is " << grad_J_fit_over_ip << std::endl;

    if (logging) std::cout << "delta_l is " << delta_l << std::endl;
    if (warnings && delta_l<delta_l_lowerbound) std::cout << "\t--> WARNING: delta_l very low\n";

    if (logging) std::cout << "Number of parameters at boundaries is " << nb_params_at_boundaries << std::endl;
    if (warnings && nb_params_at_boundaries>0) std::cout << "\t--> WARNING: at least one parameter has reached bounds\n";

    if (warnings && result.params[0]<r_0_lowerbound) std::cout << "\t--> WARNING: r_0 < " << r_0_lowerbound << " is very low\n";

    if (logging) std::cout << "Maximum angle theta where abs(P.x - max(P.x)) < " << threshold << " is " << max_theta_in_threshold << std::endl;
    if (warnings && max_theta_in_threshold>1.0f) std::cout << "\t--> WARNING: interest points seem flat on the dayside magnetopause\n";

    if (warnings && is_concave) std::cout << "\t--> WARNING: interest points seem concave on the dayside\n";

    t1 = Time::now();
    if (timing) std::cout << "Analysis done. Time taken: " << fsec((t1-t0)).count() << 's' << std::endl;



    // *** saving **********************************************************************************
    
    t0 = Time::now();

    if (save_J) save_file_bin( savepath + std::string("/J_processed_real.") + file_save_format, J_processed_real );
    if (save_B) save_file_bin( savepath + std::string("/B_processed_real.") + file_save_format, B_processed_real );
    // if (save_V) save_file_bin( savepath + std::string("/V_processed_real.") + file_save_format, V_processed_real );

    save_file_bin( savepath + std::string("/J.") + file_save_format, J );
    save_file_bin( savepath + std::string("/J_processed_sim.") + file_save_format, J_processed_sim );

    if (save_J_norm) save_file_bin( savepath + std::string("/J_norm_processed_real.") + file_save_format, J_norm_real );

    if (save_ip) save_interest_points( savepath + std::string("/interest_points_cpp.csv"), interest_points, nb_theta, nb_phi );

    if (save_params) save_parameters( savepath + std::string("/params_cpp.csv"), result.params );

    t1 = Time::now();
    if (timing) std::cout << "Interest points, parameters and file saving done. Time taken: " << fsec((t1-t0)).count() << 's' << std::endl;


    // *** freeing *********************************************************************************
    
    J.del(); 
    if (save_B) B.del(); 
    if (save_V) V.del();
    X.del(); Y.del(); Z.del();
    if (save_B) B_processed_sim.del(); 
    J_processed_sim.del(); 
    J_norm_sim.del();
    if (save_B) B_processed_real.del(); 
    J_processed_real.del(); 
    J_norm_real.del(); 
    if (save_V) V_processed_real.del();

    delete[] interest_points;
}
