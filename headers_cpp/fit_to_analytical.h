#ifndef FIT_TO_ANALYTICAL_H
#define FIT_TO_ANALYTICAL_H

#include <array>
#include <random>

#include <thread>
#include <future>

#include <fstream>

#include "matrix.h"
#include "points.h"
#include "streamlines.h"
#include "raycast.h"



#include <Eigen/Dense>
#include <ceres/ceres.h>


const float PI = 3.1415926535;



template <typename T>
T is_pos( T v ) { return T( v>=T(0) ); }


// template <typename T>
// T abs_approx( T v ) { return v*(v>=T(0)) - v*(v<T(0)); }


/// @brief analytical approximation of the Magnetopause topology
/// @param theta rotation around the \hat{y} axis in [0; \pi]
/// @param phi rotation around the \hat{x} axis in [-\pi; \pi)
/// @param params [r_0, alpha_0, alpha_1, alpha_2, d_n, l_n, s_n, d_s, l_s, s_s, e]
/// @return the radius at angle (theta, phi)
template <typename T>
T Rolland25( const T* const params, T theta, T phi )
{
    // if (theta<0 || theta>PI) { std::cout << "theta should be in [0; pi]\n"; exit(1); }
    // if (phi<-PI || phi>PI) { std::cout << "phi should be in [-pi; pi)\n"; exit(1); }

    T cos_theta = ceres::cos(theta);
    T cos_phi = ceres::cos(phi);

    return params[0] * (
        (T(1.0)+params[10]) / (T(1.0)+params[10]*cos_theta)
    ) * ceres::pow(
        T(2.0) / (T(1.0)+cos_theta), 
        params[1] + params[2]*cos_phi + params[3]*cos_phi*cos_phi
    ) - (
        params[4] * ceres::exp( -ceres::abs(theta - params[5]) / params[6] ) * is_pos<T>( cos_phi ) +
        params[7] * ceres::exp( -ceres::abs(theta - params[8]) / params[9] ) * is_pos<T>( -cos_phi )
    ) * cos_phi*cos_phi;
}


class SphericalResidual 
{
private:
    const double m_theta, m_phi, m_weight, m_observed_radius;

public:
    SphericalResidual(const InterestPoint& interest_point) 
        : m_theta(interest_point.theta), m_phi(interest_point.phi), m_observed_radius(interest_point.radius), m_weight(interest_point.weight) {;}

    SphericalResidual(double theta, double phi, double observed_radius, double weight) 
        : m_theta(theta), m_phi(phi), m_observed_radius(observed_radius), m_weight(weight) {;}

    template <typename T>
    bool operator()(const T* const params, T* residual) const 
    {
        T predicted_radius = Rolland25(params, T(m_theta), T(m_phi));
        residual[0] = (m_observed_radius - predicted_radius)*m_weight;
        return true;
    }
};



struct OptiResult
{
    std::vector<double> params;
    double cost;

    OptiResult(): cost(MAXFLOAT) {;}
    OptiResult(int nb_params): params(nb_params), cost(MAXFLOAT) {;}
};






/// @brief 
/// @tparam Residual 
/// @tparam nb_params 
/// @param interest_points containing in order (theta, phi, radius, weight)
/// @param nb_interest_points 
/// @param params 
/// @param lowerbound lowerbound of each parameter
/// @param upperbound upperbound of each parameter
/// @param max_nb_iterations_per_run 
/// @param trust_region
/// @param linear_solver 
/// @param print_progress 
/// @return 
template <typename Residual, int nb_params>
OptiResult fit_with_params( 
    InterestPoint* interest_points,
    int nb_interest_points,
    double* params, 
    double* lowerbound=nullptr, double* upperbound=nullptr,
    int max_nb_iterations_per_run=50,
    ceres::TrustRegionStrategyType trust_region=ceres::LEVENBERG_MARQUARDT,
    ceres::LinearSolverType linear_solver=ceres::DENSE_QR,
    bool print_progress=false
)
{
    ceres::Problem problem;

    for (int i=0; i<nb_interest_points; i++) 
    {
        ceres::CostFunction* cost_function = new ceres::AutoDiffCostFunction<Residual, 1, nb_params>(
            new Residual(interest_points[i])
        );
        problem.AddResidualBlock( cost_function, nullptr, params );
    }

    if (lowerbound) for (int i=0; i<nb_params; i++)
        problem.SetParameterLowerBound(params, i, lowerbound[i]);

    if (upperbound) for (int i=0; i<nb_params; i++)
        problem.SetParameterUpperBound(params, i, upperbound[i]);

    ceres::Solver::Options options;
    options.max_num_iterations = max_nb_iterations_per_run; // / 4;  // TODO: see if this is ok
    options.minimizer_progress_to_stdout = print_progress;
    options.function_tolerance = 1e-8; //1e-4;

    options.trust_region_strategy_type = trust_region;
    options.linear_solver_type = linear_solver;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    if (print_progress) std::cout << summary.BriefReport() << "\n";

    // if (summary.final_cost < best_result.cost)
    // {
    //     options.max_num_iterations = max_nb_iterations_per_run;
    //     options.function_tolerance = 1e-8;
    //     ceres::Solve(options, &problem, &summary);

    //     // if (summary.final_cost < best_result.cost)
    //     // {
    //     //     for (int i=0; i<nb_params; i++) best_result.params[i] = params[i];
    //     //     final_result.cost = summary.final_cost;
    //     // }
    // }

    OptiResult result(nb_params);
    result.cost = summary.final_cost;
    for (int i=0; i<nb_params; i++) result.params[i] = params[i];

    return result;
}







/// @brief 
/// @tparam Residual 
/// @tparam nb_params 
/// @param interest_points containing (theta, phi, radius, weight)
/// @param nb_interest_points 
/// @param params 
/// @param lowerbound lowerbound of each parameter
/// @param upperbound upperbound of each parameter
/// @param radii_of_variation how much each parameter should vary in positive and negative directions
/// @param nb_runs number of different initial conditions tried
/// @param max_nb_iterations_per_run 
/// @param trust_region
/// @param linear_solver 
/// @param print_progress 
/// @param print_results 
/// @return 
template <typename Residual, int nb_params>
OptiResult fit_MP( 
    InterestPoint* interest_points,
    int nb_interest_points,
    const double* initial_params, 
    double* lowerbound=nullptr, double* upperbound=nullptr,
    double* radii_of_variation=nullptr, int nb_runs=1,
    int max_nb_iterations_per_run=50,
    ceres::TrustRegionStrategyType trust_region=ceres::LEVENBERG_MARQUARDT,
    ceres::LinearSolverType linear_solver=ceres::DENSE_QR,
    bool print_progress=false, bool print_results=true
)
{
    if (print_progress)
    {
        std::cout << "Initial parameters:\n{ ";
        std::cout << initial_params[0];
        for (int i=1; i<nb_params; i++) std::cout << ", " << initial_params[i];
        std::cout << " }" << std::endl;
    }

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    OptiResult final_result(nb_params);

    std::vector<std::future<OptiResult>> futures;
    const int nb_threads = std::thread::hardware_concurrency();
    std::cout << "number of threads: " << nb_threads << std::endl;

    std::vector<double*> params_list(nb_runs);

    for (int run=0; run<nb_runs; run++)
    {
        params_list[run] = new double[nb_params];
        for (int i=0; i<nb_params; i++) params_list[run][i] = initial_params[i] + dist(gen) * radii_of_variation[i];

        if (futures.size() >= nb_threads) 
        {
            // Wait for one to complete
            OptiResult result = futures.front().get();

            if (print_progress)
            {
                std::cout << "\nCurrent parameters with cost " << result.cost << " :\n{ ";
                std::cout << result.params[0];
                for (int i=1; i<nb_params; i++) std::cout << ", " << result.params[i];
                std::cout << " }" << std::endl;
            }

            if (result.cost < final_result.cost) final_result = result;
            
            futures.erase(futures.begin());

            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        
        futures.push_back(std::async(std::launch::async, [=]() {
            return fit_with_params<Residual, nb_params>(  
                interest_points, nb_interest_points, 
                params_list[run], lowerbound, upperbound,
                max_nb_iterations_per_run,
                trust_region, linear_solver,
                print_progress
            );
        }));
    }

    for (auto& future : futures) 
    {
        OptiResult result = future.get();

        if (print_progress)
        {
            std::cout << "\nCurrent parameters with cost " << result.cost << " :\n{ ";
            std::cout << result.params[0];
            for (int i=1; i<nb_params; i++) std::cout << ", " << result.params[i];
            std::cout << " }" << std::endl;
        }

        if (result.cost < final_result.cost) final_result = result;
    }

    if (print_results)
    {
        std::cout << "Final parameters with average cost " << final_result.cost / nb_interest_points << " :\n{ ";
        std::cout << final_result.params[0];
        for (int i=1; i<nb_params; i++) std::cout << ", " << final_result.params[i];
        std::cout << " }" << std::endl;
    }

    for (double* params: params_list) delete[] params;
    futures.clear();

    return final_result;
}





void save_parameters( std::string filename, const std::vector<double>& params );





#endif
