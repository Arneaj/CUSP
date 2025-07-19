#include "../headers_cpp/fit_to_analytical.h"



template <typename T, int nb_params>
void fit_spherical( 
    T (*func)(const T* const, T, T), 
    double* initial_params, 
    double* lowerbound=nullptr, double* upperbound=nullptr,
    double* radii_of_variation=nullptr, int nb_runs=1,
    int max_nb_iterations_per_run=50,
    bool print_progress=false,
    ceres::TrustRegionStrategyType trust_region=ceres::LEVENBERG_MARQUARDT,
    ceres::LinearSolverType linear_solver=ceres::DENSE_QR
)
{
    
}















