#include "../headers_cpp/fit_to_analytical.h"



template <typename T, int nb_params>
OptiResult fit_MP( 
    T (*func)(const T* const, T, T),
    std::array<float, 4>* interest_points,
    int nb_interest_points,
    const double* initial_params, 
    double* lowerbound, double* upperbound,
    double* radii_of_variation, int nb_runs,
    int max_nb_iterations_per_run,
    ceres::TrustRegionStrategyType trust_region,
    ceres::LinearSolverType linear_solver,
    bool print_progress, bool print_results
)
{
    if (print_progress)
    {
        std::cout << "Initial values:\n";
        for (int i=0; i<nb_params; i++) std::cout << "param[" << i << "] = " << params[i] << '\n';
        std::cout << std::endl;
    }

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    OptiResult final_result(nb_params);

    for (int run=0; run<nb_runs; run++)
    {
        double* params(new double[nb_params]);
        for (int i=0; i<nb_params; i++) params[i] = initial_params[i] + dist(gen) * radii_of_variation[i];

        ceres::Problem problem;

        for (int i=0; i<nb_interest_points i++) 
        {
            ceres::CostFunction* cost_function = new ceres::AutoDiffCostFunction<MPResidual, 1, nb_params>(
                new MPResidual(func, interest_points[i][0], interest_points[i][1], interest_points[i][2], interest_points[i][3])
            );
            problem.AddResidualBlock( cost_function, nullptr, params );
        }

        if (lowerbound) for (int i=0; i<nb_params; i++)
            problem.SetParameterLowerBound(params, i, lowerbound[i]);

        if (upperbound) for (int i=0; i<nb_params; i++)
            problem.SetParameterUpperBound(params, i, upperbound[i]);

        ceres::Solver::Options options;
        options.max_num_iterations = max_nb_iterations_per_run;
        options.minimizer_progress_to_stdout = print_progress;

        options.trust_region_strategy_type = trust_region;
        options.linear_solver_type = linear_solver;

        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);

        if (print_progress) std::cout << summary.BriefReport() << "\n";

        if (print_results)
        {
            std::cout << "\nCurrent parameters:\n{ ";
            std::cout << params[0];
            for (int i=1; i<nb_params; i++) std::cout << "; " << params[i];
            std::cout << " }" << std::endl;
        }

        if (summary.final_cost < final_result.cost)
        {
            for (int i=0; i<nb_params; i++) final_result.params[i] = params[i];
            final_result.cost = summary.final_cost;
        }

        delete[] params;
    }

    if (print_results)
    {
        std::cout << "\nFinal parameters:\n{ ";
        std::cout << params[0];
        for (int i=1; i<nb_params; i++) std::cout << "; " << params[i];
        std::cout << " }" << std::endl;
    }

    return final_result;
}















