#include "fit_to_analytical.h"


struct SphericalResidual {
    SphericalResidual(double theta, double phi, double observed_radius)
        : theta_(theta), phi_(phi), observed_radius_(observed_radius) {}

    template <typename T>
    bool operator()(const T* const params, T* residual) const {
        T predicted_radius = analytical_function(params, T(theta_), T(phi_));
        residual[0] = observed_radius_ - predicted_radius;
        return true;
    }

private:
    const double theta_, phi_, observed_radius_;
};



// Usage
ceres::Problem problem;
for (int i = 0; i < N; ++i) {
    for (int j = 0; j < M; ++j) {
        ceres::CostFunction* cost_function = 
            new ceres::AutoDiffCostFunction<SphericalResidual, 1, 10>(
                new SphericalResidual(theta[i], phi[j], observed_radii[i][j]));
        problem.AddResidualBlock(cost_function, nullptr, parameters);
    }
}



