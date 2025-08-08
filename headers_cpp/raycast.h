#ifndef RAYCAST_H
#define RAYCAST_H

#include <vector>
#include <array>
#include <algorithm>
#include <numeric>
#include <fstream>

#include "matrix.h"
#include "points.h"
#include "streamlines.h"



float get_bowshock_radius(  const Point& projection,
                            const Matrix& Rho, const Point& earth_pos,
                            float dr,
                            bool* is_at_bounds=nullptr );

float get_bowshock_radius(  float theta, float phi,
                            const Matrix& Rho, const Point& earth_pos,
                            float dr,
                            bool* is_at_bounds=nullptr );

std::vector<Point> get_bowshock( const Matrix& Rho, const Point& earth_pos, float dr, int nb_phi, int max_nb_theta, bool is_squeezed=true );


class InterestPoint
{
public:
    float theta, phi, radius, weight;

    InterestPoint()
        : theta(0.0), phi(0.0), radius(0.0), weight(0.0) {;}

    InterestPoint(float _theta, float _phi)
        : theta(_theta), phi(_phi), radius(0.0), weight(0.0) {;}

    InterestPoint(float _theta, float _phi, float _radius, float _weight)
        : theta(_theta), phi(_phi), radius(_radius), weight(_weight) {;}
};




InterestPoint* get_interest_points( const Matrix& J_norm, const Point& earth_pos,
                                    const Point* const unsqueezed_bow_shock,
                                    float theta_min, float theta_max, 
                                    int nb_theta, int nb_phi, 
                                    float dx, float dr,
                                    float alpha_0_min, float alpha_0_max, float nb_alpha_0,
                                    float r_0_mult_min, float r_0_mult_max, float nb_r_0,
                                    float* p_avg_std_dev=nullptr );

InterestPoint* get_interest_points( const Matrix& J_norm, const Point& earth_pos,
                                    const Matrix& Rho,
                                    float theta_min, float theta_max, 
                                    int nb_theta, int nb_phi, 
                                    float dx, float dr,
                                    float alpha_0_min, float alpha_0_max, float nb_alpha_0,
                                    float r_0_mult_min, float r_0_mult_max, float nb_r_0,
                                    float* p_avg_std_dev=nullptr );



void save_interest_points( const std::string& filename, const InterestPoint* interest_points, int nb_theta, int nb_phi );



void process_interest_points(   InterestPoint* interest_points, 
                                int nb_theta, int nb_phi, 
                                const Shape& shape_sim, const Shape& shape_real,
                                const Point& earth_pos_sim, const Point& earth_pos_real );


/// @brief transforms points from the simulation coordinates to the real coordinates
/// @param points despite writing (x,y,z), points in this case are actually of coordinates (theta,phi,radius)
/// @param shape_sim 
/// @param shape_real 
/// @param earth_pos_sim 
/// @param earth_pos_real 
void process_points(    std::vector<Point>& points, 
                        const Shape& shape_sim, const Shape& shape_real,
                        const Point& earth_pos_sim, const Point& earth_pos_real );


void save_points( const std::string& filename, const std::vector<Point>& points );


#endif