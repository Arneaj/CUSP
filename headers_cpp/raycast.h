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




InterestPoint* get_interest_points( const Matrix& J_norm, Point earth_pos,
                                    float theta_min, float theta_max, 
                                    int nb_theta, int nb_phi, 
                                    float dx, float dr,
                                    float alpha_0_min, float alpha_0_max, float nb_alpha_0,
                                    float r_0_mult_min, float r_0_mult_max, float nb_r_0,
                                    float* avg_std_dev );




void save_interest_points( std::string filename, const InterestPoint* interest_points, int nb_theta, int nb_phi );



void process_interest_points(   InterestPoint* interest_points, 
                                int nb_theta, int nb_phi, 
                                const Shape& shape_sim, const Shape& shape_real,
                                const Point& earth_pos_sim, const Point& earth_pos_real );




#endif