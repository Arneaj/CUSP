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

std::array<float, 4>* get_interest_points(  const Matrix& J_norm, Point earth_pos, 
                                            int nb_theta, int nb_phi, 
                                            float dx, float dr,
                                            float alpha_0_max, float alpha_0_min, float nb_alpha_0,
                                            float r_0_max, float r_0_min, float nb_r_0 );




void save_interest_points( std::string filename, const std::array<float, 4>* interest_points, int nb_theta, int nb_phi );


#endif