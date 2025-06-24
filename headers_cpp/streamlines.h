#ifndef STREAMLINES_H
#define STREAMLINES_H

#include "matrix.h"
#include "points.h"


float interpolate(Point P, const Matrix& B, int i);

Point norm_RK4(Point P_init, const Matrix& B, float step, float sign=1.0f);


std::vector<Point> find_streamline(const Matrix& B, Point P_i, float step, int max_length, bool both_sides=true, const Point* earth_pos=nullptr);


std::vector<Point> smooth_interpolation(const std::vector<Point>& points, int kernel_radius);

std::vector<Point> find_smooth_streamline(const Matrix& B, Point P_i, float step, int kernel_radius, int max_length);


std::vector<std::vector<Point>> get_streamlines_singlethreaded(const Matrix& B, const std::vector<Point>& points, float step, int kernel_radius, int max_length);

std::vector<std::vector<Point>> get_streamlines(const Matrix& B, const std::vector<Point>& points, float step=1.0f, int max_length=1000, const Point* earth_pos=nullptr);

void save_streamlines(const std::string& filename, const std::vector<std::vector<Point>>& streamlines);


#endif
