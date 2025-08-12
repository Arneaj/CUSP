#ifndef EXTEND_STREAMLINES_H
#define EXTEND_STREAMLINES_H

#include "matrix.h"
#include "points.h"
#include "streamlines.h"


std::vector<Point> extend_streamline(const Matrix& V, const std::vector<Point>& streamline, double step=0.5, int max_length=1000);


std::vector<std::vector<Point>> get_extended_streamlines(   const Matrix& B, const Matrix& V,
                                                            const std::vector<Point>& points, 
                                                            double step=0.5, int max_length=1000,
                                                            const Point* earth_pos=nullptr  );



#endif
