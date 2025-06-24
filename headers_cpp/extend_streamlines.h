#ifndef EXTEND_STREAMLINES_H
#define EXTEND_STREAMLINES_H

#include "matrix.h"
#include "points.h"
#include "streamlines.h"


std::vector<Point> extend_streamline(const Matrix& V, const std::vector<Point>& streamline, float step=0.5f, int max_length=1000);


std::vector<std::vector<Point>> get_extended_streamlines(   const Matrix& B, const Matrix& V,
                                                            const std::vector<Point>& points, 
                                                            float step=0.5f, int max_length=1000,
                                                            const Point* earth_pos=nullptr  );



#endif
