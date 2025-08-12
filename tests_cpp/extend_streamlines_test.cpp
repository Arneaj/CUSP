#include<iostream>

#include "../headers_cpp/read_file.h"
#include "../headers_cpp/streamlines.h"
#include "../headers_cpp/extend_streamlines.h"



int main()
{
    Matrix B = read_file("../data/Run1_28800/B.txt");
    Matrix V = read_file("../data/Run1_28800/V.txt");


    std::vector<Point> points;

    double STEP = 1;

    for (double ix=B.get_shape().x-1-STEP*5; ix<B.get_shape().x-1; ix += STEP) for (double iy=1; iy<B.get_shape().y-1; iy += STEP)
    {
        points.push_back( Point(ix, iy, 1) );
        points.push_back( Point(ix, iy, B.get_shape().z-2) );
    }

    std::vector<std::vector<Point>> extended_streamlines = get_extended_streamlines(B, V, points, 0.5f, 1000, nullptr);


    save_streamlines("../data/Run1_28800/streamlines.txt", extended_streamlines);


    B.del();
    V.del();

    return 0;
}
