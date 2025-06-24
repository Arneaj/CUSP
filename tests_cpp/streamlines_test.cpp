#include<iostream>

#include "../headers_cpp/read_file.h"
#include "../headers_cpp/streamlines.h"



int main()
{
    Matrix B = read_file("../data/Run1_28800/B_processed.txt");


    std::vector<Point> points;

    float STEP = 0.2;

    // for (float ix=1; ix<B.get_shape().x-1; ix += STEP) for (float iy=1; iy<B.get_shape().y-1; iy += STEP)
    for (float ix=B.get_shape().x-1-STEP*5; ix<B.get_shape().x-1; ix += STEP) for (float iy=1; iy<B.get_shape().y-1; iy += STEP)
    {
        points.push_back( Point(ix, iy, 1) );
        points.push_back( Point(ix, iy, B.get_shape().z-2) );
    }


    std::vector<std::vector<Point>> streamlines = get_streamlines(B, points);

    save_streamlines("../data/Run1_28800/streamlines.txt", streamlines);
    

    B.del();

    return 0;
}
