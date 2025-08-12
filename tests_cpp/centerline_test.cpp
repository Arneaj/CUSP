#include<iostream>
#include<fstream>

#include "../headers_cpp/read_file.h"
#include "../headers_cpp/centerline.h"


int main()
{
    Matrix B = read_file("../data/Run1_28800/B.txt");


    std::vector<Point> points;

    double STEP = 1;

    for (double ix=1; ix<B.get_shape().x-1; ix += STEP) for (double iy=1; iy<B.get_shape().y-1; iy += STEP)
    {
        points.push_back( Point(ix, iy, 1) );
        points.push_back( Point(ix, iy, B.get_shape().z-2) );
    }

    std::vector<std::vector<Point>> streamlines = get_streamlines(B, points, 0.5, 1000);


    double x_min = 30, x_max = B.get_shape().x, step = 4;
    int length = x_to_index(x_max, x_min, step);


    std::shared_ptr<Point_array[]> interpolated_streamlines = std::shared_ptr<Point_array[]>( new Point_array[streamlines.size()] );
    for (int i=0; i<streamlines.size(); i++) 
        interpolated_streamlines.get()[i] = get_interpolated_streamline(streamlines[i], length, step, x_min, x_max);

    Point_array centerline = get_centerline(interpolated_streamlines, length, streamlines.size());


    std::ofstream fs;
    fs.open("../data/Run1_28800/centerline.txt");

    for (int i=0; i<length; i++)
    {
        fs << std::to_string( centerline.get()[i].x ).c_str();
        fs << ",";

        fs << std::to_string( centerline.get()[i].y ).c_str();
        fs << ",";

        fs << std::to_string( centerline.get()[i].z ).c_str();
        if ( i < length-1 ) fs << ",";
    }

    fs.close();


    B.del();

    return 0;
}

