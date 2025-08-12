#include <iostream>
#include <thread>

#include "../headers_cpp/streamlines.h"
#include "../headers_cpp/magnetopause.h"


#ifndef CUSTOM_PI
#define CUSTOM_PI
const double PI = 3.141592653589793238462643383279502884;
#endif



void remove_outliers( std::vector<std::vector<Point>>& streamlines, Shape shape )
{
    Point avg_start_point;
    for (const std::vector<Point>& streamline: streamlines) avg_start_point += streamline[0];
    avg_start_point /= streamlines.size();

    double valid_radius = std::sqrt( shape.y*shape.y + shape.z*shape.z ) * 0.1;

    int i = 0;
    while (true)
    {
        if ( i >= (int) streamlines.size() ) break;
        if ( (streamlines[i][0] - avg_start_point).norm() <= valid_radius ) { i++; continue; }

        streamlines.erase( streamlines.begin() + i);
    }
}



Point find_earth_pos( const Matrix& B ) { return B.index_max(); }


Point find_real_earth_pos( const Matrix& X, const Matrix& Y, const Matrix& Z ) { return Point(-X[0], -Y[0], -Z[0]); }

Point find_sim_earth_pos( Point real_earth_pos, Shape real_shape, Shape sim_shape ) 
{ 
    return real_earth_pos * sim_shape.xyz() / real_shape.xyz(); 
}



void get_close_streamlines__mutithreaded_helper(  
    const Matrix& B, const Point* earth_pos, 
    double r_step, double angle_step, double streamline_step, 
    int max_length, double theta_start, 
    int start_index, int end_index,  
    std::vector<std::vector<Point>>& streamlines    )
{
    double earth_radius = 2;                             // TODO: FIX MAGIC NUMBER

    for (int i=start_index; i<end_index; i++)     
    {
        double theta = theta_start + angle_step*i;
        std::vector<Point> furthest_streamline;

        for (double r=2*earth_radius; ; r+=r_step)
        {
            double x = - r * std::cos(theta);
            double y = r * std::sin(theta);

            Point starting_point = Point(x, y, 0) + *earth_pos;

            std::vector<Point> current_streamline = find_streamline(B, starting_point, streamline_step, max_length, true, earth_pos);

            if ( current_streamline.empty() ) continue;

            if ( (current_streamline.front()-*earth_pos).norm() > earth_radius ) break;
            if ( (current_streamline.back()-*earth_pos).norm() > earth_radius ) break;

            furthest_streamline = current_streamline;
        }

        if ( furthest_streamline.empty() ) throw std::exception();

        streamlines[i] = furthest_streamline;
    }
}



std::vector<std::vector<Point>> get_close_streamlines( const Matrix& B, const Point* earth_pos, double r_step, double angle_step, double streamline_step, int max_length )
{
    int nb_threads = std::thread::hardware_concurrency();
    double theta_end = PI*0.66;                          // TODO: FIX MAGIC NUMBER
    double theta_start = -theta_end;

    int length = (int) ( (theta_end - theta_start) / angle_step );

    std::vector<std::vector<Point>> streamlines(length);
    std::thread t[nb_threads];

    for (int i=0; i<nb_threads; i++)
    {
        int start_index = (i*length)/nb_threads;
        int end_index = ((i+1)*length)/nb_threads;

        t[i] = std::thread(
            &get_close_streamlines__mutithreaded_helper, 
            std::ref(B), std::ref(earth_pos),
            r_step, angle_step, streamline_step,
            max_length, theta_start,
            start_index, end_index,
            std::ref(streamlines)
        );
    } 

    for (int i=0; i<nb_threads; i++) t[i].join();

    return streamlines;
}






