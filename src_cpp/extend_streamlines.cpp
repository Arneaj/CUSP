#include<iostream>
#include<cmath>
#include<vector>
#include<fstream>
#include<thread>

#include"../headers_cpp/streamlines.h"
#include"../headers_cpp/extend_streamlines.h"




std::vector<Point> extend_streamline(const Matrix& V, const std::vector<Point>& streamline, float step, int max_length)
{
    float starting_x = V.get_shape().x * 0.8;         // TODO: FIX MAGIC NUMBER
    std::vector<Point> extended_streamline;

    for (int i=(int) streamline.size()-1; i>=0; i--)
    {
        if ( streamline[i].x > starting_x ) break;
        extended_streamline.push_back(streamline[i]);
    }

    if ( (int) extended_streamline.size() == 0 ) for (int i=0; i<(int) streamline.size(); i++)
    {
        if ( streamline[i].x > starting_x ) break;
        extended_streamline.push_back(streamline[i]);
    }
    
    if ( (int) extended_streamline.size() == 0 ) throw std::exception();

    std::vector<Point> v_streamline = find_streamline(V, extended_streamline.back(), step, max_length, false);

    for (const Point& point : v_streamline) extended_streamline.push_back(point);

    return extended_streamline;
}



std::vector<std::vector<Point>> get_extended_streamlines(   const Matrix& B, const Matrix& V,
                                                            const std::vector<Point>& points, 
                                                            float step, int max_length,
                                                            const Point* earth_pos  )
{
    std::vector<std::vector<Point>> streamlines = get_streamlines(B, points, step, max_length, earth_pos);
    std::vector<std::vector<Point>> extended_streamlines;
    
    for (const std::vector<Point>& streamline: streamlines)
        try
        {
            extended_streamlines.push_back( extend_streamline(V, streamline, step, max_length) );
        }
        catch(const std::exception& e) {;}

    return extended_streamlines;
}




