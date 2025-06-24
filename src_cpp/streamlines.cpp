#include<iostream>
#include<cmath>
#include<vector>
#include<fstream>
#include<thread>

#include"../headers_cpp/streamlines.h"




float interpolate(Point P, const Matrix& B, int i)
{
    int xm = (int) (P.x);
    int ym = (int) (P.y);
    int zm = (int) (P.z);

    float xd = P.x - xm;
    float yd = P.y - ym;
    float zd = P.z - zm;

    return  ( B(xm,ym,zm,i)*(1-xd) + B(xm+1,ym,zm,i)*xd )*(1-yd)*(1-zd)
        +   ( B(xm,ym+1,zm,i)*(1-xd) + B(xm+1,ym+1,zm,i)*xd )*yd*(1-zd)
        +   ( B(xm,ym,zm+1,i)*(1-xd) + B(xm+1,ym,zm+1,i)*xd )*(1-yd)*zd
        +   ( B(xm,ym+1,zm+1,i)*(1-xd) + B(xm+1,ym+1,zm+1,i)*xd )*yd*zd;
}

Point norm_RK4(Point P_init, const Matrix& B, float step, float sign)
{
    Point P = P_init;

    if (P.x>=B.get_shape().x || P.x<0) throw std::exception();
    if (P.y>=B.get_shape().y || P.y<0) throw std::exception();
    if (P.z>=B.get_shape().z || P.z<0) throw std::exception();
    
    Point k1(
        interpolate(P, B, 0),
        interpolate(P, B, 1),
        interpolate(P, B, 2)
    );
    float k1_norm = k1.norm();
    k1 /= k1_norm * sign;

    P = P_init + 0.5*step*k1;

    if (P.x>=B.get_shape().x || P.x<0) throw std::exception();
    if (P.y>=B.get_shape().y || P.y<0) throw std::exception();
    if (P.z>=B.get_shape().z || P.z<0) throw std::exception();

    Point k2(
        interpolate(P, B, 0),
        interpolate(P, B, 1),
        interpolate(P, B, 2)
    );
    k2 /= k1_norm * sign;

    P = P_init + 0.5*step*k2;

    if (P.x>=B.get_shape().x || P.x<0) throw std::exception();
    if (P.y>=B.get_shape().y || P.y<0) throw std::exception();
    if (P.z>=B.get_shape().z || P.z<0) throw std::exception();

    Point k3(
        interpolate(P, B, 0),
        interpolate(P, B, 1),
        interpolate(P, B, 2)
    );
    k3 /= k1_norm * sign;

    P = P_init + step*k3;

    if (P.x>=B.get_shape().x || P.x<0) throw std::exception();
    if (P.y>=B.get_shape().y || P.y<0) throw std::exception();
    if (P.z>=B.get_shape().z || P.z<0) throw std::exception();

    Point k4(
        interpolate(P, B, 0),
        interpolate(P, B, 1),
        interpolate(P, B, 2)
    );
    k4 /= k1_norm * sign;

    Point final_step = step*(k1+2*k2+2*k3+k4)/6;

    P = P_init + final_step / final_step.norm();

    if (P.x>=B.get_shape().x || P.x<0) throw std::exception();
    if (P.y>=B.get_shape().y || P.y<0) throw std::exception();
    if (P.z>=B.get_shape().z || P.z<0) throw std::exception();

    return P;
}



std::vector<Point> find_streamline(const Matrix& B, Point P_i, float step, int max_length, bool both_sides, const Point* earth_pos)
{
    Point P(P_i);

    std::vector<Point> points;
    points.push_back(P);
    int t = 0;

    int earth_radius;
    if ( earth_pos ) earth_radius = 2;     // TODO : FIX MAGIC NUMBER

    while (t < max_length)
    {
        try { P = norm_RK4(P, B, step); }
        catch(const std::exception& e) { break; }
        
        points.push_back(P);

        if ( earth_pos && (P-*earth_pos).norm() <= earth_radius ) break;

        t++;
    }

    if ( !both_sides ) return points;

    t = 0;
    P = P_i;

    while (t < max_length)
    {
        try { P = norm_RK4(P, B, step, -1.0f); }
        catch(const std::exception& e) { break; }
        
        points.emplace(points.begin(), P);

        if ( earth_pos && (P-*earth_pos).norm() <= earth_radius ) break;

        t++;
    }

    return points;
}



std::vector<Point> smooth_interpolation(const std::vector<Point>& points, int kernel_radius)
{
    if ( (int) points.size() < kernel_radius ) return points;

    std::vector<Point> avg_points;

    for (int pos=kernel_radius; pos+kernel_radius < (int) points.size(); pos++)
    {
        Point avg_P;

        for (int i=pos-kernel_radius; i<=pos+kernel_radius; i++) { avg_P += points[i]; }

        avg_P /= 2*kernel_radius+1;

        avg_points.push_back(avg_P);
    }

    return avg_points;
}

std::vector<Point> find_smooth_streamline(const Matrix& B, Point P_i, float step, int kernel_radius, int max_length)
{
    return smooth_interpolation(
        find_streamline(B, P_i, step, max_length), 
        kernel_radius
    );
}






std::vector<std::vector<Point>> get_streamlines_singlethreaded(const Matrix& B, const std::vector<Point>& points, float step, int kernel_radius, int max_length)
{
    std::vector<std::vector<Point>> streamlines;

    for (int i=0; i<(int) points.size(); i++)
    {
        std::vector<Point> streamline = find_smooth_streamline(B, points[i], step, kernel_radius, max_length);

        streamlines.push_back(streamline);
    }

    return streamlines;
}



void get_streamlines__mutithreaded_helper(  const Matrix& B, const std::vector<Point>& points, 
                                            float step, int max_length, 
                                            int start_index, int end_index,
                                            std::vector<std::vector<Point>>& streamlines, const Point* earth_pos  )
{
    for (int i=start_index; i<end_index; i++)
    {
        std::vector<Point> streamline = find_streamline(B, points[i], step, max_length, true, earth_pos);

        streamlines[i] = streamline;
    }
}



std::vector<std::vector<Point>> get_streamlines(const Matrix& B, const std::vector<Point>& points, 
                                                float step, int max_length, const Point* earth_pos )
{
    int nb_threads = std::thread::hardware_concurrency();

    int nb_points = (int) points.size();

    std::vector<std::vector<Point>> streamlines(nb_points);

    std::thread t[nb_threads];

    for (int i=0; i<nb_threads; i++)
    {
        int start_index = (i*nb_points)/nb_threads;
        int end_index = ((i+1)*nb_points)/nb_threads;

        t[i] = std::thread(
            &get_streamlines__mutithreaded_helper, 
            std::ref(B), std::ref(points),
            step, max_length,
            start_index, end_index,
            std::ref(streamlines),
            earth_pos
        );
    } 

    for (int i=0; i<nb_threads; i++) t[i].join();

    return streamlines;
}





void save_streamlines(const std::string& filename, const std::vector<std::vector<Point>>& streamlines)
{
    std::ofstream fs;
    fs.open(filename);

    for (int i=0; i<(int) streamlines.size(); i++)
    {
        std::vector<Point> streamline = streamlines[i];

        for (int j=0; j<(int) streamline.size(); j++) 
        {
            fs << std::to_string( streamline[j].x ).c_str();
            fs << ",";

            fs << std::to_string( streamline[j].y ).c_str();
            fs << ",";

            fs << std::to_string( streamline[j].z ).c_str();
            if (j < (int) streamline.size()-1) fs << ",";
        }

        fs << "\n";
    }

    fs.close();
}














