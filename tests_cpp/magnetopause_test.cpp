#include<iostream>

#include "../headers_cpp/read_file.h"
#include "../headers_cpp/streamlines.h"
#include "../headers_cpp/extend_streamlines.h"
#include "../headers_cpp/magnetopause.h"


#include <chrono>

typedef std::chrono::high_resolution_clock Time;
typedef std::chrono::duration<double> fsec;


int main()
{
    auto t0 = Time::now();

    Matrix B = read_file("../data/Run1_28800/B_processed.txt", 1);
    Matrix V = read_file("../data/Run1_28800/V_processed.txt", 1);

    auto t1 = Time::now();

    std::cout << "File reading done. Time taken: " << fsec((t1-t0)).count() << 's' << std::endl;


    std::vector<Point> points;

    double STEP = 0.1;

    for (double ix=B.get_shape().x-1-5; ix<B.get_shape().x-1; ix++) for (double iy=1; iy<B.get_shape().y-1; iy += STEP)
    {
        points.push_back( Point(ix, iy, 1) );
        points.push_back( Point(ix, iy, B.get_shape().z-2) );
    }

    t0 = Time::now();

    Point earth_pos = find_earth_pos(B);

    t1 = Time::now();

    std::cout << "Earth position search done. Time taken: " << fsec((t1-t0)).count() << 's' << std::endl;



    t0 = Time::now();

    std::vector<std::vector<Point>> extended_streamlines = get_extended_streamlines(B, V, points, 0.1f, 1000, &earth_pos);

    t1 = Time::now();

    std::cout << "Night streamlines search done. Time taken: " << fsec((t1-t0)).count() << 's' << std::endl;


    t0 = Time::now();

    remove_outliers( extended_streamlines, B.get_shape() );

    t1 = Time::now();

    std::cout << "Night streamlines extension done. Time taken: " << fsec((t1-t0)).count() << 's' << std::endl;



    t0 = Time::now();

    std::vector<std::vector<Point>> front_streamlines = get_close_streamlines(B, &earth_pos, 0.1f, 0.0033f, 0.1f, 1000);

    t1 = Time::now();

    std::cout << "Day streamlines search done. Time taken: " << fsec((t1-t0)).count() << 's' << std::endl;



    for (const std::vector<Point>& streamline: front_streamlines) extended_streamlines.push_back(streamline);



    t0 = Time::now();

    save_streamlines("../data/Run1_28800/streamlines.txt", extended_streamlines);

    t1 = Time::now();

    std::cout << "File saving done. Time taken: " << fsec((t1-t0)).count() << 's' << std::endl;





    B.del();
    V.del();

    return 0;
}

