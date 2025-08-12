#include<iostream>
#include<chrono>

#include "../headers_cpp/read_file.h"
#include "../headers_cpp/streamlines.h"



int main()
{
    Matrix B = read_file("../data/Run1_28800/B.txt");


    std::vector<Point> points;

    double STEP = 0.2;

    // for (double ix=1; ix<B.get_shape().x-1; ix += STEP) for (double iy=1; iy<B.get_shape().y-1; iy += STEP)
    for (double ix=B.get_shape().x-1-STEP*5; ix<B.get_shape().x-1; ix += STEP) for (double iy=1; iy<B.get_shape().y-1; iy += STEP)
    {
        points.push_back( Point(ix, iy, 1) );
        points.push_back( Point(ix, iy, B.get_shape().z-2) );
    }

    typedef std::chrono::high_resolution_clock Time;
    typedef std::chrono::duration<double> fsec;

    const int NB_RUNS = 10;






    auto t0 = Time::now(); // time before solving

	for (int i=0; i<NB_RUNS; i++) 
        std::vector<std::vector<Point>> streamlines = get_streamlines_singlethreaded(B, points, 1.0f, 0, 1000);

    auto t1 = Time::now(); // time after solving

    fsec duration_in_sec = (t1 - t0)/NB_RUNS;
    std::cout << "Average time taken for regular function: " << duration_in_sec.count() << std::endl;






    t0 = Time::now(); // time before solving

	for (int i=0; i<NB_RUNS; i++) 
        std::vector<std::vector<Point>> streamlines = get_streamlines(B, points, 1.0f, 1000);

    t1 = Time::now(); // time after solving

    duration_in_sec = (t1 - t0)/NB_RUNS;
    std::cout << "Average time taken for multithreaded function: " << duration_in_sec.count() << std::endl;


    

    

    B.del();

    return 0;
}
