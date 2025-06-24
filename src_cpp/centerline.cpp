#include<iostream>

#include"../headers_cpp/centerline.h"






inline int x_to_index( float x, float x_min, float step ) { return (int) ( (x-x_min) / step ); }




void fill_in_nulls( Point_array points, int length ) 
{
    for (int i=0; i<length; i++)
    {
        if ( !points.get()[i].is_null ) continue;

        int j_front = 1;
        bool front_in = false;

        while( true )
        {
            if ( i+j_front >= length ) break;
            if ( !points.get()[i+j_front].is_null ) { front_in = true; break; }
            j_front++;
        }

        bool back_in = i > 0;

        if ( back_in && front_in ) { points.get()[i] = (points.get()[i-1]*j_front + points.get()[i+j_front]) / (1+j_front); continue; }
        if ( back_in ) { points.get()[i] = points.get()[i-1]; continue; }
        if ( front_in ) { points.get()[i] = points.get()[i+j_front]; continue; }
        break;
        //throw std::exception();
    }
}




Point_array get_interpolated_streamline(    const std::vector<Point>& streamline, 
                                            int length, 
                                            float step, float x_min, float x_max    )
{
    float hits[length];
    Point_array interpolated_streamline = Point_array( new Point[length] );
    for (int i=0; i<length; i++)
    {
        interpolated_streamline.get()[i] = Point();
        hits[i] = 0;
    }

    float X[length];
    X[0] = x_min;
    for (int i=1; i<length; i++) X[i] = X[i-1] + step;

    for (const Point& point: streamline)
    {
        int index = x_to_index( point.x, x_min, step );
        if (index<0 || index>=length) continue;

        interpolated_streamline.get()[index] += Point( X[index], point.y, point.z );
        hits[index]++;
    }

    for (int i=0; i<length; i++) 
    {
        if (hits[i] == 0) 
        {
            interpolated_streamline.get()[i].set_null(); 
            continue;
        }

        interpolated_streamline.get()[i] /= hits[i];
    }

    // fill_in_nulls(interpolated_streamline, length);   // TODO: COULD BE A BAD IDEA

    return interpolated_streamline;
}




Point_array get_centerline( std::shared_ptr<Point_array[]> interpolated_streamlines, int length, int nb_streamlines )
{
    Point_array centerline = Point_array(new Point[length]);
    for (int i=0; i<length; i++) centerline.get()[i] = Point();

    for (int ix=0; ix<length; ix++) 
    {
        int hits = 0;
        for (int is=0; is<nb_streamlines; is++)
        {
            if ( interpolated_streamlines.get()[is].get()[ix].is_null ) continue;

            centerline.get()[ix] += interpolated_streamlines.get()[is].get()[ix];
            hits++;
        } 

        if ( hits == 0 ) { centerline.get()[ix].set_null(); continue; }

        centerline.get()[ix] /= hits;
    }

    fill_in_nulls(centerline, length);

    return centerline;
}




// Point_array get_multilayered_centerline(    const std::vector<std::vector<Point>>& streamlines, 
//                                             float step, float x_min, float x_max,
//                                             int nb_sl_min, int nb_sl_max, int nb_layers    )
// {
//     ;
// }





// Point_array thing(  const std::vector<std::vector<Point>>& streamlines, 
//                     float step, float x_min, float x_max    )
// {
//     int length = x_to_index(x_max, x_min, step);

    


// }





