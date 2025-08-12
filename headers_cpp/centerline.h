#ifndef CENTER_LINE_H
#define CENTER_LINE_H

#include<memory>

#include "streamlines.h"

#define Point_array std::shared_ptr<Point[]>


int x_to_index( double x, double x_min, double step );

void fill_in_nulls( Point_array points, int length );




Point_array get_interpolated_streamline(    const std::vector<Point>& streamline, 
                                            int length, 
                                            double step, double x_min, double x_max    );

Point_array get_centerline( std::shared_ptr<Point_array[]> interpolated_streamlines, int length, int nb_streamlines );


Point_array get_multilayered_centerline(    const std::vector<std::vector<Point>>& streamlines, 
                                            double step, double x_min, double x_max,
                                            int nb_sl_min, int nb_sl_max, int nb_layers    );


                                            
Point_array thing(  const std::vector<std::vector<Point>>& streamlines, 
                    double step, double x_min, double x_max    );



#endif
