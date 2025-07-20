#ifndef MAGNETOPAUSE_H
#define MAGNETOPAUSE_H



void remove_outliers( std::vector<std::vector<Point>>& streamlines, Shape shape );

Point find_earth_pos( const Matrix& B );

std::vector<std::vector<Point>> get_close_streamlines( const Matrix& B, const Point* earth_pos, float r_step=0.1f, float angle_step=0.1f, float streamline_step=0.5f, int max_length=1000 );


Point find_real_earth_pos( const Matrix& X, const Matrix& Y, const Matrix& Z );

Point find_sim_earth_pos( Point real_earth_pos, Shape real_shape, Shape sim_shape );


#endif
