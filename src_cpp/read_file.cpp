#include<iostream>
#include<fstream>

#include<vector>

#include "../headers_cpp/matrix.h"
#include "../headers_cpp/read_file.h"


Shape read_shape( std::ifstream& fs, int step )
{
    std::vector<int> sh;
    std::string s;

    while ( !fs.eof() )
    {
        s = "";
        while ( fs.peek() != ',' && fs.peek() != '\n' )
        {
            s.push_back( fs.get() );
        }
        sh.push_back( std::stoi(s) );

        if (fs.peek() == '\n') break;

        fs.get();
    }

    fs.get();

    Shape shape(sh);

    shape.x /= step;
    shape.y /= step;
    shape.z /= step;

    return shape;
}


Matrix read_matrix( std::ifstream& fs, Shape shape, int step )
{
    Matrix mat(shape);
    std::string s;

    for (int ix=0; ix<shape.x*step; ix++) for (int iy=0; iy<shape.y*step; iy++) for (int iz=0; iz<shape.z*step; iz++)           
        for (int i=0; i<shape.i; i++)
        {
            s = "";
            while ( fs.peek() != ',' )
            {
                s.push_back( fs.get() );
            }

            fs.get();

            if ( ix % step != 0 ) continue; 
            if ( iy % step != 0 ) continue;
            if ( iz % step != 0 ) continue;
            
            mat(ix/step,iy/step,iz/step,i) = std::stof(s);
        }

    return mat;
}


Matrix read_file( std::string filename, int step )
{
    std::ifstream fs;
    fs.open(filename);

    Shape shape = read_shape( fs, step );
    Matrix matrix = read_matrix( fs, shape, step );

    fs.close();

    return matrix;
}


void save_file( std::string filename, const Matrix& mat )
{
    std::ofstream fs;
    fs.open(filename);

    fs  << mat.get_shape().x << ','
        << mat.get_shape().y << ','
        << mat.get_shape().z << ',' 
        << mat.get_shape().i << '\n';

    for (int ix=0; ix<mat.get_shape().x; ix++) for (int iy=0; iy<mat.get_shape().y; iy++) 
        for (int iz=0; iz<mat.get_shape().z; iz++) for (int i=0; i<mat.get_shape().i; i++)
            fs << mat(ix,iy,iz,i) << ',';

    fs << '\n';

    fs.close();
}



