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




struct DataHeader {
    uint32_t magic_number = 0x12345678;  // File validity check
    uint32_t type_size;                  // sizeof(float) or sizeof(double)
    uint32_t x_dim;
    uint32_t y_dim;
    uint32_t z_dim;
    uint32_t i_dim;
    uint32_t separator[4] = {0};

    DataHeader( uint32_t _type_size, uint32_t _x_dim, uint32_t _y_dim, uint32_t _z_dim, uint32_t _i_dim )
        : type_size(_type_size), x_dim(_x_dim), y_dim(_y_dim), z_dim(_z_dim), i_dim(_i_dim) {;}

    DataHeader( uint32_t _type_size, const Shape& shape )
        : type_size(_type_size), x_dim(shape.x), y_dim(shape.y), z_dim(shape.z), i_dim(shape.i) {;}
};



void save_file_bin( std::string filename, Matrix& mat )
{
    std::ofstream fs;
    fs.open(filename, std::ios::binary);

    DataHeader header(sizeof(float), mat.get_shape());

    fs.write(reinterpret_cast<const char*>(&header), sizeof(header));
    fs.write(reinterpret_cast<const char*>(mat.get_array()), mat.get_shape().x*mat.get_shape().y*mat.get_shape().z*mat.get_shape().i * sizeof(float));

    fs.close();
}


