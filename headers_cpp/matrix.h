#ifndef MATRIX_H
#define MATRIX_H

#include <vector>
#include <string>
#include "points.h"


class exception_OOB: public std::exception
{
private:
    std::string error_message;

public:
    exception_OOB(std::string _e): error_message(_e) {;}

    virtual const char* what() const noexcept override { return error_message.c_str(); }
};


class Shape
{
public:
    int x;
    int y;
    int z;
    int i;

    Shape() : x(0), y(0), z(0), i(0) {;}

    Shape(int _x, int _y, int _z, int _i) : x(_x), y(_y), z(_z), i(_i) {;}

    Shape(const std::vector<int>& sh) : Shape(sh[0],sh[1],sh[2],sh[3]) {;}

    Point xyz() { return Point(x, y, z); }

    friend std::ostream& operator<<(std::ostream& os, const Shape& sh);
};


/// @brief  A 3D voxel grid of any size with any number of components per voxel. 
///         Fortran-style indexing if manually indexing through array Matrix::mat.
class Matrix
{
private:
    Shape shape;
    Shape strides;
    double* mat;
    int i_offset;

public:
    Matrix(): shape(), strides(), mat(nullptr), i_offset(0) {;}
    Matrix(Shape sh): shape(sh), strides(1, sh.x, sh.x*sh.y, sh.x*sh.y*sh.z), mat(new double[shape.x * shape.y * shape.z * shape.i]), i_offset(0) 
    {
        if (mat == nullptr) { std::cout << "ERROR: out of memory when allocating interest radii candidates.\n"; exit(1); };
    }
    Matrix(Shape sh, double* m): shape(sh), strides(1, sh.x, sh.x*sh.y, sh.x*sh.y*sh.z), mat(m), i_offset(0) {;}
    Matrix(Shape sh, Shape stride, double* m): shape(sh), strides(stride), mat(m), i_offset(0) {;}
    Matrix(Shape sh, double* m, int _i_offset): shape(sh), strides(1, sh.x, sh.x*sh.y, sh.x*sh.y*sh.z), mat(m), i_offset(_i_offset) {;}

    void del() { delete[] mat; }

    Shape get_shape() { return shape; }
    Shape get_shape() const { return shape; }

    Shape get_strides() { return strides; }
    Shape get_strides() const { return strides; }

    double* get_array() { return mat; }

    void flatten();

    Point index_max() const;
    Point index_max(int i) const;

    double max() const;
    double min() const;

    double max(int i) const;
    double min(int i) const;

    Matrix norm() const;

    bool is_point_OOB(const Point& p) const;


    double& operator()(int ix, int iy, int iz, int i);
    const double& operator()(int ix, int iy, int iz, int i) const;

    double operator()(const Point& p, int i) const;
    Point operator()(const Point& p) const;

    Matrix operator()(int i);

    double& operator[](int id);
    const double& operator[](int id) const;

    void operator+=(double val);
    void operator-=(double val);
    void operator*=(double val);
    void operator/=(double val);
};


#endif
