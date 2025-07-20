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

    std::string what() { return error_message; }
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

    Shape(std::vector<int> sh) : Shape(sh[0],sh[1],sh[2],sh[3]) {;}

    Point xyz() { return Point(x, y, z); }

    friend std::ostream& operator<<(std::ostream& os, Shape sh);
};


class Matrix
{
private:
    Shape shape;
    float* mat;
    int i_offset;

public:
    Matrix(): shape(), mat(new float[ shape.x * shape.y * shape.z * shape.i]), i_offset(0) {;}
    Matrix(Shape sh): shape(sh), mat(new float[ shape.x * shape.y * shape.z * shape.i]), i_offset(0) {;}
    Matrix(Shape sh, float* m): shape(sh), mat(m), i_offset(0) {;}
    Matrix(Shape sh, float* m, int _i_offset): shape(sh), mat(m), i_offset(_i_offset) {;}

    void del() { delete[] mat; }

    Shape get_shape() { return shape; }
    Shape get_shape() const { return shape; }

    void flatten();

    Point index_max() const;
    Point index_max(int i) const;

    float max() const;
    float min() const;

    float max(int i) const;
    float min(int i) const;

    Matrix norm() const;

    float& operator()(int ix, int iy, int iz, int i);
    const float& operator()(int ix, int iy, int iz, int i) const;

    Matrix operator()(int i);

    float& operator[](int id);
    const float& operator[](int id) const;

    void operator+=(float val);
    void operator-=(float val);
    void operator*=(float val);
    void operator/=(float val);
};


#endif
