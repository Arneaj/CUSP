#include<iostream>

#include<vector>

#include "../headers_cpp/points.h"
#include "../headers_cpp/matrix.h"



void Matrix::flatten()
{
    shape.x = shape.x * shape.y * shape.z * shape.i;
    shape.y = 1;
    shape.z = 1;
    shape.i = 1;
}


bool Matrix::is_point_OOB(const Point& p) const
{
    return  p.x>=shape.x-1 || p.x<0 ||
            p.y>=shape.y-1 || p.y<0 ||
            p.z>=shape.z-1 || p.z<0 ;
}


double Matrix::operator()(const Point& p, int i) const
{
    // if ( is_point_OOB(p) ) throw exception_OOB("Point is out of bounds!");

    int xm = (int) (p.x);
    int ym = (int) (p.y);
    int zm = (int) (p.z);

    double xd = p.x - xm;
    double yd = p.y - ym;
    double zd = p.z - zm;

    return  ( (*this)(xm,ym,zm,i)*(1-xd) + (*this)(xm+1,ym,zm,i)*xd )*(1-yd)*(1-zd)
        +   ( (*this)(xm,ym+1,zm,i)*(1-xd) + (*this)(xm+1,ym+1,zm,i)*xd )*yd*(1-zd)
        +   ( (*this)(xm,ym,zm+1,i)*(1-xd) + (*this)(xm+1,ym,zm+1,i)*xd )*(1-yd)*zd
        +   ( (*this)(xm,ym+1,zm+1,i)*(1-xd) + (*this)(xm+1,ym+1,zm+1,i)*xd )*yd*zd;
}


Point Matrix::operator()(const Point& p) const
{
    return Point( (*this)(p,0), (*this)(p,1), (*this)(p,2) );
}









double& Matrix::operator()(int ix, int iy, int iz, int i)
{
    if (ix<0 || ix>=shape.x) { throw exception_OOB("x out of range"); }
    if (iy<0 || iy>=shape.y) { throw exception_OOB("y out of range"); }
    if (iz<0 || iz>=shape.z) { throw exception_OOB("z out of range"); }
    if (i<0 || i>=shape.i) { throw exception_OOB("i out of range"); }

    return mat[ ix + iy*shape.x + iz*shape.x*shape.y + (i+i_offset)*shape.x*shape.y*shape.z ];
    // return mat[ ix*shape.y*shape.z*shape.i + iy*shape.z*shape.i + iz*shape.i + (i+i_offset) ];
}

const double& Matrix::operator()(int ix, int iy, int iz, int i) const
{
    if (ix<0 || ix>=shape.x) { throw exception_OOB("x out of range"); }
    if (iy<0 || iy>=shape.y) { throw exception_OOB("y out of range"); }
    if (iz<0 || iz>=shape.z) { throw exception_OOB("z out of range"); }
    if (i<0 || i>=shape.i) { throw exception_OOB("i out of range"); }

    return mat[ ix + iy*shape.x + iz*shape.x*shape.y + (i+i_offset)*shape.x*shape.y*shape.z ];
    // return mat[ ix*shape.y*shape.z*shape.i + iy*shape.z*shape.i + iz*shape.i + (i+i_offset) ];
}





double& Matrix::operator[](int id)
{
    if (id<0 || id >= shape.x*shape.y*shape.z*shape.i) { throw exception_OOB("index out of range"); }

    return mat[ id ];
}

const double& Matrix::operator[](int id) const
{
    if (id<0 || id >= shape.x*shape.y*shape.z*shape.i) { throw exception_OOB("index out of range"); }

    return mat[ id ];
}




Matrix Matrix::operator()(int i)
{
    if (i<0 || i>=shape.i) { std::cout << "i out of range\n"; throw std::exception(); }

    Shape new_shape(shape.x, shape.y, shape.z, 1);
    Matrix matrix(new_shape, mat, i);

    return matrix;
}




Point Matrix::index_max() const
{
    double max_norm = 0;
    Point max_index;

    for (int ix=0; ix<shape.x; ix++) for (int iy=0; iy<shape.y; iy++) for (int iz=0; iz<shape.z; iz++)
    {
        double norm = Point( (*this)(ix,iy,iz,0), (*this)(ix,iy,iz,1), (*this)(ix,iy,iz,2) ).norm();

        if ( norm > max_norm ) 
        {
            max_norm = norm;
            max_index = Point(ix,iy,iz);
        }
    }

    return max_index;
}

Point Matrix::index_max(int i) const
{
    double max_val = 0;
    Point max_index;

    for (int ix=0; ix<shape.x; ix++) for (int iy=0; iy<shape.y; iy++) for (int iz=0; iz<shape.z; iz++)
    {
        double val = (*this)(ix,iy,iz,i);

        if ( val > max_val ) 
        {
            max_val = val;
            max_index = Point(ix,iy,iz);
        }
    }

    return max_index;
}


double Matrix::max() const
{
    double max_norm = -INFINITY;

    for (int ix=0; ix<shape.x; ix++) for (int iy=0; iy<shape.y; iy++) for (int iz=0; iz<shape.z; iz++)
    {
        double norm = Point( (*this)(ix,iy,iz,0), (*this)(ix,iy,iz,1), (*this)(ix,iy,iz,2) ).norm();

        if ( norm > max_norm ) max_norm = norm;
    }

    return max_norm;
}

double Matrix::min() const
{
    double min_norm = INFINITY;

    for (int ix=0; ix<shape.x; ix++) for (int iy=0; iy<shape.y; iy++) for (int iz=0; iz<shape.z; iz++)
    {
        double norm = Point( (*this)(ix,iy,iz,0), (*this)(ix,iy,iz,1), (*this)(ix,iy,iz,2) ).norm();

        if ( norm < min_norm ) min_norm = norm;
    }

    return min_norm;
}


double Matrix::max(int i) const
{
    double max_norm = -INFINITY;

    for (int ix=0; ix<shape.x; ix++) for (int iy=0; iy<shape.y; iy++) for (int iz=0; iz<shape.z; iz++)
    {
        double norm = (*this)(ix,iy,iz,i);

        if ( norm > max_norm ) max_norm = norm;
    }

    return max_norm;
}

double Matrix::min(int i) const
{
    double min_norm = INFINITY;

    for (int ix=0; ix<shape.x; ix++) for (int iy=0; iy<shape.y; iy++) for (int iz=0; iz<shape.z; iz++)
    {
        double norm = (*this)(ix,iy,iz,i);

        if ( norm < min_norm ) min_norm = norm;
    }

    return min_norm;
}





Matrix Matrix::norm() const
{
    Matrix normed_mat( Shape( shape.x, shape.y, shape.z, 1 ) );

    for (int ix=0; ix<shape.x; ix++) for (int iy=0; iy<shape.y; iy++) for (int iz=0; iz<shape.z; iz++)
        normed_mat(ix, iy, iz, 0) = Point( (*this)(ix,iy,iz,0), (*this)(ix,iy,iz,1), (*this)(ix,iy,iz,2) ).norm();

    return normed_mat;
}





void Matrix::operator+=(double val)
{
    for (int ix=0; ix<shape.x; ix++) for (int iy=0; iy<shape.y; iy++) for (int iz=0; iz<shape.z; iz++)
        for (int i=0; i<shape.i; i++)
            (*this)(ix,iy,iz,i) += val; 
}


void Matrix::operator-=(double val)
{
    for (int ix=0; ix<shape.x; ix++) for (int iy=0; iy<shape.y; iy++) for (int iz=0; iz<shape.z; iz++)
        for (int i=0; i<shape.i; i++)
            (*this)(ix,iy,iz,i) -= val; 
}


void Matrix::operator*=(double val)
{
    for (int ix=0; ix<shape.x; ix++) for (int iy=0; iy<shape.y; iy++) for (int iz=0; iz<shape.z; iz++)
        for (int i=0; i<shape.i; i++)
            (*this)(ix,iy,iz,i) *= val; 
}


void Matrix::operator/=(double val)
{
    for (int ix=0; ix<shape.x; ix++) for (int iy=0; iy<shape.y; iy++) for (int iz=0; iz<shape.z; iz++)
        for (int i=0; i<shape.i; i++)
            (*this)(ix,iy,iz,i) /= val; 
}




std::ostream& operator<<(std::ostream& os, const Shape& sh)
{
    os << "(" << sh.x << ", " 
              << sh.y << ", " 
              << sh.z << ", "
              << sh.i << ")";

    return os;
}