#include<iostream>
#include"../headers_cpp/points.h"




Point Point::operator+(const Point p)
{
    return Point(
        x + p.x,
        y + p.y,
        z + p.z
    );  
}

Point Point::operator+=(Point p)
{
    x += p.x;
    y += p.y;
    z += p.z;

    return *this;
}



Point Point::operator-(Point p)
{
    return Point(
        x - p.x,
        y - p.y,
        z - p.z
    );
}

Point Point::operator-=(Point p)
{
    x -= p.x;
    y -= p.y;
    z -= p.z;

    return *this;
}



Point Point::operator/(float v)
{
    return Point(
        x / v,
        y / v,
        z / v
    );
}

Point Point::operator/=(float v)
{
    x /= v;
    y /= v;
    z /= v;

    return *this;
}



Point Point::operator*(float v)
{
    return Point(
        x * v,
        y * v,
        z * v
    );
}

Point operator*(float v, Point p)
{
    return Point(
        p.x * v,
        p.y * v,
        p.z * v
    );
}



float Point::norm()
{
    return std::sqrt(x*x+y*y+z*z);
}




