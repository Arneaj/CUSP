#include<iostream>
#include"../headers_cpp/points.h"




Point Point::operator+(const Point& p)
{
    return Point(
        x + p.x,
        y + p.y,
        z + p.z
    );  
}

Point Point::operator+(const Point& p) const
{
    return Point(
        x + p.x,
        y + p.y,
        z + p.z
    );  
}

Point Point::operator+=(const Point& p)
{
    x += p.x;
    y += p.y;
    z += p.z;

    return *this;
}



Point Point::operator-(const Point& p)
{
    return Point(
        x - p.x,
        y - p.y,
        z - p.z
    );
}

Point Point::operator-(const Point& p) const
{
    return Point(
        x - p.x,
        y - p.y,
        z - p.z
    );
}

Point Point::operator-=(const Point& p)
{
    x -= p.x;
    y -= p.y;
    z -= p.z;

    return *this;
}



Point Point::operator/(double v)
{
    return Point(
        x / v,
        y / v,
        z / v
    );
}

Point Point::operator/=(double v)
{
    x /= v;
    y /= v;
    z /= v;

    return *this;
}

Point Point::operator*=(double v)
{
    x *= v;
    y *= v;
    z *= v;

    return *this;
}



Point Point::operator*(double v)
{
    return Point(
        x * v,
        y * v,
        z * v
    );
}

Point Point::operator*(const Point& p)
{
    return Point(
        x * p.x,
        y * p.y,
        z * p.z
    );
}

Point Point::operator/(const Point& p)
{
    return Point(
        x / p.x,
        y / p.y,
        z / p.z
    );
}


Point operator*(double v, const Point& p)
{
    return Point(
        p.x * v,
        p.y * v,
        p.z * v
    );
}



double Point::norm()
{
    return std::sqrt(x*x+y*y+z*z);
}




