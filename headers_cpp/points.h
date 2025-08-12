#ifndef POINTS_H
#define POINTS_H

#include<cmath>
#include<iostream>

class Point
{
public:
    double x;
    double y;
    double z;

    bool is_null;

    Point(): x(0.0), y(0.0), z(0.0), is_null(false) {;}
    Point(double _x, double _y, double _z): x(_x), y(_y), z(_z), is_null(false) {;}
    // Point(const Point& _p): x(_p.x), y(_p.y), z(_p.z) {;}

    bool operator==(const Point& p) { return x==p.x && y==p.y && z==p.z && is_null==p.is_null; }

    Point operator+(const Point& p);
    Point operator+(const Point& p) const;
    Point operator+=(const Point& p);

    Point operator-(const Point& p);
    Point operator-(const Point& p) const;
    Point operator-=(const Point& p);

    Point operator/(double v);
    Point operator/=(double v);

    Point operator*=(double v);

    Point operator*(const Point& p);
    Point operator/(const Point& p);

    Point operator*(double v);
    friend Point operator*(double v, const Point& p);

    friend std::ostream& operator<<(std::ostream& io, const Point& point) 
    { 
        io << '(' << point.x << ", " << point.y << ", " << point.z << ')';
        return io;
    }

    double norm();

    void set_null() { is_null = true; }
};





#endif
