#ifndef POINTS_H
#define POINTS_H

#include<cmath>
#include<iostream>

class Point
{
public:
    float x;
    float y;
    float z;

    bool is_null;

    Point(): x(0.0f), y(0.0f), z(0.0f), is_null(false) {;}
    Point(float _x, float _y, float _z): x(_x), y(_y), z(_z), is_null(false) {;}
    // Point(const Point& _p): x(_p.x), y(_p.y), z(_p.z) {;}

    Point operator+(const Point& p);
    Point operator+(const Point& p) const;
    Point operator+=(const Point& p);

    Point operator-(const Point& p);
    Point operator-(const Point& p) const;
    Point operator-=(const Point& p);

    Point operator/(float v);
    Point operator/=(float v);

    Point operator*=(float v);

    Point operator*(const Point& p);
    Point operator/(const Point& p);

    Point operator*(float v);
    friend Point operator*(float v, const Point& p);

    friend std::ostream& operator<<(std::ostream& io, const Point& point) 
    { 
        io << '(' << point.x << ", " << point.y << ", " << point.z << ')';
        return io;
    }

    float norm();

    void set_null() { is_null = true; }
};





#endif
