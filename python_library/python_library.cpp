#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "../headers_cpp/points.h"
#include "../headers_cpp/matrix.h"
#include "../headers_cpp/streamlines.h"
#include "../headers_cpp/magnetopause.h"
#include "../headers_cpp/read_file.h"
// #include "../headers_cpp/read_pvtr.h"
// #include "../headers_cpp/reader_writer.h"
#include "../headers_cpp/preprocessing.h"
#include "../headers_cpp/raycast.h"
// #include "../headers_cpp/fit_to_analytical.h"
#include "../headers_cpp/analysis.h"


pybind11::array_t<float> array_from_matrix( Matrix& matrix )
{
    const Shape& sh = matrix.get_shape();

    return pybind11::array_t<float>(
        {sh.x, sh.y, sh.z, sh.i},                                                                       // shape
        {sizeof(float), sizeof(float)*sh.x, sizeof(float)*sh.x*sh.y, sizeof(float)*sh.x*sh.y*sh.z},     // strides
        matrix.get_array(),                                                                             // data pointer
        pybind11::cast(matrix)                                                                          // parent object (keeps data alive)
    );
}


pybind11::array_t<float> array_from_point_vec( std::vector<Point> points )
{
    int length = points.size();
    float* arr = new float[length*3];

    for (int i=0; i<length; i++)
    {
        arr[3*i] = points[i].x;
        arr[3*i+1] = points[i].y;
        arr[3*i+2] = points[i].z;
    }

    return pybind11::array_t<float>(
        {length, 3},                        // shape
        {sizeof(float)*3, sizeof(float)},   // strides
        arr,                                // data pointer
        pybind11::cast(arr)                 // parent object (keeps data alive)
    );
}

Matrix matrix_from_array( pybind11::array_t<float> arr )
{
    auto buf = arr.unchecked<4>();
    Shape sh( buf.shape(0), buf.shape(1), buf.shape(2), buf.shape(3) );

    int total_size = buf.shape(0)*buf.shape(1)*buf.shape(2)*buf.shape(3);

    float* mat = new float[total_size];
    std::memcpy( mat, arr.data(), sizeof(float)*total_size );

    return Matrix( sh, mat );
}


float get_bowshock_radius_numpy(  float theta, float phi,
                            const pybind11::array_t<float>& Rho, const Point& earth_pos,
                            float dr )
{
    Matrix rho = matrix_from_array( Rho );
    return get_bowshock_radius(theta, phi, rho, earth_pos, dr);
}

pybind11::array_t<float> get_bowshock_numpy( const pybind11::array_t<float>& Rho, const Point& earth_pos, float dr, int nb_phi, int max_nb_theta )
{
    Matrix rho = matrix_from_array( Rho );
    return array_from_point_vec( get_bowshock(rho, earth_pos, dr, nb_phi, max_nb_theta) );
}





PYBIND11_MODULE(topology_analysis, m)
{
    m.doc() = "Topology analysis module for magnetic field data";
    
    // Bind classes first
    pybind11::class_<Point>(m, "Point")
        .def(pybind11::init<float, float, float>())
        .def_readwrite("x", &Point::x)
        .def_readwrite("y", &Point::y)
        .def_readwrite("z", &Point::z);

    m.def("get_bowshock_radius", &get_bowshock_radius_numpy);
    m.def("get_bowshock", &get_bowshock_numpy);
}

