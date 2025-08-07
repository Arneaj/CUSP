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


namespace casters
{
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

    pybind11::array_t<float> array_from_interest_point_vec( InterestPoint* interest_points, int nb_interest_points )
    {
        float* arr = new float[nb_interest_points*4];

        for (int i=0; i<nb_interest_points; i++)
        {
            arr[4*i] = interest_points[i].theta;
            arr[4*i+1] = interest_points[i].phi;
            arr[4*i+2] = interest_points[i].radius;
            arr[4*i+3] = interest_points[i].radius;
        }

        delete[] interest_points;

        return pybind11::array_t<float>(
            {nb_interest_points, 4},            // shape
            {sizeof(float)*4, sizeof(float)},   // strides
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

    std::vector<InterestPoint> ip_vec_from_array( pybind11::array_t<float> arr )
    {
        std::vector<InterestPoint> ip(arr.shape(0));
        
        for (int i=0; i<arr.shape(0); i++)
            ip[i] = { *arr.data(4*i), *arr.data(4*i+1), *arr.data(4*i+2), *arr.data(4*i+3) };

        return ip;
    }

    std::vector<Point> point_vec_from_array( pybind11::array_t<float> arr )
    {
        std::vector<Point> points(arr.shape(0));
        
        for (int i=0; i<arr.shape(0); i++)
            points[i] = { *arr.data(3*i), *arr.data(3*i+1), *arr.data(3*i+2) };

        return points;
    }

    Point point_from_array( pybind11::array_t<float> point )
    {
        pybind11::ssize_t nb_dims = point.ndim();
        if ( nb_dims > 1 || point.shape(0) != 3 )
        {
            throw pybind11::index_error("Point needs to be an array of shape (3)");
        }

        return Point( *point.data(0), *point.data(1), *point.data(2) );
    }

    Shape shape_from_array( pybind11::array_t<float> shape )
    {
        pybind11::ssize_t nb_dims = shape.ndim();
        if ( nb_dims > 1 || shape.shape(0) != 4 )
        {
            throw pybind11::index_error("Point needs to be an array of shape (3)");
        }

        return Shape( *shape.data(0), *shape.data(1), *shape.data(2), *shape.data(3) );
    }
}


namespace raycasting
{
    float get_bowshock_radius_numpy(  
        float theta, float phi,
        const pybind11::array_t<float>& Rho, const pybind11::array_t<float>& earth_pos,
        float dr )
    {
        Matrix rho = casters::matrix_from_array( Rho );
        Point _earth_pos = casters::point_from_array( earth_pos );

        return get_bowshock_radius(theta, phi, rho, _earth_pos, dr);
    }

    pybind11::array_t<float> get_bowshock_numpy( 
        const pybind11::array_t<float>& Rho, const pybind11::array_t<float>& earth_pos, 
        float dr, int nb_phi, int max_nb_theta )
    {
        Matrix rho = casters::matrix_from_array( Rho );
        Point _earth_pos = casters::point_from_array( earth_pos );

        return casters::array_from_point_vec( get_bowshock(rho, _earth_pos, dr, nb_phi, max_nb_theta) );
    }



    pybind11::array_t<float> get_interest_points_numpy( 
        const pybind11::array_t<float>& J_norm, const pybind11::array_t<float>& earth_pos,
        float theta_min, float theta_max, 
        int nb_theta, int nb_phi, 
        float dx, float dr,
        float alpha_0_min, float alpha_0_max, float nb_alpha_0,
        float r_0_mult_min, float r_0_mult_max, float nb_r_0,
        float& avg_std_dev )
    {
        Matrix _J_norm = casters::matrix_from_array( J_norm );
        Point _earth_pos = casters::point_from_array( earth_pos );

        return casters::array_from_interest_point_vec( get_interest_points(
            _J_norm, _earth_pos, 
            theta_min, theta_max,
            nb_theta, nb_phi,
            dx, dr, alpha_0_min, alpha_0_max, nb_alpha_0,
            r_0_mult_min, r_0_mult_max, nb_r_0,
            &avg_std_dev
        ), nb_theta*nb_phi);
    }

    pybind11::array_t<float> get_interest_points_numpy_no_std_dev( 
        const pybind11::array_t<float>& J_norm, const pybind11::array_t<float>& earth_pos,
        float theta_min, float theta_max, 
        int nb_theta, int nb_phi, 
        float dx, float dr,
        float alpha_0_min, float alpha_0_max, float nb_alpha_0,
        float r_0_mult_min, float r_0_mult_max, float nb_r_0 )
    {
        Matrix _J_norm = casters::matrix_from_array( J_norm );
        Point _earth_pos = casters::point_from_array( earth_pos );

        return casters::array_from_interest_point_vec( get_interest_points(
            _J_norm, _earth_pos, 
            theta_min, theta_max,
            nb_theta, nb_phi,
            dx, dr, alpha_0_min, alpha_0_max, nb_alpha_0,
            r_0_mult_min, r_0_mult_max, nb_r_0,
            nullptr
        ), nb_theta*nb_phi);
    }


    pybind11::array_t<float> process_interest_points_numpy(   
        const pybind11::array_t<float>& interest_points, 
        int nb_theta, int nb_phi, 
        const pybind11::array_t<float>& shape_sim, const pybind11::array_t<float>& shape_real,
        const pybind11::array_t<float>& earth_pos_sim, const pybind11::array_t<float>& earth_pos_real )
    {
        std::vector<InterestPoint> _interest_points = casters::ip_vec_from_array(interest_points);
        Shape _shape_sim = casters::shape_from_array(shape_sim), _shape_real = casters::shape_from_array(shape_real);
        Point _earth_pos_sim = casters::point_from_array(earth_pos_sim), _earth_pos_real = casters::point_from_array(earth_pos_real);

        process_interest_points( _interest_points.data(), nb_theta, nb_phi, _shape_sim, _shape_real, _earth_pos_sim, _earth_pos_real );

        return casters::array_from_interest_point_vec( _interest_points.data(), _interest_points.size() );
    }

    pybind11::array_t<float> process_points_numpy(    
        const pybind11::array_t<float>& points, 
        const pybind11::array_t<float>& shape_sim, const pybind11::array_t<float>& shape_real,
        const pybind11::array_t<float>& earth_pos_sim, const pybind11::array_t<float>& earth_pos_real )
    {
        std::vector<Point> _points = casters::point_vec_from_array( points );
        Shape _shape_sim = casters::shape_from_array(shape_sim), _shape_real = casters::shape_from_array(shape_real);
        Point _earth_pos_sim = casters::point_from_array(earth_pos_sim), _earth_pos_real = casters::point_from_array(earth_pos_real);

        process_points( _points, _shape_sim, _shape_real, _earth_pos_sim, _earth_pos_real );

        return casters::array_from_point_vec( _points );
    }
}





PYBIND11_MODULE(topology_analysis, m)
{
    m.doc() = "Topology analysis module for magnetic field data";
    
    // Bind classes first
    // pybind11::class_<Point>(m, "Point")
    //     .def(pybind11::init<float, float, float>())
    //     .def_readwrite("x", &Point::x)
    //     .def_readwrite("y", &Point::y)
    //     .def_readwrite("z", &Point::z);

    m.def("get_bowshock_radius", &raycasting::get_bowshock_radius_numpy);
    m.def("get_bowshock", &raycasting::get_bowshock_numpy);

    m.def("get_interest_points", &raycasting::get_interest_points_numpy);
    m.def("get_interest_points", &raycasting::get_interest_points_numpy_no_std_dev);

    m.def("process_interest_points", &raycasting::process_interest_points_numpy);
    m.def("process_points", &raycasting::process_points_numpy);
}

