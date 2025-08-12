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
    double* rearrange_data( double* vec, const Shape& shape )
    {
        // #pragma omp parallel for
        // for (int ix = 0; ix < shape.x; ix++) for (int iy = 0; iy < shape.y; iy++) 
        //     for (int iz = 0; iz < shape.z; iz++) for (int i = 0; i < shape.i; i++) {
        //         int srcIndex = ((iz*shape.y + iy)*shape.x + ix)*shape.i + i;
        //         int dstIndex = ((i*shape.z + iz)*shape.y + iy)*shape.x + ix;
                
        //         finalData[dstIndex] = extractedData[srcIndex];
        //     }
    }

    pybind11::array_t<double> array_from_matrix( Matrix& matrix )
    {
        const Shape& sh = matrix.get_shape();

        return pybind11::array_t<double>(
            {sh.x, sh.y, sh.z, sh.i},                                                                       // shape
            {sizeof(double), sizeof(double)*sh.x, sizeof(double)*sh.x*sh.y, sizeof(double)*sh.x*sh.y*sh.z},     // strides
            matrix.get_array(),                                                                             // data pointer
            pybind11::cast(matrix.get_array())                                                              // parent object (keeps data alive)
        );
    }

    pybind11::array_t<double> array_from_point_vec( const std::vector<Point>& points )
    {
        int length = points.size();
        double* arr = new double[length*3];

        for (int i=0; i<length; i++)
        {
            arr[3*i] = points[i].x;
            arr[3*i+1] = points[i].y;
            arr[3*i+2] = points[i].z;
        }

        return pybind11::array_t<double>(
            {length, 3},                        // shape
            {sizeof(double)*3, sizeof(double)},   // strides
            arr,                                // data pointer
            pybind11::cast(arr)                 // parent object (keeps data alive)
        );
    }

    pybind11::array_t<double> array_from_interest_point_vec( InterestPoint* interest_points, int nb_interest_points )
    {
        double* arr = new double[nb_interest_points*4];

        for (int i=0; i<nb_interest_points; i++)
        {
            arr[4*i] = interest_points[i].theta;
            arr[4*i+1] = interest_points[i].phi;
            arr[4*i+2] = interest_points[i].radius;
            arr[4*i+3] = interest_points[i].weight;
        }

        delete[] interest_points;

        return pybind11::array_t<double>(
            {nb_interest_points, 4},            // shape
            {sizeof(double)*4, sizeof(double)},   // strides
            arr,                                // data pointer
            pybind11::cast(arr)                 // parent object (keeps data alive)
        );
    }


    Matrix matrix_from_array( const pybind11::array_t<double>& arr )
    {
        int nb_dim = arr.ndim();
        Shape sh;

        if (nb_dim==4) { sh = Shape( arr.shape(0), arr.shape(1), arr.shape(2), arr.shape(3) ); }
        else if (nb_dim==3) { sh = Shape( arr.shape(0), arr.shape(1), arr.shape(2), 1 ); }
        else if (nb_dim==2) { sh = Shape( arr.shape(0), arr.shape(1), 1, 1 ); }
        else if (nb_dim==1) { sh = Shape( arr.shape(0), 1, 1, 1 ); }

        int total_size = sh.x*sh.y*sh.z*sh.i;

        double* mat = new double[total_size];
        std::memcpy( mat, arr.data(), sizeof(double)*total_size );

        return Matrix( sh, mat );
    }

    std::vector<InterestPoint> ip_vec_from_array( const pybind11::array_t<double>& arr )
    {
        std::vector<InterestPoint> ip(arr.shape(0));
        
        for (int i=0; i<arr.shape(0); i++)
            ip[i] = { *arr.data(4*i), *arr.data(4*i+1), *arr.data(4*i+2), *arr.data(4*i+3) };

        return ip;
    }

    std::vector<Point> point_vec_from_array( const pybind11::array_t<double>& arr )
    {
        std::vector<Point> points(arr.shape(0));
        
        for (int i=0; i<arr.shape(0); i++)
            points[i] = { *arr.data(3*i), *arr.data(3*i+1), *arr.data(3*i+2) };

        return points;
    }

    Point point_from_array( const pybind11::array_t<double>& point )
    {
        pybind11::ssize_t nb_dims = point.ndim();
        if ( nb_dims > 1 || point.shape(0) != 3 )
        {
            throw pybind11::index_error("Point needs to be an array of shape (3)");
        }

        return Point( *point.data(0), *point.data(1), *point.data(2) );
    }

    Shape shape_from_array( const pybind11::array_t<double>& shape )
    {
        pybind11::ssize_t nb_dims = shape.ndim();
        if ( nb_dims > 1 || shape.shape(0) != 4 )
        {
            throw pybind11::index_error("Shape needs to be an array of shape (4)");
        }

        return Shape( *shape.data(0), *shape.data(1), *shape.data(2), *shape.data(3) );
    }
}


namespace preprocessing
{
    pybind11::array_t<double> orthonormalise_numpy( 
        const pybind11::array_t<double>& mat, 
        const pybind11::array_t<double>& X, const pybind11::array_t<double>& Y, const pybind11::array_t<double>& Z, 
        const pybind11::array_t<double>& new_shape )
    {
        Shape _shape = casters::shape_from_array( new_shape );
        Matrix _mat = casters::matrix_from_array( mat );
        Matrix _X = casters::matrix_from_array( X );
        Matrix _Y = casters::matrix_from_array( Y );
        Matrix _Z = casters::matrix_from_array( Z );

        _shape.i = _mat.get_shape().i;

        Matrix new_mat = orthonormalise( _mat, _X, _Y, _Z, &_shape ); 

        pybind11::array_t<double> ret = casters::array_from_matrix( new_mat );
        
        _X.del(); _Y.del(); _Z.del();
        _mat.del();

        return ret;
    }
}


namespace raycasting
{
    double get_bowshock_radius_numpy(  
        double theta, double phi,
        const pybind11::array_t<double>& Rho, const pybind11::array_t<double>& earth_pos,
        double dr )
    {
        Matrix _Rho = casters::matrix_from_array( Rho );
        Point _earth_pos = casters::point_from_array( earth_pos );

        double rad = get_bowshock_radius(theta, phi, _Rho, _earth_pos, dr);

        _Rho.del();

        return rad;
    }

    pybind11::array_t<double> get_bowshock_numpy( 
        const pybind11::array_t<double>& Rho, const pybind11::array_t<double>& earth_pos, 
        double dr, int nb_phi, int max_nb_theta )
    {
        Matrix _Rho = casters::matrix_from_array( Rho );
        Point _earth_pos = casters::point_from_array( earth_pos );

        pybind11::array_t<double> ret = casters::array_from_point_vec( get_bowshock(_Rho, _earth_pos, dr, nb_phi, max_nb_theta) );

        _Rho.del();

        return ret;
    }



    pybind11::array_t<double> get_interest_points_numpy( 
        const pybind11::array_t<double>& J_norm, const pybind11::array_t<double>& earth_pos,
        const pybind11::array_t<double>& Rho,
        double theta_min, double theta_max, 
        int nb_theta, int nb_phi, 
        double dx, double dr,
        double alpha_0_min, double alpha_0_max, double nb_alpha_0,
        double r_0_mult_min, double r_0_mult_max, double nb_r_0,
        double& avg_std_dev )
    {
        Matrix _Rho = casters::matrix_from_array( Rho );
        Matrix _J_norm = casters::matrix_from_array( J_norm );
        Point _earth_pos = casters::point_from_array( earth_pos );

        pybind11::array_t<double> ret = casters::array_from_interest_point_vec( get_interest_points(
            _J_norm, _earth_pos, 
            _Rho,
            theta_min, theta_max,
            nb_theta, nb_phi,
            dx, dr, alpha_0_min, alpha_0_max, nb_alpha_0,
            r_0_mult_min, r_0_mult_max, nb_r_0,
            &avg_std_dev
        ), nb_theta*nb_phi);

        _Rho.del();
        _J_norm.del();

        return ret;
    }

    pybind11::array_t<double> get_interest_points_numpy_no_std_dev( 
        const pybind11::array_t<double>& J_norm, const pybind11::array_t<double>& earth_pos,
        const pybind11::array_t<double>& Rho,
        double theta_min, double theta_max, 
        int nb_theta, int nb_phi, 
        double dx, double dr,
        double alpha_0_min, double alpha_0_max, double nb_alpha_0,
        double r_0_mult_min, double r_0_mult_max, double nb_r_0 )
    {
        Matrix _J_norm = casters::matrix_from_array( J_norm );
        Matrix _Rho = casters::matrix_from_array( Rho );
        Point _earth_pos = casters::point_from_array( earth_pos );

        pybind11::array_t<double> ret = casters::array_from_interest_point_vec( get_interest_points(
            _J_norm, _earth_pos, 
            _Rho,
            theta_min, theta_max,
            nb_theta, nb_phi,
            dx, dr, alpha_0_min, alpha_0_max, nb_alpha_0,
            r_0_mult_min, r_0_mult_max, nb_r_0,
            nullptr
        ), nb_theta*nb_phi);

        _Rho.del();
        _J_norm.del();

        return ret;
    }


    pybind11::array_t<double> process_interest_points_numpy(   
        const pybind11::array_t<double>& interest_points, 
        int nb_theta, int nb_phi, 
        const pybind11::array_t<double>& shape_sim, const pybind11::array_t<double>& shape_real,
        const pybind11::array_t<double>& earth_pos_sim, const pybind11::array_t<double>& earth_pos_real )
    {
        std::vector<InterestPoint> _interest_points = casters::ip_vec_from_array(interest_points);
        Shape _shape_sim = casters::shape_from_array(shape_sim), _shape_real = casters::shape_from_array(shape_real);
        Point _earth_pos_sim = casters::point_from_array(earth_pos_sim), _earth_pos_real = casters::point_from_array(earth_pos_real);

        process_interest_points( _interest_points.data(), nb_theta, nb_phi, _shape_sim, _shape_real, _earth_pos_sim, _earth_pos_real );

        return casters::array_from_interest_point_vec( _interest_points.data(), _interest_points.size() );
    }

    pybind11::array_t<double> process_points_numpy(    
        const pybind11::array_t<double>& points, 
        const pybind11::array_t<double>& shape_sim, const pybind11::array_t<double>& shape_real,
        const pybind11::array_t<double>& earth_pos_sim, const pybind11::array_t<double>& earth_pos_real )
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

    // pybind11::class_<Matrix>(m, "Matrix").
    //     .def(py::init<>());
        // .def("myFunction");

    m.def("preprocess", &preprocessing::orthonormalise_numpy);

    m.def("get_bowshock_radius", &raycasting::get_bowshock_radius_numpy);
    m.def("get_bowshock", &raycasting::get_bowshock_numpy);

    m.def("get_interest_points", &raycasting::get_interest_points_numpy);
    m.def("get_interest_points", &raycasting::get_interest_points_numpy_no_std_dev);

    m.def("process_interest_points", &raycasting::process_interest_points_numpy);
    m.def("process_points", &raycasting::process_points_numpy);
}

