#include "../headers_cpp/raycast.h"

#include <omp.h>

#ifndef CUSTOM_PI
#define CUSTOM_PI
const float PI = 3.141592653589793238462643383279502884f;
#endif



float Shue97(float r_0, float alpha_0, float one_plus_cos_theta)
{
    return r_0 * std::pow( 2.0/one_plus_cos_theta, alpha_0 );
}


float get_bowshock_radius(  const Point& projection,
                            const Matrix& Rho, const Point& earth_pos,
                            float dr,
                            bool* is_at_bounds )
{
    float bow_r = 0.0f;
    float r = 0.0f;
    Point p = r * projection + earth_pos;

    float min_value = 0.0f;
    float previous_rho = Rho(p, 0);

    r += dr;

    while ( !Rho.is_point_OOB(p) )
    {
        float rho = Rho(p, 0);
        float value = (rho - previous_rho) * r*r*r; 
        // bit horrifying, but this is to try and ignore things near the earth

        if ( value < min_value )
        {
            min_value = value;
            bow_r = r;
        }

        r += dr;
        p = r * projection + earth_pos;

        previous_rho = rho;
    }

    if ( is_at_bounds != nullptr && std::abs( r-dr-bow_r ) < 0.1f*dr ) *is_at_bounds = true;
    
    return bow_r;
}


float get_bowshock_radius(  float theta, float phi,
                            const Matrix& Rho, const Point& earth_pos,
                            float dr,
                            bool* is_at_bounds )
{
    float sin_theta = std::sin(theta);

    Point proj = Point(
        -std::cos(theta),
        sin_theta*std::sin(phi),
        sin_theta*std::cos(phi)
    );

    return get_bowshock_radius(proj, Rho, earth_pos, dr, is_at_bounds);
}


inline void squeeze_vector( std::vector<Point>& points )
{
    Point empty_p{};
    points.erase(std::remove(points.begin(), points.end(), empty_p), points.end());
}


std::vector<Point> get_bowshock( const Matrix& Rho, const Point& earth_pos, float dr, int nb_phi, int max_nb_theta, bool is_squeezed )
{
    std::vector<Point> bs_points( nb_phi*max_nb_theta );

    float shue97_radii[max_nb_theta];

    float dphi = 2.0f*PI / nb_phi;
    float dtheta = PI / max_nb_theta;

    float theta=0.0f;

    #pragma omp parallel for
    for (int itheta=0; itheta<max_nb_theta; itheta++)
    {
        shue97_radii[itheta] = Shue97(5.0f, 0.7f, 1.0f+std::cos(theta));
        theta += dtheta;
    }

    #pragma omp parallel for
    for (int iphi=0; iphi<nb_phi; iphi++)
    {
        float phi = iphi*dphi - PI;
        float sin_phi = std::sin(phi);
        float cos_phi = std::cos(phi);

        theta = 0.0f;

        bool is_at_bounds = false;

        for (int itheta=0; itheta<max_nb_theta; itheta++)
        {
            float sin_theta = std::sin(theta);

            Point proj = Point(
                -std::cos(theta),
                sin_theta*sin_phi,
                sin_theta*cos_phi
            );

            float r = get_bowshock_radius(proj, Rho, earth_pos, dr, &is_at_bounds);

            if ( is_at_bounds || r < shue97_radii[itheta] ) break;

            bs_points[itheta*nb_phi + iphi] = Point(theta, phi, r);

            theta += dtheta;
        }
    }

    if (is_squeezed) squeeze_vector(bs_points);

    return bs_points;
}





void interest_points_helper(    float r_0, float alpha_0, 
                                std::vector<float>* interest_points,
                                const Point* const unsqueezed_bow_shock,
                                const Matrix& J_norm, const Point& earth_pos, 
                                float theta_min,
                                int nb_theta, int nb_phi, 
                                float dr, float dtheta, float dphi )
{
    float theta = theta_min;

    #pragma omp parallel for
    for (int itheta=0; itheta<nb_theta; itheta++)
    {
        theta = theta_min + (itheta+1)*dtheta;

	    // if (std::abs(theta) < 0.05) continue;

        float sin_theta = std::sin(theta);
        float cos_theta = std::cos(theta);

        if ( 1 + cos_theta < 1e-3 ) continue;

        float initial_r = Shue97(r_0, alpha_0, 1 + cos_theta);
	    // float final_r = Shue97(12, 1, 1 + cos_theta);
        // float final_r = Shue97(20, 1, 1 + cos_theta);

        float phi = -PI;

        for (int iphi=0; iphi<nb_phi; iphi++)
        {
            phi += dphi;

            const Point& bs = unsqueezed_bow_shock[itheta*nb_phi+iphi];

            float final_r;

            if (bs == Point()) final_r = Shue97(r_0, alpha_0, 1 + cos_theta);
            else final_r = bs.z -1.0f;  // get radius of the bowshock at (theta,phi)  //    add a bit of extra space to be sure not to get the bowshock
            //                      -> NOT SURE THIS IS A GOOD IDEA
            
            float max_value = 0;
            float max_r = initial_r;
            float r = max_r;

            Point proj = Point(
                -cos_theta,
                sin_theta*std::sin(phi),
                sin_theta*std::cos(phi)
            );

            Point p = r * proj + earth_pos;

            while ( r <= final_r && !J_norm.is_point_OOB(p) )
            {
                float value = J_norm(p, 0); // interpolate( p, J_norm, 0 );

                if ( value > max_value )
                {
                    max_value = value;
                    max_r = r;
                }

                r += dr;
                p = r * proj + earth_pos;
            }

            if ( max_r == initial_r ) continue;

            interest_points[ itheta*nb_phi + iphi ].push_back( max_r );
            // interest_points.push_back( max_r * proj + (*earth_pos) );
        }
    }
}




float get_median( std::vector<float>& vec )
{
    std::sort( vec.begin(), vec.end() );

    return vec[ vec.size() / 2 ];
}


float get_std_dev( std::vector<float>& vec )
{
    float avg = std::accumulate( vec.begin(), vec.end(), 0.0f ) / vec.size();

    for (float& val: vec) val = (val-avg)*(val-avg);
    
    return std::sqrt( std::accumulate( vec.begin(), vec.end(), 0.0f ) / vec.size() );
}







InterestPoint* get_interest_points( const Matrix& J_norm, const Point& earth_pos,
                                    const Point* const unsqueezed_bow_shock,
                                    float theta_min, float theta_max, 
                                    int nb_theta, int nb_phi, 
                                    float dx, float dr,
                                    float alpha_0_min, float alpha_0_max, float nb_alpha_0,
                                    float r_0_mult_min, float r_0_mult_max, float nb_r_0,
                                    float* p_avg_std_dev )
{
    std::vector<float>* interest_radii_candidates = new std::vector<float>[ nb_theta*nb_phi ];

    float x = earth_pos.x;

    if (p_avg_std_dev) *p_avg_std_dev = 0;
    

    while ( x>0 and x<J_norm.get_shape().x )
    {
        if ( J_norm(x, J_norm.get_shape().y/2, J_norm.get_shape().z/2, 0) > 1e-18 ) break;
        x += dx;
    }

    float r_inner = x - earth_pos.x;       
    float dr_0_mult = (r_0_mult_max - r_0_mult_min) / nb_r_0;
    float dalpha_0 = (alpha_0_max - alpha_0_min) / nb_alpha_0;

    float dtheta = (theta_max - theta_min) / nb_theta;
    float dphi = 2.0*PI / nb_phi;

    for (float r_0_mult=r_0_mult_min; r_0_mult<=r_0_mult_max; r_0_mult+=dr_0_mult) for (float alpha_0=alpha_0_min; alpha_0<=alpha_0_max; alpha_0+=dalpha_0)
        interest_points_helper( r_0_mult * r_inner, alpha_0,
                                interest_radii_candidates,
                                unsqueezed_bow_shock,
                                J_norm, earth_pos,
                                theta_min,
                                nb_theta, nb_phi, 
                                dr, dtheta, dphi );


    InterestPoint* interest_points = new InterestPoint[ nb_theta*nb_phi ];
    float theta = theta_min;

    for (int itheta=0; itheta<nb_theta; itheta++)
    {
        theta += dtheta;
        float phi = -PI;

//        float sin_theta = std::abs( std::sin(theta) );

        for (int iphi=0; iphi<nb_phi; iphi++)
        {
            phi += dphi;

            if ( interest_radii_candidates[ itheta*nb_phi + iphi ].size() == 0 ) 
            {
                interest_points[ itheta*nb_phi + iphi ] = InterestPoint( theta, phi );
                continue;
            }

            float interest_radius = get_median( interest_radii_candidates[ itheta*nb_phi + iphi ] );
            float std_dev = get_std_dev( interest_radii_candidates[ itheta*nb_phi + iphi ] );

            if (p_avg_std_dev) *p_avg_std_dev += std_dev;

            // float weight = std::exp( -std_dev );  // TODO: change weights
            float weight = 1.0f / (1.0f + std_dev);
            // float weight = 5.0f / (5.0f + std_dev*std_dev);
            // if (std::abs(theta) < 0.5*PI) weight *= sin_theta;

            interest_points[ itheta*nb_phi + iphi ] = InterestPoint( theta, phi, interest_radius, weight ); 
        }
    }

    if (p_avg_std_dev) *p_avg_std_dev /= nb_theta*nb_phi;

    delete[] interest_radii_candidates;

    return interest_points;
}

InterestPoint* get_interest_points( const Matrix& J_norm, const Point& earth_pos,
                                    const Matrix& Rho,
                                    float theta_min, float theta_max, 
                                    int nb_theta, int nb_phi, 
                                    float dx, float dr,
                                    float alpha_0_min, float alpha_0_max, float nb_alpha_0,
                                    float r_0_mult_min, float r_0_mult_max, float nb_r_0,
                                    float* p_avg_std_dev )
{
    std::vector<Point> bow_shock = get_bowshock( Rho, earth_pos, dr, nb_phi, nb_theta, false );

    return get_interest_points( J_norm, earth_pos, bow_shock.data(), theta_min, theta_max, nb_theta, nb_phi, dx, dr, alpha_0_min, alpha_0_max, nb_alpha_0, r_0_mult_min, r_0_mult_max, nb_r_0, p_avg_std_dev);
}





void process_interest_points(   InterestPoint* interest_points, 
                                int nb_theta, int nb_phi, 
                                const Shape& shape_sim, const Shape& shape_real,
                                const Point& earth_pos_sim, const Point& earth_pos_real )
{
    // std::cout << shape_sim << std::endl;
    // std::cout << shape_real << std::endl;
    // std::cout << earth_pos_sim << std::endl;
    // std::cout << earth_pos_real << std::endl;

    for (int itheta=0; itheta<nb_theta; itheta++) for (int iphi=0; iphi<nb_phi; iphi++)
    {
        Point point;

        float sin_theta = std::sin(interest_points[itheta*nb_phi + iphi].theta);

        point.x = std::cos(interest_points[itheta*nb_phi + iphi].theta);
        point.y = sin_theta * std::sin(interest_points[itheta*nb_phi + iphi].phi);
        point.z = sin_theta * std::cos(interest_points[itheta*nb_phi + iphi].phi);

        point *= interest_points[itheta*nb_phi + iphi].radius;
        point += earth_pos_sim;

        point.x *= float(shape_real.x) / float(shape_sim.x);
        point.y *= float(shape_real.y) / float(shape_sim.y);
        point.z *= float(shape_real.z) / float(shape_sim.z);

        // std::cout << point << std::endl;

        point -= earth_pos_real;

        interest_points[itheta*nb_phi + iphi].radius = point.norm();
        interest_points[itheta*nb_phi + iphi].theta = std::acos( point.x / std::max(0.1f, interest_points[itheta*nb_phi + iphi].radius) );
        interest_points[itheta*nb_phi + iphi].phi = std::acos( point.z / std::max(0.1f, std::sqrt( point.y*point.y + point.z*point.z )) );
        interest_points[itheta*nb_phi + iphi].phi *= (point.y>0) - (point.y<=0);
    }
}






void save_interest_points( const std::string& filename, const InterestPoint* interest_points, int nb_theta, int nb_phi )
{
    std::ofstream fs;
    fs.open(filename);

    for (int itheta=0; itheta<nb_theta; itheta++) for (int iphi=0; iphi<nb_phi; iphi++)
    {
        fs  << interest_points[ itheta*nb_phi + iphi ].theta << ','
            << interest_points[ itheta*nb_phi + iphi ].phi << ','
            << interest_points[ itheta*nb_phi + iphi ].radius << ','
            << interest_points[ itheta*nb_phi + iphi ].weight << '\n';
    }

    fs.close();
}




void process_points(    std::vector<Point>& points, 
                        const Shape& shape_sim, const Shape& shape_real,
                        const Point& earth_pos_sim, const Point& earth_pos_real )
{
    for (Point& p: points)
    {
        Point point;

        float sin_theta = std::sin(p.x);

        point.x = std::cos(p.x);
        point.y = sin_theta * std::sin(p.y);
        point.z = sin_theta * std::cos(p.y);

        point *= p.z;
        point += earth_pos_sim;

        point.x *= float(shape_real.x) / float(shape_sim.x);
        point.y *= float(shape_real.y) / float(shape_sim.y);
        point.z *= float(shape_real.z) / float(shape_sim.z);

        // std::cout << point << std::endl;

        point -= earth_pos_real;

        p.z = point.norm();
        p.x = std::acos( point.x / std::max(0.1f, p.z) );
        p.y = std::acos( point.z / std::max(0.1f, std::sqrt( point.y*point.y + point.z*point.z )) );
        p.y *= (point.y>0) - (point.y<=0);
    }
}






void save_points( const std::string& filename, const std::vector<Point>& points )
{
    std::ofstream fs;
    fs.open(filename);

    for (const Point& p: points)
    {
        fs  << p.x << ','
            << p.y << ','
            << p.z << '\n';
    }

    fs.close();
}
