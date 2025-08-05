#include "../headers_cpp/raycast.h"

#include <omp.h>

const float PI = 3.141592653589793238462643383279502884f;



float Shue97(float r_0, float alpha_0, float one_plus_cos_theta)
{
    return r_0 * std::pow( 2.0/one_plus_cos_theta, alpha_0 );
}


float get_bowshock_radius(  const Point& projection,
                            const Matrix& Rho, const Point& earth_pos,
                            float dr )
{
    float bow_r = 0.0f;
    float r = 0.0f;
    Point p = r * projection + earth_pos;

    float min_delta_rho = 0.0f;
    float previous_rho = Rho(p, 0);

    r += dr;

    while ( !Rho.is_point_OOB(p) )
    {
        float rho = Rho(p, 0);
        float delta_rho = rho - previous_rho;

        if ( delta_rho < min_delta_rho ) min_delta_rho = delta_rho;

        r += dr;
        p = r * projection + earth_pos;

        previous_rho = rho;
    }
    
    return 0.0f;
}


float get_bowshock_radius(  float theta, float phi,
                            const Matrix& Rho, const Point& earth_pos,
                            float dr )
{
    float sin_theta = std::sin(theta);

    Point proj = Point(
        -std::cos(theta),
        sin_theta*std::sin(phi),
        sin_theta*std::cos(phi)
    );

    return get_bowshock_radius(proj, Rho, earth_pos, dr);
}


void interest_points_helper(    float r_0, float alpha_0, 
                                std::vector<float>* interest_points,
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
        float final_r = Shue97(20, 1, 1 + cos_theta);

        float phi = -PI;

        for (int iphi=0; iphi<nb_phi; iphi++)
        {
            phi += dphi;

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
                                    float theta_min, float theta_max, 
                                    int nb_theta, int nb_phi, 
                                    float dx, float dr,
                                    float alpha_0_min, float alpha_0_max, float nb_alpha_0,
                                    float r_0_mult_min, float r_0_mult_max, float nb_r_0,
                                    float* avg_std_dev )
{
    std::vector<float>* interest_radii_candidates = new std::vector<float>[ nb_theta*nb_phi ];

    float x = earth_pos.x;

    if (avg_std_dev) *avg_std_dev = 0;
    

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

            if (avg_std_dev) *avg_std_dev += std_dev;

            // float weight = std::exp( -std_dev );  // TODO: change weights
            float weight = 1.0f / (1.0f + std_dev);
            // float weight = 5.0f / (5.0f + std_dev*std_dev);
            // if (std::abs(theta) < 0.5*PI) weight *= sin_theta;

            interest_points[ itheta*nb_phi + iphi ] = InterestPoint( theta, phi, interest_radius, weight ); 
        }
    }

    if (avg_std_dev) *avg_std_dev /= nb_theta*nb_phi;

    delete[] interest_radii_candidates;

    return interest_points;
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
