#include "../headers_cpp/raycast.h"


const float PI = 3.1415926535;


float get_r(float r_0, float alpha_0, float one_plus_cos_theta)
{
    return r_0 * std::pow( 2/one_plus_cos_theta, alpha_0 );
}


void interest_points_helper(    float r_0, float alpha_0, 
                                std::vector<float>* interest_points,
                                const Matrix& J_norm, Point earth_pos, 
                                int nb_theta, int nb_phi, 
                                float dr, float dtheta, float dphi )
{
    float theta = -PI;

    for (int itheta=0; itheta<nb_theta; itheta++)
    {
        theta += dtheta;

	    if (std::abs(theta) < 0.1) continue;

        float sin_theta = std::sin(theta);
        float cos_theta = std::cos(theta);

        if ( 1 + cos_theta < 1e-3 ) continue;

        float initial_r = get_r(r_0, alpha_0, 1 + cos_theta);
	    // float final_r = get_r(12, 1, 1 + cos_theta);
        float final_r = get_r(20, 1, 1 + cos_theta);

        float phi = 0;

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

            while ( r <= final_r &&
		    p.x >= 0 && p.x < J_norm.get_shape().x-1 &&
                    p.y >= 0 && p.y < J_norm.get_shape().y-1 &&
                    p.z >= 0 && p.z < J_norm.get_shape().z-1 )
            {
                float value = interpolate( p, J_norm, 0 );

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



std::array<float, 4>* get_interest_points(  const Matrix& J_norm, Point earth_pos, 
                                            int nb_theta, int nb_phi, 
                                            float dx, float dr,
                                            float alpha_0_min, float alpha_0_max, float nb_alpha_0,
                                            float r_0_mult_min, float r_0_mult_max, float nb_r_0 )
{
    std::vector<float>* interest_radii_candidates = new std::vector<float>[ nb_theta*nb_phi ];

    float x = earth_pos.x;
    

    while ( x>0 and x<J_norm.get_shape().x )
    {
        if ( J_norm(x, J_norm.get_shape().y/2, J_norm.get_shape().z/2, 0) > 1e-18 ) break;
        x += dx;
    }

    float r_inner = x - earth_pos.x;       
    float dr_0_mult = (r_0_mult_max - r_0_mult_min) / nb_r_0;
    float dalpha_0 = (alpha_0_max - alpha_0_min) / nb_alpha_0;

    float dtheta = 2*PI / nb_theta;
    float dphi = PI / nb_phi;

    for (float r_0_mult=r_0_mult_min; r_0_mult<=r_0_mult_max; r_0_mult+=dr_0_mult) for (float alpha_0=alpha_0_min; alpha_0<=alpha_0_max; alpha_0+=dalpha_0)
        interest_points_helper( r_0_mult * r_inner, alpha_0,
                                interest_radii_candidates, 
                                J_norm, earth_pos,
                                nb_theta, nb_phi, 
                                dr, dtheta, dphi );


    std::array<float, 4>* interest_points = new std::array<float, 4>[ nb_theta*nb_phi ];
    float theta = -PI;

    for (int itheta=0; itheta<nb_theta; itheta++)
    {
        theta += dtheta;
        float phi = 0;

//        float sin_theta = std::abs( std::sin(theta) );

        for (int iphi=0; iphi<nb_phi; iphi++)
        {
            phi += dphi;

            if ( interest_radii_candidates[ itheta*nb_phi + iphi ].size() == 0 ) 
            {
                interest_points[ itheta*nb_phi + iphi ] = std::array<float, 4>{ theta, phi, 0, 0 };
                continue;
            }

            float interest_radius = get_median( interest_radii_candidates[ itheta*nb_phi + iphi ] );
            float std_dev = get_std_dev( interest_radii_candidates[ itheta*nb_phi + iphi ] );

            // float weight = std::exp( -std_dev );  // TODO: change weights
            // float weight = 1.0f / (1.0f + std_dev);
            float weight = 5.0f / (5.0f + std_dev*std_dev);
            // if (std::abs(theta) < 0.5*PI) weight *= sin_theta;

            interest_points[ itheta*nb_phi + iphi ] = std::array<float, 4>{ theta, phi, interest_radius, weight }; 
        }
    }

    delete[] interest_radii_candidates;

    return interest_points;
}




void save_interest_points( std::string filename, const std::array<float, 4>* interest_points, int nb_theta, int nb_phi )
{
    std::ofstream fs;
    fs.open(filename);

    // for (const Point& p: interest_points)
    // {
    //     fs  << p.x << ','
    //         << p.y << ','
    //         << p.z << '\n';
    // }

    for (int itheta=0; itheta<nb_theta; itheta++) for (int iphi=0; iphi<nb_phi; iphi++)
    {
        fs  << interest_points[ itheta*nb_phi + iphi ][0] << ','
            << interest_points[ itheta*nb_phi + iphi ][1] << ','
            << interest_points[ itheta*nb_phi + iphi ][2] << ','
            << interest_points[ itheta*nb_phi + iphi ][3] << '\n';
    }

    fs.close();
}
