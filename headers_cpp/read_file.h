#ifndef READ_FILE_H
#define READ_FILE_H

#include "matrix.h"


Matrix read_file( std::string filename, int step=1 );

void save_file( std::string filename, const Matrix& mat );

void save_file_bin( std::string filename, Matrix& mat );


struct SolarWindInputs
{
    Point B;
    Point V;
    float rho;
    float Ti;
    float Te;  
};

/// @brief reads the solar wind input written in the format of the Gorgon benchmark outputs. 
///        This function is a bit too JavaScripty for me but it'll do
/// @param filepath 
/// @param timestep 
/// @return `SolarWindInputs` struct containing B, V, rho, Ti and Te
SolarWindInputs read_Gorgon_inputs( std::string filepath, std::string timestep );


#endif
