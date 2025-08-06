#include "../headers_cpp/raycast.h"
#include "../headers_cpp/read_file.h"
#include "../headers_cpp/read_pvtr.h"
#include "../headers_cpp/reader_writer.h"
#include "../headers_cpp/magnetopause.h"
#include "../headers_cpp/preprocessing.h"
#include "../headers_cpp/fit_to_analytical.h"
#include "../headers_cpp/analysis.h"

#include <iostream>


#include <chrono>

typedef std::chrono::high_resolution_clock Time;
typedef std::chrono::duration<float> fsec;

#ifndef CUSTOM_PI
#define CUSTOM_PI
const float PI = 3.141592653589793238462643383279502884f;
#endif


int main(int argc, char* argv[])
{
    // "/rds/general/user/avr24/projects/swimmr-sage/live/mheyns/benchmarking/runs/Run1/MS/x00_Bvec_c-21000.pvtr"

    if (argc < 1) { std::cout << "Please provide the filepath.\n"; exit(1); }
    if (argc < 2) { std::cout << "Please provide the savepath.\n"; exit(1); }

    std::string filepath(argv[1]);
    std::string savepath(argv[2]);

    // *** file reading ****************************************************************************
    
    PVTRReaderBinWriter reader_writer;    

    Matrix M;
    reader_writer.read(filepath, M);

    Matrix X;
    Matrix Y;
    Matrix Z;
    reader_writer.get_coordinates(filepath, X, Y, Z);

    // *** preprocessing ***************************************************************************
    
    Point p_min( X[0], Y[0], Z[0] );
    Point p_max( X[ X.get_shape().x-1 ], Y[ Y.get_shape().x-1 ], Z[ Z.get_shape().x-1 ] );
    Point p_range = p_max - p_min;

    Shape new_shape_real(std::round(p_range.x), std::round(p_range.y), std::round(p_range.z), M.get_shape().i);

    Matrix M_processed_real = orthonormalise(M, X, Y, Z, &new_shape_real);

    // *** saving **********************************************************************************
    
    reader_writer.write( savepath, M_processed_real );

    // *** freeing *********************************************************************************
    
    M.del(); 
    X.del(); Y.del(); Z.del();
    M_processed_real.del(); 
}
