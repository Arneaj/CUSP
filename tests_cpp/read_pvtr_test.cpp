#include "../headers_cpp/read_pvtr.h"


int main()
{
    std::string filename("/rds/general/user/avr24/projects/swimmr-sage/live/mheyns/benchmarking/runs/Run1/MS/x00_Bvec_c-21000.pvtr");

    Matrix B = read_pvtr(filename);
    
    std::array<Matrix, 3> XYZ = get_coord(filename);

    std::cout << "Shape of B: " << B.get_shape() << std::endl;
    std::cout << "B(0,0,0,0) = " << B(0,0,0,0) << std::endl;
}

