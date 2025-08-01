#include "../headers_cpp/read_pvtr.h"
#include "../headers_cpp/read_file.h"


int main(int argc, char* argv[])
{
    std::string filepath("../test_data/x00_jvec-21000.pvtr");

    Matrix J = read_pvtr(filepath);

    Matrix X;
    Matrix Y;
    Matrix Z;
    
    get_coord(X, Y, Z, filepath);

    std::cout << "Shape of J: " << J.get_shape() << std::endl;
    std::cout << "J(0,0,0,0) = " << J(0,0,0,0) << std::endl;

    J.del();
    X.del(); Y.del(); Z.del();
}

