#include "../headers_cpp/read_pvtr.h"
#include "../headers_cpp/read_file.h"


int main(int argc, char* argv[])
{
    if (argc < 1) { std::cout << "Please provide input file path.\n"; exit(1); }
    if (argc < 2) { std::cout << "Please provide output file path.\n"; exit(1); }

    std::string filepath(argv[1]);
    std::string savepath(argv[1]);

    Matrix M = read_pvtr(filepath);

    Matrix X;
    Matrix Y;
    Matrix Z;
    
    get_coord(X, Y, Z, filepath);

    std::cout << "Shape of M: " << M.get_shape() << std::endl;
    std::cout << "M(0,0,0,0) = " << M(0,0,0,0) << std::endl;

    save_file_bin( savepath, M );
}

