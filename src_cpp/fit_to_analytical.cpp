#include "../headers_cpp/fit_to_analytical.h"




void save_parameters( std::string filename, const std::vector<double>& params )
{
    std::ofstream fs;
    fs.open(filename);

    fs << params[0];
    for (int i=1; i<params.size(); i++) fs << ',' << params[i];
    fs << std::endl;

    fs.close();
}















