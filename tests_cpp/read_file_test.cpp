#include<iostream>

#include"../headers_cpp/read_file.h"



int main()
{
    Matrix matrix = read_file( "../data/Run1_28800/B.txt" );

    std::cout << matrix(0,0,0,0) << std::endl;

    matrix.del();
    
    return 0;
}

