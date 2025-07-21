#ifndef READ_FILE_H
#define READ_FILE_H

#include "matrix.h"


Matrix read_file( std::string filename, int step=1 );

void save_file( std::string filename, const Matrix& mat );


void save_file_bin( std::string filename, Matrix& mat );


#endif
