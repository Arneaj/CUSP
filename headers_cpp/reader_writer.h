#ifndef READER_WRITER_H
#define READER_WRITER_H

#include "matrix.h"

#include "../headers_cpp/read_pvtr.h"
#include "../headers_cpp/read_file.h"


/// @brief interface to read and write matrices from and to the file format of your choice
class ReaderWriter
{
public:
    virtual ~ReaderWriter() = default;

    /// @brief function that needs to read the file at the given filepath and write it to M
    /// @param filepath path to the file containing your data
    /// @param M Matrix containing a Shape and a float* containing the data.
    ///          The array given to the matrix should be Fortran indexed, i.e.: 
    ///          M(ix,iy,iz,i) = M.mat[ix + iy*shape.x + iz*shape.x*shape.y + i*shape.x*shape.y*shape.z] 
    virtual void read( std::string filepath, Matrix& M ) = 0;

    /// @brief function that needs to read the file at the given filepath and write the coordinates 
    ///        corresponding to each cell of the read matrix. This is needed for the preprocessing
    ///        of the data from a non-uniform grid to a uniform one.
    /// @param filepath path to the file containing your data
    /// @param X Matrix of shape (M.get_shape().x,1,1,1) containing the coordinate of each cell of the corresponding axis.
    /// @param Y Matrix of shape (M.get_shape().y,1,1,1) containing the coordinate of each cell of the corresponding axis.
    /// @param Z Matrix of shape (M.get_shape().z,1,1,1) containing the coordinate of each cell of the corresponding axis.
    virtual void get_coordinates( std::string filepath, Matrix& X, Matrix& Y, Matrix& Z ) = 0;

    /// @brief function that needs to write the Matrix M to the file format of your choice
    /// @param savepath path to the file that will contain the data
    /// @param M Matrix containing a Shape and a float* containing the data
    virtual void write( std::string savepath, Matrix& M ) = 0;
};


/// @brief example of an implementation of the interface
class PVTRReaderBinWriter: public ReaderWriter 
{
public:
    virtual void read( std::string filepath, Matrix& M ) { M = read_pvtr( filepath ); }
    virtual void get_coordinates( std::string filepath, Matrix& X, Matrix& Y, Matrix& Z ) { get_coord( X, Y, Z, filepath ); }
    virtual void write( std::string savepath, Matrix& M ) { save_file_bin( savepath, M ); }
};


#endif
