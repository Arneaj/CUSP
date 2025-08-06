#include "../headers_cpp/read_file.h"
#include "../headers_cpp/preprocessing.h"

Matrix orthonormalise( const Matrix& mat, Matrix& X, Matrix& Y, Matrix& Z, const Shape* new_shape )
{
    Shape shape;
    if ( new_shape == nullptr ) shape = mat.get_shape();
    else shape = *new_shape;

    shape.i = mat.get_shape().i;

    Matrix new_mat( shape );

    float X_max = X.max(0), X_min = X.min(0), inv_dX = 1.0f / (X_max - X_min);
    float Y_max = Y.max(0), Y_min = Y.min(0), inv_dY = 1.0f / (Y_max - Y_min);
    float Z_max = Z.max(0), Z_min = Z.min(0), inv_dZ = 1.0f / (Z_max - Z_min);

    X -= X_min; X *= inv_dX; X *= shape.x;
    Y -= Y_min; Y *= inv_dY; Y *= shape.y;
    Z -= Z_min; Z *= inv_dZ; Z *= shape.z;

    std::vector<int> iX(shape.x);
    std::vector<int> iY(shape.y);
    std::vector<int> iZ(shape.z);

    std::vector<float> dX(shape.x);
    std::vector<float> dY(shape.y);
    std::vector<float> dZ(shape.z);

    #pragma omp parallel for
    for (int sx=0; sx<shape.x; sx++) for (int i=0; i<X.get_shape().x-1; i++)
    {
        if ( sx > X[i+1] ) continue;
        iX[sx] = i;
        dX[sx] = ( sx - X[i]) / (X[i+1] - X[i]);
        break;
    }

    #pragma omp parallel for
    for (int sy=0; sy<shape.y; sy++) for (int i=0; i<Y.get_shape().x-1; i++)
    {
        if ( sy > Y[i+1] ) continue;
        iY[sy] = i;
        dY[sy] = ( sy - Y[i]) / (Y[i+1] - Y[i]);
        break;
    }

    #pragma omp parallel for
    for (int sz=0; sz<shape.z; sz++) for (int i=0; i<Z.get_shape().x-1; i++)
    {
        if ( sz > Z[i+1] ) continue;
        iZ[sz] = i;
        dZ[sz] = ( sz - Z[i]) / (Z[i+1] - Z[i]);
        break;
    }

    #pragma omp parallel for
    for (int sx=0; sx<shape.x; sx++) for (int sy=0; sy<shape.y; sy++)
        for (int sz=0; sz<shape.z; sz++) for (int i=0; i<shape.i; i++)
            new_mat(sx,sy,sz,i) =   ( mat(iX[sx],iY[sy],iZ[sz],i)*(1-dX[sx]) + mat(iX[sx]+1,iY[sy],iZ[sz],i)*dX[sx] )*(1-dY[sy])*(1-dZ[sz])
                                +   ( mat(iX[sx],iY[sy]+1,iZ[sz],i)*(1-dX[sx]) + mat(iX[sx]+1,iY[sy]+1,iZ[sz],i)*dX[sx] )*dY[sy]*(1-dZ[sz])
                                +   ( mat(iX[sx],iY[sy],iZ[sz]+1,i)*(1-dX[sx]) + mat(iX[sx]+1,iY[sy],iZ[sz]+1,i)*dX[sx] )*(1-dY[sy])*dZ[sz]
                                +   ( mat(iX[sx],iY[sy]+1,iZ[sz]+1,i)*(1-dX[sx]) + mat(iX[sx]+1,iY[sy]+1,iZ[sz]+1,i)*dX[sx] )*dY[sy]*dZ[sz];   
    
    return new_mat;
}







