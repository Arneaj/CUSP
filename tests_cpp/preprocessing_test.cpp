#include <iostream>

#include "../headers_cpp/preprocessing.h"

int main()
{
    Matrix B = read_file("../data/Run1_18000/B.txt");
    Matrix J = read_file("../data/Run1_18000/J.txt");
    // Matrix V = read_file("../data/Run1_28800/V.txt");

    Matrix X = read_file("../data/Run1_18000/X.txt");
    Matrix Y = read_file("../data/Run1_18000/Y.txt");
    Matrix Z = read_file("../data/Run1_18000/Z.txt");

    std::cout << "Reading files done." << std::endl;

    Point p_min( X[0], Y[0], Z[0] );
    Point p_max( X[ X.get_shape().x-1 ], Y[ Y.get_shape().x-1 ], Z[ Z.get_shape().x-1 ] );
    Point p_range = p_max - p_min;

    Shape new_shape(std::round(p_range.x), std::round(p_range.y), std::round(p_range.z), B.get_shape().i);
    // Shape new_shape(B.get_shape().x*1.2, B.get_shape().y*1.2, B.get_shape().z*1.2, B.get_shape().i);

    Matrix B_processed = orthonormalise(B, X, Y, Z, &new_shape);
    Matrix J_processed = orthonormalise(J, X, Y, Z, &new_shape);

    Matrix J_norm_processed = J_processed.norm();
    // Matrix V_processed = orthonormalise(V, X, Y, Z, &new_shape);

    std::cout << "Preprocessing files done." << std::endl;

    save_file("../data/Run1_18000/B_processed_real.txt", B_processed);
    save_file("../data/Run1_18000/J_processed_real.txt", J_processed);
    save_file("../data/Run1_18000/J_norm_processed_real.txt", J_norm_processed);
    // save_file("../data/Run1_18000/V_processed.txt", V_processed);

    std::cout << "Saving files done." << std::endl;

    B.del(); J.del(); // V.del();
    X.del(); Y.del(); Z.del();
    B_processed.del(); J_processed.del(); // V_processed.del();
    J_norm_processed.del();
}

