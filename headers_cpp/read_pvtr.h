#include <vtkSmartPointer.h>
#include <vtkXMLPRectilinearGridReader.h>
#include <vtkXMLPStructuredGridReader.h>
#include <vtkXMLPUnstructuredGridReader.h>
#include <vtkRectilinearGrid.h>
#include <vtkStructuredGrid.h>
#include <vtkUnstructuredGrid.h>
#include <vtkDataSet.h>
#include <vtkPointData.h>
#include <vtkCellData.h>
#include <vtkDataArray.h>
#include <vtkFloatArray.h>
#include <vtkDoubleArray.h>
#include <iostream>
#include <string>
#include <array>
#include <memory>
#include <vector>
#include <cstring>
#include <algorithm>

#include "../headers_cpp/matrix.h"


Matrix read_pvtr(const std::string& filename);

void get_coord(Matrix& X, Matrix& Y, Matrix& Z, const std::string& filename);



