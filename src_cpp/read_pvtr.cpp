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


Matrix read_pvtr(std::string filename)
{
    vtkSmartPointer<vtkXMLPRectilinearGridReader> reader = 
        vtkSmartPointer<vtkXMLPRectilinearGridReader>::New();
    
    reader->SetFileName(filename.c_str());
    reader->Update();
    
    // Get the output
    vtkRectilinearGrid* grid = reader->GetOutput();

    if (!grid) 
    {
        std::cerr << "Error: Could not read file " << filename << std::endl;
        exit(1);
    }

    // Get grid dimensions
    int* dims = grid->GetDimensions();
    std::array<int, 4> dimensions = {dims[0], dims[1], dims[2], 1}; // 4th dimension for components
    
    // Get information about cell data arrays
    vtkCellData* cellData = grid->GetCellData();
    int numCellArrays = cellData->GetNumberOfArrays();

    if (numCellArrays <= 0) exit(1);

    vtkDataArray* firstArray = cellData->GetArray(0);
    
    // Get array information
    vtkIdType numTuples = firstArray->GetNumberOfTuples();
    int numComponents = firstArray->GetNumberOfComponents();
    vtkIdType totalSize = numTuples * numComponents;
    
    // Update dimensions array with actual number of components
    dimensions[3] = numComponents;
    
    std::cout << "\nExtracting array: " << firstArray->GetName() << std::endl;
    std::cout << "Total size: " << totalSize << std::endl;
    
    // Allocate memory for the extracted data
    float* extractedData(new float[totalSize]);
    
    // Method 1: Direct pointer access (fastest, but type-dependent)
    if (firstArray->GetDataType() == VTK_FLOAT) 
    {
        float* vtkData = static_cast<vtkFloatArray*>(firstArray)->GetPointer(0);
        std::memcpy(extractedData, vtkData, totalSize * sizeof(float));
    }
    else if (firstArray->GetDataType() == VTK_DOUBLE) 
    {
        double* vtkData = static_cast<vtkDoubleArray*>(firstArray)->GetPointer(0);
        // Convert from double to float
        for (vtkIdType j = 0; j < totalSize; j++) {
            extractedData[j] = static_cast<float>(vtkData[j]);
        }
    }
    else 
    {
        // Method 2: Generic approach using GetTuple (slower but works for any type)
        std::vector<double> tuple(numComponents);
        for (vtkIdType tupleIdx = 0; tupleIdx < numTuples; tupleIdx++) {
            firstArray->GetTuple(tupleIdx, tuple.data());
            for (int comp = 0; comp < numComponents; comp++) {
                extractedData[tupleIdx * numComponents + comp] = static_cast<float>(tuple[comp]);
            }
        }
    }

    Shape sh( dimensions[0], dimensions[1], dimensions[2], dimensions[3] );

    return Matrix( sh, extractedData );
}


std::array<Matrix, 3> get_coord(std::string filename)
{
    vtkSmartPointer<vtkXMLPRectilinearGridReader> reader = 
        vtkSmartPointer<vtkXMLPRectilinearGridReader>::New();
    
    reader->SetFileName(filename.c_str());
    reader->Update();
    
    // Get the output
    vtkRectilinearGrid* grid = reader->GetOutput();
    
    if (!grid) 
    {
        std::cerr << "Error: Could not read file " << filename << std::endl;
        exit(1);
    }
    
    // Get grid dimensions
    int* dims = grid->GetDimensions();
        
    // Extract coordinate arrays (X, Y, Z)
    vtkDataArray* xCoords = grid->GetXCoordinates();
    vtkDataArray* yCoords = grid->GetYCoordinates();
    vtkDataArray* zCoords = grid->GetZCoordinates();

    // Extract coordinate arrays to float*
    auto extractCoordinates = [](vtkDataArray* coordArray) -> float* {
        vtkIdType numValues = coordArray->GetNumberOfTuples();
        float* coords(new float[numValues]);
        
        if (coordArray->GetDataType() == VTK_FLOAT) 
        {
            float* vtkData = static_cast<vtkFloatArray*>(coordArray)->GetPointer(0);
            std::memcpy(coords, vtkData, numValues * sizeof(float));
        }
        else if (coordArray->GetDataType() == VTK_DOUBLE) 
        {
            double* vtkData = static_cast<vtkDoubleArray*>(coordArray)->GetPointer(0);
            for (vtkIdType i = 0; i < numValues; i++) {
                coords[i] = static_cast<float>(vtkData[i]);
            }
        }
        else 
        {
            // Generic method
            for (vtkIdType i = 0; i < numValues; i++) {
                coords[i] = static_cast<float>(coordArray->GetTuple1(i));
            }
        }
        
        return coords;
    };
    
    float* xArray = extractCoordinates(xCoords);
    float* yArray = extractCoordinates(yCoords);
    float* zArray = extractCoordinates(zCoords);

    Shape sh_x( dims[0], 1, 1, 1 );
    Shape sh_y( dims[1], 1, 1, 1 );
    Shape sh_z( dims[2], 1, 1, 1 );

    return std::array<Matrix, 3>({
        Matrix(sh_x, xArray),
        Matrix(sh_y, yArray),
        Matrix(sh_z, zArray)
    });
}


int main()
{
    std::string filename("/rds/general/user/avr24/projects/swimmr-sage/live/mheyns/benchmarking/runs/Run1/MS/x00_Bvec_c-21000.pvtr");

    Matrix B = read_pvtr(filename);
    
    std::array<Matrix, 3> XYZ = get_coord(filename);

    std::cout << "Shape of B: " << B.get_shape() << std::endl;
    std::cout << "B(0,0,0,0) = " << B(0,0,0,0) << std::endl;
}


// void processCellData(float* data, const std::array<int, 4>& dimensions, 
//                     float* xCoords, float* yCoords, float* zCoords);

// int main(int argc, char* argv[]) {
//     if (argc != 2) {
//         std::cerr << "Usage: " << argv[0] << " <pvtr_file>" << std::endl;
//         return 1;
//     }

//     std::string filename = argv[1];
    
//     // Create a smart pointer to the appropriate reader
//     // For PVTR files, you typically use vtkXMLPRectilinearGridReader
//     vtkSmartPointer<vtkXMLPRectilinearGridReader> reader = 
//         vtkSmartPointer<vtkXMLPRectilinearGridReader>::New();
    
//     reader->SetFileName(filename.c_str());
//     reader->Update();
    
//     // Get the output
//     vtkRectilinearGrid* grid = reader->GetOutput();
    
//     if (!grid) {
//         std::cerr << "Error: Could not read file " << filename << std::endl;
//         return 1;
//     }
    
//     // Print basic information about the dataset
//     std::cout << "Number of points: " << grid->GetNumberOfPoints() << std::endl;
//     std::cout << "Number of cells: " << grid->GetNumberOfCells() << std::endl;
    
//     // Get grid dimensions
//     int* dims = grid->GetDimensions();
//     std::array<int, 4> dimensions = {dims[0], dims[1], dims[2], 1}; // 4th dimension for components
    
//     std::cout << "Grid dimensions: " << dimensions[0] << " x " << dimensions[1] << " x " << dimensions[2] << std::endl;
    
//     // Extract coordinate arrays (X, Y, Z)
//     vtkDataArray* xCoords = grid->GetXCoordinates();
//     vtkDataArray* yCoords = grid->GetYCoordinates();
//     vtkDataArray* zCoords = grid->GetZCoordinates();
    
//     std::cout << "\nCoordinate arrays:" << std::endl;
//     std::cout << "X coordinates: " << xCoords->GetNumberOfTuples() << " values" << std::endl;
//     std::cout << "Y coordinates: " << yCoords->GetNumberOfTuples() << " values" << std::endl;
//     std::cout << "Z coordinates: " << zCoords->GetNumberOfTuples() << " values" << std::endl;
    
//     // Extract coordinate arrays to float*
//     auto extractCoordinates = [](vtkDataArray* coordArray) -> std::unique_ptr<float[]> {
//         vtkIdType numValues = coordArray->GetNumberOfTuples();
//         std::unique_ptr<float[]> coords(new float[numValues]);
        
//         if (coordArray->GetDataType() == VTK_FLOAT) {
//             float* vtkData = static_cast<vtkFloatArray*>(coordArray)->GetPointer(0);
//             std::memcpy(coords.get(), vtkData, numValues * sizeof(float));
//         }
//         else if (coordArray->GetDataType() == VTK_DOUBLE) {
//             double* vtkData = static_cast<vtkDoubleArray*>(coordArray)->GetPointer(0);
//             for (vtkIdType i = 0; i < numValues; i++) {
//                 coords[i] = static_cast<float>(vtkData[i]);
//             }
//         }
//         else {
//             // Generic method
//             for (vtkIdType i = 0; i < numValues; i++) {
//                 coords[i] = static_cast<float>(coordArray->GetTuple1(i));
//             }
//         }
        
//         return coords;
//     };
    
//     std::unique_ptr<float[]> xArray = extractCoordinates(xCoords);
//     std::unique_ptr<float[]> yArray = extractCoordinates(yCoords);
//     std::unique_ptr<float[]> zArray = extractCoordinates(zCoords);
    
//     // Print first few coordinate values
//     std::cout << "\nFirst 5 X coordinates: ";
//     for (int i = 0; i < std::min(5, dimensions[0]); i++) {
//         std::cout << xArray[i] << " ";
//     }
//     std::cout << std::endl;
    
//     std::cout << "First 5 Y coordinates: ";
//     for (int i = 0; i < std::min(5, dimensions[1]); i++) {
//         std::cout << yArray[i] << " ";
//     }
//     std::cout << std::endl;
    
//     std::cout << "First 5 Z coordinates: ";
//     for (int i = 0; i < std::min(5, dimensions[2]); i++) {
//         std::cout << zArray[i] << " ";
//     }
//     std::cout << std::endl;
    
//     // Print information about cell data arrays
//     vtkCellData* cellData = grid->GetCellData();
//     int numCellArrays = cellData->GetNumberOfArrays();
//     std::cout << "\nNumber of cell data arrays: " << numCellArrays << std::endl;
    
//     for (int i = 0; i < numCellArrays; i++) {
//         vtkDataArray* array = cellData->GetArray(i);
//         std::cout << "Cell array " << i << ": " << array->GetName() 
//                   << " (components: " << array->GetNumberOfComponents() << ")" << std::endl;
//     }
    
//     // Example: Extract first cell data array to float*
//     if (numCellArrays > 0) {
//         vtkDataArray* firstArray = cellData->GetArray(0);
        
//         // Get array information
//         vtkIdType numTuples = firstArray->GetNumberOfTuples();
//         int numComponents = firstArray->GetNumberOfComponents();
//         vtkIdType totalSize = numTuples * numComponents;
        
//         // Update dimensions array with actual number of components
//         dimensions[3] = numComponents;
        
//         std::cout << "\nExtracting array: " << firstArray->GetName() << std::endl;
//         std::cout << "Tuples: " << numTuples << ", Components: " << numComponents << std::endl;
//         std::cout << "Total size: " << totalSize << std::endl;
        
//         // Allocate memory for the extracted data
//         std::unique_ptr<float[]> extractedData(new float[totalSize]);
        
//         // Method 1: Direct pointer access (fastest, but type-dependent)
//         if (firstArray->GetDataType() == VTK_FLOAT) {
//             float* vtkData = static_cast<vtkFloatArray*>(firstArray)->GetPointer(0);
//             std::memcpy(extractedData.get(), vtkData, totalSize * sizeof(float));
//         }
//         else if (firstArray->GetDataType() == VTK_DOUBLE) {
//             double* vtkData = static_cast<vtkDoubleArray*>(firstArray)->GetPointer(0);
//             // Convert from double to float
//             for (vtkIdType j = 0; j < totalSize; j++) {
//                 extractedData[j] = static_cast<float>(vtkData[j]);
//             }
//         }
//         else {
//             // Method 2: Generic approach using GetTuple (slower but works for any type)
//             std::vector<double> tuple(numComponents);
//             for (vtkIdType tupleIdx = 0; tupleIdx < numTuples; tupleIdx++) {
//                 firstArray->GetTuple(tupleIdx, tuple.data());
//                 for (int comp = 0; comp < numComponents; comp++) {
//                     extractedData[tupleIdx * numComponents + comp] = static_cast<float>(tuple[comp]);
//                 }
//             }
//         }
        
//         // Print first few values as verification
//         std::cout << "First 10 values: ";
//         for (int j = 0; j < std::min(10, static_cast<int>(totalSize)); j++) {
//             std::cout << extractedData[j] << " ";
//         }
//         std::cout << std::endl;
        
//         // Example of how to use the extracted data in another function
//         processCellData(extractedData.get(), dimensions, xArray.get(), yArray.get(), zArray.get());
//     }
    
//     return 0;
// }

// // Example function that processes the extracted cell data
// void processCellData(float* data, const std::array<int, 4>& dimensions, 
//                     float* xCoords, float* yCoords, float* zCoords) {
//     std::cout << "\nProcessing cell data with dimensions: " 
//               << dimensions[0] << " x " << dimensions[1] << " x " << dimensions[2] 
//               << " x " << dimensions[3] << " components" << std::endl;
    
//     // For cell data, the grid dimensions are reduced by 1 in each direction
//     // Grid dimensions: 181 x 101 x 101 points -> 180 x 100 x 100 cells
//     int cellDimX = dimensions[0] - 1;  // 180
//     int cellDimY = dimensions[1] - 1;  // 100
//     int cellDimZ = dimensions[2] - 1;  // 100
    
//     std::cout << "Cell dimensions: " << cellDimX << " x " << cellDimY << " x " << cellDimZ << std::endl;
    
//     // Example: Access data at specific cell (i, j, k) and component c
//     auto getCellValue = [&](int i, int j, int k, int c) -> float {
//         // For cell data, indexing is similar but with cell dimensions
//         int index = ((k * cellDimY + j) * cellDimX + i) * dimensions[3] + c;
//         return data[index];
//     };
    
//     // Function to get cell center coordinates
//     auto getCellCenter = [&](int i, int j, int k) -> std::array<float, 3> {
//         // Cell centers are at the midpoint between adjacent grid points
//         float x = (xCoords[i] + xCoords[i + 1]) / 2.0f;
//         float y = (yCoords[j] + yCoords[j + 1]) / 2.0f;
//         float z = (zCoords[k] + zCoords[k + 1]) / 2.0f;
//         return {x, y, z};
//     };
    
//     // Example usage - access the Bvec_c vector at cell (0,0,0) and its coordinates
//     if (cellDimX > 0 && cellDimY > 0 && cellDimZ > 0 && dimensions[3] == 3) {
//         auto cellCenter = getCellCenter(0, 0, 0);
//         std::cout << "Cell (0,0,0) center at: (" 
//                   << cellCenter[0] << ", " << cellCenter[1] << ", " << cellCenter[2] << ")" << std::endl;
//         std::cout << "Bvec_c at cell (0,0,0): [" 
//                   << getCellValue(0, 0, 0, 0) << ", "
//                   << getCellValue(0, 0, 0, 1) << ", "
//                   << getCellValue(0, 0, 0, 2) << "]" << std::endl;
        
//         // Show a few more examples
//         for (int example = 1; example < std::min(3, cellDimX); example++) {
//             auto center = getCellCenter(example, 0, 0);
//             std::cout << "Cell (" << example << ",0,0) center at: (" 
//                       << center[0] << ", " << center[1] << ", " << center[2] << ")" << std::endl;
//         }
//     }
// }
