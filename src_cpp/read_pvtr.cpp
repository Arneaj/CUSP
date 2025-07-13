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

void processCellData(float* data, const std::array<int, 4>& dimensions);

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <pvtr_file>" << std::endl;
        return 1;
    }

    std::string filename = argv[1];
    
    // Create a smart pointer to the appropriate reader
    // For PVTR files, you typically use vtkXMLPRectilinearGridReader
    vtkSmartPointer<vtkXMLPRectilinearGridReader> reader = 
        vtkSmartPointer<vtkXMLPRectilinearGridReader>::New();
    
    reader->SetFileName(filename.c_str());
    reader->Update();
    
    // Get the output
    vtkRectilinearGrid* grid = reader->GetOutput();
    
    if (!grid) {
        std::cerr << "Error: Could not read file " << filename << std::endl;
        return 1;
    }
    
    // Print basic information about the dataset
    std::cout << "Number of points: " << grid->GetNumberOfPoints() << std::endl;
    std::cout << "Number of cells: " << grid->GetNumberOfCells() << std::endl;
    
    // Get grid dimensions
    int* dims = grid->GetDimensions();
    std::array<int, 4> dimensions = {dims[0], dims[1], dims[2], 1}; // 4th dimension for components
    
    std::cout << "Grid dimensions: " << dimensions[0] << " x " << dimensions[1] << " x " << dimensions[2] << std::endl;
    
    // Print information about cell data arrays
    vtkCellData* cellData = grid->GetCellData();
    int numCellArrays = cellData->GetNumberOfArrays();
    std::cout << "\nNumber of cell data arrays: " << numCellArrays << std::endl;
    
    for (int i = 0; i < numCellArrays; i++) {
        vtkDataArray* array = cellData->GetArray(i);
        std::cout << "Cell array " << i << ": " << array->GetName() 
                  << " (components: " << array->GetNumberOfComponents() << ")" << std::endl;
    }
    
    // Example: Extract first cell data array to float*
    if (numCellArrays > 0) {
        vtkDataArray* firstArray = cellData->GetArray(0);
        
        // Get array information
        vtkIdType numTuples = firstArray->GetNumberOfTuples();
        int numComponents = firstArray->GetNumberOfComponents();
        vtkIdType totalSize = numTuples * numComponents;
        
        // Update dimensions array with actual number of components
        dimensions[3] = numComponents;
        
        std::cout << "\nExtracting array: " << firstArray->GetName() << std::endl;
        std::cout << "Tuples: " << numTuples << ", Components: " << numComponents << std::endl;
        std::cout << "Total size: " << totalSize << std::endl;
        
        // Allocate memory for the extracted data
        float* extractedData(new float[totalSize]);
        
        // Method 1: Direct pointer access (fastest, but type-dependent)
        if (firstArray->GetDataType() == VTK_FLOAT) {
            float* vtkData = static_cast<vtkFloatArray*>(firstArray)->GetPointer(0);
            std::memcpy(extractedData, vtkData, totalSize * sizeof(float));
        }
        else if (firstArray->GetDataType() == VTK_DOUBLE) {
            double* vtkData = static_cast<vtkDoubleArray*>(firstArray)->GetPointer(0);
            // Convert from double to float
            for (vtkIdType j = 0; j < totalSize; j++) {
                extractedData[j] = static_cast<float>(vtkData[j]);
            }
        }
        else {
            // Method 2: Generic approach using GetTuple (slower but works for any type)
            std::vector<double> tuple(numComponents);
            for (vtkIdType tupleIdx = 0; tupleIdx < numTuples; tupleIdx++) {
                firstArray->GetTuple(tupleIdx, tuple.data());
                for (int comp = 0; comp < numComponents; comp++) {
                    extractedData[tupleIdx * numComponents + comp] = static_cast<float>(tuple[comp]);
                }
            }
        }
        
        // Print first few values as verification
        // std::cout << "First 10 values: ";
        // for (int j = 0; j < std::min(10, static_cast<int>(totalSize)); j++) {
        //     std::cout << extractedData[j] << " ";
        // }
        // std::cout << std::endl;
        
        // Example of how to use the extracted data in another function
        processCellData(extractedData, dimensions);
    }
    
    
    
    return 0;
}

// Example function that processes the extracted cell data
void processCellData(float* data, const std::array<int, 4>& dimensions) {
    std::cout << "\nProcessing cell data with dimensions: " 
              << dimensions[0] << " x " << dimensions[1] << " x " << dimensions[2] 
              << " x " << dimensions[3] << " components" << std::endl;
    
    // For cell data, the grid dimensions are reduced by 1 in each direction
    // Grid dimensions: 181 x 101 x 101 points -> 180 x 100 x 100 cells
    int cellDimX = dimensions[0] - 1;  // 180
    int cellDimY = dimensions[1] - 1;  // 100
    int cellDimZ = dimensions[2] - 1;  // 100
    
    std::cout << "Cell dimensions: " << cellDimX << " x " << cellDimY << " x " << cellDimZ << std::endl;
    
    // Example: Access data at specific cell (i, j, k) and component c
    auto getCellValue = [&](int i, int j, int k, int c) -> float {
        // For cell data, indexing is similar but with cell dimensions
        int index = ((k * cellDimY + j) * cellDimX + i) * dimensions[3] + c;
        return data[index];
    };
    
    // Example usage - access the Bvec_c vector at cell (0,0,0)
    if (cellDimX > 0 && cellDimY > 0 && cellDimZ > 0 && dimensions[3] == 3) {
        std::cout << "Bvec_c at cell (0,0,0): [" 
                  << getCellValue(0, 0, 0, 0) << ", "
                  << getCellValue(0, 0, 0, 1) << ", "
                  << getCellValue(0, 0, 0, 2) << "]" << std::endl;
    }
}

// Example function that processes the extracted point data
void processData(float* data, const std::array<int, 4>& dimensions) {
    std::cout << "\nProcessing point data with dimensions: " 
              << dimensions[0] << " x " << dimensions[1] << " x " << dimensions[2] 
              << " x " << dimensions[3] << " components" << std::endl;
    
    // Example: Access data at specific grid point (i, j, k) and component c
    auto getValue = [&](int i, int j, int k, int c) -> float {
        // For rectilinear grids, data is typically stored in row-major order
        int index = ((k * dimensions[1] + j) * dimensions[0] + i) * dimensions[3] + c;
        return data[index];
    };
    
    // Example usage
    if (dimensions[0] > 0 && dimensions[1] > 0 && dimensions[2] > 0) {
        std::cout << "Value at (0,0,0), component 0: " << getValue(0, 0, 0, 0) << std::endl;
    }
}

// Alternative extraction function for different data types
template<typename T>
std::unique_ptr<float[]> extractVTKArrayToFloat(vtkDataArray* array, vtkIdType& totalSize) {
    vtkIdType numTuples = array->GetNumberOfTuples();
    int numComponents = array->GetNumberOfComponents();
    totalSize = numTuples * numComponents;
    
    std::unique_ptr<float[]> result(new float[totalSize]);
    
    // Direct cast and copy if already the right type
    if constexpr (std::is_same_v<T, float>) {
        if (array->GetDataType() == VTK_FLOAT) {
            T* vtkData = static_cast<vtkFloatArray*>(array)->GetPointer(0);
            std::memcpy(result.get(), vtkData, totalSize * sizeof(float));
            return result;
        }
    }
    
    // Generic conversion
    std::vector<double> tuple(numComponents);
    for (vtkIdType i = 0; i < numTuples; i++) {
        array->GetTuple(i, tuple.data());
        for (int c = 0; c < numComponents; c++) {
            result[i * numComponents + c] = static_cast<float>(tuple[c]);
        }
    }
    
    return result;
}