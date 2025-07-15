#include "../headers_cpp/read_pvtr.h"





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
    
    // Get information about cell data arrays
    vtkCellData* cellData = grid->GetCellData();
    int numCellArrays = cellData->GetNumberOfArrays();

    if (numCellArrays <= 0) exit(1);

    vtkDataArray* firstArray = cellData->GetArray(0);
    
    // Get array information
    vtkIdType numTuples = firstArray->GetNumberOfTuples();
    int numComponents = firstArray->GetNumberOfComponents();
    vtkIdType totalSize = numTuples * numComponents;
    
    // std::cout << "\nExtracting array: " << firstArray->GetName() << std::endl;
    // std::cout << "Total size: " << totalSize << std::endl;
    
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

    int cellDimX = dims[0]-1; 
    int cellDimY = dims[1]-1; 
    int cellDimZ = dims[2]-1; 

    float* finalData(new float[totalSize]);
    
    for (int ix = 0; ix < cellDimX; ix++) {
        for (int iy = 0; iy < cellDimY; iy++) {
            for (int iz = 0; iz < cellDimZ; iz++) {
                for (int i = 0; i < numComponents; i++) {
                    int srcIndex = ((iz*cellDimY + iy)*cellDimX + ix)*numComponents + i;
                    int dstIndex = ((i*cellDimZ + iz)*cellDimY + iy)*cellDimX + ix;
                    
                    finalData[dstIndex] = extractedData[srcIndex];
                }
            }
        }
    }
    
    delete[] extractedData;

    std::cout << "hello4" << std::endl;

    Shape sh( cellDimX, cellDimY, cellDimZ, numComponents );

    return Matrix( sh, finalData );
}


void get_coord(Matrix& X, Matrix& Y, Matrix& Z, std::string filename)
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

    Shape sh_x( dims[0]-1, 1, 1, 1 );
    Shape sh_y( dims[1]-1, 1, 1, 1 );
    Shape sh_z( dims[2]-1, 1, 1, 1 );

    X = Matrix(sh_x, xArray);
    Y = Matrix(sh_y, yArray);
    Z = Matrix(sh_z, zArray);
}

