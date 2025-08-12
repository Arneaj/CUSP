#include "../headers_cpp/read_pvtr.h"





Matrix read_pvtr( const std::string& filename )
{
    vtkSmartPointer<vtkXMLPRectilinearGridReader> reader = 
        vtkSmartPointer<vtkXMLPRectilinearGridReader>::New();
    
    reader->SetFileName(filename.c_str());
    reader->Update();
    
    // Get the output
    vtkRectilinearGrid* grid = reader->GetOutput();

    if (!grid) { std::cerr << "ERROR: could not read file " << filename << std::endl; exit(1); }

    // Get grid dimensions
    int* dims = grid->GetDimensions();
    
    // Get information about cell data arrays
    vtkCellData* cellData = grid->GetCellData();
    int numCellArrays = cellData->GetNumberOfArrays();

    if (numCellArrays <= 0) { std::cout << "ERROR: no cell arrays available\n"; exit(1); }

    vtkDataArray* firstArray = cellData->GetArray(0);
    
    // Get array information
    vtkIdType numTuples = firstArray->GetNumberOfTuples();
    int numComponents = firstArray->GetNumberOfComponents();
    vtkIdType totalSize = numTuples * numComponents;
    
    // std::cout << "\nExtracting array: " << firstArray->GetName() << std::endl;
    // std::cout << "Total size: " << totalSize << std::endl;
    
    // Allocate memory for the extracted data
    double* extractedData(new double[totalSize]);
    
    // Method 1: Direct pointer access (fastest, but type-dependent)
    if (firstArray->GetDataType() == VTK_DOUBLE) 
    {
        double* vtkData = static_cast<vtkDoubleArray*>(firstArray)->GetPointer(0);
        std::memcpy(extractedData, vtkData, totalSize * sizeof(double));
    }
    else if (firstArray->GetDataType() == VTK_FLOAT) 
    {
        double* vtkData = static_cast<vtkDoubleArray*>(firstArray)->GetPointer(0);
        // Convert from float to double
        for (vtkIdType j = 0; j < totalSize; j++) {
            extractedData[j] = static_cast<double>(vtkData[j]);
        }
    }
    else 
    {
        // Method 2: Generic approach using GetTuple (slower but works for any type)
        std::vector<double> tuple(numComponents);
        for (vtkIdType tupleIdx = 0; tupleIdx < numTuples; tupleIdx++) {
            firstArray->GetTuple(tupleIdx, tuple.data());
            for (int comp = 0; comp < numComponents; comp++) {
                extractedData[tupleIdx * numComponents + comp] = static_cast<double>(tuple[comp]);
            }
        }
    }

    int cellDimX = dims[0]-1; 
    int cellDimY = dims[1]-1; 
    int cellDimZ = dims[2]-1; 

    double* finalData(new double[totalSize]);
    
    #pragma omp parallel for
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

    Shape sh( cellDimX, cellDimY, cellDimZ, numComponents );

    return Matrix( sh, finalData );
}


void get_coord(Matrix& X, Matrix& Y, Matrix& Z, const std::string& filename)
{
    vtkSmartPointer<vtkXMLPRectilinearGridReader> reader = 
        vtkSmartPointer<vtkXMLPRectilinearGridReader>::New();
    
    reader->SetFileName(filename.c_str());
    reader->Update();
    
    // Get the output
    vtkRectilinearGrid* grid = reader->GetOutput();
    
    if (!grid) { std::cerr << "ERROR: could not read file " << filename << std::endl; exit(1); }
    
    // Get grid dimensions
    int* dims = grid->GetDimensions();
        
    // Extract coordinate arrays (X, Y, Z)
    vtkDataArray* xCoords = grid->GetXCoordinates();
    vtkDataArray* yCoords = grid->GetYCoordinates();
    vtkDataArray* zCoords = grid->GetZCoordinates();

    // Extract coordinate arrays to double*
    auto extractCoordinates = [](vtkDataArray* coordArray) -> double* {
        vtkIdType numValues = coordArray->GetNumberOfTuples();
        double* coords(new double[numValues]);
        
        if (coordArray->GetDataType() == VTK_DOUBLE) 
        {
            double* vtkData = static_cast<vtkDoubleArray*>(coordArray)->GetPointer(0);
            std::memcpy(coords, vtkData, numValues * sizeof(double));
        }
        else if (coordArray->GetDataType() == VTK_FLOAT) 
        {
            double* vtkData = static_cast<vtkDoubleArray*>(coordArray)->GetPointer(0);
            for (vtkIdType i = 0; i < numValues; i++) {
                coords[i] = static_cast<double>(vtkData[i]);
            }
        }
        else 
        {
            // Generic method
            for (vtkIdType i = 0; i < numValues; i++) {
                coords[i] = static_cast<double>(coordArray->GetTuple1(i));
            }
        }

        double* cell_coords(new double[numValues-1]);

        for (int i=0; i<numValues-1; i++) cell_coords[i] = (coords[i]+coords[i+1])*0.5;

        delete[] coords;
        
        return cell_coords;
    };
    
    double* xArray = extractCoordinates(xCoords);
    double* yArray = extractCoordinates(yCoords);
    double* zArray = extractCoordinates(zCoords);

    Shape sh_x( dims[0]-1, 1, 1, 1 );
    Shape sh_y( dims[1]-1, 1, 1, 1 );
    Shape sh_z( dims[2]-1, 1, 1, 1 );

    X = Matrix(sh_x, xArray);
    Y = Matrix(sh_y, yArray);
    Z = Matrix(sh_z, zArray);
}

