from gorgon import import_from, filename


B,V,_,_ = import_from(filename)


STEP = 2


with open("../data/B.txt", "w") as f:
    f.write( f"{B.shape[0]//STEP},{B.shape[1]//STEP},{B.shape[2]//STEP},{B.shape[3]}\n" )
    
    for ix in range(0, 240, STEP):
        print( f"{ round(100*ix/240, 2) }%" )
        for iy in range(0, 160, STEP):
            for iz in range(0, 160, STEP):
                for i in range(3):
                    f.write( str(B[ix,iy,iz,i]) )
                    f.write( "," )

print("finished writing B")
print()


with open("../data/V.txt", "w") as f:
    f.write( f"{V.shape[0]//STEP},{V.shape[1]//STEP},{V.shape[2]//STEP},{V.shape[3]}\n" )
    
    for ix in range(0, 240, STEP):
        print( f"{ round(100*ix/240, 2) }%" )
        for iy in range(0, 160, STEP):
            for iz in range(0, 160, STEP):
                for i in range(3):
                    f.write( str(V[ix,iy,iz,i]) )
                    f.write( "," )


print("finished writing V")


