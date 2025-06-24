    module null_finder
    implicit none
    
    double precision, parameter :: one = 1., err=-999.
    
    contains
    
    subroutine find_nulls(arr, nx, ny, nz, nulls, n_nulls)
    double precision, intent(in), dimension(nx, ny, nz, 3) :: arr
    integer, intent(in) :: nx, ny, nz
    integer :: ix, iy, iz
    double precision, dimension(2, 2, 2, 3) :: arr_i
    logical :: is_null
    integer, intent(out), dimension(nx*ny*nz, 3) :: nulls
    integer, intent(out) :: n_nulls
    
    !open(unit=101, file='nulls.txt', status='replace')
    !open(unit=100, file='nulls2.txt', status='replace')
    !close(100)
    
    n_nulls = 0    
    do iz=1,nz-1
        do iy=1,ny-1
            do ix=1,nx-1
                
                arr_i = arr(ix:ix+1, iy:iy+1, iz:iz+1, :)
                
                call check_signs(arr_i, is_null)
                !write(101, 200) ix, iy, iz, is_null
                
                if(is_null) then
                    call check_null(arr_i, is_null)
                    if(is_null) then
                        n_nulls = n_nulls+1
                        nulls(n_nulls, :) = (/ix-1, iy-1, iz-1/)
                    end if
                end if
                
            end do
        end do 
    end do
    
    !close(101)
200 format(3(I5, ','), L5)
    
    end subroutine find_nulls
    
    subroutine check_signs(cell, is_sign_change)
    double precision, dimension(2, 2, 2, 3), intent(in) :: cell
    integer :: ix, iy, iz, ic
    logical, intent(out) :: is_sign_change
    double precision :: si, s0
    
    
    !open(unit=100, file='nulls2.txt', access='append')
    
    do ic=1,3
        s0 = sign(one, cell(2, 2, 2, ic))
        
        is_sign_change = .false.
        do iz=1,2
            do iy=1,2
                do ix=1,2
                    
                    si = sign(one, cell(ix, iy, iz, ic))
                    
                    !write(100, 200) '', ic, ix, iy, iz, s0, si, s0.ne.si
                    
                    ! If signs aren't equal, a sign change has occured
                    if(s0.ne.si) then
                        is_sign_change = .true.
                        exit
                    end if     
                    
                end do
                if(is_sign_change) exit
            end do
            if(is_sign_change) exit
        end do
        
        ! All three components must have a sign change
        !check_signs = check_signs.and.is_sign_change
        
        ! If one component has signs that are all equal, no null point can exist
        !   i.e. is_sign_change == .false.
        !write(100, 201) '', ic, is_sign_change
        if(.not.is_sign_change) exit ! If 
        
    end do
    
200 format(A5, 4(I5, ','), 2(F6.2, ','), L5)
201 format(A5, I5, ',', L5)
    
    !write(100, *) is_sign_change
    !close(100)
    
    end subroutine check_signs
    
    subroutine check_null(cell, is_null)
    double precision, dimension(2, 2, 2, 3), intent(in) :: cell
    logical, intent(out) :: is_null
    double precision, dimension(6, 4, 3) :: coeff
    integer :: ic, ic1, ic2, ir, i
    double precision, dimension(6, 2) :: x0, y0, vi, sign_vi
    logical, dimension(6, 2) :: root
    double precision :: s0
    
    is_null = .false.
    
    !open(unit=100, file='nulls3.txt', status='replace')
    
    ! Find the bilinear coefficients for all faces
    do ic=1,3
        call face_coeff(cell(:, :, :, ic), coeff(:, :, ic))
        !write(100, 200) ic
        !do i=1,6
            !write(100, 201) '', i, coeff(i, :, ic)
        !end do 
    end do
200 format(I3)
201 format(A3, I3, 4(',', F6.2))
    
    ! Find all roots    
    do ic=1,3
        
        ic1 = mod(ic, 3)+1
        ic2 = mod(ic+1, 3)+1
        !write(100, 203) ic, ic1, ic2
        
        ! For all the faces, need to find the root of two bilinear functions, then find 
        ! the value of the 3rd component at those points.
        ! If a pair of them changes sign, there must be a null inside
        do i=1,6
            ! Calculate the roots
            call find_root(coeff(i, :, ic), coeff(i, :, ic1), x0(i, :), y0(i, :), root(i, :))
            ! Filter roots which are out of bounds
            call filter_roots(x0(i, :), y0(i, :), root(i, :))
            
            ! Find the sign of the 3rd component
            do ir=1,2
                if(root(i, ir)) then
                    sign_vi(i, ir) = bilinear_interp(x0(i, ir), y0(i, ir), coeff(i, :, ic2))
                    sign_vi(i, ir) = sign(one, sign_vi(i, ir))
                else
                    vi(i, ir) = err
                    sign_vi(i, ir) = err
                end if
            end do
            
            !write(100, 202) '', i, x0(i, :), y0(i, :), root(i, :), sign_vi(i, :)
        end do
        
        
        ! Compare the signs of the 3rd component's values
        s0 = 0.
        do i=1,6
            do ir=1,2
                if(root(i, ir)) then
                    s0 = sign_vi(i, ir)
                    exit
                end if
            end do
            if(s0 .ne. 0.) exit
        end do
            
        if(s0.eq.0.) then
            !write(100, *) 'No root in ', ic, ic1, ic2
            cycle
        end if
        
        do i=1,6
            do ir=1,2
                if(root(i, ir)) then
                    is_null = s0.ne.sign_vi(i, ir)
                end if
                if(is_null) exit
            end do
            if(is_null) exit
        end do
        
        if(is_null) exit
    end do
    
    !close(100)
    
203 format(3(I2))
202 format(A3, I2, 4(',', F10.2), 2(',', L3), 2(',', F10.2))
    
    
    end subroutine check_null
    
    subroutine face_coeff(cell, coeff)
    ! Calculates the bilinear coefficients on each face of a cell
    double precision, dimension(2, 2, 2), intent(in) :: cell
    double precision, dimension(6, 4), intent(out) :: coeff
    
    call bilinear_coeff(cell(1, :, :), coeff(1, :))
    call bilinear_coeff(cell(2, :, :), coeff(2, :))
    
    call bilinear_coeff(cell(:, 1, :), coeff(3, :))
    call bilinear_coeff(cell(:, 2, :), coeff(4, :))
    
    call bilinear_coeff(cell(:, :, 1), coeff(5, :))
    call bilinear_coeff(cell(:, :, 2), coeff(6, :))
    
    end subroutine face_coeff
    
    subroutine bilinear_coeff(f, coeff)
    ! Calculates the bilinear coefficients
    double precision, dimension(2, 2), intent(in) :: f
    double precision, dimension(4), intent(out) :: coeff
    
    coeff(1) = f(1, 1) ! a
    coeff(2) = f(2, 1) - f(1, 1) ! b
    coeff(3) = f(1, 2) - f(1, 1) ! c
    coeff(4) = f(2, 2) - f(1, 2) - f(2, 1) + f(1, 1) ! d
    
    end subroutine bilinear_coeff
    
    double precision function bilinear_interp(xi, yi, coeff)
    double precision, intent(in) :: xi, yi
    double precision, dimension(4), intent(in) :: coeff
    
    bilinear_interp = coeff(1)+coeff(2)*xi+coeff(3)*yi+coeff(4)*xi*yi
    
    end function bilinear_interp
    
    subroutine find_root(c1, c2, x0, y0, root)
    double precision, intent(in), dimension(4) :: c1, c2
    double precision, intent(out), dimension(2) :: x0, y0
    logical, dimension(2), intent(out) :: root
    double precision :: A, B, C, det
    integer :: i
    
    ! Find A, B and C coeff
    C = c1(1)*c2(3) - c1(3)*c2(1)
    B = c1(1)*c2(4) - c1(4)*c2(1) + c1(2)*c2(3) - c1(3)*c2(2)
    A = c1(2)*c2(4) - c1(4)*c2(2)
    
    ! Find roots
    if(A.ne.0.) then ! Quadratic Roots
        det = B**2-4*A*C
        if(det.gt.0) then
            x0(1) = 0.5*(-B + sqrt(det))/A
            x0(2) = 0.5*(-B - sqrt(det))/A
            root = (/.true., .true./)
        elseif(det.eq.0.) then
            x0(1) = -0.5*B/A
            x0(2) = err
            root = (/.true., .false./)
        else
            x0 = (/err, err/)
            root = (/.false., .false./)
        end if
        
    elseif(B.ne.0.) then ! Linear Root
            x0(1) = -C/B
            x0(2) = err
            root = (/.true., .false./)
            
    else ! No roots
        x0 = (/err, err/)
        root = (/.false., .false./)
    end if
    
    
    ! Calculate y
    do i=1,2
        if(root(i)) then
            if(c1(3).ne.0..or.c1(4).ne.0.) then
                y0(i) = -(c1(1) + c1(2)*x0(i))/(c1(3) + c1(4)*x0(i))
            else
                y0(i) = -(c2(1) + c2(2)*x0(i))/(c2(3) + c2(4)*x0(i))
            end if
        else
            y0(i) = err
        end if
    end do
    
    end subroutine find_root
    
    subroutine filter_roots(x0, y0, root)
    double precision, intent(inout), dimension(2) :: x0, y0
    logical, dimension(2), intent(inout) :: root
    integer :: i
    
    do i=1,2
        if(x0(i).le.0. .or. x0(i).ge.1. .or. &
           y0(i).le.0. .or. y0(i).ge.1.) then
            x0(i) = err
            y0(i) = err
            root(i) = .false.
        end if            
    end do
    
    end subroutine filter_roots
    
    end module null_finder