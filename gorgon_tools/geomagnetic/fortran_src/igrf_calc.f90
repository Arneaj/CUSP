module igrf_functions
contains
    double precision function factorial(n)
    ! Returns the factorial of a given integer n.
    implicit none
    integer, intent(in) :: n
    integer :: i
    
    factorial = 1.
    do i = 1,n
        factorial = factorial*i
    end do
    end function
    
    double precision function legml(m,l,x)
    ! Solves the associated legendre function of order m and degree l
    ! for a given x. 
    implicit none
    integer, intent(in) :: m, l
    double precision, intent(in) :: x
    integer :: n
    double precision :: leg_sum, x_tmp
    logical :: flag
    
    if(x >= 0) then
        x_tmp = -x
        flag = .true.
    else
        x_tmp = x
        flag = .false.
    end if
    
    leg_sum = 0.
    do n=0,l
        leg_sum = leg_sum + factorial(l+n)/factorial(l-n) * (-1)**n/(factorial(m+n)*factorial(n)) * ((1.-x_tmp)/2.)**n
    end do
    legml = ((1.-x_tmp)/(1.+x_tmp))**(m/2.)*leg_sum
    if(m>0) legml = (-1)**m*factorial(l+m)/factorial(l-m)*legml
    if(flag) legml = (-1)**(l+m)*legml
    end function

end module

module igrf_subs
    contains

    subroutine igrf_grid(r, th, az, n, g, h, order, Br, Bth, Baz)
    ! Calculates the IGRF-13 magnetic field components on a grid
    ! in spherical coordinates with arrays of length n.
        use igrf_functions
        IMPLICIT NONE
        integer, intent(in) :: n, order
        double precision, intent(in), dimension(0:n-1) :: r, th, az
        double precision, intent(in), dimension(0:order-1,0:order) :: g, h
        double precision, intent(out), dimension(0:n-1) :: Br, Bth, Baz
        integer :: i
        double precision :: M, th_dip, az_dip
        !call read_gauss_coeffs(g, h, order)
        M = sqrt(g(0,0)**2+g(0,1)**2+h(0,1)**2)
        th_dip = acos(-g(0,0)/M)
        az_dip = asin(-h(0,1)/sqrt(g(0,1)**2+h(0,1)**2))
        do i=0,n-1
            call igrf_expansion(r(i), th(i), az(i), g, h, order, Br(i), Bth(i), Baz(i))
        end do
    end subroutine

    subroutine read_gauss_coeffs(g, h, order)
    ! Reads data files containing an array of Gauss coefficients g^n_m, h^n_m
    ! where the arrays have size order x (order+1) and are in nT. 
        IMPLICIT NONE
        integer :: i
        integer, intent(in) :: order
        double precision, dimension(0:order-1,0:order), intent(out) :: g, h

        ! g coefficients
        open (unit=100, file='IGRF-13_g_2020.dat')
        !--- Read values into arrays ----------------
        do i = 1, order
            read(100,200) g(i-1,0:order)
        end do
    200         format(ES12.5, 13(1x, ES12.5))
        close (100)
        
        ! h coefficients
        open (unit=100, file='IGRF-13_h_2020.dat')
        !--- Read values into arrays ----------------
        do i = 1, order
            read(100,201) h(i-1,0:order)
        end do
    201         format(ES12.5, 13(1x, ES12.5))
        close (100)
    end subroutine

    subroutine igrf_expansion(r, th, az, g, h, order, Br, Bth, Baz)
    ! Performs a spherical harmonic expansion to solve B = -grad(U),
    ! where U(r,theta,phi) is magnetic scalar potential. Requires
    ! Schmidt semi-normalised Gauss coefficients g^m_n, h^m_n taken
    ! from e.g. IGRF-13.
        use igrf_functions
        implicit none
        integer, intent(in) :: order
        double precision, intent(in) :: r, th, az
        double precision, dimension(0:order-1,0:order), intent(in) :: g, h
        double precision, intent(out) :: Br, Bth, Baz
        integer :: i, j, j_fix
        double precision :: z, s, dleg, cos_j, sin_j, coeff1, coeff2, coeff3, coeff4, coeff5, coeff6
        double precision, dimension(0:order+1) :: sleg, leg1, leg2   

        z = cos(th)
        s = sin(th)
        Br = 0.; Bth = 0.; Baz = 0.

        do i=1,order
            call schmidtlegendres(i,z,sleg)
            call legendres(i,z,leg1)
            call legendres(i+1,z,leg2)
            do j=1,i+1
                if(j==1) then
                    j_fix = 0
                else
                    j_fix = 1   
                end if
                dleg = (j_fix*(-1)**(j+1)*sqrt(2.*factorial(i-j+1)/factorial(i+j-1)) &
                    + (-j_fix+1))*((i+1)*z*leg1(j-1)-(i-j+2)*leg2(j-1))/(1.-z**2)
                cos_j = cos((j-1)*az)
                sin_j = sin((j-1)*az)
                coeff1 = (i+1)*(1./r)**(i+2)*cos_j*sleg(j-1)
                coeff2 = (i+1)*(1./r)**(i+2)*sin_j*sleg(j-1)
                coeff3 = (1./r)**(i+2)*cos_j*dleg*s
                coeff4 = (1./r)**(i+2)*sin_j*dleg*s
                coeff5 = (j-1)*(1./r)**(i+2)*sin_j*sleg(j-1)/s
                coeff6 = -(j-1)*(1./r)**(i+2)*cos_j*sleg(j-1)/s

                Br  = Br  + coeff1*g(i-1,j-1) + coeff2*h(i-1,j-1)
                Bth = Bth + coeff3*g(i-1,j-1) + coeff4*h(i-1,j-1)
                Baz = Baz + coeff5*g(i-1,j-1) + coeff6*h(i-1,j-1)
            end do
        end do
    end subroutine

    subroutine legendres(N, x, P)
    ! Calculates the associated legendre polynomial for a given order n
    ! and all degrees m, returned as an array of length n+1.
        use igrf_functions
        implicit none
        integer, intent(in) :: N
        double precision, intent(in) :: x
        double precision, dimension(0:N), intent(out) :: P   
        integer :: m
                
        do m=0,N
            P(m) = legml(m,N,x)
        end do    
    end subroutine
        
    subroutine schmidtlegendres(N, x, S)
    ! Calculates the Schmidt-normalised associated legendre polynomial 
    ! for a given order n and all degrees m, returned as an array of 
    ! length n+1.
        use igrf_functions
        implicit none
        integer, intent(in) :: N
        double precision, intent(in) :: x
        double precision, dimension(0:N), intent(out) :: S    
        integer :: m
        double precision :: C
        
        do m=0,N
            ! normalisation constants
            C = (-1)**m*sqrt(2.*factorial(N-m)/factorial(N+m))
            if(m==0) C = 1            
            S(m) = C*legml(m,N,x)
        end do
    end subroutine 
end module 