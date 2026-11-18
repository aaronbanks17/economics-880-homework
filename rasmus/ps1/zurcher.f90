module zurcher_mod
    use, intrinsic :: iso_fortran_env, only: dp => real64
    implicit none
    type busdata
        integer :: id
        integer :: type
        integer :: start_month
        type(bus_monthly_obs), allocatable :: m(:)
    end type busdata

    type bus_monthly_obs
        integer :: replace
        integer :: odo_primo
        integer :: odo_ultimo
        integer :: odo_ultimo_total
        integer :: odo_delta
    end type bus_monthly_obs
contains
    subroutine readbusdata(mybus)
        type(busdata), allocatable, intent(out) :: mybus(:)
        type busdata_line
            integer :: id
            integer :: type
            integer :: year
            integer :: month
            integer :: replace
            integer :: odo_primo
            integer :: odo_ultimo
            integer :: odo_ultimo_total
            integer :: odo_delta
        end type busdata_line
        
        type(busdata_line) :: obs 
        type(busdata_line) :: obs_stack(200) !!Make large enough to hold all monthly observations of any bus.
        type(busdata) :: bus_temp(500) !Make large enough to hold all buses temporarily

        integer :: b, i, j

        open(unit=10, file='busdata.csv', status='old', action='read')
        read(10,*) obs_stack(1)

        i=1
        b=0
        do
            read(10,*,iostat=j) obs
            if (j /= 0) exit
            if(obs%id == obs_stack(i)%id) then
                i = i + 1
                obs_stack(i) = obs
            else
                call add_to_bus_temp
                i = 1
                obs_stack(i) = obs
            end if
        end do
        call add_to_bus_temp
        close(10)
        allocate(mybus(b))
        mybus = bus_temp(1:b)
    contains
        subroutine add_to_bus_temp
            b=b+1
            bus_temp(b)%id = obs_stack(1)%id
            bus_temp(b)%type = obs_stack(1)%type
            bus_temp(b)%start_month = (obs_stack(1)%year-75)*12 + obs_stack(1)%month-6 !month 1 is July 1975
            allocate(bus_temp(b)%m(i))
            do j=1,i
                bus_temp(b)%m(j)%replace = obs_stack(j)%replace
                bus_temp(b)%m(j)%odo_primo = obs_stack(j)%odo_primo
                bus_temp(b)%m(j)%odo_ultimo = obs_stack(j)%odo_ultimo
                bus_temp(b)%m(j)%odo_ultimo_total = obs_stack(j)%odo_ultimo_total
                bus_temp(b)%m(j)%odo_delta = obs_stack(j)%odo_delta
            end do
        end subroutine add_to_bus_temp
    end subroutine readbusdata  
end module zurcher_mod   
