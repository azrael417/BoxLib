# MPI reconfiguration...

ifndef MPI
$(error THIS FILE SHOULD ONLY BE LOADED WITH MPI defined)
endif

mpi_libraries += -lmpi_mpifh

FC  = mpif90
F90 = mpif90
