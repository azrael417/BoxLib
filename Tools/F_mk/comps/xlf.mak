  ifdef OMP
    FC  := xlf95_r
    F90 := xlf95_r
    CC  := xlc_r
    CXX := xlC_r

    F90FLAGS += -qsmp=noauto:omp
    FFLAGS   += -qsmp=noauto:omp
    CFLAGS   += -qsmp=noauto:omp
    CXXFLAGS += -qsmp=noauto:omp
  else
    FC  := xlf95_r
    F90 := xlf95_r
    CC  := xlc_r
    CXX := xlC_r
  endif 

  F90FLAGS += -I $(mdir) -qmoddir=$(mdir)
  FFLAGS   += -I $(mdir) -qmoddir=$(mdir)

  F_C_LINK := LOWERCASE

  ifdef NDEBUG
    F90FLAGS += -O 
    FFLAGS += -O 
    CFLAGS += -O
    CXXFLAGS += -O
  else
    F90FLAGS += -g 
    FFLAGS += -g 
    CFLAGS += -g
    CXXFLAGS += -g
  endif
