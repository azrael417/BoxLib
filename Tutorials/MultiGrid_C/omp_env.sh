# When 1 MPI task running on 1 power chip                                                               

export OMP_NESTED=TRUE                                                                                 
export OMP_NUM_THREADS=20,4                                                                            
export OMP_PLACES=cores                                                                                
export OMP_THREAD_LIMIT=80                                                                             
export OMP_PROC_BIND=spread,close   

