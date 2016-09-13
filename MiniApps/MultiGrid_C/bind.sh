#!/bin/bash
#--------------------------------------------------------------------------------
# mpirun -np $nmpi bind.sh your.exe [args]
# optionally set BIND_BASE in your job script
# optionally set BIND_STRIDE in your job script
# optionally set BIND_POLICY=packed in your job script
# optionally set BIND_CPU_LIST="cpu0 cpu1 ..." in your job script
# Note : for some OpenMP implementations (GNU OpenMP) use mpirun --bind-to none
#--------------------------------------------------------------------------------
cpus_per_node=`cat /proc/cpuinfo | grep processor | wc -l`

if [ -z "OMPI_COMM_WORLD_LOCAL_SIZE" ]; then

  let OMPI_COMM_WORLD_LOCAL_SIZE=1

  let OMPI_COMM_WORLD_LOCAL_RANK=0

fi

# if OMP_NUM_THREADS is not set, assume no threading and bind with taskset
if [ -z "$OMP_NUM_THREADS" ]; then

  if [ "$OMPI_COMM_WORLD_RANK" == "0" ]; then
    echo OMP_NUM_THREADS is not set ... assuming one thread per MPI rank
  fi

  if [ -z "$BIND_CPU_LIST" ]; then

    if [ -z "$BIND_BASE" ]; then
      let BIND_BASE=0
    fi

    if [ -z "$BIND_STRIDE" ]; then
      let stride=$cpus_per_node/$OMPI_COMM_WORLD_LOCAL_SIZE
    else
      let stride=$BIND_STRIDE
    fi

    let core=$BIND_BASE+$stride*$OMPI_COMM_WORLD_LOCAL_RANK

    taskset -c $core $@

  else

    declare -a cpulist=($BIND_CPU_LIST)

    taskset -c ${cpulist[$OMPI_COMM_WORLD_LOCAL_RANK]} $@

  fi

else
#if OMP_NUM_THREADS is set, bind using OMP_PLACES

  if [ -z "$BIND_STRIDE" ]; then
    let cpus_required=$OMP_NUM_THREADS*$OMPI_COMM_WORLD_LOCAL_SIZE
    let BIND_STRIDE=$cpus_per_node/$cpus_required
  fi

  if [ "${BIND_POLICY}" == "packed" ]; then
    let cpus_per_rank=$OMP_NUM_THREADS*$BIND_STRIDE
  else
    let cpus_per_rank=$cpus_per_node/$OMPI_COMM_WORLD_LOCAL_SIZE
  fi

  if [ -z "$BIND_BASE" ]; then
    let base=0;
  else
    let base=$BIND_BASE;
  fi

  let start_cpu=$base+$OMPI_COMM_WORLD_LOCAL_RANK*$cpus_per_rank

  export OMP_PLACES={$start_cpu:$OMP_NUM_THREADS:$BIND_STRIDE}

  export OMP_PROC_BIND=true
  "$@"

fi

