#!/bin/bash

#set -e

export OMP_NUM_THREADS=68
export OPENBLAS_NUM_THREADS=68
export BLIS_NUM_THREADS=68
export BLIS_JC_NT=4 BLIS_IC_NT=17

#export KMP_AFFINITY=$NULL
#export OMP_MAX_ACTIVE_LEVELS=2
#export KMP_HOT_TEAMS_MODE=1
#export KMP_HOT_TEAMS_MAX_LEVEL=2
#export OMP_PLACES="cores(68)"
#export OMP_PROC_BIND=spread,close
#export OMP_NUM_THREADS=68



OUT_PATH=./out
LOG_PATH=./log

#for kernel in 'mkl' 'blis' 'openblas' 'userdgemm' 'kernel.01.mc'
for kernel in 'kernel.01.mc'
do
    make -s -j102 ${kernel}

    if [ "${kernel}" = "openblas" ]; then
        export -n OMP_PROC_BIND
        export -n OMP_PLACES
    else
        export OMP_PROC_BIND=close
        export OMP_PLACES=cores
    fi

#	for size in {40000..40000..2000}
	for size in {1000..10000..1000}
	do
		m=$size 
		n=$size
		k=$size 

        #perf stat -e cache-misses,cache-references,cpu-cycles,instructions,context-switches,L1-dcache-load-misses,L1-icache-load-misses,L1-icache-loads,LLC-loads,LLC-stores,branch-load-misses,branch-loads,dTLB-load-misses,iTLB-load-misses,iTLB-loads,l2_requests.miss,l2_requests.reference,mem_uops_retired.all_loads,mem_uops_retired.all_stores,mem_uops_retired.hitm,mem_uops_retired.l1_miss_loads,mem_uops_retired.l2_hit_loads,mem_uops_retired.l2_miss_loads,mem_uops_retired.utlb_miss_loads,mem_uops_retired.dtlb_miss_loads,page_walks.cycles taskset -c 33 ${OUT_PATH}/dgemm_flops_${kernel}.out 1 Col N N ${m} ${n} ${k} -1.0 1.0 5
		#taskset -c 33 ${OUT_PATH}/dgemm_flops_${kernel}.out 1 Col N N ${m} ${n} ${k} -1.0 1.0 5
		${OUT_PATH}/dgemm_flops_${kernel}.out 68 Col N N ${m} ${n} ${k} -1.0 1.0 5
	done
done
