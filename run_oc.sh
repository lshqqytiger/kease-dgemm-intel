#!/bin/bash

set -e

export OMP_NUM_THREADS=1

OUT_PATH=./out
LOG_PATH=./log

#for kernel in 'mkl' 'blis' 'openblas' 'kernel.01.oc' 'userdgemm'
for kernel in 'mkl'
do
    make -s ${kernel}

	for size in {100..1000..100}
	do
		m=$size 
		n=$size
		k=$size 

#        perf stat -e cache-misses,cache-references,cpu-cycles,instructions,context-switches,L1-dcache-load-misses,L1-icache-load-misses,L1-icache-loads,LLC-loads,LLC-stores,branch-load-misses,branch-loads,dTLB-load-misses,iTLB-load-misses,iTLB-loads,l2_requests.miss,l2_requests.reference,mem_uops_retired.all_loads,mem_uops_retired.all_stores,mem_uops_retired.hitm,mem_uops_retired.l1_miss_loads,mem_uops_retired.l2_hit_loads,mem_uops_retired.l2_miss_loads,mem_uops_retired.utlb_miss_loads,mem_uops_retired.dtlb_miss_loads,page_walks.cycles taskset -c 33 ${OUT_PATH}/dgemm_flops_${kernel}.out 1 Col N N ${m} ${n} ${k} -1.0 1.0 50
#		taskset -c 33 ${OUT_PATH}/dgemm_flops_${kernel}.out 1 Col N N ${m} ${n} ${k} -1.0 1.0 50
		${OUT_PATH}/dgemm_flops_${kernel}.out 1 Col N N ${m} ${n} ${k} -1.0 1.0 20
	done
done
