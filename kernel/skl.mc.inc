#define TOTAL_CORE 36
#define NUMA_NODE 2

#ifndef CM
#define CM 18
#endif
#define CN (TOTAL_CORE / CM)

#ifndef MB
#define MB (MR * 1)
#endif

#ifndef NB
#define NB (NR * 1)
#endif

#ifndef KB
#define KB 4
#endif

#ifndef MK_UNROLL_DEPTH
#define MK_UNROLL_DEPTH 1
#endif

// prefetch hints
// wt1, w, t0, t1, t2, nta

// #define MK_PREFETCH_A0
// #define MK_PREFETCH_A1

#ifndef MK_PREFETCH_A_HINT
#define MK_PREFETCH_A_HINT nta
#endif

#ifndef MK_PREFETCH_NEXT_A_DEPTH
#define MK_PREFETCH_NEXT_A_DEPTH 0
#endif

#ifndef MK_PREFETCH_NEXT_A_LOCALITY
#define MK_PREFETCH_NEXT_A_LOCALITY LOCALITY_NONE
#endif

#ifndef MK_PREFETCH_C_DEPTH
#define MK_PREFETCH_C_DEPTH 0
#endif

#ifndef MK_PREFETCH_C_HINT
#define MK_PREFETCH_C_HINT nta
#endif

#ifndef ACC_PREFETCH_DEPTH
#define ACC_PREFETCH_DEPTH 0
#endif

#ifndef ACC_PREFETCH_HINT
#define ACC_PREFETCH_HINT nta
#endif

// #define INNER_MN
// #define PACKACC_M_FIRST
