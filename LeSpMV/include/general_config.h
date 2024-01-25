#ifndef GENERAL_CONFIG_H
#define GENERAL_CONFIG_H
////////////////////////////////////////////////////////////////
//   General defines
////////////////////////////////////////////////////////////////

// experimental setting
#define MAT_GRID_SIZE 512  //not use
#define MAX_DIAG_NUM 10240
#define MAX_ITER 1000
#define MIN_ITER 500 

#define TIME_LIMIT 20.0  
#define NUM_FORMATS 7

// hyperpramaters for SpMV algorithms
#define SELL_SIGMA 512   // (2^9), 4096 (2^12) , and 16384 (2^14)
#define CHUNK_SIZE 8
#define NTRATIO (0.6)

//  general setting in Liu weifeng's library
//  TILE size: CSR5_SIGMA x CSR5_OMEGA  column major
//                      y_offset                    seg_offset      bit_flag
// Foa a column: [log(CSR5_SIGMA x CSR5_OMEGA) + log(CSR5_OMEGA) + CSR5_SIGMA] (bits)
#define CSR5_SIGMA   16     // can change to 12 or 16
#define X86_CACHELINE   64

// OMP paramaters
#define OMP_ROWS_SIZE 64

// Kernel Flag : 0 = serial simple implementation
//               1 = *default* simple omp implementations
//               2 = load balanced omp implementation
#ifndef KERNEL_FLAG
    #define KERNEL_FLAG 1
#endif // !KERNEL_FLAG



// =======================================================
//   OMP Scheduling strategy: stcont, static or dynamic
// =======================================================
// static contiguous
#ifdef STCONT
    #define SCHEDULE_STRATEGY static
#endif

// static with chunk size
#ifdef STATIC
    #define SCHEDULE_STRATEGY static, CHUNK_SIZE
#endif // ST_CHUNK

#ifdef DYN
    #define SCHEDULE_STRATEGY dynamic
#endif

#ifndef SCHEDULE_STRATEGY
    #define SCHEDULE_STRATEGY static
#endif // !SCHEDULE_STRATEGY

// SCHE_MODE :  0 = omp_set_schedule(omp_sched_static, 0);
//              1 = omp_set_schedule(omp_sched_static, chunk_size);
//              2 = omp_set_schedule(omp_sched_dynamic, chunk_size);
//              3 = omp_set_schedule(omp_sched_guided, chunk_size);
#define SCHE_MODE 1

#define MAT_FEATURES "./features/mat_features.txt"

#endif /* GENERAL_CONFIG_H */
