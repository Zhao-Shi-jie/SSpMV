#ifndef GENERAL_CONFIG_H
#define GENERAL_CONFIG_H
////////////////////////////////////////////////////////////////
//   General defines
////////////////////////////////////////////////////////////////

// experimental setting
#define MAT_GRID_SIZE 512  //not use
#define MAX_DIAG_NUM 10240
#define MAX_ITER 800
#define MIN_ITER 500 
#define MAX_R 8
#define MAX_C 8

#define TIME_LIMIT 10.0  
#define NUM_FORMATS 7

// hyperpramaters for SpMV algorithms
#define CHUNK_SIZE 8
#define ALIGNMENT_NUM 16
#define NTRATIO (0.6)

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


#define MAT_FEATURES "./features/mat_features.txt"

#endif /* GENERAL_CONFIG_H */
