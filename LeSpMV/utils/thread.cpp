
#include"../include/thread.h"
#include<stdio.h>

int _thread_num;

int Le_get_core_num()
{
#ifdef _OPENMP
    return omp_get_num_procs();
#else
    return 1;
#endif
}

void Le_set_thread_num(const int thread_num)
{
#ifdef _OPENMP
    _thread_num = thread_num;
#else
    _thread_num = 1;
#endif
}

int Le_get_thread_num()
{
#ifdef _OPENMP
    return _thread_num == 0 ? Le_get_core_num() : _thread_num;
#else
    return 1;
#endif
}
int Le_get_thread_id()
{
#ifdef _OPENMP
    return omp_get_thread_num();
#else
    return 0;
#endif
}

void set_omp_schedule(int sche_mode, int chunk_size) {
#ifdef _OPENMP
    switch (sche_mode) {
        case 0:
            omp_set_schedule(omp_sched_static, 0); // 使用默认的chunk size
            printf("-- OMP Static schedule strategy --\n");
            break;
        case 1:
            omp_set_schedule(omp_sched_static, chunk_size);
            printf("-- OMP StaticConst schedule strategy --\n");
            break;
        case 2:
            omp_set_schedule(omp_sched_dynamic, chunk_size);
            printf("-- OMP Dynamic schedule strategy --\n");
            break;
        default:
            // 报告错误或使用默认调度策略
            break;
    }
#endif
}