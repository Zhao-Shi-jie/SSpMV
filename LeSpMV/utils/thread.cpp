
#include"../include/thread.h"

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