/**
 * @file benchmark_spmv_csr5.cpp
 * @author your name (you@domain.com)
 * @brief 
 * @version 0.1
 * @date 2024-01-07
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include<iostream>
#include<cstdio>
#include"../include/LeSpMV.h"
#include"../include/cmdline.h"

void usage(int argc, char** argv)
{
    std::cout << "Usage:\n";
    std::cout << "\t" << argv[0] << " with following parameters:\n";
    std::cout << "\t" << " my_matrix.mtx\n";
    std::cout << "\t" << " --precision=64(now only support 64 by AVX512)\n";
    std::cout << "\t" << " --threads= define the num of omp threads\n";
    std::cout << "Note: my_matrix.mtx must be real-valued sparse matrix in the MatrixMarket file format.\n"; 
}

template <typename IndexType, typename UIndexType, typename ValueType>
void run_csr5_kernels(int argc, char **argv)
{
    char * mm_filename = NULL;
    for(int i = 1; i < argc; i++){
        if(argv[i][0] != '-'){
            mm_filename = argv[i];
            break;
        }
    }

    if(mm_filename == NULL)
    {
        printf("You need to input a matrix file!\n");
        return;
    }

    // reference CSR kernel for csr5 test
    CSR_Matrix<IndexType, ValueType> csr;
    csr = read_csr_matrix<IndexType, ValueType> (mm_filename);

    printf("Using %d-by-%d matrix with %d nonzero values\n", csr.num_rows, csr.num_cols, csr.num_nnzs); 

    fflush(stdout);
    
    // for(IndexType methods = 0; methods <= 2; ++methods){
        // not even need the kernel_tag=0 & schedulemod=0
        test_csr5_matrix_kernels<IndexType, UIndexType, ValueType>(csr, 0, 0);
        fflush(stdout);
    // }

    delete_csr_matrix(csr);
}

int main(int argc, char** argv)
{
    if (get_arg(argc, argv, "help") != NULL){
        usage(argc, argv);
        return EXIT_SUCCESS;
    }

    int precision = 64;
    char * precision_str = get_argval(argc, argv, "precision");
    if(precision_str != NULL)
        precision = atoi(precision_str);

    // 不用超线程，只计算真实CORE
    Le_set_thread_num(CPU_SOCKET * CPU_CORES_PER_SOC * CPU_HYPER_THREAD);

    char * threads_str = get_argval(argc, argv, "threads");
    if(threads_str != NULL)
        Le_set_thread_num(atoi(threads_str));

    printf("\nUsing %d-bit floating point precision, threads = %d\n\n", precision, Le_get_thread_num());

    if(precision ==  32){
        
        printf(" Sorry, not finish 32-bit CSR5 yet, see --help for more settings.\n");
        return EXIT_FAILURE;
    }
    else if(precision == 64){
        
        run_csr5_kernels<int, uint32_t, double>(argc,argv);
    }
    else{
        usage(argc, argv);
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}