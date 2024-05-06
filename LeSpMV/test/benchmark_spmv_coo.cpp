/**
 * @file benchmark_spmv_coo.cpp for running the coo test routine.
 * @author Shengle Lin (lsl036@hnu.edu.cn)
 * @brief 
 * @version 0.1
 * @date 2023-11-16
 * 
 * @copyright Copyright (c) 2023
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
    std::cout << "\t" << " --matID     = m_num, giving the matrix ID number in dataset (default 0).\n";
    std::cout << "\t" << " --Index     = 0 (int:default) or 1 (long long)\n";
    std::cout << "\t" << " --precision = 32(or 64)\n";
    std::cout << "\t" << " --threads   = define the num of omp threads\n";
    std::cout << "Note: my_matrix.mtx must be real-valued sparse matrix in the MatrixMarket file format.\n"; 
}

template <typename IndexType, typename ValueType>
void run_coo_kernels(int argc, char **argv)
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

    std::string matrixName = extractFileNameWithoutExtension(mm_filename);

    int matID = 0;
    char * matID_str = get_argval(argc, argv, "matID");
    if(matID_str != NULL)
    {
        matID = atoi(matID_str);
    }

    CSR_Matrix <IndexType, ValueType> csr_ref;
    csr_ref = read_csr_matrix<IndexType, ValueType> (mm_filename);

    if constexpr(std::is_same<IndexType, int>::value) {
        printf("Using %d-by-%d matrix with %d nonzero values\n", csr_ref.num_rows, csr_ref.num_cols, csr_ref.num_nnzs);
    }
    else if constexpr(std::is_same<IndexType, long long>::value) {
        printf("Using %lld-by-%lld matrix with %lld nonzero values\n", csr_ref.num_rows, csr_ref.num_cols, csr_ref.num_nnzs);
    }

    // fill matrix with random values: some matrices have extreme values, 他要替换稀疏矩阵的值，这里跳过
    // which makes correctness testing difficult, especially in single precision
    /*
    srand(13);
    for(IndexType i = 0; i < coo.num_nnzs; i++){
        coo.values[i] = 1.0 - 2.0 * (rand() / (RAND_MAX + 1.0)); 
    }  */
    fflush(stdout);
    // timer run_time_struct;
    // double coo_gflops= 0.0;

    // int coo_kernel_tag = 1; // 测试omp的simple kernel

    // int coo_kernel_tag = 2; // 测试omp的alphasparse

    // test_coo_matrix_kernels(coo, coo_kernel_tag);
    // fflush(stdout);
    // 保存测试性能结果
    
    FILE *save_perf = fopen(MAT_PERFORMANCE, "a");
    if ( save_perf == nullptr)
    {
        std::cout << "Unable to open perf-saved file: "<< MAT_PERFORMANCE << std::endl;
        return ;
    }

    double msec_per_iteration;
    double sec_per_iteration;

    // 0: 串行， 1：omp simple， 3：alphaspasre COO
    // Our : {St,StCont, Dyn, guided} x {omp}
    for (int sche_mode = 0 ; sche_mode < 4; ++sche_mode){
    for(int methods = 1; methods < 2; ++methods){
        msec_per_iteration = test_coo_matrix_kernels(csr_ref, methods, sche_mode);
        fflush(stdout);
        sec_per_iteration = msec_per_iteration / 1000.0;
        double GFLOPs = (sec_per_iteration == 0) ? 0 : (2.0 * (double) csr_ref.num_nnzs / sec_per_iteration) / 1e9;
        // 输出格式： 【Mat Format Method Schedule Time Performance】
        fprintf(save_perf, "%d %s COO %d %d %8.4f %5.4f \n", matID, matrixName.c_str(), methods, sche_mode, msec_per_iteration, GFLOPs);
    }
    }

    fclose(save_perf);
    delete_host_matrix(csr_ref);
}

int main(int argc, char** argv)
{
    if (get_arg(argc, argv, "help") != NULL){
        usage(argc, argv);
        return EXIT_SUCCESS;
    }

    int precision = 32;
    char * precision_str = get_argval(argc, argv, "precision");
    if(precision_str != NULL)
        precision = atoi(precision_str);
    
    // 包括超线程
    Le_set_thread_num(CPU_SOCKET * CPU_CORES_PER_SOC * CPU_HYPER_THREAD);

    char * threads_str = get_argval(argc, argv, "threads");
    if(threads_str != NULL)
        Le_set_thread_num(atoi(threads_str));

    int Index = 0;
    char * Index_str = get_argval(argc, argv, "Index");
    if(Index_str != NULL)
        Index = atoi(Index_str);

    printf("\nUsing %d-bit floating point precision, %d-bit Index, threads = %d\n\n", precision, (Index+1)*32 , Le_get_thread_num());

    if (Index == 0 && precision ==  32){
        run_coo_kernels<int, float>(argc,argv);
    }
    else if (Index == 0 && precision == 64){
        run_coo_kernels<int, double>(argc,argv);
    }
    else if (Index == 1 && precision ==  32){
        run_coo_kernels<long long, float>(argc,argv);
    }
    else if (Index == 1 && precision == 64){
        run_coo_kernels<long long, double>(argc,argv);
    }
    else{
        usage(argc, argv);
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

