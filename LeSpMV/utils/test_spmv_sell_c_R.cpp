/**
 * @file test_spmv_sell_c_R.cpp
 * @author your name (you@domain.com)
 * @brief Test routine for spmv_sell_c_R.cpp
 * @version 0.1
 * @date 2024-01-29
 * 
 * @copyright Copyright (c) 2024
 * 
 */
#include"../include/LeSpMV.h"
#include<iostream>

template <typename IndexType, typename ValueType>
int test_sell_c_R_matrix_kernels(const CSR_Matrix<IndexType,ValueType> &csr_ref, int kernel_tag, int schedule_mod)
{
    std::cout << "=====  Testing SELL-c-R Kernels  =====" << std::endl;

    SELL_C_R_Matrix<IndexType,ValueType> sell_c_R;

    FILE* save_features = fopen(MAT_FEATURES,"w");

    IndexType chunkwidth = CHUNK_SIZE;
    IndexType alignment  = SIMD_WIDTH/8/sizeof(ValueType);

    sell_c_R = csr_to_sell_c_R(csr_ref, save_features, chunkwidth, alignment);

    fclose(save_features);

    // 测试这个routine 要我们测的 kernel_tag
    sell_c_R.kernel_flag = kernel_tag;

    if ( 0 == sell_c_R.kernel_flag){
        std::cout << "\n===  Compared SELL-c-R serial with csr default  ===" << std::endl;
        // test correctness
        test_spmv_kernel(csr_ref, LeSpMV_csr<IndexType, ValueType>,
                         sell_c_R, LeSpMV_sell_c_R<IndexType, ValueType>,
                         "sell_c_R_serial_simple");

        std::cout << "\n===  Performance of SELL-c-R serial simple  ===" << std::endl;
        // count performance of Gflops and Gbytes
        benchmark_spmv_on_host(sell_c_R, LeSpMV_sell_c_R<IndexType, ValueType>,"sell_c_R_serial_simple");

    }
    else if ( 1 == sell_c_R.kernel_flag)
    {
        std::cout << "\n===  Compared SELL-c-R omp with csr default  ===" << std::endl;
        // 设置 omp 调度策略
        const IndexType thread_num = Le_get_thread_num();
        const IndexType chunk_size = std::max(1, sell_c_R.validchunkNum/thread_num);
        set_omp_schedule(schedule_mod, chunk_size);

        // test correctness
        test_spmv_kernel(csr_ref, LeSpMV_csr<IndexType, ValueType>,
                         sell_c_R, LeSpMV_sell_c_R<IndexType, ValueType>,
                         "sell_c_R_omp_simple");

        std::cout << "\n===  Performance of SELL-c-R omp simple  ===" << std::endl;
        // count performance of Gflops and Gbytes
        benchmark_spmv_on_host(sell_c_R, LeSpMV_sell_c_R<IndexType, ValueType>,"sell_c_R_omp_simple");
        
    }
    else if ( 2 == sell_c_R.kernel_flag)
    {
        std::cout << "\n===  Compared SELL-c-R Load-Balance with csr default  ===" << std::endl;

        // test correctness
        test_spmv_kernel(csr_ref, LeSpMV_csr<IndexType, ValueType>,
                         sell_c_R, LeSpMV_sell_c_R<IndexType, ValueType>,
                         "sell_c_R_omp_ld");

        std::cout << "\n===  Performance of SELL-c-R omp simple  ===" << std::endl;
        // count performance of Gflops and Gbytes
        benchmark_spmv_on_host(sell_c_R, LeSpMV_sell_c_R<IndexType, ValueType>,"sell_c_R_omp_ld");
    }

    delete_host_matrix(sell_c_R);
    return 0;
}

template int test_sell_c_R_matrix_kernels<int,float>(const CSR_Matrix<int,float> &csr_ref, int kernel_tag, int sche);

template int test_sell_c_R_matrix_kernels<int,double>(const CSR_Matrix<int,double> &csr_ref, int kernel_tag, int sche);