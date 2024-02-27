/**
 * @file test_spmv_bsr.cpp
 * @author Shengle Lin (lsl036@hnu.edu.cn)
 * @brief Test routine for spmv_bsr.cpp
 * @version 0.1
 * @date 2024-02-21
 * 
 * @copyright Copyright (c) 2023
 * 
 */
#include"../include/LeSpMV.h"
#include<iostream>

template <typename IndexType, typename ValueType>
int test_bsr_matrix_kernels(const CSR_Matrix<IndexType,ValueType> &csr_ref, int kernel_tag, int schedule_mod)
{
    std::cout << "=====  Testing BSR Kernels  =====" << std::endl;

    BSR_Matrix<IndexType,ValueType> bsr;

    IndexType alignment = (SIMD_WIDTH/8/sizeof(ValueType));

    bsr = csr_to_bsr(csr_ref, BSR_BlockDimRow, 1*alignment);

    // 测试这个routine 要我们测的 kernel_tag
    bsr.kernel_flag = kernel_tag;

    if(0 == bsr.kernel_flag){
        std::cout << "\n===  Compared BSR serial with csr default  ===" << std::endl;
        // test correctness
        test_spmv_kernel(csr_ref, LeSpMV_csr<IndexType, ValueType>,
                         bsr, LeSpMV_bsr<IndexType, ValueType>,
                         "bsr_serial_simple");

        std::cout << "\n===  Performance of BSR serial simple  ===" << std::endl;
        // count performance of Gflops and Gbytes
        benchmark_spmv_on_host(bsr, LeSpMV_bsr<IndexType, ValueType>,"bsr_serial_simple");
    }
    else if (1 == bsr.kernel_flag)
    {
        std::cout << "\n===  Compared BSR omp with csr default  ===" << std::endl;

        // 设置 omp 调度策略
        const IndexType thread_num = Le_get_thread_num();
        
        IndexType chunk_size = OMP_ROWS_SIZE;
        chunk_size = std::max(chunk_size, bsr.mb/thread_num);

        set_omp_schedule(schedule_mod, chunk_size);

        // test correctness
        test_spmv_kernel(csr_ref, LeSpMV_csr<IndexType, ValueType>,
                         bsr, LeSpMV_bsr<IndexType, ValueType>,
                         "bsr_omp_simple");

        std::cout << "\n===  Performance of BSR omp simple  ===" << std::endl;
        // count performance of Gflops and Gbytes
        benchmark_spmv_on_host(bsr, LeSpMV_bsr<IndexType, ValueType>,"bsr_omp_simple");
        
    }

    delete_host_matrix(bsr);
    return 0;
}

template int test_bsr_matrix_kernels<int,float>(const CSR_Matrix<int,float> &csr_ref, int kernel_tag, int sche);

template int test_bsr_matrix_kernels<int,double>(const CSR_Matrix<int,double> &csr_ref, int kernel_tag, int sche);