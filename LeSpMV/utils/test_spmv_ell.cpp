/**
 * @file test_spmv_ell.cpp
 * @author Shengle Lin (lsl036@hnu.edu.cn)
 * @brief Test routine for spmv_ell.cpp
 * @version 0.1
 * @date 2023-11-17
 * 
 * @copyright Copyright (c) 2023
 * 
 */
#include"../include/LeSpMV.h"
#include<iostream>


/**
 * @brief Input CSR format for reference. Inside we make an ELL format
 * 
 * @tparam IndexType 
 * @tparam ValueType 
 * @param csr 
 * @param kernel_tag 
 * @return int 
 */
template <typename IndexType, typename ValueType>
int test_ell_matrix_kernels(const CSR_Matrix<IndexType,ValueType> &csr_ref, int kernel_tag, LeadingDimension ld)
{
    std::cout << "=====  Testing ELL Kernels  =====" << std::endl;

    ELL_Matrix<IndexType,ValueType> ell;
    ell = csr_to_ell(csr_ref, ld);
    
    // 测试这个routine 要我们测的 kernel_tag
    ell.kernel_flag = kernel_tag;

    if(0 == ell.kernel_flag){
        std::cout << "\n===  Compared ell serial with csr default  ===" << std::endl;
        // test correctness
        test_spmv_kernel(csr_ref, LeSpMV_csr<IndexType, ValueType>,
                         ell, LeSpMV_ell<IndexType, ValueType>,
                         "ell_serial_simple");

        std::cout << "\n===  Performance of ELL serial simple  ===" << std::endl;
        // count performance of Gflops and Gbytes
        benchmark_spmv_on_host(ell,LeSpMV_ell<IndexType, ValueType>,"ell_serial_simple");
    }
    else if (1 == ell.kernel_flag)
    {
        std::cout << "\n===  Compared ell omp with csr default  ===" << std::endl;
        // test correctness
        test_spmv_kernel(csr_ref, LeSpMV_csr<IndexType, ValueType>,
                         ell, LeSpMV_ell<IndexType, ValueType>,
                         "ell_omp_simple");

        std::cout << "\n===  Performance of ELL omp simple  ===" << std::endl;
        // count performance of Gflops and Gbytes
        benchmark_spmv_on_host(ell,LeSpMV_ell<IndexType, ValueType>,"ell_omp_simple");     
    }
    else if (2 == ell.kernel_flag)
    {
        std::cout << "\n===  Compared ell Load-Balance with csr default  ===" << std::endl;
        // test correctness
        test_spmv_kernel(csr_ref, LeSpMV_csr<IndexType, ValueType>,
                         ell, LeSpMV_ell<IndexType, ValueType>,
                         "ell_omp_ld");

        std::cout << "\n===  Performance of ELL omp Load-Balance  ===" << std::endl;
        // count performance of Gflops and Gbytes
        benchmark_spmv_on_host(ell,LeSpMV_ell<IndexType, ValueType>,"ell_omp_ld");
    }

    delete_ell_matrix(ell);
    return 0;
}

template int test_ell_matrix_kernels<int,float>(const CSR_Matrix<int,float> &csr_ref, int kernel_tag, LeadingDimension ld);

template int test_ell_matrix_kernels<int,double>(const CSR_Matrix<int,double> &csr_ref, int kernel_tag, LeadingDimension ld);