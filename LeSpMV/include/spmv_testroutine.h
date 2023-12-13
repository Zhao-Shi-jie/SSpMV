#ifndef SPMV_TESTROUTINE_H
#define SPMV_TESTROUTINE_H

#include"sparse_format.h"
#include<cstdio>

template <typename IndexType, typename ValueType>
int test_coo_matrix_kernels(const CSR_Matrix<IndexType,ValueType> &csr_ref, int kernel_tag);

template <typename IndexType, typename ValueType>
int test_csr_matrix_kernels(const CSR_Matrix<IndexType,ValueType> &csr_ref, int kernel_tag, int schedule_mod);

/**
 * @brief Input CSR format for reference. Inside we make an ELL matrix
 * 
 * @tparam IndexType 
 * @tparam ValueType 
 * @param csr 
 * @param kernel_tag 
 * @return int 
 */
template <typename IndexType, typename ValueType>
int test_ell_matrix_kernels(const CSR_Matrix<IndexType,ValueType> &csr_ref, int kernel_tag, LeadingDimension ld, int schedule_mod);

template <typename IndexType, typename ValueType>
int test_dia_matrix_kernels(const CSR_Matrix<IndexType,ValueType> &csr_ref, int kernel_tag);

/**
 * @brief Input CSR format for reference. Inside we make an S_ELL matrix for testing
 * 
 * @tparam IndexType 
 * @tparam ValueType 
 * @param csr_ref 
 * @param kernel_tag 
 * @return int 
 */
template <typename IndexType, typename ValueType>
int test_s_ell_matrix_kernels(const CSR_Matrix<IndexType,ValueType> &csr_ref, int kernel_tag, int schedule_mod);

#endif /* SPMV_TESTROUTINE_H */
