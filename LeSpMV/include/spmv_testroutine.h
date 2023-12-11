#ifndef SPMV_TESTROUTINE_H
#define SPMV_TESTROUTINE_H

#include"sparse_format.h"
#include<cstdio>

template <typename IndexType, typename ValueType>
int test_coo_matrix_kernels(const CSR_Matrix<IndexType,ValueType> &csr_ref, int kernel_tag);

template <typename IndexType, typename ValueType>
int test_csr_matrix_kernels(const CSR_Matrix<IndexType,ValueType> &csr_ref, int kernel_tag);

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
int test_ell_matrix_kernels(const CSR_Matrix<IndexType,ValueType> &csr_ref, int kernel_tag, LeadingDimension ld);

template <typename IndexType, typename ValueType>
int test_dia_matrix_kernels(const CSR_Matrix<IndexType,ValueType> &csr_ref, int kernel_tag);

#endif /* SPMV_TESTROUTINE_H */
