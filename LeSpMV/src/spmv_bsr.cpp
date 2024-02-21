/**
 * @file spmv_bsr.cpp
 * @author your name (you@domain.com)
 * @brief Simple implementation of SpMV in Sliced SELL-c-R format.
 *         Using OpenMP automatic parallism and vectorization.
 * @version 0.1
 * @date 2024-02-21
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include"../include/LeSpMV.h"

#include"../include/thread.h"

template <typename IndexType, typename ValueType>
void __spmv_bsr_serial_simple(  const IndexType num_rows,
                                const IndexType blockDimRow,
                                const IndexType blockDimCol,
                                const IndexType mb,
                                const IndexType nb,
                                const IndexType nnzb,
                                const ValueType alpha,
                                const IndexType *row_ptr,
                                const IndexType *col_index,
                                const ValueType *values,
                                const ValueType *x,
                                const ValueType beta,
                                ValueType *y)
{
    // 先对y进行缩放
    // for (IndexType i = 0; i < num_rows; ++i) {
    //     y[i] *= beta;
    // }

    for (IndexType i = 0; i < mb; i++)
    {
        IndexType start = row_ptr[i];
        IndexType end   = row_ptr[i+1];

        std::vector<ValueType> tmp(blockDimRow,0);

        for (IndexType j = start; j < end; j++)
        {
            // 获取当前块的列索引
            IndexType block_col = col_index[j];

            // 执行块与向量的乘法
            for (IndexType br = 0; br < blockDimRow; ++br) {
                for (IndexType bc = 0; bc < blockDimCol; ++bc) {
                    // 计算输出向量的索引
                    // IndexType y_index = i * blockDimRow + br;
                    // 计算输入向量的索引
                    IndexType x_index = block_col * blockDimCol + bc;
                    // 累加结果
                    // y[y_index] += alpha * values[j * blockDimRow * blockDimCol + br * blockDimCol + bc] * x[x_index];
                    tmp[br] += alpha * values[j * blockDimRow * blockDimCol + br * blockDimCol + bc] * x[x_index];
                }
            }
        }

        for (IndexType br = 0; br < blockDimRow; br++)
        {
            // 计算输出向量的索引
            IndexType y_index = i * blockDimRow + br;
            if (y_index < num_rows)
            {
                y[y_index] = tmp[br] + beta * y[y_index];
            }
        }
    }
}

template <typename IndexType, typename ValueType>
void __spmv_bsr_omp_simple( const IndexType num_rows,
                            const IndexType blockDimRow,
                            const IndexType blockDimCol,
                            const IndexType mb,
                            const IndexType nb,
                            const IndexType nnzb,
                            const ValueType alpha,
                            const IndexType *row_ptr,
                            const IndexType *col_index,
                            const ValueType *values,
                            const ValueType *x,
                            const ValueType beta,
                            ValueType *y)
{
    const IndexType m_inner = num_rows;
    const IndexType thread_num = Le_get_thread_num();

}

template <typename IndexType, typename ValueType>
void LeSpMV_bsr(const ValueType alpha, const BSR_Matrix<IndexType, ValueType>& bsr, const ValueType *x, const ValueType beta, ValueType *y)
{
    if ( 0 == bsr.kernel_flag)
    {
        __spmv_bsr_serial_simple(bsr.num_rows, bsr.blockDim_r, bsr.blockDim_c, bsr.mb, bsr.nb, bsr.nnzb, alpha, bsr.row_ptr, bsr.block_colindex, bsr.block_data, x, beta, y);
    }
    else if (1 == bsr.kernel_flag)
    {
        __spmv_bsr_omp_simple(bsr.num_rows, bsr.blockDim_r, bsr.blockDim_c, bsr.mb, bsr.nb, bsr.nnzb, alpha, bsr.row_ptr, bsr.block_colindex, bsr.block_data, x, beta, y);
    }
    else
    {
        __spmv_bsr_omp_simple(bsr.num_rows, bsr.blockDim_r, bsr.blockDim_c, bsr.mb, bsr.nb, bsr.nnzb, alpha, bsr.row_ptr, bsr.block_colindex, bsr.block_data, x, beta, y);
    }
}

template void LeSpMV_bsr<int, float>(const float alpha, const BSR_Matrix<int, float>& bsr, const float *x, const float beta, float *y);

template void LeSpMV_bsr<int, double>(const double alpha, const BSR_Matrix<int, double>& bsr, const double *x, const double beta, double *y);