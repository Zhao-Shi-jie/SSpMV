/**
 * @file spmv_sell_c_sigma.cpp
 * @author your name (you@domain.com)
 * @brief Simple implementation of SpMV in Sliced SELL-c-sigma format.
 *         Using OpenMP automatic parallism and vectorization.
 * @version 0.1
 * @date 2024-01-22
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include"../include/LeSpMV.h"

#include"../include/thread.h"

template <typename IndexType, typename ValueType>
void __spmv_sell_cs_serial_simple( const IndexType * Reorder,
                                   const IndexType num_rows,
                                   const IndexType chunk_rowNum,
                                   const IndexType total_chunk_num,
                                   const ValueType alpha,
                                   const IndexType *max_row_width,
                                   const IndexType * const *col_index,
                                   const ValueType * const *values,
                                   const ValueType * x, 
                                   const ValueType beta, 
                                   ValueType * y)
{
    for ( IndexType chunkID = 0; chunkID < total_chunk_num; ++chunkID)
    {
        IndexType chunk_width = max_row_width[chunkID];
        IndexType chunk_start_row = chunkID * chunk_rowNum;

        for (IndexType row = 0; row < chunk_rowNum; row++)
        {
            IndexType global_row = chunk_start_row + row;
            if ( global_row >= num_rows) break; // 越界检查
            
            IndexType sumPos = Reorder[global_row];
            ValueType sum    = 0;

            for (IndexType i = 0; i < chunk_width; i++)
            {
                IndexType col_index_pos = row * chunk_width + i;
                IndexType col = col_index[chunkID][col_index_pos];

                if (col >= 0) { // 检查是否为填充的空位
                    sum += values[chunkID][col_index_pos] * x[col];
                }
            }

            y[sumPos] = alpha * sum + beta * y[sumPos];
        }   
    }
}

template <typename IndexType, typename ValueType>
void __spmv_sell_cs_omp_simple( const IndexType * Reorder,
                                const IndexType num_rows,
                                const IndexType chunk_rowNum,
                                const IndexType total_chunk_num,
                                const ValueType alpha,
                                const IndexType *max_row_width,
                                const IndexType * const *col_index,
                                const ValueType * const *values,
                                const ValueType * x, 
                                const ValueType beta, 
                                ValueType * y)
{
    const IndexType thread_num = Le_get_thread_num();

    #pragma omp parallel for num_threads(thread_num)
    for ( IndexType chunkID = 0; chunkID < total_chunk_num; ++chunkID)
    {
        IndexType chunk_width = max_row_width[chunkID];
        IndexType chunk_start_row = chunkID * chunk_rowNum;

        for (IndexType row = 0; row < chunk_rowNum; row++)
        {
            IndexType global_row = chunk_start_row + row;
            if ( global_row >= num_rows) break; // 越界检查
            
            IndexType sumPos = Reorder[global_row];
            ValueType sum    = 0;

            #pragma omp simd reduction(+:sum)
            for (IndexType i = 0; i < chunk_width; i++)
            {
                IndexType col_index_pos = row * chunk_width + i;
                IndexType col = col_index[chunkID][col_index_pos];

                if (col >= 0) { // 检查是否为填充的空位
                    sum += values[chunkID][col_index_pos] * x[col];
                }
            }

            y[sumPos] = alpha * sum + beta * y[sumPos];
        }   
    }
}

template <typename IndexType, typename ValueType>
void LeSpMV_sell_c_sigma(const ValueType alpha, const SELL_C_Sigma_Matrix<IndexType, ValueType>& sell_c_sigma, const ValueType *x, const ValueType beta, ValueType *y)
{
    if (0 == sell_c_sigma.kernel_flag)
    {
        __spmv_sell_cs_serial_simple(sell_c_sigma.reorder, sell_c_sigma.num_rows, sell_c_sigma.chunkWidth_C, sell_c_sigma.validchunkNum, alpha, sell_c_sigma.chunk_len, sell_c_sigma.col_index, sell_c_sigma.values, x, beta, y);
    }
    else if (1 == sell_c_sigma.kernel_flag)
    {
        __spmv_sell_cs_omp_simple(sell_c_sigma.reorder, sell_c_sigma.num_rows, sell_c_sigma.chunkWidth_C, sell_c_sigma.validchunkNum, alpha, sell_c_sigma.chunk_len, sell_c_sigma.col_index, sell_c_sigma.values, x, beta, y);
    }
    else{
        //DEFAULT: omp simple implementation
        __spmv_sell_cs_omp_simple(sell_c_sigma.reorder, sell_c_sigma.num_rows, sell_c_sigma.chunkWidth_C, sell_c_sigma.validchunkNum, alpha, sell_c_sigma.chunk_len, sell_c_sigma.col_index, sell_c_sigma.values, x, beta, y);
    }
}

template void LeSpMV_sell_c_sigma<int, float>(const float alpha, const SELL_C_Sigma_Matrix<int, float>& sell, const float * x, const float beta, float * y);

template void LeSpMV_sell_c_sigma<int, double>(const double alpha, const SELL_C_Sigma_Matrix<int, double>& sell, const double * x, const double beta, double * y);