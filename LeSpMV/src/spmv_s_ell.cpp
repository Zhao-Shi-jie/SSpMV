/**
 * @file spmv_s_ell.cpp
 * @author Shengle Lin (lsl036@hnu.edu.cn)
 * @brief  Simple implementation of SpMV in Sliced ELL format.
 *         Using OpenMP automatic parallism and vectorization.
 * @version 0.1
 * @date 2023-12-11
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include"../include/LeSpMV.h"

#include"../include/thread.h"

template <typename IndexType, typename ValueType>
void __spmv_sell_serial_simple( const IndexType num_rows,
                                const IndexType row_num_perC,
                                const IndexType total_chunk_num,
                                const ValueType alpha,
                                const IndexType *max_row_width,
                                const IndexType * const *col_index,
                                const ValueType * const *values,
                                const ValueType * x, 
                                const ValueType beta, 
                                ValueType * y)
{
    for (IndexType chunk = 0; chunk < total_chunk_num; ++chunk)
    {
        IndexType chunk_width = max_row_width[chunk];
        IndexType chunk_start_row = chunk * row_num_perC;

        for (IndexType row = 0; row < row_num_perC; ++row)
        {
            ValueType sum = 0;
            IndexType global_row = chunk_start_row + row;
            if (global_row >= num_rows) break; // 越界检查

            // #pragma omp simd reduction(+:sum)
            for (IndexType i = 0; i < chunk_width; ++i) 
            {
                IndexType col_index_pos = row * chunk_width + i;
                IndexType col = col_index[chunk][col_index_pos];

                if (col >= 0) { // 检查是否为填充的空位
                    sum += values[chunk][col_index_pos] * x[col];
                }
            }

            y[global_row] = alpha * sum + beta * y[global_row];
        }
    }

}


/**
 * @brief 
 * 
 * @tparam IndexType 
 * @tparam ValueType 
 * @param row_num_perC      一个 chunk 内的 行数目
 * @param total_chunk_num 
 * @param alpha 
 * @param col_index 
 * @param values 
 * @param x 
 * @param beta 
 * @param y 
 */
template <typename IndexType, typename ValueType>
void __spmv_sell_omp_simple(const IndexType num_rows,
                            const IndexType row_num_perC,
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
    const IndexType chunk_size = std::max(1, total_chunk_num/ thread_num);

    // const int colindex_align_bytes = ALIGNMENT_NUM * sizeof(IndexType);
    // const int values_align_bytes   = ALIGNMENT_NUM * sizeof(ValueType);

    //  Only spmv for row major SELL
    // #pragma omp parallel for num_threads(thread_num) schedule(static,chunk_size)
    #pragma omp parallel for num_threads(thread_num) schedule(SCHEDULE_STRATEGY)
    for (IndexType chunk = 0; chunk < total_chunk_num; ++chunk)
    {
        IndexType chunk_width = max_row_width[chunk];
        IndexType chunk_start_row = chunk * row_num_perC;

        for (IndexType row = 0; row < row_num_perC; ++row)
        {
            ValueType sum = 0;
            IndexType global_row = chunk_start_row + row;
            if (global_row >= num_rows) break; // 越界检查

            #pragma omp simd reduction(+:sum)
            for (IndexType i = 0; i < chunk_width; ++i) 
            {
                IndexType col_index_pos = row * chunk_width + i;
                IndexType col = col_index[chunk][col_index_pos];

                if (col >= 0) { // 检查是否为填充的空位
                    sum += values[chunk][col_index_pos] * x[col];
                }
            }

            y[global_row] = alpha * sum + beta * y[global_row];
        }
    }
}

template <typename IndexType, typename ValueType>
void LeSpMV_sell(const ValueType alpha, const S_ELL_Matrix<IndexType, ValueType>& sell, const ValueType * x, const ValueType beta, ValueType * y){
    if (0 == sell.kernel_flag)
    {
        __spmv_sell_serial_simple(sell.num_rows, sell.sliceWidth, sell.chunk_num, alpha, sell.row_width, sell.col_index, sell.values, x, beta, y);

    }
    else if(1 == sell.kernel_flag)
    {
        __spmv_sell_omp_simple(sell.num_rows, sell.sliceWidth, sell.chunk_num, alpha, sell.row_width, sell.col_index, sell.values, x, beta, y);

    }
    else if(2 == sell.kernel_flag)
    {
        // load balanced ? maybe no need foe sell
    }
    else{
        //DEFAULT: omp simple implementation
        __spmv_sell_omp_simple(sell.num_rows, sell.sliceWidth, sell.chunk_num, alpha, sell.row_width, sell.col_index, sell.values, x, beta, y);
    }
}

template void LeSpMV_sell<int, float>(const float alpha, const S_ELL_Matrix<int, float>& sell, const float * x, const float beta, float * y);

template void LeSpMV_sell<int, double>(const double alpha, const S_ELL_Matrix<int, double>& sell, const double * x, const double beta, double * y);