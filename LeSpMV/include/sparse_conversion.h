#ifndef SPARSE_CONVERSION_H
#define SPARSE_CONVERSION_H

#include"sparse_format.h"
#include"sparse_operation.h"
#include"general_config.h"
#include"plat_config.h"
#include"thread.h"
#include"memopt.h"
#include <vector>
#include <algorithm>
#include <stdexcept>

template <class IndexType, class ValueType>
CSR_Matrix<IndexType, ValueType> coo_to_csr( const COO_Matrix<IndexType, ValueType> &coo, bool compact = false)
{
    CSR_Matrix<IndexType, ValueType> csr;

    csr.num_rows = coo.num_rows;
    csr.num_cols = coo.num_cols;
    csr.num_nnzs = coo.num_nnzs;

    csr.tag = 0;
    csr.row_offset = new_array<IndexType> (csr.num_rows + 1);
    csr.col_index  = new_array<IndexType> (csr.num_nnzs);
    csr.values     = new_array<ValueType> (csr.num_nnzs);

    //========== Rowoffset calculation ==========
    for (IndexType i = 0; i < csr.num_rows; i++){
        csr.row_offset[i] = 0;
    }

    // Get each row's nnzs
    for (IndexType i = 0; i < csr.num_nnzs; i++){
        csr.row_offset[ coo.row_index[i] ]++;
    }

    //  sum to get row_offset
    for(IndexType i = 0, cumsum = 0; i < csr.num_rows; i++){
        IndexType temp = csr.row_offset[i];
        csr.row_offset[i] = cumsum;
        cumsum += temp;
    }
    csr.row_offset[csr.num_rows] = csr.num_nnzs;

    // ========== write col_index and values ==========
    for (IndexType i = 0; i < csr.num_nnzs; i++){
        IndexType rowIndex  = coo.row_index[i];
        IndexType destIndex = csr.row_offset[rowIndex];

        csr.col_index[destIndex] = coo.col_index[i];
        csr.values[destIndex]    = coo.values[i];
    
        csr.row_offset[rowIndex]++;  // row_offset move behind
    }

    // Restore the row_offset
    for(IndexType i = 0, last = 0; i <= csr.num_rows; i++){
        IndexType temp = csr.row_offset[i];
        csr.row_offset[i] = last;
        last = temp;
    }

    // ========== Compact Situation ==========
    if(compact) {
        //sum duplicates together 是累加！！
        sum_csr_duplicates(csr.num_rows, csr.num_cols, 
                           csr.row_offset, csr.col_index, csr.values);

        csr.num_nnzs = csr.row_offset[csr.num_rows];
    }
    
    return csr;
}

template <class IndexType, class ValueType>
ELL_Matrix<IndexType, ValueType> coo_to_ell( const COO_Matrix<IndexType, ValueType> &coo, const LeadingDimension ld = RowMajor)
{
    ELL_Matrix<IndexType,ValueType> ell;

    ell.num_rows = coo.num_rows;
    ell.num_cols = coo.num_cols;
    ell.num_nnzs = coo.num_nnzs;

    ell.tag = 0;
    ell.ld = RowMajor;
    std::vector<IndexType> rowCounts (ell.num_rows, 0);

    for (IndexType i = 0; i < coo.num_nnzs; i++){
        rowCounts[coo.row_index[i]]++;
    }

    ell.max_row_width = *std::max_element(rowCounts.begin(), rowCounts.end());
    // 分配矩阵空间
    ell.col_index = new_array<IndexType> (ell.num_rows * ell.max_row_width);
    ell.values    = new_array<ValueType> (ell.num_rows * ell.max_row_width);

    // 初始化ELL格式的数组
    std::fill_n(ell.col_index, ell.num_rows*ell.max_row_width, static_cast<IndexType> (-1)); // 使用 -1 作为填充值，因为它不是有效的列索引
    std::fill_n(ell.values, ell.num_rows*ell.max_row_width, static_cast<ValueType> (0)); // 零填充values

    // 用COO数据填充ELL数组
    std::vector<IndexType> currentPos(ell.num_rows, 0); // 跟踪每行当前填充位置
    if (ColMajor == ell.ld){
        for (IndexType i = 0; i < ell.num_nnzs; ++i) {
            IndexType row = coo.row_index[i];
            IndexType pos = row + currentPos[row] * ell.num_rows;
            ell.values[pos] = coo.values[i];
            ell.col_index[pos] = coo.col_index[i];
            currentPos[row]++;
        }
    }
    else if (RowMajor == ell.ld){
        for (IndexType i = 0; i < ell.num_nnzs; ++i) {
            IndexType row = coo.row_index[i];
            IndexType pos = row * ell.max_row_width + currentPos[row];
            ell.values[pos] = coo.values[i];
            ell.col_index[pos] = coo.col_index[i];
            currentPos[row]++;
        }
    }

    return ell;
}

template <class IndexType, class ValueType>
COO_Matrix<IndexType, ValueType> csr_to_coo( const CSR_Matrix<IndexType, ValueType> &csr)
{
    COO_Matrix<IndexType, ValueType> coo;

    coo.num_rows = csr.num_rows;
    coo.num_cols = csr.num_cols;
    coo.num_nnzs = csr.num_nnzs;

    coo.row_index  = new_array<IndexType> (coo.num_nnzs);
    coo.col_index  = new_array<IndexType> (coo.num_nnzs);
    coo.values     = new_array<ValueType> (coo.num_nnzs);

    // 转换，按 row index 递增顺序来存 COO
    for (IndexType row = 0; row < coo.num_rows; ++row) {
        for (IndexType i = csr.row_offset[row]; i < csr.row_offset[row + 1]; ++i) {
            coo.row_index[i] = row;             // 行索引
            coo.col_index[i] = csr.col_index[i];    // 列索引
            coo.values[i] = csr.values[i];          // 非零值
        }
    }
    return coo;
}

/**
 * @brief Create the ELL format matrix from CSR format in row-major
 *        This routine do not delete the CSR_Matrix handle
 * 
 *         x  0  x  0  x   line: 0                      csr.row_offset = [0, 3, 5, 8, 12, 15] 
 *         x  x  0  0  0         1               csr.col = [0,2,4, 0,1, 0,3,4, 1,2,3,4, 1,2,4]
 *   A =   x  0  0  x  x         2              max_nnz_per_row = 4, row_num = 5
 *         0  x  x  x  x         3         ell.col_index = [0,0,0,1,1,  2,1,3,2,2,  4,-,4,3,4,  -,-,-,4,-]   col-major
 *         0  x  x  0  x         4         ell.col_index = [0,2,4,-,  0,1,-,-,  0,3,4,-,  1,2,3,4,  1,2,4,-] row-major
 * 
 * @tparam IndexType 
 * @tparam ValueType 
 * @param csr 
 * @return ELL_Matrix<IndexType, ValueType> 
 */
template <class IndexType, class ValueType>
ELL_Matrix<IndexType, ValueType> csr_to_ell(const CSR_Matrix<IndexType, ValueType> &csr, const LeadingDimension ld = RowMajor)
{
    ELL_Matrix<IndexType,ValueType> ell;

    ell.num_rows = csr.num_rows;
    ell.num_cols = csr.num_cols;
    ell.num_nnzs = csr.num_nnzs;

    ell.tag = 0;
    ell.ld = ld;
    IndexType max_nnz_per_row = 0;
    //计算每行的非零元数量并找到最大值
    for (IndexType i = 0; i < csr.num_rows; i++)
    {
        max_nnz_per_row = std::max( max_nnz_per_row, csr.row_offset[i+1] - csr.row_offset[i]);
    }
    ell.max_row_width = max_nnz_per_row;

    // 分配矩阵空间
    ell.col_index = new_array<IndexType> (ell.num_rows * ell.max_row_width);
    ell.values    = new_array<ValueType> (ell.num_rows * ell.max_row_width);

    ell.partition = copy_array<IndexType>(csr.partition, Le_get_thread_num()+1);

    // 初始化ELL格式的数组
    std::fill_n(ell.col_index, ell.num_rows*ell.max_row_width, static_cast<IndexType> (-1)); // 使用 -1 作为填充值，因为它不是有效的列索引
    std::fill_n(ell.values, ell.num_rows*ell.max_row_width, static_cast<ValueType> (0)); // 零填充values

    if (ColMajor == ld)
    {
        // 给ELL格式的两个数组进行赋值, col-major
        for (IndexType rowId = 0; rowId < ell.num_rows; ++rowId)
        {
            IndexType ellIndex = rowId;
            // 遍历 CSR row_ptr中 这行的所有非零元素
            for (IndexType csrIndex = csr.row_offset[rowId]; csrIndex < csr.row_offset[rowId+1]; ++csrIndex) 
            {
                ell.col_index[ellIndex] = csr.col_index[csrIndex];
                ell.values[ellIndex]    =    csr.values[csrIndex];
                ellIndex += ell.num_rows;
            }
        }
    }
    else if (RowMajor == ld)
    {
        // 给ELL格式的两个数组进行赋值, row-major
        for (IndexType rowId = 0; rowId < ell.num_rows; ++rowId)
        {
            IndexType ellIndex = rowId * ell.max_row_width;
            // 遍历 CSR row_ptr中 这行的所有非零元素
            for (IndexType csrIndex = csr.row_offset[rowId]; csrIndex < csr.row_offset[rowId+1]; ++csrIndex) 
            {
                ell.col_index[ellIndex] = csr.col_index[csrIndex];
                ell.values[ellIndex]    =    csr.values[csrIndex];
                ellIndex ++;
            }   
        }
    }
    return ell;
}

/**
 * @brief CSR to S_ELL format conversion
 *        Alignment in AVX512: float should be 4 bytes * 16 = 64 bytes. 
 *                             double should be 8 bytes * 8 = 64 bytes. 
 * @tparam IndexType 
 * @tparam ValueType 
 */
template <class IndexType, class ValueType>
S_ELL_Matrix<IndexType, ValueType> csr_to_sell(const CSR_Matrix<IndexType, ValueType> &csr, FILE *fp_feature, const int chunkwidth = CHUNK_SIZE,  const IndexType alignment = (SIMD_WIDTH/sizeof(ValueType)))
{
    S_ELL_Matrix<IndexType, ValueType> sell;

    sell.num_rows = csr.num_rows;
    sell.num_cols = csr.num_cols;
    sell.num_nnzs = csr.num_nnzs;

    sell.tag = 0;

    sell.sliceWidth = chunkwidth;
    sell.alignment  = alignment;

    // 确定需要多少个slice/chunk
    sell.chunk_num = (csr.num_rows + sell.sliceWidth - 1) / sell.sliceWidth; // 分块数向上取整

    // sell.row_width.resize (sell.chunk_num, 0);
    sell.row_width = new_array<IndexType>(sell.chunk_num);
    memset(sell.row_width, 0 , sell.chunk_num * sizeof(IndexType));

    for (IndexType row = 0; row < csr.num_rows; ++row) {
        IndexType chunk_id = row / sell.sliceWidth;
        IndexType row_nnz = csr.row_offset[row + 1] - csr.row_offset[row];
        sell.row_width[chunk_id] = std::max(sell.row_width[chunk_id], row_nnz);
    }
    // 对每个chunk的最大行宽度进行对齐
    // for (IndexType& width : sell.row_width) {
    //     width = ((width + sell.alignment - 1) / sell.alignment) * sell.alignment;
    // }
    for (IndexType i = 0; i < sell.chunk_num; i++)
    {
        sell.row_width[i] = ((sell.row_width[i] + sell.alignment - 1) / sell.alignment) * sell.alignment;
    }
    
    // sell.col_index.resize(sell.chunk_num);
    // sell.values.resize(sell.chunk_num);
    // // 初始化col_index (-1) 和 values (0)
    // for (IndexType chunk = 0; chunk < sell.chunk_num; ++chunk) {
    //     sell.col_index[chunk].resize(sell.row_width[chunk] * sell.sliceWidth, IndexType(-1));
    //     sell.values[chunk].resize(sell.row_width[chunk] * sell.sliceWidth, ValueType(0));
    // }
    // 为每个chunk的行指针数组分配内存
    sell.col_index = new IndexType*[sell.chunk_num];
    sell.values = new ValueType*[sell.chunk_num];
    for (IndexType chunk = 0; chunk < sell.chunk_num; ++chunk) {
        size_t elem_nums = sell.row_width[chunk] * sell.sliceWidth;

        sell.col_index[chunk] = new_array<IndexType> (elem_nums);
        // 初始化col_index中的每个元素为-1
        std::fill_n(sell.col_index[chunk], elem_nums, static_cast<IndexType>(-1));

        sell.values[chunk] = new_array<ValueType> (elem_nums);
        // 初始化values中的每个元素为0
        std::fill_n(sell.values[chunk], elem_nums, ValueType(0));
    }


    //转换 CSR 到 S-ELL
    for (IndexType row = 0; row < csr.num_rows; ++row)
    {
        IndexType chunk_id         = row / sell.sliceWidth;    // 所属的 chunk 号
        IndexType row_within_chunk = row % sell.sliceWidth; // chunk 内部的行号 0 ~ sliceWidth-1
        IndexType row_start        = csr.row_offset[row];
        IndexType row_end          = csr.row_offset[row+1];

        for (IndexType idx = row_start; idx < row_end; idx++)
        {
            IndexType col = csr.col_index[idx];
            ValueType val = csr.values[idx];

            IndexType pos = row_within_chunk * sell.row_width[chunk_id] + (idx - row_start);
            sell.col_index[chunk_id][pos] = col;
            sell.values[chunk_id][pos] = val;
        } 
    }
    
    return sell;
}

template <class IndexType, class ValueType>
DIA_Matrix<IndexType, ValueType> csr_to_dia(const CSR_Matrix<IndexType, ValueType> &csr, const IndexType max_diags, FILE *fp_feature, const IndexType alignment = (SIMD_WIDTH/sizeof(ValueType)))
{
    DIA_Matrix<IndexType, ValueType> dia;

    dia.num_rows     = csr.num_rows;
    dia.num_cols     = csr.num_cols;
    dia.num_nnzs     = csr.num_nnzs;
    dia.diag_offsets = nullptr;
    dia.diag_data    = nullptr;
    dia.tag          = 0;

    // compute number of occupied diagonals and enumerate them
    IndexType complete_ndiags = 0;
    const IndexType unmarked = (IndexType) -1;

    IndexType* diag_map = new_array<IndexType> (dia.num_rows + dia.num_cols);
    std::fill(diag_map, diag_map + dia.num_rows + dia.num_cols, unmarked);

    IndexType* diag_map_2 = new_array<IndexType> (dia.num_rows + dia.num_cols);
    std::fill(diag_map_2, diag_map_2 + dia.num_rows + dia.num_cols, 0);

    for (IndexType i = 0; i < dia.num_rows; i++)
    {
        //  遍历 csr 的第 i 行元素
        for (IndexType jj = csr.row_offset[i]; jj < csr.row_offset[i+1]; jj++)
        {
            IndexType j = csr.col_index[jj]; // j : 元素的列序号
            IndexType map_index = (csr.num_rows - i) + j; //offset shifted by + num_rows

            if( diag_map[map_index] == unmarked)
            {
                diag_map[map_index] = complete_ndiags;
                complete_ndiags ++;
            }
            diag_map_2[map_index] ++;
        }
    }

    IndexType j_ndiags = 0;
    double ratio;
    IndexType NTdiags = 0;
    double* array_ndiags = new_array<double>(10);
    std::fill(array_ndiags, array_ndiags + 10, 0.0);

    for(IndexType i = 0; i < dia.num_rows + dia.num_cols; ++i){
        //  此条对角线非空
        if( diag_map_2[i] != 0 )
        {
            j_ndiags ++;
            ratio = (double) diag_map_2[i] / csr.num_rows;

            if (ratio < 0.1 )
                array_ndiags[0] ++;
            else if (ratio < 0.2 )
                array_ndiags[1] ++;
            else if (ratio < 0.3 )
                array_ndiags[2] ++;
            else if (ratio < 0.4 )
                array_ndiags[3] ++;
            else if (ratio < 0.5 )
                array_ndiags[4] ++;
            else if (ratio < 0.6 )
                array_ndiags[5] ++;
            else if (ratio < 0.7 )
                array_ndiags[6] ++;
            else if (ratio < 0.8 )
                array_ndiags[7] ++;
            else if (ratio < 0.9 )
                array_ndiags[8] ++;
            else if (ratio <= 1.0 )
                array_ndiags[9] ++;

            if (ratio >= NTRATIO )
                NTdiags ++;
        }
    }
    assert( j_ndiags == complete_ndiags);
    delete_array (diag_map_2);
#ifdef COLLECT_FEATURES
        fprintf(fp_feature, "Ndiags : %d\n", complete_ndiags );
#endif
    for ( int i=0; i<10; i++)
    {
        array_ndiags[i] /= complete_ndiags;
// 对角线稠密范围
#ifdef COLLECT_FEATURES
          if ( i == 0 )
            fprintf(fp_feature, "Num_diags ER in ( %d %%, %d %% ) : %lf \n", i*10, (i+1)*10, array_ndiags[i] );
          else if ( i == 9 )
            fprintf(fp_feature, "Num_diags ER in [ %d %%, %d %% ] : %lf \n", i*10, (i+1)*10, array_ndiags[i] );
          else
            fprintf(fp_feature, "Num_diags ER in [ %d %%, %d %% ) : %lf \n", i*10, (i+1)*10, array_ndiags[i] );
#endif
    }
    
#ifdef COLLECT_FEATURES
        // 达到 NT 比例的对角线占比
        double NTdiags_ratio = (double) NTdiags/ complete_ndiags;
        // DIA 格式下的稠密度
        double ER_DIA = (double) dia.num_nnzs / (complete_ndiags * dia.num_rows);
        fprintf(fp_feature, "NTdiags_ratio : %lf  ( TH is 0.6 )\n", NTdiags_ratio );
        fprintf(fp_feature, "ER_DIA : %lf\n", ER_DIA );
#endif
    delete_array(array_ndiags);
    dia.complete_ndiags = complete_ndiags;

    if(complete_ndiags > max_diags)
    {
        printf("\tNumber of diagonals (%d) excedes limit (%d)\n", dia.complete_ndiags, max_diags);
        // dia.num_rows     = 0;
        // dia.num_cols     = 0;
        // dia.num_nnzs     = 0;
        dia.stride       = 0; 
        dia.gflops	= 0;
        delete_array(diag_map);                                     
        return dia;
    }

    // length of each diagonal in memory, 按照 alignment 对齐
    dia.stride = alignment * ((dia.num_rows + alignment - 1)/ alignment);

    dia.diag_offsets = new_array<int>       (dia.complete_ndiags);
    dia.diag_data    = new_array<ValueType> (dia.complete_ndiags * dia.stride);

    std::fill(dia.diag_data, dia.diag_data + dia.complete_ndiags * dia.stride, ValueType(0));

    for(IndexType n = 0; n < dia.num_rows + dia.num_cols; n++)
        if(diag_map[n] != unmarked) // 算出offset
            dia.diag_offsets[diag_map[n]] = (int) n - (int) dia.num_rows;
        
    for (IndexType i = 0; i < csr.num_rows; i++)
    {
        for(IndexType jj = csr.row_offset[i]; jj < csr.row_offset[i+1]; jj++){
            IndexType j = csr.col_index[jj];
            IndexType map_index = (csr.num_rows - i) + j; //offset shifted by + num_rows
            IndexType diag = diag_map[map_index];
            dia.diag_data[diag*dia.stride + i] = csr.values[jj];
        }
    }
    
    delete_array(diag_map);

    return dia;

}

template <class IndexType, typename UIndexType, class ValueType>
CSR5_Matrix<IndexType, UIndexType, ValueType> csr_to_csr5(const CSR_Matrix<IndexType, ValueType> &csr, FILE *fp_feature)
{
    CSR5_Matrix<IndexType, UIndexType, ValueType> csr5;

    csr5.num_rows = csr.num_rows;
    csr5.num_cols = csr.num_cols;
    csr5.num_nnzs = csr.num_nnzs;

    // original CSR format array
    csr5.row_offset = copy_array(csr.row_offset, csr.num_rows + 1);
    csr5.col_index  = copy_array(csr.col_index , csr.num_nnzs);
    csr5.values     = copy_array(csr.values    , csr.num_nnzs);

    csr5.tile_ptr  = NULL;
    csr5.tile_desc = NULL;
    csr5.tile_desc_offset_ptr = NULL;
    csr5.tile_desc_offset     = NULL;
    csr5.calibrator           = NULL;

    // store sigma and omega (tiles row and column number)
    csr5.omega = SIMD_WIDTH / 8 / sizeof(ValueType);
    csr5.sigma = CSR5_SIGMA;  // fixed in paper   12 or 16
/*
    //  heuristic in ALSPARSE
    int r = 4;
    int s = 32;
    int t = 256;
    int u = 6;
    IndexType csr_nnz_per_row = csr5.num_nnzs / csr5.num_rows;
    if (csr_nnz_per_row <= r)
        csr5.sigma = r;
    else if (csr_nnz_per_row > r && csr_nnz_per_row <= s)
        csr5.sigma = csr_nnz_per_row;
    else if (csr_nnz_per_row <= t && csr_nnz_per_row > s)
        csr5.sigma = s;
    else // csr_nnz_per_row > t
        csr5.sigma = u;
*/
    // compute how many bits required for `y_offset' and `seg_offset'
    IndexType base = 2;
    csr5.bit_y_offset = 1;
    while (base < csr5.sigma * csr5.omega) {
        base *= 2; ++csr5.bit_y_offset;
    }

    base = 2;
    csr5.bit_scansum_offset = 1;
    while (base < csr5.omega){
        base *= 2; ++csr5.bit_scansum_offset;
    }
    
    if( csr5.bit_y_offset + csr5.bit_scansum_offset > sizeof(UIndexType) * 8 - 1) //the 1st bit of bit-flag should be in the first packet
    {
        printf("error: UNSUPPORTED CSR5 OMEGA for bit saving\n");
        return -1;
    }

    int bit_all = csr5.bit_y_offset + csr5.bit_scansum_offset + csr5.sigma;
    csr5.num_packets = ceil((double)bit_all / (double)(sizeof(UIndexType) * 8));
    
    // calculate the number of partitions
    csr5._p = ceil( (double) csr5.num_nnzs/ (double) (csr5.omega * csr5.sigma));

    // malloc the newly added arrays for CSR5
    csr5.tile_ptr = memalign(X86_CACHELINE, (uint64_t) (csr5._p + 1) * sizeof(UIndexType));
    if (csr5.tile_ptr == NULL){
        printf("error: UNABLE TO ASIGN MEMORY IN CSR5 tile_ptr \n");
        return -2;
    }
    for(IndexType i = 0; i < csr5._p + 1; i++) {
        csr5.tile_ptr[i] = 0;
    }

    csr5.tile_desc = memalign(X86_CACHELINE, (uint64_t)( csr5._p * csr5.omega * csr5.num_packets) * sizeof(UIndexType));
    if (csr5.tile_desc == NULL){
        printf("error: UNABLE TO ASIGN MEMORY IN CSR5 tile_desc \n");
        return -2;
    }
    memset(csr5.tile_desc, 0, csr5._p * csr5.omega * csr5.num_packets * sizeof(UIndexType));

    int num_thread = Le_get_thread_num();
    csr5.calibrator = memalign(X86_CACHELINE, (uint64_t)(num_thread * X86_CACHELINE));
    if (csr5.tile_desc == NULL){
        printf("error: UNABLE TO ASIGN MEMORY IN CSR5 calibrator \n");
        return -2;
    }
    memset(csr5.calibrator, 0, num_thread * X86_CACHELINE);

    csr.tile_desc_offset_ptr = memalign(X86_CACHELINE, (uint64_t) (csr5._p + 1) * sizeof(IndexType));
    if (csr.tile_desc_offset_ptr == NULL){
        printf("error: UNABLE TO ASIGN MEMORY IN CSR5 tile_desc_offset_ptr \n");
        return -2;
    }
    memset(csr.tile_desc_offset_ptr, 0, (csr5._p + 1) * sizeof(IndexType));

    // convert csr data to csr5 data (3 steps)
    // step 1. generate partition pointer
    // ... need test above routine

    return csr5;
}

#endif /* SPARSE_CONVERSION_H */
