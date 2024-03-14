/**
 * @file sparse_features.cpp
 * @author your name (you@domain.com)
 * @brief  用于提取稀疏矩阵的待分析 features， 参考solver组实现
 * @version 0.1
 * @date 2024-01-23
 * 
 * @copyright Copyright (c) 2024
 * 
 */
#include"sparse_io.h"
#include"sparse_format.h"
#include"../include/thread.h"
#include"../include/sparse_features.h"
#include"../include/sparse_partition.h"
#include"../include/thread.h"

#include <cassert>
#include <cmath>
#include <fstream>
#include <string>
#include <algorithm>
#include <numeric>

std::string my_to_String(int n)
{
    char s[20];  // 20 字符的缓冲区足够容纳 64 位整数
    int m = n;
    int i = 0, j = 0;

    if (m == 0) {
        s[i++] = '0';  // 处理零的情况
    } else if (m < 0) {
        s[i++] = '-';  // 处理负数的情况
        m = -m;  // 将负数转化为正数
    }

    while (m > 0) {
        s[i++] = m % 10 + '0';
        m /= 10;
    }

    s[i] = '\0';
    i = i - 1;
    
    char ss[20];  // 同样大小的缓冲区
    while (i >= 0) {
        ss[j++] = s[i--];
    }
    ss[j] = '\0';

    return std::string(ss);
}

template <typename IndexType, typename ValueType>
void MTX<IndexType, ValueType>::Stringsplit(const std::string& s, const char split, std::vector<std::string>& res) {
      
      std::string str = s;
      if (str == "")		return;
      str.erase(0,str.find_first_not_of(" "));
      str.erase(str.find_last_not_of(" ")+1);//清除line两端的空格
      //在字符串末尾也加入分隔符，方便截取最后一段
      std::string strs = str + split;
      size_t pos = strs.find(split);

      // 若找不到内容则字符串搜索函数返回 npos
      while (pos != strs.npos)
      {
          std::string temp = strs.substr(0, pos);
          res.push_back(temp);
          //去掉已分割的字符串,在剩下的字符串中进行分割
          strs = strs.substr(pos + 1, strs.size());
          strs.erase(0,strs.find_first_not_of(" "));
          pos = strs.find(split);
      }
}
/*
             __ --|
            |     |
y_values_0  |     |  y_values_1
            |     |
            |_____|  _____   _____   _____
            x_step
*/
template <typename ValueType>
double trapezoidalRule(const std::vector<ValueType>& y_values, const ValueType x_length) {
    double area = 0.0;
    double x_step = (double) x_length / y_values.size(); // should be 1 here.

    area += (y_values[0] + 0) / 2.0;
    for (size_t i = 0; i < y_values.size() - 1; ++i) {
        double y_avg = (double) (y_values[i] + y_values[i + 1]) / 2.0;
        area += y_avg * x_step;
    }

    return area;
}

std::string extractFileNameWithoutExtension(const std::string& filePath) {
    // 查找最后一个路径分隔符
    size_t lastSlash = filePath.find_last_of("/\\");
    if (lastSlash == std::string::npos) {
        lastSlash = 0; // 没有路径分隔符
    } else {
        lastSlash += 1; // 移过路径分隔符
    }

    // 查找最后一个点（扩展名的开始）
    size_t lastDot = filePath.find_last_of('.');
    if (lastDot == std::string::npos || lastDot < lastSlash) {
        lastDot = filePath.length(); // 没有扩展名
    }

    // 提取文件名
    return filePath.substr(lastSlash, lastDot - lastSlash);
}

template <typename IndexType, typename ValueType>
bool MTX<IndexType, ValueType>::MtxLoad(const char* mat_path) 
{
    int retcode = 0;
    MM_typecode mat_code;
    int is_integer = 0, is_real = 0, is_pattern = 0;

    FILE *mtx_file = fopen(mat_path, "r");
    if(mtx_file == NULL){
        std::cout << "Unable to open mtx file: "<< mat_path << std::endl;
        exit(1);
    }
    matrixName = extractFileNameWithoutExtension(mat_path);

    // Get the matrix type
    if (mm_read_banner(mtx_file, &mat_code) != 0){
        std::cout << "Could not process Matrix Market banner." << std::endl;
        exit(1);
    }

    if(!mm_is_valid(mat_code)){
        std::cout << "Invalid Matrix" << std::endl;
        exit(1);
    }
    
    if (!((mm_is_real(mat_code) || mm_is_integer(mat_code) || mm_is_pattern(mat_code)) && mm_is_coordinate(mat_code) && mm_is_sparse(mat_code) ) ){
        printf("Sorry, this application does not support ");
        printf("Market Market type: [%s]\n", mm_typecode_to_str(mat_code));
        printf("Only sparse real-valued or pattern coordinate matrices are supported\n");
        exit(1);
    }

    if (mm_is_complex(mat_code)) { return false; } 
    if (mm_is_pattern(mat_code))  { is_pattern = 1; }
    if (mm_is_real(mat_code)) { is_real = 1; }
    if (mm_is_integer (mat_code) ) { is_integer = 1; }

    if (mm_is_symmetric(mat_code) || mm_is_hermitian( mat_code )) {
        is_symmetric_ = true;
    } else {
        is_symmetric_ = false;
    }

    // Get size of sparse matrix.
    if (mm_read_mtx_crd_size(mtx_file, &num_rows, &num_cols, &nnz_mtx_) != 0)
    {
        std::cout << "The line of rows, cols, nnzs is in wrong format" << std::endl;
        exit(1);
    }

    nnz_by_row_.resize(num_rows, 0);
    nnz_by_col_.resize(num_cols, 0);
    Diag_Dom.resize(num_rows, 0.0);
    max_each_row_.resize(num_rows, 0.0);
    min_each_row_.resize(num_rows, 0.0);
    max_each_col_.resize(num_cols, 0.0);
    min_each_col_.resize(num_cols, 0.0);

    symm_pair_.reserve(nnz_mtx_);

    //  判断矩阵是否足够大分 tile 读取 tile特征
    bool tile_flag = (num_rows >= t_num_blocks) && (num_cols >= t_num_blocks);
    
    //  可以存 tile features
    if (tile_flag){
        // 为避免空块，只能向下取整； 多出的元素均匀分给 0 ~ t_mod_RB-1 行，和 0 ~ t_mod_CB-1 列
        t_num_RB = num_rows / t_num_blocks; t_mod_RB = num_rows % t_num_blocks;
        t_num_CB = num_cols / t_num_blocks; t_mod_CB = num_cols % t_num_blocks;
        nnz_by_Tiles_.resize(t_num_blocks * t_num_blocks, 0);
        nnz_by_RB_.resize(t_num_blocks, 0);
        nnz_by_CB_.resize(t_num_blocks, 0);
    }

    IndexType row_idx, col_idx, diaoffset;
    IndexType t_rowidx, t_colidx;   // tiles 中的序号
    // rowidx < threshold  tile_size = (t_num_RB + 1); else tile_size = t_num_RB
    IndexType RB_threshold = t_mod_RB * (t_num_RB + 1);
    IndexType CB_threshold = t_mod_CB * (t_num_CB + 1);
    ValueType value;
    ValueType value_abs;

    std::cout << "- Reading sparse matrix from file: "<< mat_path << std::endl;
    fflush(stdout);

    if(mm_is_pattern(mat_code)){        // 二进制矩阵， 元素只有0/1
        for (IndexType i = 0; i < nnz_mtx_; i++)
        {
            if constexpr(std::is_same<IndexType, int>::value) {
                assert(fscanf(mtx_file,"%d %d\n", &row_idx, &col_idx) == 2);
            } else if constexpr(std::is_same<IndexType, long long>::value) {
                assert(fscanf(mtx_file,"%lld %lld\n", &row_idx, &col_idx) == 2);
            }
            // adjust from 1-based to 0-based indexing
            --row_idx;
            --col_idx;
            diaoffset = col_idx - row_idx;
            value = 1.0;
            value_abs = 1.0;
            
            nnz_by_row_[row_idx]++; // 本行的 nnz 加一
            nnz_by_col_[col_idx]++; // 本列的 nnz 加一
            // 记录对角距离的分布频率
            diag_offset_[diaoffset]++;
            
            // 存一下分tile的信息
            if (tile_flag){
                t_rowidx = (row_idx < RB_threshold)? (row_idx / (t_num_RB+1)):(t_mod_RB + (row_idx - RB_threshold)/t_num_RB);
                t_colidx = (col_idx < CB_threshold)? (col_idx / (t_num_CB+1)):(t_mod_CB + (col_idx - CB_threshold)/t_num_CB);
                nnz_by_Tiles_[t_rowidx * t_num_blocks + t_colidx]++;
                nnz_by_RB_[t_rowidx]++;
                nnz_by_CB_[t_colidx]++;
            }

            if(is_symmetric_){              // 对称矩阵情况
                if(row_idx == col_idx){     // 对角线
                    nnz_diagonal_ ++;
                    Diag_Dom[row_idx] += value_abs;
                    max_value_diagonal_ = 1.0; // pattern matrix only have 1.0
                } else{                     // 非对角线
                    nnz_by_row_[col_idx]++;     // 对称的情况，把列号所在的nnz也加进来
                    nnz_by_col_[row_idx]++;     // 对称的情况，把行号所在的nnz也加进来
                    // 记录对角距离的分布频率
                    diag_offset_[-diaoffset]++;

                    // 存一下 tile 的信息
                    if(tile_flag)
                    {
                        nnz_by_Tiles_[t_colidx * t_num_blocks + t_rowidx]++;
                        nnz_by_RB_[t_colidx]++;
                        nnz_by_CB_[t_rowidx]++;
                    }
                    nnz_lower_ ++;
                    nnz_upper_ ++;
                    Diag_Dom[row_idx] -= 1.0;
                    Diag_Dom[col_idx] -= 1.0;
                    max_value_offdiag_ = 1.0;
                }
            }else { // 非对称矩阵
                if (row_idx == col_idx)  // 行 == 列， 对角线
                {
                    nnz_diagonal_ ++;
                    max_value_diagonal_ = max_value_diagonal_ > value_abs ? max_value_diagonal_ : value_abs;
                    Diag_Dom[row_idx] += value_abs;
                } else {                    // 非对角线情况
                    if (row_idx > col_idx) { // 行 > 列，元素在下三角
                        nnz_lower_ ++;
                    } else{                  // 行 < 列，元素在上三角
                        nnz_upper_ ++;
                    }
                    max_value_offdiag_ = max_value_offdiag_ > value_abs ? max_value_offdiag_ : value_abs;
                    Diag_Dom[row_idx] -= value_abs;
                }
                char buffer[100];
                sprintf(buffer, "%lf", value);
                std::string str_value  = buffer;
                std::string str_insert = my_to_String(row_idx) + "_" + my_to_String(col_idx);
                m_.insert(std::make_pair(str_insert, str_value));
                symm_pair_.push_back(str_insert);
            }
        }
    }else if (mm_is_real(mat_code) || mm_is_integer(mat_code)){
        for( IndexType i = 0; i < nnz_mtx_; i++ ){
            IndexType row_id, col_id;
            double V;
            if constexpr(std::is_same<IndexType, int>::value) {
                assert(fscanf(mtx_file, "%d %d %lf\n", &row_id, &col_id, &V) == 3);
            } else if constexpr(std::is_same<IndexType, long long>::value) {
                assert(fscanf(mtx_file, "%lld %lld %lf\n", &row_id, &col_id, &V) == 3);
            }

            row_idx   = (IndexType) row_id - 1;
            col_idx   = (IndexType) col_id - 1;
            value     = (ValueType) V;
            value_abs = std::abs(value);
            diaoffset = col_idx - row_idx;

            nnz_by_row_[row_idx]++;
            nnz_by_col_[col_idx]++;
            // 记录对角距离的分布频率
            diag_offset_[diaoffset]++;

            // 存一下分tile的信息
            if (tile_flag){
                t_rowidx = (row_idx < RB_threshold)? (row_idx / (t_num_RB+1)):(t_mod_RB + (row_idx - RB_threshold)/t_num_RB);
                t_colidx = (col_idx < CB_threshold)? (col_idx / (t_num_CB+1)):(t_mod_CB + (col_idx - CB_threshold)/t_num_CB);

                nnz_by_Tiles_[t_rowidx * t_num_blocks + t_colidx]++;
                nnz_by_RB_[t_rowidx]++;
                nnz_by_CB_[t_colidx]++;
            }

            if(is_symmetric_){
                if(row_idx == col_idx){
                    nnz_diagonal_ ++;
                    Diag_Dom[row_idx] += value_abs;
                    max_value_diagonal_ = max_value_diagonal_ > value_abs ? max_value_diagonal_ : value_abs;

                } else{
                    nnz_by_row_[col_idx]++;
                    nnz_by_col_[row_idx]++;
                    // 记录对角距离的分布频率
                    diag_offset_[-diaoffset]++;
                    if(tile_flag)
                    {
                        nnz_by_Tiles_[t_colidx * t_num_blocks + t_rowidx]++;
                        nnz_by_RB_[t_colidx]++;
                        nnz_by_CB_[t_rowidx]++;
                    }
                    nnz_lower_ ++;
                    nnz_upper_ ++;
                    Diag_Dom[row_idx] -= value_abs;
                    Diag_Dom[col_idx] -= value_abs;
                    max_value_offdiag_ = max_value_offdiag_ > value_abs ? max_value_offdiag_ : value_abs;
                }
                // 记录row-variability 和 col-variability : log10(max/min)
                if(max_each_row_[row_idx] < log10(value_abs)) { max_each_row_[row_idx] = log10(value_abs); }
                if(max_each_row_[col_idx] < log10(value_abs)) { max_each_row_[col_idx] = log10(value_abs); }
                if(value_abs > 0.0 && min_each_row_[row_idx] > log10(value_abs)) { min_each_row_[row_idx] = log10(value_abs); }
                if(value_abs > 0.0 && min_each_row_[col_idx] > log10(value_abs)) { min_each_row_[col_idx] = log10(value_abs); }

                if(max_each_col_[col_idx] < log10(value_abs)) { max_each_col_[col_idx] = log10(value_abs); }
                if(max_each_col_[row_idx] < log10(value_abs)) { max_each_col_[row_idx] = log10(value_abs); }
                if(value_abs > 0.0 && min_each_col_[col_idx] > log10(value_abs)) { min_each_col_[col_idx] = log10(value_abs); }
                if(value_abs > 0.0 && min_each_col_[row_idx] > log10(value_abs)) { min_each_col_[row_idx] = log10(value_abs); }
            }
            else { // 非对称矩阵
                if (row_idx == col_idx)  // 行 == 列， 对角线
                {
                    nnz_diagonal_ ++;
                    max_value_diagonal_ = max_value_diagonal_ > value_abs ? max_value_diagonal_ : value_abs;
                    Diag_Dom[row_idx] += value_abs;
                } else {                    // 非对角线情况
                    if (row_idx > col_idx) { // 行 > 列，元素在下三角
                        nnz_lower_ ++;
                    } else{                  // 行 < 列，元素在上三角
                        nnz_upper_ ++;
                    }
                    max_value_offdiag_ = max_value_offdiag_ > value_abs ? max_value_offdiag_ : value_abs;
                    Diag_Dom[row_idx] -= value_abs;
                }

                char buffer[100];
                sprintf(buffer, "%lf", value);
                std::string str_value  = buffer;
                std::string str_insert = my_to_String(row_idx) + "_" + my_to_String(col_idx);
                m_.insert(std::make_pair(str_insert, str_value));
                symm_pair_.push_back(str_insert);

                // 记录row-variability 和 col-variability : log10(max/min)
                if(max_each_row_[row_idx] < log10(value_abs)) { max_each_row_[row_idx] = log10(value_abs); }
                if(value_abs > 0.0 && min_each_row_[row_idx] > log10(value_abs)) { min_each_row_[row_idx] = log10(value_abs); }

                if(max_each_col_[col_idx] < log10(value_abs)) { max_each_col_[col_idx] = log10(value_abs); }
                if(value_abs > 0.0 && min_each_col_[col_idx] > log10(value_abs)) { min_each_col_[col_idx] = log10(value_abs); }
            }
        }
    }else{
        std::cout << "Unsupported data type" << std::endl;
        exit(1);
    }

    if(is_symmetric_){
        num_nnzs = nnz_lower_*2 + nnz_diagonal_;
    }else{
        num_nnzs = nnz_mtx_;
    }

    fclose(mtx_file);
    return true;
}

template bool MTX<int, float>::MtxLoad(const char* mat_path);
template bool MTX<int, double>::MtxLoad(const char* mat_path);
template bool MTX<long long, float>::MtxLoad(const char* mat_path);
template bool MTX<long long, double>::MtxLoad(const char* mat_path);

template <typename IndexType, typename ValueType>
void P_ratioAndGini(const std::vector<IndexType> vec, const IndexType num_nnzs, ValueType &p_ratio, ValueType &Gini)
{
    std::vector<IndexType> ordered_vec = vec;
    std::sort(ordered_vec.begin(), ordered_vec.end());

    IndexType length = vec.size();
    IndexType quit = length;
    IndexType step_numnnzs = 0, count = 0;
    ValueType p_tmp = 0.0, one_minus_p = 0.0;

    do
    {
        --quit; ++count;
        step_numnnzs += ordered_vec[quit];
        p_tmp       = (double) count / length;
        one_minus_p = (double) step_numnnzs / num_nnzs;
    } while (p_tmp + one_minus_p < 1.0);
    // calculate p-ratio
    p_ratio = p_tmp;

    // 累计和
    std::vector<IndexType> cumulative_sum(ordered_vec.size());
    std::partial_sum(ordered_vec.begin(), ordered_vec.end(), cumulative_sum.begin());
    // 梯形规则（trapezoidal rule）近似计算面积B
    ValueType Area_B     = trapezoidalRule(cumulative_sum, length);
    ValueType Area_total = (ValueType) num_nnzs * length / 2.0;
    // calculate Gini coefficient
    Gini = (Area_total - Area_B) / Area_total;
}

template <typename IndexType, typename ValueType>
bool MTX<IndexType, ValueType>::CalculateFeatures() 
{
    IndexType nz_row_num = 0, nz_col_num = 0, dominance = 0;
    IndexType diaglineNum = num_rows + num_cols - 1; // 矩阵的主对角线条数
    IndexType close_threshold =  diaglineNum / 20; // 5% 比率靠近对角线作为阈值
    IndexType close_nnz = 0;
    // Calculate total NNZ and NNZ close to the diagonal
    for (const auto& offset_count : diag_offset_) {
        if (std::abs(offset_count.first) <= close_threshold) {
            close_nnz += offset_count.second;
        }
    }
    diag_close_ratio_ = (ValueType) close_nnz / num_nnzs;

    // 对于digital matrix， 这两个值不会变动，结果为 -1
    ValueType row_divide_max = -1.0;// 确保起始值足够低
    ValueType col_divide_max = -1.0;// 确保起始值足够低

    IndexType threadNum = Le_get_thread_num();

    nnz_ratio_ =  (ValueType) num_nnzs / ( (ValueType) num_rows * num_cols);
    ave_nnz_each_row_ = (ValueType) num_nnzs / num_rows;
    ave_nnz_each_col_ = (ValueType) num_nnzs / num_cols;
    
    for (IndexType i = 0; i < num_rows; i++)
    {
        min_nnz_each_row_ = std::min(min_nnz_each_row_, nnz_by_row_[i]);
        max_nnz_each_row_ = std::max(max_nnz_each_row_, nnz_by_row_[i]);

        if (nnz_by_row_[i])
        {
            nz_row_num++;
        }

        if ( min_each_row_[i] ){       // 计算 row_divide_max
            ValueType tmp = max_each_row_[i] - min_each_row_[i];
            row_divide_max = std::max(row_divide_max, tmp);
        }
        ValueType diff = nnz_by_row_[i] - ave_nnz_each_row_; // 计算每行nnz和平均值的差
        var_nnz_each_row_ += diff * diff;
        if ( Diag_Dom[i] > 0)
        {
            dominance ++;
        }
    }
    // 计算 row nz_ratio 和 其他统计信息
    nz_row_ratio_ = (ValueType) nz_row_num/ num_rows;
    row_variability_ = row_divide_max;
    var_nnz_each_row_ /= num_rows;
    standard_dev_row_ = std::sqrt(var_nnz_each_row_);

    // calculate P-ratio and Gini index
    P_ratioAndGini(nnz_by_row_, num_nnzs, P_ratio_row_, Gini_row_);

    for (IndexType j = 0; j < num_cols; j++)
    {
        min_nnz_each_col_ = std::min(min_nnz_each_col_, nnz_by_col_[j]);
        max_nnz_each_col_ = std::max(max_nnz_each_col_, nnz_by_col_[j]);

        if(nnz_by_col_[j])
        {
            nz_col_num++;
        }
        if ( min_each_col_[j] ){       // 计算 col_divide_max
            ValueType tmp = max_each_col_[j] - min_each_col_[j];
            col_divide_max = std::max(col_divide_max, tmp);
        }
        ValueType diff = nnz_by_col_[j] - ave_nnz_each_col_; // 计算每列nnz和平均值的差
        var_nnz_each_col_ += diff * diff;
    }
    // 计算 col nz_ratio 和 其他统计信息
    nz_col_ratio_ = (ValueType) nz_col_num/ num_cols;
    col_variability_ = col_divide_max;
    var_nnz_each_col_ /= num_cols;
    standard_dev_col_ = std::sqrt(var_nnz_each_col_);

    // calculate P-ratio and Gini index
    P_ratioAndGini(nnz_by_col_, num_nnzs, P_ratio_col_, Gini_col_);

    // 计算对角占优比例
    diagonal_dominant_ratio_ = (ValueType) dominance/ std::min(num_rows,num_cols);

    if ( is_symmetric_){
        pattern_symm_ = 1.0;
        value_symm_   = 1.0;
    } 
    else if (num_rows == num_cols) 
    {
        IndexType symm_num_pattern = 0;
        IndexType symm_num_value   = 0;

        #pragma omp parallel for num_threads(threadNum) reduction(+:symm_num_pattern,symm_num_value)
        for (IndexType i = 0; i < nnz_mtx_; i++)
        {
            std::vector<std::string> vec;
            std::string key_copy = symm_pair_[i];
            Stringsplit(key_copy,'_',vec);
            if (vec[1] == vec[0])//不用计算对角线元素
            {
                continue;
            }
            std::string new_key = vec[1] + "_" + vec[0];
            auto index = m_.find(new_key);

            if (index !=m_.end()) //非对焦元素模式对称
            {
                symm_num_pattern ++;
                if(m_[new_key] == m_[key_copy])
                    symm_num_value ++;
            }
        }
        pattern_symm_ = (ValueType) symm_num_pattern/ (nnz_mtx_ - nnz_diagonal_);
        value_symm_   = (ValueType) symm_num_value  / (nnz_mtx_ - nnz_diagonal_);
        
    }
    else{       // rectangular
        pattern_symm_ = 0.0;
        value_symm_ = 0.0;
    }

    if ( pattern_symm_ == 1.0 && value_symm_ == 1.0)
        is_symmetric_ = true;
    else
        is_symmetric_ = false;


    if ((num_rows >= t_num_blocks) && (num_cols >= t_num_blocks) )
    {
        CalculateTilesFeatures();
    }

    return true;
}
template bool MTX<int, float>::CalculateFeatures();
template bool MTX<int, double>::CalculateFeatures();
template bool MTX<long long, float>::CalculateFeatures();
template bool MTX<long long, double>::CalculateFeatures();

template <typename IndexType, typename ValueType>
bool MTX<IndexType, ValueType>::CalculateTilesFeatures()
{
    t_ave_nnz_all_tiles = (ValueType) num_nnzs / ((ValueType) t_num_blocks * t_num_blocks);
    t_ave_nnz_RB        = (ValueType) num_nnzs / t_num_blocks;
    t_ave_nnz_CB        = (ValueType) num_nnzs / t_num_blocks;

    ValueType diff_tiles, diff_RB, diff_CB;

    // Count RB and CB features
    for (size_t i = 0; i < t_num_blocks; i++)
    {
        if (nnz_by_RB_[i])
            ++t_nz_ratio_RB_ ;
        // else
        //     printf("zero RB = %d\n", i);

        if (nnz_by_CB_[i])
            ++t_nz_ratio_CB_ ;

        diff_RB = nnz_by_RB_[i] - t_ave_nnz_RB;
        diff_CB = nnz_by_CB_[i] - t_ave_nnz_CB;
        t_var_nnz_RB += diff_RB * diff_RB;
        t_var_nnz_CB += diff_CB * diff_CB;

        t_min_nnz_each_RB_ = std::min(t_min_nnz_each_RB_, nnz_by_RB_[i]);
        t_max_nnz_each_RB_ = std::max(t_max_nnz_each_RB_, nnz_by_RB_[i]);

        t_min_nnz_each_CB_ = std::min(t_min_nnz_each_CB_, nnz_by_CB_[i]);
        t_max_nnz_each_CB_ = std::max(t_max_nnz_each_CB_, nnz_by_CB_[i]);
    }
    t_nz_ratio_RB_ /= (ValueType) t_num_blocks;
    t_var_nnz_RB   /= (ValueType) t_num_blocks;
    t_standard_dev_RB = std::sqrt(t_var_nnz_RB);

    t_nz_ratio_CB_ /= (ValueType) t_num_blocks;
    t_var_nnz_CB   /= (ValueType) t_num_blocks;
    t_standard_dev_CB = std::sqrt(t_var_nnz_CB);

    // Count Tile features
    for (size_t i = 0; i < t_num_blocks*t_num_blocks; i++)
    {
        if (nnz_by_Tiles_[i])
            ++t_nz_ratio_tiles_ ;

        diff_tiles = nnz_by_Tiles_[i] - t_ave_nnz_all_tiles;
        t_var_nnz_all_tiles += diff_tiles * diff_tiles;

        t_min_nnz_all_tiles_ = std::min(t_min_nnz_all_tiles_, nnz_by_Tiles_[i]);
        t_max_nnz_all_tiles_ = std::max(t_max_nnz_all_tiles_, nnz_by_Tiles_[i]);
    }
    t_nz_ratio_tiles_   /= ((ValueType) t_num_blocks*t_num_blocks);
    t_var_nnz_all_tiles /= ((ValueType) t_num_blocks*t_num_blocks);
    t_standard_dev_all_tiles = std::sqrt(t_var_nnz_all_tiles);
    // calculate Tiles' P-ratio and Gini index
    P_ratioAndGini(nnz_by_Tiles_, num_nnzs, t_P_ratio_all_tiles_, t_Gini_all_tiles_);

    // calculate RB's P-ratio and Gini index
    P_ratioAndGini(nnz_by_RB_, num_nnzs, t_P_ratio_RB_, t_Gini_RB_);

    // calculate CB's P-ratio and Gini index
    P_ratioAndGini(nnz_by_CB_, num_nnzs, t_P_ratio_CB_, t_Gini_CB_);

    return true;
}
template bool MTX<int, float>::CalculateTilesFeatures();
template bool MTX<int, double>::CalculateTilesFeatures();
template bool MTX<long long, float>::CalculateTilesFeatures();
template bool MTX<long long, double>::CalculateTilesFeatures();

// Finding the best Dim that make block_num closed to 2048
template <typename IndexType>
IndexType BestDimForBSR( IndexType nums, IndexType target){
    IndexType bestDim = 0;
    IndexType bestDiff = 10000;
    for (IndexType Dim = 1; Dim < nums; Dim++)
    {
        IndexType calculated_mb = (nums + Dim - 1) / Dim;
        IndexType diff = std::abs(calculated_mb - target);
        
        if (diff < bestDiff)
        {
            bestDiff = diff;
            bestDim = Dim;
        }

        if (diff == 0)
            break;
    }
    return bestDim;
}
template int BestDimForBSR( int nums, int target);
template long long BestDimForBSR( long long nums, long long target);

template <typename IndexType, typename ValueType>
void AnalyzeTile_Group(const ValueType* values, const IndexType blockDimRow, const IndexType blockDimCol, const IndexType GroupNum, IndexType& GrX_uniqRB, IndexType& GrX_uniqCB, bool flag_Row, bool flag_Col)
{
    if (flag_Row){
        for (size_t bc = 0; bc < blockDimCol; ++bc) {
            for (size_t br = 0; br < blockDimRow; br++) {
                // size_t idx = br * blockDimCol + bc; // 行优先存储的 索引
                if( values[br * blockDimCol + bc] ){ // 当前元素非零，往后看 GroupNum 行的元素
                    if(br + GroupNum <= blockDimRow) // 还在tile row范围内，分析
                    {
                        size_t step;
                        for (step = 1; step < GroupNum; step++)
                        {
                            if( values[(br+step) * blockDimCol + bc] == 0)
                                break;
                        }
                        if(step == GroupNum){   // 连续 GroupNum 行非零
                            ++GrX_uniqRB;
                        }
                        br += step; // 遍历过的这一段行就不用再check了
                    }
                    else    // 往后的一个group超过tile的大小了，跳过这个行分析
                    {   break; }
                }
            }
        }
    }

    if (flag_Col){
        for (size_t br = 0; br < blockDimRow; br++){
            for (size_t bc = 0; bc < blockDimCol; ++bc) {
                if( values[br * blockDimCol + bc] ){ // 当前元素非零，往后看 GroupNum 列的元素
                    if(bc + GroupNum <= blockDimCol) // 还在tile col范围内，分析
                    {
                        size_t step;
                        for (step = 1; step < GroupNum; step++)
                        {
                            if( values[br * blockDimCol + bc + step] == 0)
                                break;
                        }
                        if(step == GroupNum){   // 连续 GroupNum 列非零
                            ++GrX_uniqCB;
                        }
                        bc += step; // 遍历过的这一段列就不用再check了
                    }
                    else    // 往后的一个group超过tile的大小了，跳过这个列分析
                    {   break; }
                }
            }
        }
    }
}
template void AnalyzeTile_Group<int, float>(const float*, const int, const int, const int, int&, int&, bool, bool);
template void AnalyzeTile_Group<int, double>(const double*, const int, const int, const int, int&, int&, bool, bool);
template void AnalyzeTile_Group<long long, float>(const float*, const long long, const long long, const long long, long long&, long long&, bool, bool);
template void AnalyzeTile_Group<long long, double>(const double*, const long long, const long long, const long long, long long&, long long&, bool, bool);


template <typename IndexType, typename ValueType>
bool MTX<IndexType, ValueType>::CalculateTilesExtraFeatures(const char* mat_path)
{
    // 使用 BSR 做分块的信息统计， 但并不严格的是 2048 * 2048 块了
    //  应该用 BSR来统计更好， 04. March. 2024
    BSR_Matrix<IndexType, ValueType> bsr;
    IndexType tileDim_row = BestDimForBSR(num_rows, t_num_blocks);
    IndexType tileDim_col = BestDimForBSR(num_cols, t_num_blocks);

    bsr = read_bsr_matrix<IndexType, ValueType>(mat_path, tileDim_row, tileDim_col);

    // tile的size 必须要比 Group 大，统计Grx_uniq才有意义
    bool flag_GrX_uniqRB = (bsr.blockDim_r >= GrX);
    bool flag_GrX_uniqCB = (bsr.blockDim_c >= GrX);
    IndexType GrxRB_perT, GrxCB_perT;

    uniq_RB.resize(bsr.mb * bsr.nb, 0);  // number of total blocks in mat
    uniq_CB.resize(bsr.mb * bsr.nb, 0);
    
    if( flag_GrX_uniqRB ) {
        GrxRB_perT = (bsr.blockDim_r + GrX - 1)/GrX;
        GrX_uniqRB.resize(bsr.mb * bsr.nb, 0);
        GrX_potReuseRB.resize(bsr.mb * GrxRB_perT, 0);
    }
    if( flag_GrX_uniqCB ) {
        GrxCB_perT = (bsr.blockDim_c + GrX - 1)/GrX;
        GrX_uniqCB.resize(bsr.mb * bsr.nb, 0);
        GrX_potReuseCB.resize(bsr.nb * GrxCB_perT, 0);
    }


    IndexType threadNum = Le_get_thread_num();

    #pragma omp parallel for num_threads(threadNum)
    for (size_t i = 0; i < bsr.mb; i++)
    {
        // 遍历 第i行的行块
        size_t start = bsr.row_ptr[i];
        size_t end   = bsr.row_ptr[i+1];

        for (size_t j = start; j < end; j++)
        {
            // 获取当前块的列索引
            size_t block_col = bsr.block_colindex[j];
            // 存储在 uniq 中的 tile ID 位置
            size_t tileID = i * bsr.nb  + block_col;
            // 存储在 GrX_potReuseRB 和 GrX_potReuseCB 中的 起始 行/列 位置
            // size_t GrX_potRowID = i * GrxRB_perT;
            // size_t GrX_potColID = block_col * GrxCB_perT;

            // 每一个tile 内的 标记矩阵，用于统计uniqR行 or uniqC列 是否已经统计过
            std::vector<bool> flag_R(bsr.blockDim_r, true);
            std::vector<bool> flag_C(bsr.blockDim_c, true);

            std::vector<bool> flag_potGrXR(GrxRB_perT, true);
            std::vector<bool> flag_potGrXC(GrxCB_perT, true);

            // 遍历块的内部
            for (size_t br = 0; br < bsr.blockDim_r; ++br) {
                for (size_t bc = 0; bc < bsr.blockDim_c; ++bc) {
                    // 存在 bsr.values 中的 index
                    size_t value_id = j * bsr.blockDim_r * bsr.blockDim_c + br * bsr.blockDim_c + bc;
                    
                    // 统计 uniqRB 和 uniqCB, 统计后标记数组更新false
                    if( bsr.block_data[value_id]){  // nnz
                        // uniq_RB 判别
                        if (flag_R[br])  // true: count the none zero row
                        {
                           uniq_RB[tileID]++;
                           flag_R[br] = false;  // convert to false
                        }
                        // uniq_CB 判别
                        if (flag_C[bc])  // true: count the none zero col
                        {
                           uniq_CB[tileID]++;
                           flag_C[bc] = false;  // convert to false
                        }
                        if( flag_GrX_uniqRB && flag_potGrXR[br/GrX]){   // true: count the none zero tile in GrX_row
                            #pragma omp atomic
                            GrX_potReuseRB[i *  GrxRB_perT + br/GrX]++;
                            flag_potGrXR[br/GrX] = false;
                        }
                        if( flag_GrX_uniqCB && flag_potGrXC[bc/GrX]){   // true: count the none zero tile in GrX_col
                            #pragma omp atomic
                            GrX_potReuseCB[block_col * GrxCB_perT + bc/GrX]++;
                            flag_potGrXC[bc/GrX] = false;
                        }
                    }
                }
            }
            // Count the GrX_uniqRB and GrX_uniqCB
            AnalyzeTile_Group(&bsr.block_data[j * bsr.blockDim_r * bsr.blockDim_c], bsr.blockDim_r, bsr.blockDim_c, GrX, GrX_uniqRB[tileID], GrX_uniqCB[tileID], flag_GrX_uniqRB, flag_GrX_uniqCB);
        }
    }
    
    uniqR = (ValueType) std::accumulate(uniq_RB.begin(), uniq_RB.end(), 0)/num_nnzs;
    uniqC = (ValueType) std::accumulate(uniq_CB.begin(), uniq_CB.end(), 0)/num_nnzs;
    potReuseR = (ValueType) std::accumulate(uniq_RB.begin(), uniq_RB.end(), 0)/num_rows;
    potReuseC = (ValueType) std::accumulate(uniq_CB.begin(), uniq_CB.end(), 0)/num_cols;
    
    if( flag_GrX_uniqRB )
    {
        GrX_uniqR = (ValueType) std::accumulate(GrX_uniqRB.begin(), GrX_uniqRB.end(), 0) / num_nnzs;
        GrX_potReuseR = (ValueType) std::accumulate(GrX_potReuseRB.begin(), GrX_potReuseRB.end(), 0) / GrX_potReuseRB.size();
    }
    if( flag_GrX_uniqCB )
    {
        GrX_uniqC = (ValueType) std::accumulate(GrX_uniqCB.begin(), GrX_uniqCB.end(), 0) / num_nnzs;
        GrX_potReuseC = (ValueType) std::accumulate(GrX_potReuseCB.begin(), GrX_potReuseCB.end(), 0) / GrX_potReuseCB.size();
    }
    delete_bsr_matrix(bsr);
    return true;
}
template bool MTX<int, float>::CalculateTilesExtraFeatures(const char* mat_path);
template bool MTX<int, double>::CalculateTilesExtraFeatures(const char* mat_path);
template bool MTX<long long, float>::CalculateTilesExtraFeatures(const char* mat_path);
template bool MTX<long long, double>::CalculateTilesExtraFeatures(const char* mat_path);

template <typename IndexType, typename ValueType>
bool MTX<IndexType, ValueType>::PrintImage(std::string& outputpath){
    std::string image_s = "image=";
    setlocale(LC_ALL, "en_US.utf8"); // 设置UTF-8编码
    for (int i = 0; i < image_.size(); ++i) {
        for (int j = 0; j < image_[i].size(); ++j) {
        image_s+=my_to_String(image_[i][j]);
        image_s+=" ";
        // out_stream << image[i][j] << "\n";
        }
    }
    std::cout<<image_s<<std::endl;
    return true;
}

template bool MTX<int, float>::PrintImage(std::string& outputpath);
template bool MTX<int, double>::PrintImage(std::string& outputpath);
template bool MTX<long long, float>::PrintImage(std::string& outputpath);
template bool MTX<long long, double>::PrintImage(std::string& outputpath);

template <typename IndexType, typename ValueType>
bool MTX<IndexType, ValueType>::FeaturesWrite(const char* file_path)
{
    // std::ofstream f_write(file_path);

    // f_write << matrixID_ << matrixName ;
    
    FILE *save_features = fopen(file_path, "a");
    if ( save_features == nullptr)
    {
        std::cout << "Unable to open features saved file: "<< file_path << std::endl;
        return false;
    }

    if constexpr(std::is_same<IndexType, int>::value) {
        fprintf(save_features, "%d %s ", matrixID_, matrixName.c_str());
        fprintf(save_features, "%d %d ", num_rows, num_cols);
        fprintf(save_features, "%d %lf %lf ", num_nnzs, nnz_ratio_, diag_close_ratio_);

        fprintf(save_features, "%d %lf %lf ", is_symmetric_, pattern_symm_, value_symm_);
        
        fprintf(save_features, "%d %d %d ", nnz_lower_, nnz_upper_, nnz_diagonal_);
        
        // row statistic features
        fprintf(save_features, "%.3f %d %d %lf %lf %lf %lf %lf ", nz_row_ratio_, min_nnz_each_row_, max_nnz_each_row_, ave_nnz_each_row_, var_nnz_each_row_, standard_dev_row_, P_ratio_row_, Gini_row_);

        // col statistic features
        fprintf(save_features, "%.3f %d %d %lf %lf %lf %lf %lf ", nz_col_ratio_, min_nnz_each_col_, max_nnz_each_col_, ave_nnz_each_col_, var_nnz_each_col_, standard_dev_col_, P_ratio_col_, Gini_col_);

        // values features
        // fprintf(save_features, "%lg %lg ", max_value_offdiag_, max_value_diagonal_);
        // fprintf(save_features, "%lf ", diagonal_dominant_ratio_);

        // fprintf(save_features, "%lf %lf \n", row_variability_, col_variability_);

        //  Tile features
        fprintf(save_features, "%.3f %d %d %lf %lf %lf %lf %lf ", t_nz_ratio_tiles_, t_min_nnz_all_tiles_, t_max_nnz_all_tiles_, t_ave_nnz_all_tiles, t_var_nnz_all_tiles, t_standard_dev_all_tiles, t_P_ratio_all_tiles_, t_Gini_all_tiles_);

        fprintf(save_features, "%.3f %d %d %lf %lf %lf %lf %lf ", t_nz_ratio_RB_, t_min_nnz_each_RB_, t_max_nnz_each_RB_, t_ave_nnz_RB, t_var_nnz_RB, t_standard_dev_RB, t_P_ratio_RB_, t_Gini_RB_);

        fprintf(save_features, "%.3f %d %d %lf %lf %lf %lf %lf ", t_nz_ratio_CB_, t_min_nnz_each_CB_, t_max_nnz_each_CB_, t_ave_nnz_CB, t_var_nnz_CB, t_standard_dev_CB, t_P_ratio_CB_, t_Gini_CB_);

        // Tile extra fetures
        fprintf(save_features, "%lf %lf %lf %lf %lf %lf %lf %lf \n", uniqR, uniqC, GrX_uniqR, GrX_uniqC, potReuseR, potReuseC, GrX_potReuseR, GrX_potReuseC);
    } 
    else if constexpr(std::is_same<IndexType, long long>::value) {
        fprintf(save_features, "%lld %s ", matrixID_, matrixName.c_str());
        fprintf(save_features, "%lld %lld ", num_rows, num_cols);
        fprintf(save_features, "%lld %lf ", num_nnzs, nnz_ratio_);

        fprintf(save_features, "%d %lf %lf ", is_symmetric_, pattern_symm_, value_symm_);
        
        fprintf(save_features, "%lld %lld %lld ", nnz_lower_, nnz_upper_, nnz_diagonal_);
        
        // row statistic features
        fprintf(save_features, "%.3f %lld %lld %lf %lf %lf %lf %lf ", nz_row_ratio_, min_nnz_each_row_, max_nnz_each_row_, ave_nnz_each_row_, var_nnz_each_row_, standard_dev_row_, P_ratio_row_, Gini_row_);

        // col statistic features
        fprintf(save_features, "%.3f %lld %lld %lf %lf %lf %lf %lf ", nz_col_ratio_, min_nnz_each_col_, max_nnz_each_col_, ave_nnz_each_col_, var_nnz_each_col_, standard_dev_col_, P_ratio_col_, Gini_col_);

        // values features
        // fprintf(save_features, "%lg %lg ", max_value_offdiag_, max_value_diagonal_);
        // fprintf(save_features, "%lf ", diagonal_dominant_ratio_);
        // fprintf(save_features, "%lf %lf \n", row_variability_, col_variability_);

        //  Tile features
        fprintf(save_features, "%.3f %lld %lld %lf %lf %lf %lf %lf ", t_nz_ratio_tiles_, t_min_nnz_all_tiles_, t_max_nnz_all_tiles_, t_ave_nnz_all_tiles, t_var_nnz_all_tiles, t_standard_dev_all_tiles, t_P_ratio_all_tiles_, t_Gini_all_tiles_);

        fprintf(save_features, "%.3f %lld %lld %lf %lf %lf %lf %lf ", t_nz_ratio_RB_, t_min_nnz_each_RB_, t_max_nnz_each_RB_, t_ave_nnz_RB, t_var_nnz_RB, t_standard_dev_RB, t_P_ratio_RB_, t_Gini_RB_);

        fprintf(save_features, "%.3f %lld %lld %lf %lf %lf %lf %lf ", t_nz_ratio_CB_, t_min_nnz_each_CB_, t_max_nnz_each_CB_, t_ave_nnz_CB, t_var_nnz_CB, t_standard_dev_CB, t_P_ratio_CB_, t_Gini_CB_);

        // Tile extra fetures
        fprintf(save_features, "%lf %lf %lf %lf %lf %lf %lf %lf \n", uniqR, uniqC, GrX_uniqR, GrX_uniqC, potReuseR, potReuseC, GrX_potReuseR, GrX_potReuseC);

    }
        fclose(save_features);
    return true;
}

template bool MTX<int, float>::FeaturesWrite(const char* file_path);
template bool MTX<int, double>::FeaturesWrite(const char* file_path);
template bool MTX<long long, float>::FeaturesWrite(const char* file_path);
template bool MTX<long long, double>::FeaturesWrite(const char* file_path);