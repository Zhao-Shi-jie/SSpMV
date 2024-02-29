#ifndef SPARSE_FEATURES_H
#define SPARSE_FEATURES_H
#include<iostream>
#include<vector>
#include<string>
#include <map>
#include <unordered_map>

#include"general_config.h"
#include"sparse_format.h"
#include"timer.h"

template <typename IndexType, typename ValueType>
class MTX{
    public:
        MTX(IndexType matID = 0) {
            matrixID_ = matID;
            std::cout << "== MTX Features Extraction ==\n" << std::endl;
        }
        bool MtxLoad(const char* file_path);
        bool FeaturesWrite(const char* file_path);
        bool ConvertToCSR(CSR_Matrix<IndexType, ValueType> &csr);
        void Stringsplit(const std::string& s, const char split, std::vector<std::string>& res);
        bool CalculateFeatures();
        bool CalculateTilesFeatures();
        bool PrintImage(std::string& outputpath);
        double MtxLoad_time_= 0.0;
        double CalculateFeatures_time_= 0.0;
        double ConvertToCSR_time_= 0.0;

        void FeaturesPrint()
        {
            std::cout << "[" << matrixID_ << "]  " << matrixName << std::endl;
            std::cout << "is_symmetric " << is_symmetric_ << std::endl;
            std::cout << "pattern_symm " << pattern_symm_*100 << "%"<< std::endl;
            std::cout << "value_symm   " << value_symm_*100 << "%"<< std::endl<< std::endl;

            std::cout<< "num_rows  = "<< num_rows << std::endl;
            std::cout<< "num_cols  = "<< num_cols << std::endl;
            std::cout<< "nnz_mtx   = " << nnz_mtx_ << std::endl;
            std::cout<< "real_nnz  = "<< num_nnzs << std::endl;
            std::cout<< "nnz_ratio = " << nnz_ratio_*100 << "%"<< std::endl<< std::endl;

            std::cout<< "nnz_lower    : " << nnz_lower_ << std::endl;
            std::cout<< "nnz_upper    : " << nnz_upper_ << std::endl;
            std::cout<< "nnz_diagonal : " << nnz_diagonal_ << std::endl<< std::endl;
            
            // row statistic features
            std::cout<< "NoneZero_Row_Ratio      : " << nz_row_ratio_*100 << "%"<< std::endl;
            std::cout<< "min_nnz_each_row        : " << min_nnz_each_row_ << std::endl;
            std::cout<< "max_nnz_each_row        : " << max_nnz_each_row_ << std::endl;
            std::cout<< "ave_nnz_each_row        : " << ave_nnz_each_row_ << std::endl;
            std::cout<< "var_nnz_each_row        : " << var_nnz_each_row_ << std::endl;
            std::cout<< "standard_dev_row        : " << standard_dev_row_ << std::endl;
            std::cout<< "P-ratio_row             : " << P_ratio_row_ << std::endl;
            std::cout<< "Gini_coeff_row          : " << Gini_row_ << std::endl<< std::endl;

            // col statistic features
            std::cout<< "NoneZero_Col_Ratio      : " << nz_col_ratio_*100 << "%"<< std::endl;
            std::cout<< "min_nnz_each_col        : " << min_nnz_each_col_ << std::endl;
            std::cout<< "max_nnz_each_col        : " << max_nnz_each_col_ << std::endl;
            std::cout<< "ave_nnz_each_col        : " << ave_nnz_each_col_ << std::endl;
            std::cout<< "var_nnz_each_col        : " << var_nnz_each_col_ << std::endl;
            std::cout<< "standard_dev_col        : " << standard_dev_col_ << std::endl;
            std::cout<< "P-ratio_col             : " << P_ratio_col_ << std::endl;
            std::cout<< "Gini_coeff_col          : " << Gini_col_ << std::endl<< std::endl;

            std::cout<< "diagonal_dominant_ratio : " << diagonal_dominant_ratio_*100 << "%"<< std::endl<< std::endl;

            std::cout<< "max_value_offdiag  = " << max_value_offdiag_ << std::endl;
            std::cout<< "max_value_diagonal = " << max_value_diagonal_ << std::endl;
            std::cout<< "row_variability    = " << row_variability_ << std::endl;
            std::cout<< "col_variability    = " << col_variability_ << std::endl;
            
        }

    private:
        bool is_symmetric_ = false;
        std::string matrixName;
        IndexType matrixID_ = 0;

        IndexType num_rows = 0;
        IndexType num_cols = 0;
        IndexType num_nnzs = 0;         // 真实的 nnz 数目
        IndexType nnz_mtx_ = 0;         // mtx 文件里显示的 nnz数目
        IndexType nnz_lower_ = 0;       // 下三角 非零元的数目
        IndexType nnz_upper_ = 0;       // 上三角 非零元的数目
        IndexType nnz_diagonal_ = 0;    // 对角线上的 nnz 数目
        IndexType min_nnz_each_row_ = 100000000;    // 各行中最少的 nnz 数目
        IndexType max_nnz_each_row_ = 0;    // 各行中最大的 nnz 数目
        IndexType min_nnz_each_col_ = 100000000;    // 各列中最大的 nnz 数目
        IndexType max_nnz_each_col_ = 0;    // 各列中最大的 nnz 数目

    // Structure features
        ValueType pattern_symm_ = 0.0;      // 模式对称比例
        ValueType value_symm_   = 0.0;      // 数值对称比例
        ValueType nnz_ratio_ = 0.0;         // 稠密度， = 1 - 稀疏度

        // nnzs skew statistic features
        ValueType nz_row_ratio_ = 0.0;
        ValueType ave_nnz_each_row_ = 0.0;  // 每行平均的 nnz 数目
        ValueType var_nnz_each_row_ = 0.0;  // 每行 nnz 的 方差
        ValueType standard_dev_row_ = 0.0;  // 每行 nnz 的 标准差
        ValueType Gini_row_ = 0.0;          // [0, 1] -> [balanced ~ imbalanced]
        ValueType P_ratio_row_ = 0.0;       // p fraction of rows have (1-p) fraction of nnzs in the matrix   [0, 0.5] -> [imbalanced ~ balanced]

        ValueType nz_col_ratio_ = 0.0;
        ValueType ave_nnz_each_col_ = 0.0;  // 每列平均的 nnz 数目
        ValueType var_nnz_each_col_ = 0.0;  // 每列 nnz 的 方差
        ValueType standard_dev_col_ = 0.0;  // 每列 nnz 的 标准差
        ValueType Gini_col_ = 0.0;
        ValueType P_ratio_col_ = 0.0;       // p fraction of cols have (1-p) fraction of nnzs in the matrix [0, 0.5] -> [imbalanced ~ balanced]

        ValueType diagonal_dominant_ratio_ = 0.0;  // 对角占优比率

    // Values features
        ValueType max_value_offdiag_  = -std::numeric_limits<ValueType>::max();
        ValueType max_value_diagonal_ = -std::numeric_limits<ValueType>::max();

        // 对于digital matrix， 这两个值不会变动，结果为 -1
        ValueType row_variability_ = -1.0;
        ValueType col_variability_ = -1.0;

    // Intermediate variables
        std::vector<IndexType> nnz_by_row_;     // 保存每行的 nnz 数目
        std::vector<IndexType> nnz_by_col_;     // 保存每列的 nnz 数目

        // 每行中 最大值、 最小值的 log10（value）
        std::vector<ValueType> max_each_row_;
        std::vector<ValueType> min_each_row_;
        // 每列中 最大值、 最小值的 log10（value）
        std::vector<ValueType> max_each_col_; 
        std::vector<ValueType> min_each_col_;
        std::vector<ValueType> Diag_Dom; // 计算每行的对角占优值， diag - other_row_sum

        std::vector<std::string> symm_pair_;
        std::unordered_map<std::string,std::string> m_;
        std::vector<std::vector<int> > image_;

    /*

    RB means consider row block including whole cols:
    (Likewise, CB means consider col block including whole rows)
    Here,  t_num_rows = 2.
        ________________________
          x  x  x  x  x  x  x  x |
          x  x  x  x  x  x  x  x |  ---> RB
        ------------------------   
          x  x  x  x  x  x  x  x
          x  x  x  x  x  x  x  x
        ------------------------
          x  x  x  x  x  x  x  x
          x  x  x  x  x  x  x  x
        ________________________
    */
    // Tiles features 默认 2048*2048 tiles
        IndexType t_num_blocks = MAT_TILE_SIZE;
        IndexType t_num_RB = -1;        // tiles 内的行数目
        IndexType t_num_CB = -1;        // tiles 内的列数目
        IndexType t_num_lastRB = -1;    // 最后一个块的行数目
        IndexType t_num_lastCB = -1;    // 最后一个块的列数目

        // ave_nnz
        ValueType t_ave_nnz_all_tiles = 0.0;
        ValueType t_ave_nnz_RB = 0.0;   // row block
        ValueType t_ave_nnz_CB = 0.0;   // col block

        // var_nnz
        ValueType t_var_nnz_all_tiles = 0.0;
        ValueType t_var_nnz_RB = 0.0;
        ValueType t_var_nnz_CB = 0.0;

        // standard diviation
        ValueType t_standard_dev_all_tiles = 0.0;
        ValueType t_standard_dev_RB = 0.0;
        ValueType t_standard_dev_CB = 0.0;
    
    // Intermediate variables
        std::vector<IndexType> nnz_by_RB_;     // 保存每个行块的 nnz 数目
        std::vector<IndexType> nnz_by_RB_;     // 保存每个列块的 nnz 数目

        // min and max nnz for each 
        IndexType t_min_nnz_all_tiles_ = 100000000;
        IndexType t_max_nnz_all_tiles_ = 0.0;
        IndexType t_min_nnz_each_RB_   = 100000000;
        IndexType t_max_nnz_each_RB_   = 0;
        IndexType t_min_nnz_each_CB_   = 100000000;
        IndexType t_max_nnz_each_CB_   = 0;

        // Gini index  [0, 1] -> [balanced ~ imbalanced]
        ValueType t_Gini_all_tiles_ = 0.0;
        ValueType t_Gini_RB_ = 0.0;          
        ValueType t_Gini_CB_ = 0.0;

        // p-ratio [0, 0.5] -> [imbalanced ~ balanced]
        ValueType t_P_ratio_all_tiles_ = 0.0;
        ValueType t_P_ratio_RB_ = 0.0;
        ValueType t_P_ratio_CB_ = 0.0;

        // none zero ratio
        ValueType t_nz_ratio_tiles_ = 0.0;
        ValueType t_nz_ratio_RB_ = 0.0;
        ValueType t_nz_ratio_CB_ = 0.0;

    // aditional information
        // uniq
        std::vector<IndexType> uniq_RB;  // 记录每个tiles的非零行数
        std::vector<IndexType> uniq_CB;  // 记录每个tiles的非零列数
        ValueType uniqR = 0.0;           // sum devide nnz
        ValueType uniqC = 0.0;           // sum devide nnz

        // GrX_uniq  ; for cacheline evaluate
        IndexType GrX = CACHE_LINE / sizeof(ValueType);
        std::vector<IndexType> GrX_uniqRB;
        std::vector<IndexType> GrX_uniqCB;
        ValueType GrX_uniqR = 0.0;       // sum devide nnz
        ValueType GrX_uniqC = 0.0;       // sum devide nnz

        // porReuse ; for data reuse in the LLC
        // 记录这一 行/列 中 有几个 tiles中的 行/列 非零
        std::vector<IndexType> porReuseRB;  // size: num_rows
        std::vector<IndexType> porReuseCB;  // size: num_cols
        ValueType potReuseR = 0.0;      // sum devide num of rows
        ValueType potReuseC = 0.0;      // sum devide num of cols

        // GrX_porReuse ; for data reuse in the LLC with more coarse granularity
        std::vector<IndexType> GrX_porReuseRB;  // size: num_GrXrows
        std::vector<IndexType> GrX_porReuseCB;  // size: num_GrXcols
        ValueType GrX_potReuseR = 0.0;      // sum devide num of GrXrows
        ValueType GrX_potReuseC = 0.0;      // sum devide num of GrXcols

};

#endif /* SPARSE_FEATURES_H */
