import subprocess
import os
import pandas as pd

def read_full_dataset(excel_path):
    df = pd.read_excel(excel_path)
    full_dataset = set(df[['Id', 'Name']].itertuples(index=False, name=None))
    return full_dataset

if __name__ == "__main__":
    All_Suite_data = read_full_dataset("./SuiteSparse_Matrix.xlsx")
    
    # 读取已测试数据集的Excel文件
    tested_df = pd.read_excel('CSR5 Res.xlsx')
    
    # 去除重复项，只保留唯一的ID和Name组合
    unique_tested_dataset = tested_df[['Id', 'Name']].drop_duplicates()
    
    # 将去除重复数据后的已测试数据集转换为集合
    unique_tested_set = set(unique_tested_dataset.itertuples(index=False, name=None))
    
    # 获取未测试的数据集
    untested_datasets = All_Suite_data - unique_tested_set
    
    # 打印结果
    print("Untested datasets:")
    for untested in untested_datasets:
        print(untested)