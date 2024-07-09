import os
import pandas as pd
# 9 labels of different formats
label_file = "Suite_Best_Result.xlsx"
Gen_label_file = "Gen_Best_Result.xlsx"
label_suffix = ".format_label"

Prob_file = "Suite_Probability_Vectors.xlsx"
Gen_Prob_file = "Gen_Probability_Vectors.xlsx"
Prob_suffix  = ".prob_label"

root_dir = "/data/lsl/MModel-Data"

def write_prob_to_file(data_list):
    df = pd.read_excel(data_list)
    
    # 遍历每一行数据
    for index, row in df.iterrows():
        # 文件夹路径
        name_str = str(row['Name'])
        # format_str = str(row['Best'])
        
        dir_path = os.path.join(root_dir, name_str)
        
        # 如果文件夹不存在，创建新文件夹
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        
        # 文件路径
        file_path = os.path.join(dir_path, f"{name_str}{Prob_suffix}")
        
        # 写入文件
        with open(file_path, 'w') as file:
            # 从'Best'字段开始，接着是'CSR'到'BSR'的值
            data_to_write = [row['Best']] + [row[col] for col in ['COO', 'CSR', 'DIA', 'ELL', 'S-ELL', 'S-ELL-sigma', 'S-ELL-R', 'CSR5', 'BSR']]
            file.write(' '.join(map(str, data_to_write)) + '\n')
        
        print("Writing prob", row['Name'] ,"successfully")

def write_label_to_file(data_list):
    df = pd.read_excel(data_list)
    
    # 遍历每一行数据
    for index, row in df.iterrows():
        # 文件夹路径
        name_str = str(row['Name'])
        format_str = str(row['Format'])
        dir_path = os.path.join(root_dir, name_str)
        
        # 如果文件夹不存在，创建新文件夹
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        
        # 文件路径
        file_path = os.path.join(dir_path, f"{name_str}{label_suffix}")
        
        # 写入文件
        with open(file_path, 'w') as file:
            value = get_value_based_on_format(format_str)  # 需要定义这个函数来根据Format获取value
            file.write(f"{format_str} {value}\n")
        print("Writing label", row['Name'] ,"successfully")

def get_value_based_on_format(format):
    format_to_value ={
        'COO':         '0',
        'CSR':         '1',
        'DIA':         '2',
        'ELL':         '3',
        'S-ELL':       '4',
        'S-ELL-sigma': '5',
        'S-ELL-R':     '6',
        'CSR5':        '7',
        'BSR':         '8',
    }
    return format_to_value.get(format, 'UnknownFormat')

if __name__ == "__main__":
    # write_label_to_file(label_file)
    # write_prob_to_file(Prob_file)
    write_label_to_file(Gen_label_file)
    write_prob_to_file(Gen_Prob_file)