import os
import pandas as pd

lable_file = "Best_Result.xlsx"
root_dir = "/data/lsl/MModel-Data"
label_suffix = ".format_label"

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
        print("Writing", row['Name'] ,"successfully")

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
    write_label_to_file(lable_file)