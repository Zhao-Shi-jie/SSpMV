import subprocess
import os
import pandas as pd

# def Read_GenDataset(excel_path):
#     # 读取Excel文件
#     df = pd.read_excel(excel_path)

#     # 选择'Id'和'Name'列
#     mtx_list = df[['Id', 'Matrix']].values.tolist()

#     # 生成（ID, Name）元组列表和对应的文件路径
#     mtx_info = [(mtx_id, path) for mtx_id, path in mtx_list]

#     return mtx_info

def Run_with_prediction(excel_path):
    # Read the dataset from the Excel file
    data = pd.read_excel(excel_path)
    
    # Define the format mapping if necessary
    format_mapping = {
        'BSR': 'bsr',
        'CSR': 'csr',
        'COO': 'coo',
        'DIA': 'dia',
        'ELL': 'ell',
        'S-ELL': 'sell',
        'S-ELL-sigma': 'sell_c_sigma',
        'S-ELL-R': 'sell_c_R',
        'CSR5': 'csr5'
        # Add more mappings as needed
    }
    
    index_len = 0
    precision = 64
    
    # Iterate over each row in the DataFrame
    for index, row in data.iterrows():
        matID = row['Id']
        mtx_path = row['Matrix']
        raw_format = row['Format']
        sche_mode = row['sche_mode']
        
        # Map the format using the dictionary
        command_format = format_mapping.get(raw_format, raw_format.lower())
        
        # Construct the command string
        command = f"./benchmark_spmv_{command_format} {mtx_path} --matID={matID} --Index={index_len} --precision={precision} --sche={sche_mode}"
        
        print(f"Executing: {command}")
        
        # Execute the command
        try:
            result = subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            print(f"Command finished with return code {result.returncode}")
            if result.stdout:
                print(f"Output: {result.stdout.decode('utf-8')}")
            if result.stderr:
                print(f"Errors: {result.stderr.decode('utf-8')}")
        except subprocess.CalledProcessError as e:
            print(f"An error occurred while executing {command}")
            print(e)
        

if __name__ == "__main__":
    Run_with_prediction("./All_testmtx.xlsx")
    