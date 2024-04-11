import subprocess
import os
import pandas as pd

def Read_TestDataset(excel_path):
    # 读取Excel文件
    df = pd.read_excel(excel_path)

    # 选择'Id'和'Name'列
    mtx_list = df[['Id', 'Name']].values.tolist()

    # 生成（ID, Name）元组列表和对应的文件路径
    mtx_info = [(mtx_id, name, f"/data/suitesparse_collection/{name}/{name}.mtx") for mtx_id, name in mtx_list]

    return mtx_info
    

def Run_CSR_intKernel(excel_path):
    
    # 读dataset list 并生成数据组：
    mtx_info = Read_TestDataset(excel_path)
    
    # 定义测试命令中不变的参数
    index = 0
    precision = 64
    sche = range(0,4)
    
    for matID, name, mtx_path in mtx_info:
        for sche_mod in sche:
            # 构建并运行命令
            command = f"./benchmark_spmv_csr {mtx_path} --matID={matID} --Index={index} --precision={precision} --sche={sche_mod}"
            print(f"Executing: {command}")
            
            # 执行命令
            try:
                result = subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                # 打印命令执行结果
                print(f"Command finished with return code {result.returncode}")
                if result.stdout:
                    print(f"Output: {result.stdout.decode('utf-8')}")
                if result.stderr:
                    print(f"Errors: {result.stderr.decode('utf-8')}")
            except subprocess.CalledProcessError as e:
                print(f"An error occurred while executing {command}")
                print(e)

def Run_BSR_intKernel(excel_path):
    
    # 读dataset list 并生成数据组：
    mtx_info = Read_TestDataset(excel_path)
    
    # 定义测试命令中不变的参数
    index = 0
    precision = 64
    sche = range(0,4)
    
    for matID, name, mtx_path in mtx_info:
        for sche_mod in sche:
            # 构建并运行命令
            command = f"./benchmark_spmv_bsr {mtx_path} --matID={matID} --Index={index} --precision={precision} --sche={sche_mod}"
            print(f"Executing: {command}")
            
            # 执行命令
            try:
                result = subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                # 打印命令执行结果
                print(f"Command finished with return code {result.returncode}")
                if result.stdout:
                    print(f"Output: {result.stdout.decode('utf-8')}")
                if result.stderr:
                    print(f"Errors: {result.stderr.decode('utf-8')}")
            except subprocess.CalledProcessError as e:
                print(f"An error occurred while executing {command}")
                print(e)

def Run_CSR5_intKernel(excel_path):
    
    # 读dataset list 并生成数据组：
    mtx_info = Read_TestDataset(excel_path)
    
    # 定义测试命令中不变的参数
    index = 0
    precision = 64
    
    for matID, name, mtx_path in mtx_info:
        # 构建并运行命令
        command = f"./benchmark_spmv_csr5 {mtx_path} --matID={matID} --Index={index} --precision={precision}"
        print(f"Executing: {command}")
        
        # 执行命令
        try:
            result = subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            # 打印命令执行结果
            print(f"Command finished with return code {result.returncode}")
            if result.stdout:
                print(f"Output: {result.stdout.decode('utf-8')}")
            if result.stderr:
                print(f"Errors: {result.stderr.decode('utf-8')}")
        except subprocess.CalledProcessError as e:
            print(f"An error occurred while executing {command}")
            print(e)

def Run_DIA_intKernel(excel_path):
    
    # 读dataset list 并生成数据组：
    mtx_info = Read_TestDataset(excel_path)
    
    # 定义测试命令中不变的参数
    index = 0
    precision = 64
    
    for matID, name, mtx_path in mtx_info:
        # 构建并运行命令
        command = f"./benchmark_spmv_dia {mtx_path} --matID={matID} --Index={index} --precision={precision}"
        # 不再需要 --sche是因为 benchmark_spmv_dia 中自动跑四次不同的调度模式
        print(f"Executing: {command}")
        
        # 执行命令
        try:
            result = subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            # 打印命令执行结果
            print(f"Command finished with return code {result.returncode}")
            if result.stdout:
                print(f"Output: {result.stdout.decode('utf-8')}")
            if result.stderr:
                print(f"Errors: {result.stderr.decode('utf-8')}")
        except subprocess.CalledProcessError as e:
            print(f"An error occurred while executing {command}")
            print(e)
       
if __name__ == "__main__":
    # Run_CSR_intKernel("./SuiteSparse_Matrix.xlsx")
    # Run_BSR_intKernel("./Error_dataset.xlsx")
    # Run_CSR5_intKernel("./SuiteSparse_Matrix.xlsx")
    Run_DIA_intKernel("./SuiteSparse_Matrix.xlsx")
    