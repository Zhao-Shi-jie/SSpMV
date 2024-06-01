import subprocess
import os
import pandas as pd

def Read_GenDataset(excel_path):
    # 读取Excel文件
    df = pd.read_excel(excel_path)

    # 选择'Id'和'Name'列
    mtx_list = df[['Id', 'Matrix']].values.tolist()

    # 生成（ID, Name）元组列表和对应的文件路径
    mtx_info = [(mtx_id, path) for mtx_id, path in mtx_list]

    return mtx_info

def Run_COO_intKernel(excel_path):
    
    # 读dataset list 并生成数据组：
    mtx_info = Read_GenDataset(excel_path)
    
    # 定义测试命令中不变的参数
    index = 0
    precision = 64
    sche = range(0,4)
    
    for matID, mtx_path in mtx_info:
        # 构建并运行命令
        command = f"./benchmark_spmv_coo {mtx_path} --matID={matID} --Index={index} --precision={precision}"
        # 不再需要 --sche是因为 benchmark_spmv_dia 中自动跑四次不同的调度模式
        print(f"Executing: {command}")
        
        # 执行命令
        try:
            # 设置timeout为1200秒
            result = subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=1200)
            # 打印命令执行结果
            print(f"Command finished with return code {result.returncode}")
            if result.stdout:
                print(f"Output: {result.stdout.decode('utf-8')}")
            if result.stderr:
                print(f"Errors: {result.stderr.decode('utf-8')}")
        except subprocess.TimeoutExpired:
            print(f"Timeout: Command {command} exceeded the limit of 20 minutes.")
        except subprocess.CalledProcessError as e:
            print(f"An error occurred while executing {command}")
            print(e)
        except Exception as e:
            print(f"An unexpected error occurred: {str(e)}")
            
def Run_CSR_intKernel(excel_path):
    
    # 读dataset list 并生成数据组：
    mtx_info = Read_GenDataset(excel_path)
    
    # 定义测试命令中不变的参数
    index = 0
    precision = 64
    sche = range(0,4)
    
    for matID, mtx_path in mtx_info:
        # 构建并运行命令
        command = f"./benchmark_spmv_csr {mtx_path} --matID={matID} --Index={index} --precision={precision}"
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
            
def Run_CSR5_intKernel(excel_path):
    
    # 读dataset list 并生成数据组：
    mtx_info = Read_GenDataset(excel_path)
    
    # 定义测试命令中不变的参数
    index = 0
    precision = 64
    sche = range(0,4)
    
    for matID, mtx_path in mtx_info:
        # 构建并运行命令
        command = f"./benchmark_spmv_csr5 {mtx_path} --matID={matID} --Index={index} --precision={precision}"
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

def Run_DIA_intKernel(excel_path):
    # 读dataset list 并生成数据组：
    mtx_info = Read_GenDataset(excel_path)
    
    # 定义测试命令中不变的参数
    index = 0
    precision = 64
    
    for matID, mtx_path in mtx_info:
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
            
def Run_ELL_intKernel(excel_path):
    # 读dataset list 并生成数据组：
    mtx_info = Read_GenDataset(excel_path)
    
    # 定义测试命令中不变的参数
    index = 0
    precision = 64
    
    for matID, mtx_path in mtx_info:
        # 构建并运行命令
        command = f"./benchmark_spmv_ell {mtx_path} --matID={matID} --Index={index} --precision={precision}"
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

def Run_SELL_cs_intKernel(excel_path):
    # 读dataset list 并生成数据组：
    mtx_info = Read_GenDataset(excel_path)
    
    # 定义测试命令中不变的参数
    index = 0
    precision = 64
    
    for matID, mtx_path in mtx_info:
        # 构建并运行命令
        command = f"./benchmark_spmv_sell_c_sigma {mtx_path} --matID={matID} --Index={index} --precision={precision}"
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

def Run_SELL_intKernel(excel_path):
    # 读dataset list 并生成数据组：
    mtx_info = Read_GenDataset(excel_path)
    
    # 定义测试命令中不变的参数
    index = 0
    precision = 64
    
    for matID, mtx_path in mtx_info:
        # 构建并运行命令
        command = f"./benchmark_spmv_sell {mtx_path} --matID={matID} --Index={index} --precision={precision}"
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

def Run_SELL_cR_intKernel(excel_path):
    # 读dataset list 并生成数据组：
    mtx_info = Read_GenDataset(excel_path)
    
    # 定义测试命令中不变的参数
    index = 0
    precision = 64
    
    for matID, mtx_path in mtx_info:
        # 构建并运行命令
        command = f"./benchmark_spmv_sell_c_R {mtx_path} --matID={matID} --Index={index} --precision={precision}"
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
    # Run_COO_intKernel("./Gen_Matrix.xlsx")
    # Run_CSR_intKernel("./Gen_Matrix.xlsx")
    # Run_CSR5_intKernel("./Gen_Matrix.xlsx")
    # Run_DIA_intKernel("./Gen_Matrix.xlsx")
    # Run_ELL_intKernel("./Gen_Matrix.xlsx")
    # Run_SELL_cs_intKernel("./Gen_Matrix.xlsx")
    Run_SELL_intKernel("./Gen_Matrix.xlsx")
    # Run_SELL_cR_intKernel("./Gen_Matrix.xlsx")