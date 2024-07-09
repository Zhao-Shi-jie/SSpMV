import os
import pandas as pd

Suite_dataset = "SuiteSparse_Matrix.xlsx"
Gen_dataset   = "ValidGen_Matrix.xlsx"

root_dir = "/data/lsl/MModel-Data"
features_suf = ".features"

def Read_GenDataset(excel_path):
    # 读取Excel文件
    df = pd.read_excel(excel_path)

    # 选择'Id'和'Name'列
    mtx_list = df[['Id', 'Matrix']].values.tolist()

    # 生成（ID, Name）元组列表和对应的文件路径
    # 生成（ID, Name）元组列表和对应的文件路径
    mtx_info = []
    for mtx_id, path in mtx_list:
        if isinstance(path, str):
            # 提取文件名
            file_name = os.path.basename(path)
            # 去除扩展名
            name, _ = os.path.splitext(file_name)

            # 添加到列表
            mtx_info.append((int(mtx_id), str(name)))
        else:
            print(f"Warning: Skipping non-string path at ID {mtx_id}")
            
    return mtx_info

def Read_SuiteDataset(excel_path):
    # 读取Excel文件
    df = pd.read_excel(excel_path)

    # 选择'Id'和'Name'列
    mtx_list = df[['Id', 'Name']].values.tolist()

    # 生成（ID, Name）元组列表和对应的文件路径
    # mtx_info = [(mtx_id, name, f"/data/suitesparse_collection/{name}/{name}.mtx") for mtx_id, name in mtx_list]
    mtx_info = [(int(mtx_id), str(name)) for mtx_id, name in mtx_list]

    return mtx_info

def Read_Features(root_dir, gen_datalist):
    all_features = []

    for mtx_id, name in gen_datalist:
        feature_file_path = os.path.join(root_dir, name, f"{name}.features")
        if os.path.exists(feature_file_path):
            with open(feature_file_path, 'r') as file:
                lines = file.readlines()[1:]  # 跳过第一行
                features = {'Id': int(mtx_id), 'Name': name}
                for line in lines:
                    key, value = line.split()
                    features[key] = float(value)
                all_features.append(features)
        else:
            print(f"Feature file not found for {name} at ID {mtx_id}")

    return pd.DataFrame(all_features)


if __name__ == "__main__":
    # code to extract all features of datasets
    Suite_datalist = Read_SuiteDataset(Suite_dataset)
    Gen_datalist = Read_GenDataset(Gen_dataset)
    
    print(Suite_datalist[:20])
    print(Gen_datalist[:20])
    
    Suite_features_df = Read_Features(root_dir, Suite_datalist)
    Gen_features_df = Read_Features(root_dir, Gen_datalist)
    
    # 将特征保存到新的Excel文件
    Suite_outpath = './Suite_features.xlsx'  # 替换为您想要保存的路径
    Suite_features_df.to_excel(Suite_outpath, index=False)
    print(f"Features saved to {Suite_outpath}")
    
    Gen_outpath = './Gen_features.xlsx'
    Gen_features_df.to_excel(Gen_outpath, index=False)
    print(f"Features saved to {Gen_outpath}")