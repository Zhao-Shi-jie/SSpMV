import os
import pandas as pd

# 21 labels described in paper
Suite_label_file = "Suite_Best_Result.xlsx"
Gen_label_file = "Gen_Best_Result.xlsx"
detailed_label_suffix = ".det_format_label"

Suite_Prob_file = "Suite_Probability_Vectors.xlsx"
Gen_Prob_file = "Gen_Probability_Vectors.xlsx"
detailed_Prob_suffix  = ".det_prob_label"

root_dir = "/data/lsl/MModel-Data"

# Function to generate labels based on Format, sche_mode, and Sigma(Reorder Seg)
def generate_labels(row):
    format = row['Format']
    sche_mode = row['sche_mode']
    sigma = row['Sigma(Reorder Seg)']
    
    if format == 'COO':
        return sche_mode
    elif format == 'CSR':
        return sche_mode + 4
    elif format == 'DIA':
        return 8
    elif format == 'ELL':
        return 9 if sche_mode in [0, 1] else 10
    elif format == 'S-ELL':
        return 11 if sche_mode in [0, 1] else 12
    elif format == 'S-ELL-R':
        return 13 if sche_mode in [0, 1] else 14
    elif format == 'S-ELL-sigma':
        if sche_mode in [0, 1]:
            if sigma == 512:
                return 15
            elif sigma in [4096, 16384]:
                return 16
        elif sche_mode in [2, 3]:
            if sigma == 512:
                return 17
            elif sigma in [4096, 16384]:
                return 18
    elif format == 'BSR':
        return 19
    elif format == 'CSR5':
        return 20
    return None

def write_label_to_file(data_list):
    df = pd.read_excel(data_list)
    
    # 遍历每一行数据
    for index, row in df.iterrows():
        # 文件夹路径
        name_str = str(row['Name'])
        format_str = str(row['Format'])
        label_str = str(row['Label'])
        dir_path = os.path.join(root_dir, name_str)
        
        # 如果文件夹不存在，创建新文件夹
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        file_path = os.path.join(dir_path, f"{name_str}{detailed_label_suffix}")
        
        print(file_path)
        
        with open(file_path, "w") as file:
            file.write(f"{format_str} {label_str}\n")
        print("Writing detailed label of", row['Name'],"successfully")

if __name__ == "__main__":
    
    # detailed_labelfile_name = 'Labeled_Suite_Best_Result.xlsx'
    # # Load your data
    # SuiteSparse_data = pd.read_excel(Suite_label_file)

    # # Apply the function to the DataFrame
    # SuiteSparse_data['Label'] = SuiteSparse_data.apply(generate_labels, axis=1)

    # # Save the final DataFrame with labels to a new Excel file
    # SuiteSparse_data.to_excel(detailed_labelfile_name, index=False)
    # write_label_to_file(detailed_labelfile_name)
    
    Gen_detailed_labelfile_name = 'Labeled_Gen_Best_Result.xlsx'
    # Load your data
    GenMTX_data = pd.read_excel(Gen_label_file)

    # Apply the function to the DataFrame
    GenMTX_data['Label'] = GenMTX_data.apply(generate_labels, axis=1)

    # Save the final DataFrame with labels to a new Excel file
    GenMTX_data.to_excel(Gen_detailed_labelfile_name, index=False)
    write_label_to_file(Gen_detailed_labelfile_name)
