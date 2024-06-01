features_dim = 40
number_of_labels = 9
num_of_detailed_labels = 19

label_file_suffix = ".format_label"
detailed_file_suffix = ".detailed_label"

feat_data ="/data/lsl/feature"
label_data = "/data/lsl/label"
image_data = "/data/lsl/image"

# feat data  存储为 18 行，每行： 特征名字 <空格> 特征值 
# label data 存储为 预条件名 <空格> 对应算法号
# image data 存储为 128 *128 = 16384 行