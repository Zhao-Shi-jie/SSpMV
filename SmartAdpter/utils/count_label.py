import os

def count_label(label_path="/data1/xionghantao/data/HUAWEI_data/test_data/label"):
  files = os.listdir(label_path)
  file_map = {}
  for file_ in files:
    file_path = os.path.join(label_path, file_)
    with open(file_path, "r") as f_read:
      line = f_read.readline()
      com = line.split(" ")[0]
      label = line.split(" ")[1]
      if label not in file_map:
        file_map[label] = com
  for i in sorted (file_map.keys()) : 
    print((i, file_map[i])) 


def traverse_label(label_path="/data1/xionghantao/data/HUAWEI_data/test_data/label"):
  files = os.listdir(label_path)
  file_map = {}
  for file_ in files:
    file_path = os.path.join(label_path, file_)
    with open(file_path, "r") as f_read:
      line = f_read.readline()
      com = line.split(" ")[0]
      label = line.split(" ")[1]
      if label not in file_map:
        file_map[label] = 1
      else:
        file_map[label] += 1
  num = 0
  for i in sorted (file_map.keys()):
    num = num + int(file_map[i]) 
    print((i, file_map[i])) 
  print(num)


def traverse_label_list(list_path="/data1/xionghantao/works/JPDC_special_issue/data/testing_list/test_list_3.txt"):
  file_dir = "/data1/xionghantao/data/HUAWEI_data/train_data/label/"
  file_map = {}
  with open(list_path, "r") as f_read:
    lines = f_read.readlines()
    for line in lines:
      line = line.strip()
      path = os.path.join(file_dir, line + ".final_com_label")
      with open(path, "r") as f:
        line = f.readline()
        com = line.split(" ")[0]
        label = line.split(" ")[1]
        if label + com not in file_map:
          file_map[label + com] = 1
        else:
          file_map[label + com] += 1
  num = 0
  for i in sorted (file_map.keys()):
    num = num + int(file_map[i]) 
    print((i, file_map[i])) 
  print(num)       

      


if __name__ == "__main__":
  traverse_label_list()