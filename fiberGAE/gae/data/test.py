import pickle
import sys

# 从pickle文件加载对象
print(sys.version_info)
with open("ind.citeseer.x".format('citeseer', 'x'), 'rb') as file:
    loaded_data = pickle.load(file)

print(loaded_data)
print(loaded_data.shape)
