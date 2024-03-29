import pandas as pd
import pickle

df = pd.read_csv('FiberClusterAnnotation_k0800_v1.0.csv')
print(df)

cols = df['Annotation']
cols_list = cols.to_list()

string_to_number = {}
next_number = 0  # 初始化下一个可用的数字

# 遍历字符串列表
for string in cols_list:
    # 如果字符串还没有被映射到数字，则将它映射到下一个可用的数字
    if string not in string_to_number:
        string_to_number[string] = next_number
        next_number += 1

# 将列表中的每个字符串替换为相应的数字
numeric_list = [string_to_number[string] for string in cols_list]

print(cols_list, len(cols_list))
print(numeric_list, len(numeric_list))

with open('fiber_bundle_ids.pkl', 'wb') as f:
    pickle.dump(numeric_list, f, protocol=2)

with open('fiber_bundle_names.pkl', 'wb') as f:
    pickle.dump(cols_list, f, protocol=2)
