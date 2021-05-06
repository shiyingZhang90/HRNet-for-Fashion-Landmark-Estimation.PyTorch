import os
import shutil
import json

root = './results_mask'
target_root = './results_expanded_mask'

i = 0
name_list = os.listdir(root)
for name in name_list:
    path = os.path.join(root, name)
    for file in os.listdir(path):
        # print(os.path.join(path, file), os.path.join(target_root, file))
        shutil.copy(os.path.join(path, file), os.path.join(target_root, file))
