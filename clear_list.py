# 데이터 일괄삭제용

import os
import shutil

clear_list = [
]

results_path = './results/'
test_results_path = './test_results/'

results_dirs = os.listdir(results_path)

result_dirs_TCN = [di for di in results_dirs if '360' in di]

print(result_dirs_TCN)
print(len(result_dirs_TCN))

clear_list = result_dirs_TCN

for d_path in clear_list:
    if os.path.exists(results_path + d_path):
        shutil.rmtree(results_path + d_path)
    if os.path.exists(test_results_path + d_path):
        shutil.rmtree(test_results_path + d_path)
