import argparse
from sklearn.metrics import mean_absolute_error
from os.path import join
import pandas as pd
import numpy as np

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str,default=r"D:\XZX\PPG-more\ppg_dalia_cwt (2)\simple_cnn")
    return parser.parse_args()
args=get_args()

dense_root=args.input
LD_root = r"D:\XZX\PPG-more\self_LD_result"

all_LD_mae=[]
all_Dense_mae=[]
all_Merge_mae=[]

for i in range(0,14):
    subject_dense=join(dense_root,f"subject_{i+1}.csv")
    subject_ld=join(LD_root,f"pre{i}.csv")

    subject_dense_data=pd.read_csv(subject_dense).sort_values(by='seg',key=lambda x: x.str.split('-').str[1].astype(int))  # 提取"-"后的数字并转为整数)
    subject_ld_data=pd.read_csv(subject_ld,header=None).values.squeeze()

    label=subject_dense_data['ecg'].values
    subject_dense_hr=subject_dense_data['hr'].values

    subject_merge_hr=(subject_ld_data+subject_dense_hr)/2

    ld_mae=mean_absolute_error(subject_ld_data,label)
    Dense_mae=mean_absolute_error(subject_dense_hr,label)
    merge_mae=mean_absolute_error((subject_ld_data+subject_dense_hr)/2,label)
    print(f"subject:{i},LD_mae:{Dense_mae:.2f},merge_mae:{merge_mae:.2f}")
    df_save=pd.DataFrame({
        'seg':subject_dense_data['seg'].values,
        'ecg':label,
        'hr':subject_merge_hr
    })
    df_save.to_csv(f"./self_merge/subject_{i+1}.csv",index=False)
    all_LD_mae.append(ld_mae)
    all_Dense_mae.append(Dense_mae)
    all_Merge_mae.append(merge_mae)
# with open(join(dense_root, "LOSO.txt"), 'a+') as file:
#     file.write(f"Merge_All:LD_mae:{np.mean(all_LD_mae):.2f},Dense_mae:{np.mean(all_Dense_mae):.2f}Dense_LD_merge_mae:{np.mean(all_Merge_mae):.2f}")
print(f"Merge_All:LD_mae:{np.mean(all_LD_mae):.2f},Dense_mae:{np.mean(all_Dense_mae):.2f}Dense_LD_merge_mae:{np.mean(all_Merge_mae):.2f}")