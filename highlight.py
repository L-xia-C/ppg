# 使用beliefppg的预测值提亮对应的CWT图片
import os
import pandas as pd
import cv2
import argparse
from tqdm import tqdm
import glob, re


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--number', type=int, default=1)
    return parser.parse_args()


def extract_hr_from_filename(path: str):
    """从类似 7-1249[73.42].jpg 中提取 73.42 -> float；失败返回 None"""
    m = re.search(r'\[(\d+(?:\.\d+)?)\]\.jpg$', os.path.basename(path))
    return float(m.group(1)) if m else None


for alpha in [1.2]:
    for beta in [30]:
        belief_root = r"D:\XZX\PPG\PPG-back\q-ppg-main\dataset\saved_models_PIT\subject_csv"
        # r"D:\XZX\PPG\PPG\belief-0513\new_result15"
        self_cwt_root = r"D:\XZX\PPG\PPG\cwt_picture\ppg_dalia"
        # merge_hr = r"D:\XZX\PPG\PPG\dalia_merge\embed_result_final_last"
        self_cwt_highlight_root = f"D:\XZX\PPG\PPG\cwt_picture\qppg\ppg_dalia_highlight_alpha-{alpha}_beta-{beta}"
        os.makedirs(self_cwt_highlight_root, exist_ok=True)

        args = get_args()
        i = args.number

        total_number = 0
        f_h = 2.8
        f_l = 0.6
        half_height = 12
        total_count = 0

        subject_belief = os.path.join(belief_root, f"P{i}.csv")
        # subject_hz = os.path.join(merge_hr, f"pre{i - 1}_mean.csv")

        belief_subject_data = pd.read_csv(subject_belief)
        # merge_subject_data = pd.read_csv(subject_hz, header=None)
        for index, row in tqdm(belief_subject_data.iterrows()):
            # merge_hr = merge_subject_data.iloc[index, 0]
            belief_hr = row['hr']
            belief_lable = f"{row['ecg']:.2f}"
            label_ecg_plus = float(belief_lable) + 0.01
            label_ecg_minus = float(belief_lable) - 0.01
            segment_cwt = os.path.join(self_cwt_root, f"{i}-{index}[{belief_lable}].jpg")
            segment_cwt_plus = os.path.join(self_cwt_root, f"{i}-{index}[{label_ecg_plus:.2f}].jpg")
            segment_cwt_minus = os.path.join(self_cwt_root, f"{i}-{index}[{label_ecg_minus:.2f}].jpg")
            target_path = os.path.join(self_cwt_highlight_root, f"{i}-{index}[{belief_lable}].jpg")
            # 可以找到对应的图片
            if (os.path.exists(segment_cwt)):
                img = cv2.imread(segment_cwt)
            elif (os.path.exists(segment_cwt_plus)):
                img = cv2.imread(segment_cwt_plus)
            elif os.path.exists(segment_cwt_minus):
                img = cv2.imread(segment_cwt_minus)
            else:
                print(segment_cwt, segment_cwt_plus, segment_cwt_minus)
                continue
            [height, width] = img.shape[0], img.shape[1]
            try:
                length = int(((half_height / 60) / (f_h - f_l)) * height)
                y = int((1 - ((belief_hr / 60) - f_l) / (f_h - f_l)) * height)
                y_up = max(y - length, 0)
                y_down = min(y + length, height - 1)
                roi = img[y_up:y_down, 0:width]
                roi = cv2.convertScaleAbs(roi, alpha=alpha, beta=beta)
                img[y_up:y_down, 0:width] = roi
                cv2.imwrite(target_path, img)
            except Exception as e:
                print(segment_cwt)
            # else:
            #     pattern = os.path.join(self_cwt_root, f"{i}-{index}[[]*[]].jpg")
            #     candidates = glob.glob(pattern)
            #     if len(candidates) == 0:
            #         print(f"[MISS] 未找到文件：{pattern}")
            #         continue
            #
            #     if len(candidates) > 1:
            #         # 严格判定：一个都不能多
            #         raise RuntimeError(f"[DUP] 期望唯一文件，但找到 {len(candidates)} 个：{candidates}")
            #     real_path = candidates[0]
            #     file_hr = extract_hr_from_filename(real_path)
            #     if file_hr is None:
            #         raise ValueError(f"[PARSE-ERR] 无法从文件名解析HR：{real_path}")
            #     label_ecg = float(row['ecg'])
            #     diff = file_hr - label_ecg
            #     print(f"[OK] 使用 {real_path} | fileHR={file_hr:.2f}, label={label_ecg:.2f}, diff={diff}")
            #     # target_path=f"{i}-{index}"
            #     # matching_files = [file for file in os.listdir(self_cwt_root) if file.split("[")[0]==target_path]
            #     # os.rename(os.path.join(self_cwt_root,matching_files[0]),segment_cwt)
