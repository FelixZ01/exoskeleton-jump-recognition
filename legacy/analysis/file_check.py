import os
import pandas as pd

# 对齐后数据的目录
aligned_root = "data/legacy_aligned"

imu_first = 0
semg_first = 0
same_start = 0
total_pairs = 0

# 遍历所有受试者和动作
for subject in os.listdir(aligned_root):
    subject_path = os.path.join(aligned_root, subject)
    if not os.path.isdir(subject_path):
        continue

    for motion in os.listdir(subject_path):
        motion_path = os.path.join(subject_path, motion)
        if not os.path.isdir(motion_path):
            continue

        for folder in os.listdir(motion_path):
            folder_path = os.path.join(motion_path, folder)
            imu_path = os.path.join(folder_path, f"aligned_IMU_{folder}.csv")
            semg_path = os.path.join(folder_path, f"aligned_sEMG_{folder}.csv")

            if os.path.exists(imu_path) and os.path.exists(semg_path):
                total_pairs += 1
                imu_df = pd.read_csv(imu_path)
                semg_df = pd.read_csv(semg_path)

                imu_start = imu_df.iloc[0, 0]
                semg_start = semg_df.iloc[0, 0]

                if imu_start < semg_start:
                    imu_first += 1
                elif imu_start > semg_start:
                    semg_first += 1
                else:
                    same_start += 1
            else:
                print(f"⚠️ 未找到对齐文件：{folder_path}")

# 输出结果
print("\n=== ✅ 对齐检查结果 ===")
print(f"总共成功对齐的文件组数: {total_pairs}")
print(f"IMU先开始的次数: {imu_first}")
print(f"sEMG先开始的次数: {semg_first}")
print(f"同时开始的次数: {same_start}")
print("========================")
