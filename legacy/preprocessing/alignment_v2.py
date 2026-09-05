import os
import pandas as pd
from datetime import timedelta

base_dir = "data/interim"
imu_root = os.path.join(base_dir, "IMU_data_copy")
semg_root = os.path.join(base_dir, "sEMG_data_copy")
output_dir = "data/legacy_aligned"
os.makedirs(output_dir, exist_ok=True)

def generate_time_series(start_time, n):
    """若IMU只有一个时间戳，则生成连续1ms间隔时间序列"""
    return [start_time + timedelta(milliseconds=i) for i in range(n)]

# 遍历IMU受试者
for subject_folder in os.listdir(imu_root):
    subject_path = os.path.join(imu_root, subject_folder)
    if not os.path.isdir(subject_path):
        continue

    subject_id = subject_folder.split("_")[0]  # Example: participant folder -> numeric prefix
    semg_subject = os.path.join(semg_root, f"{subject_id}_sEMG_data")
    if not os.path.exists(semg_subject):
        print(f"⚠️ 未找到对应sEMG目录: {subject_id}")
        continue

    # 遍历动作
    for action_folder in os.listdir(subject_path):
        imu_action_path = os.path.join(subject_path, action_folder)
        semg_action_path = os.path.join(semg_subject, action_folder.replace("IMU_", "sEMG_").replace("imu", "semg").lower())
        if not os.path.exists(semg_action_path):
            print(f"⚠️ 未找到对应动作文件夹: {semg_action_path}")
            continue

        # 遍历时间戳文件
        for imu_file in os.listdir(imu_action_path):
            if not imu_file.startswith("IMU_") or not imu_file.endswith(".csv"):
                continue

            timestamp = imu_file.replace("IMU_", "").replace(".csv", "")
            semg_file = f"processed_data_{timestamp}.csv"

            imu_path = os.path.join(imu_action_path, imu_file)
            semg_path = os.path.join(semg_action_path, semg_file)
            if not os.path.exists(semg_path):
                print(f"⚠️ 未找到对应sEMG文件: {semg_file}")
                continue

            # ======== 读取IMU数据 ========
            imu = pd.read_csv(imu_path)
            time_col = imu.columns[0]
            imu[time_col] = pd.to_datetime(imu[time_col], errors='coerce')

            # 自动补全时间戳
            if imu[time_col].nunique() == 1 or imu[time_col].isna().all():
                start_time = pd.to_datetime(imu.iloc[0, 0], errors='coerce')
                imu['timestamp'] = generate_time_series(start_time, len(imu))
                print(f"🕒 {timestamp}: 自动补全IMU时间戳 {len(imu)} 行")
            else:
                imu['timestamp'] = imu[time_col]

            # ======== 读取sEMG数据 ========
            semg = pd.read_csv(semg_path)
            semg_time_col = semg.columns[0]
            semg['timestamp'] = pd.to_datetime(semg[semg_time_col], errors='coerce')

            # 对齐
            t_start, t_end = imu['timestamp'].iloc[0], imu['timestamp'].iloc[-1]
            aligned_semg = semg[(semg['timestamp'] >= t_start) & (semg['timestamp'] <= t_end)]

            if len(aligned_semg) > len(imu):
                aligned_semg = aligned_semg.iloc[:len(imu)]

            # ======== 保存 ========
            save_folder = os.path.join(output_dir, subject_folder, action_folder)
            os.makedirs(save_folder, exist_ok=True)
            imu_save_path = os.path.join(save_folder, os.path.basename(imu_path))
            semg_save_path = os.path.join(save_folder, os.path.basename(semg_path))

            imu.to_csv(imu_save_path, index=False)
            aligned_semg.to_csv(semg_save_path, index=False)
            print(f"✅ 已对齐完成: {subject_folder}/{action_folder}/{timestamp}")

print("\n🎯 全部处理完成！输出目录：", output_dir)
