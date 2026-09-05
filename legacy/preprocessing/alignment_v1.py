import os
import pandas as pd
from datetime import timedelta

# ======== 路径配置 ========
base_dir = "data/interim"
imu_root = os.path.join(base_dir, "IMU_data_copy")
semg_root = os.path.join(base_dir, "sEMG_data_copy")
output_root = "data/legacy_aligned"
os.makedirs(output_root, exist_ok=True)

# ======== 时间戳补全函数 ========
def generate_time_series(start_time, n):
    """若IMU只有一个时间戳，则生成连续1ms间隔时间序列"""
    return [start_time + timedelta(milliseconds=i) for i in range(n)]

# ======== 主流程 ========
for subject in os.listdir(imu_root):
    if not subject.endswith("_IMU_data"):
        continue
    subject_id = subject.replace("_IMU_data", "")
    imu_subject_path = os.path.join(imu_root, subject)
    semg_subject_path = os.path.join(semg_root, f"{subject_id}_sEMG_data")

    if not os.path.exists(semg_subject_path):
        print(f"⚠️ 未找到对应 sEMG 文件夹：{semg_subject_path}")
        continue

    # 遍历动作类型（跳高/跳远）
    for action_folder in os.listdir(imu_subject_path):
        imu_action_path = os.path.join(imu_subject_path, action_folder)
        if not os.path.isdir(imu_action_path):
            continue

        # 生成 sEMG 对应文件夹路径
        semg_action_path = os.path.join(
            semg_subject_path,
            action_folder.replace("IMU_", "sEMG_").lower()
        )

        if not os.path.exists(semg_action_path):
            print(f"⚠️ 未找到对应 sEMG 动作文件夹：{semg_action_path}")
            continue

        # 遍历时间戳文件夹
        for timestamp_folder in os.listdir(imu_action_path):
            imu_timestamp_path = os.path.join(imu_action_path, timestamp_folder)
            semg_timestamp_path = os.path.join(semg_action_path, timestamp_folder)

            if not os.path.isdir(imu_timestamp_path) or not os.path.isdir(semg_timestamp_path):
                continue

            # 匹配文件名
            imu_file = os.path.join(imu_timestamp_path, f"IMU_{timestamp_folder}.csv")
            semg_file = os.path.join(semg_timestamp_path, f"processed_data_{timestamp_folder}.csv")

            if not os.path.exists(imu_file) or not os.path.exists(semg_file):
                print(f"⚠️ 缺少文件，跳过：{imu_file} 或 {semg_file}")
                continue

            # ======== 读取 IMU 数据 ========
            imu = pd.read_csv(imu_file)
            imu_time_col = imu.columns[0]
            imu[imu_time_col] = pd.to_datetime(imu[imu_time_col], errors='coerce')

            # 若IMU时间戳只有一个，则补全
            if imu[imu_time_col].nunique() == 1 or imu[imu_time_col].isna().all():
                start_time = pd.to_datetime(imu.iloc[0, 0], errors='coerce')
                imu['timestamp'] = generate_time_series(start_time, len(imu))
                print(f"🕒 自动补全IMU时间戳：{subject_id} {action_folder} {timestamp_folder} （共 {len(imu)} 行）")
            else:
                imu['timestamp'] = imu[imu_time_col]

            # ======== 读取 sEMG 数据 ========
            semg = pd.read_csv(semg_file)
            semg_time_col = semg.columns[0]
            semg['timestamp'] = pd.to_datetime(semg[semg_time_col], errors='coerce')

            # ======== 对齐：裁剪sEMG到IMU时间范围 ========
            t_start, t_end = imu['timestamp'].iloc[0], imu['timestamp'].iloc[-1]
            aligned_semg = semg[(semg['timestamp'] >= t_start) & (semg['timestamp'] <= t_end)]

            # 若长度不匹配，裁剪为相同长度
            min_len = min(len(imu), len(aligned_semg))
            imu = imu.iloc[:min_len].reset_index(drop=True)
            aligned_semg = aligned_semg.iloc[:min_len].reset_index(drop=True)

            # ======== 保存 ========
            save_dir = os.path.join(output_root, subject_id, action_folder, timestamp_folder)
            os.makedirs(save_dir, exist_ok=True)
            imu_save_path = os.path.join(save_dir, f"aligned_{os.path.basename(imu_file)}")
            semg_save_path = os.path.join(save_dir, f"aligned_{os.path.basename(semg_file)}")

            imu.to_csv(imu_save_path, index=False)
            aligned_semg.to_csv(semg_save_path, index=False)

            print(f"✅ 已对齐完成：{subject_id}/{action_folder}/{timestamp_folder}")

print("\n🎯 全部对齐完成！输出目录：", output_root)
