import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ======== 1️⃣ 数据读取 ========
semg = pd.read_csv("data/example/sEMG.csv")
imu_raw = pd.read_csv("data/example/IMU.csv")
print("数据读取完成")

# ======== 2️⃣ IMU解析 ========
def parse_imu_data(row):
    try:
        parts = row.split(',')
        parts = parts[1:]  # 去掉Time部分
        return [float(x) for x in parts]
    except:
        return [np.nan] * 19

imu_values = imu_raw['Data'].apply(parse_imu_data)
imu_values_df = pd.DataFrame(imu_values.tolist())
imu_values_df.columns = [f'IMU_{i+1}' for i in range(imu_values_df.shape[1])]
imu = pd.concat([imu_raw['First_Timestamp_ms'], imu_values_df], axis=1)
imu.rename(columns={'First_Timestamp_ms': 'timestamp'}, inplace=True)
print("IMU 数据解析完成")

# ======== 3️⃣ 对齐与截断处理 ========
semg.rename(columns={'Timestamp': 'timestamp'}, inplace=True)

cutoff_len = imu.shape[0]  # IMU 数据行数（2095）
print(f"⚙️ 截断点：{cutoff_len} 行")

# 截断前后的 sEMG 数据
semg_before = semg.copy()
semg_after = semg.iloc[:cutoff_len].reset_index(drop=True)

# 同步 IMU 时间
time_imu = imu['timestamp']
time_before = semg_before['timestamp']
time_after = time_imu  # 这里明确：截断后用 IMU 的时间作为时间轴

# ============ 可视化函数 ============
def plot_imu_groups(df, time, title_prefix):
    plt.figure(figsize=(14,6))
    group_labels = ['L1','L2','L3','R1','R2','R3']
    for i in range(6):
        cols = [f'IMU_{i*3+1}', f'IMU_{i*3+2}', f'IMU_{i*3+3}']
        for c in cols:
            if c in df.columns:
                plt.plot(time, df[c], label=f'{group_labels[i]}-{c}')
    plt.title(f'{title_prefix} IMU 六个关节')
    plt.xlabel('Timestamp (ms)')
    plt.ylabel('IMU Values')
    plt.legend(fontsize=8, ncol=3)
    plt.tight_layout()
    plt.show()

def plot_semg_channels(df, time, title_prefix):
    plt.figure(figsize=(14,5))
    for col in [f'Channel_{i}' for i in range(1,9)]:
        if col in df.columns:
            plt.plot(time, df[col], label=col)
    plt.title(f'{title_prefix} sEMG Channels 1–8')
    plt.xlabel('Timestamp (ms)')
    plt.ylabel('Amplitude')
    plt.legend(ncol=4, fontsize=8)
    plt.tight_layout()
    plt.show()

def plot_foot_pressure(df, time, title_prefix):
    plt.figure(figsize=(14,5))
    for col in [f'FP_CH5{i}' for i in range(1,9)]:
        if col in df.columns:
            plt.plot(time, df[col], label=col)
    plt.title(f'{title_prefix} Foot Pressure FP_CH51–FP_CH58')
    plt.xlabel('Timestamp (ms)')
    plt.ylabel('Force')
    plt.legend(ncol=4, fontsize=8)
    plt.tight_layout()
    plt.show()

def plot_combined(df_semg, df_imu, time, title_prefix):
    # 关键修正：确保所有信号长度一致
    min_len = min(len(time), len(df_semg), len(df_imu))
    time = time.iloc[:min_len] if hasattr(time, "iloc") else time[:min_len]
    df_semg = df_semg.iloc[:min_len]
    df_imu = df_imu.iloc[:min_len]

    plt.figure(figsize=(14,5))
    plt.plot(time, df_semg['Channel_1'], label='sEMG_Channel_1', alpha=0.7)
    plt.plot(time, df_imu['IMU_1'], label='IMU_1', alpha=0.7)
    if 'FP_CH51' in df_semg.columns:
        plt.plot(time, df_semg['FP_CH51'], label='FP_CH51', alpha=0.7)
    plt.title(f'{title_prefix} sEMG + IMU + Foot Pressure')
    plt.xlabel('Timestamp (ms)')
    plt.legend()
    plt.tight_layout()
    plt.show()

# ============ 截断前可视化 ============
print("\n🟢 截断前可视化：")
plot_imu_groups(imu, time_imu, title_prefix='Before Truncation')
plot_semg_channels(semg_before, time_before, title_prefix='Before Truncation')
plot_foot_pressure(semg_before, time_before, title_prefix='Before Truncation')
plot_combined(semg_before, imu, time_imu, title_prefix='Before Truncation')

# ============ 截断后可视化 ============
print("\n🔵 截断后可视化（sEMG 与 IMU 对齐）：")
plot_imu_groups(imu, time_imu, title_prefix='After Truncation')
plot_semg_channels(semg_after, time_after, title_prefix='After Truncation')
plot_foot_pressure(semg_after, time_after, title_prefix='After Truncation')
plot_combined(semg_after, imu, time_imu, title_prefix='After Truncation')

print("\n✅ 全部流程执行完毕！")
