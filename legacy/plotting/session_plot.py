import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


semg = pd.read_csv("data/example/sEMG.csv")
imu_raw = pd.read_csv("data/example/IMU.csv")
print("数据读取完成")


# 解析 IMU Data 列
# 提取逗号分隔的数值（忽略 "Time--..." 前缀）
def parse_imu_data(row):
    try:
        parts = row.split(',')
        # 去除第一个字符串 "Time--xx:xx:xx:xxx"
        parts = parts[1:]
        # 转为数值
        return [float(x) for x in parts]
    except:
        return [np.nan] * 19  # 如果异常，返回19个NaN（根据你的Data里数字数量）

# 应用解析
imu_values = imu_raw['Data'].apply(parse_imu_data)
imu_values_df = pd.DataFrame(imu_values.tolist())

# 命名列（假设19个数值）
imu_values_df.columns = [f'IMU_{i+1}' for i in range(imu_values_df.shape[1])]

# 合并时间戳
imu = pd.concat([imu_raw['First_Timestamp_ms'], imu_values_df], axis=1)
imu.rename(columns={'First_Timestamp_ms': 'timestamp'}, inplace=True)

print("IMU 数据解析完成")
print(imu.head())


#对齐时间戳并合并
#确保时间戳列名称一致
semg.rename(columns={'Timestamp': 'timestamp'}, inplace=True)

# 按时间戳合并（内连接，因为已对齐）
merged = pd.merge(semg, imu, on='timestamp', how='inner')
print(f"✅ 合并完成，行数：{merged.shape[0]}，列数：{merged.shape[1]}")


#基础统计量
print("\n=== sEMG 统计量 ===")
print(semg.describe().T)

print("\n=== IMU 统计量 ===")
print(imu.describe().T)


#可视化部分
time = semg['timestamp']

# ----- (a) sEMG 8个主通道 -----
plt.figure(figsize=(12, 5))
for col in [f'Channel_{i}' for i in range(1, 9)]:
    if col in semg.columns:
        plt.plot(time, semg[col], label=col, alpha=0.7)
plt.xlabel('Timestamp (ms)')
plt.ylabel('Amplitude')
plt.title('sEMG Channels 1–8')
plt.legend(ncol=4, fontsize=8)
plt.tight_layout()
plt.show()

# ----- (b) FP_CH41~FP_CH46 -----
plt.figure(figsize=(12, 4))
for col in [f'FP_CH4{i}' for i in range(1, 7)]:
    plt.plot(time, semg[col], label=col)
plt.title('sEMG FP_CH41–FP_CH46')
plt.legend()
plt.tight_layout()
plt.show()

# ----- (c) FP_CH51~FP_CH58 -----
plt.figure(figsize=(12, 4))
for col in [f'FP_CH5{i}' for i in range(1, 9)]:
    plt.plot(time, semg[col], label=col)
plt.title('sEMG FP_CH51–FP_CH58')
plt.legend()
plt.tight_layout()
plt.show()

# ----- (d) IMU 前 3 通道 -----
plt.figure(figsize=(12, 4))
for col in ['IMU_1', 'IMU_2', 'IMU_3']:
    plt.plot(time, imu[col], label=col)
plt.title('IMU Channels 1–3')
plt.legend()
plt.tight_layout()
plt.show()

# =========================
#相关性分析
# =========================
# 计算每个 sEMG 通道与每个 IMU 通道的相关系数
semg_cols = [c for c in semg.columns if c.startswith('Channel_') or c.startswith('FP_')]
imu_cols = [c for c in imu.columns if c.startswith('IMU_')]

corr_matrix = pd.DataFrame(index=semg_cols, columns=imu_cols)

for s_col in semg_cols:
    for i_col in imu_cols:
        corr_matrix.loc[s_col, i_col] = semg[s_col].corr(imu[i_col])

corr_matrix = corr_matrix.astype(float)
print("\n=== sEMG 与 IMU 相关性矩阵（部分） ===")
print(corr_matrix.iloc[:8, :6])  # 打印前几列看下趋势


#特征提取
def rms(x):
    return np.sqrt(np.mean(x**2))

semg_rms = semg[semg_cols].apply(rms)
imu_rms = imu[imu_cols].apply(rms)

print("\n=== sEMG RMS 特征 ===")
print(semg_rms)

print("\n=== IMU RMS 特征 ===")
print(imu_rms)


#联合可视化（示例：Channel_1 与 IMU_1）
plt.figure(figsize=(12,4))
plt.plot(time, semg['Channel_1'], label='sEMG Channel_1', alpha=0.7)
plt.plot(time, imu['IMU_1'], label='IMU_1', alpha=0.7)
plt.xlabel('Timestamp (ms)')
plt.title('sEMG Channel_1 vs IMU_1')
plt.legend()
plt.tight_layout()
plt.show()

print("\n全部流程执行完毕！")
