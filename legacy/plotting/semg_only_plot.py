import pandas as pd
import matplotlib.pyplot as plt

# =========================
# 1. 读取 sEMG 数据
# =========================
semg_path = "data/example/sEMG.csv"
semg = pd.read_csv(semg_path)
print("sEMG 数据读取完成")
print(f"数据维度: {semg.shape}")
print(f"列名: {list(semg.columns)}")

# =========================
# 2. 时间戳列处理
# =========================
# 根据文件列名自动判断时间戳字段（有的叫 timestamp，有的叫 Data）
if 'timestamp' in semg.columns:
    time_col = 'timestamp'
elif 'Timestamp' in semg.columns:
    time_col = 'Timestamp'
elif 'Data' in semg.columns:
    time_col = 'Data'
else:
    raise ValueError("未找到时间戳列，请检查文件！")

time = semg[time_col]

# =========================
# 3. sEMG 通道可视化
# =========================

# --- (a) 主通道 Channel_1~Channel_8 ---
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

# --- (b) FP_CH41~FP_CH46 ---
plt.figure(figsize=(12, 4))
for col in [f'FP_CH4{i}' for i in range(1, 7)]:
    if col in semg.columns:
        plt.plot(time, semg[col], label=col)
plt.title('sEMG FP_CH41–FP_CH46')
plt.legend(ncol=3, fontsize=8)
plt.tight_layout()
plt.show()

# --- (c) FP_CH51~FP_CH58 ---
plt.figure(figsize=(12, 4))
for col in [f'FP_CH5{i}' for i in range(1, 9)]:
    if col in semg.columns:
        plt.plot(time, semg[col], label=col)
plt.title('sEMG FP_CH51–FP_CH58')
plt.legend(ncol=4, fontsize=8)
plt.tight_layout()
plt.show()

# =========================
# 4. 基础统计信息
# =========================
semg_cols = [c for c in semg.columns if c.startswith('Channel_') or c.startswith('FP_')]
print("\n=== sEMG 统计量 ===")
print(semg[semg_cols].describe().T)

print("\n全部流程执行完毕！")
