import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# ===================== 1️⃣ 读取 CSV =====================
path = "data/example/motion_capture.csv"
raw = pd.read_csv(path, header=7224)

# ===================== 2️⃣ 提取 Frame 与 Sub Frame =====================
frame = pd.Series(range(1, 723))  # Frame 从 1 到 722
sub_frame = pd.Series([0] * 722)  # 全为 0

# ===================== 3️⃣ 提取每个关节的 Y 高度 =====================
joint_names = ['LASI', 'RASI', 'LPSI', 'RPSI', 'LTHI', 'LKNE', 'LTIB', 'LANK', 'LHEE', 'LTOE',
               'RTHI', 'RKNE', 'RTIB', 'RANK', 'RHEE', 'RTOE']

joint_y = {}
for i, joint in enumerate(joint_names):
    col_y = 3 + i * 3  # 根据列索引规则
    joint_y[joint] = raw.iloc[1:723, col_y].astype(float).reset_index(drop=True)

# ===================== 4️⃣ 可视化关键关节高度曲线，并标出跳跃关键帧 =====================
key_joints = ['LASI', 'LHEE', 'RHEE']  # 可根据需求调整

plt.figure(figsize=(14, 6))

for joint in key_joints:
    y_data = joint_y[joint]

    # 找跳跃最高点（局部最大）
    peaks, _ = find_peaks(y_data, distance=10)
    # 找起跳和落地最低点（局部最小）
    troughs, _ = find_peaks(-y_data, distance=10)

    # 绘制曲线
    plt.plot(frame, y_data, label=joint)

    # 绘制红点（最高点）
    plt.plot(frame.iloc[peaks], y_data.iloc[peaks], 'ro')
    # 绘制绿点（起跳/落地最低点）
    plt.plot(frame.iloc[troughs], y_data.iloc[troughs], 'go')

    # 在点上标出帧数
    for p in peaks:
        plt.text(frame.iloc[p], y_data.iloc[p] + 0.5, str(frame.iloc[p]), color='red', fontsize=8, ha='center')
    for t in troughs:
        plt.text(frame.iloc[t], y_data.iloc[t] - 0.5, str(frame.iloc[t]), color='green', fontsize=8, ha='center')

plt.xlabel('Frame')
plt.ylabel('Height (Y)')
plt.title('Jump Analysis - Key Marker Heights with Peaks/Troughs and Frame Labels')
plt.legend()
plt.grid(True)
plt.show()
