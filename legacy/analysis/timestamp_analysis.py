import pandas as pd
import os
from pathlib import Path
import re
from datetime import datetime


def analyze_timestamp_alignment(base_path, sample_count=None):
    """
    分析IMU和sEMG时间戳的对齐情况
    """
    base_path = Path(base_path)
    imu_base = base_path / "IMU_data_copy"
    semg_base = base_path / "sEMG_data_copy"

    print("=" * 80)
    print("时间戳对齐情况分析")
    print("=" * 80)

    # 查找所有匹配的文件对
    file_pairs = []
    for imu_file in imu_base.rglob("IMU_*.csv"):
        timestamp_match = re.search(r'IMU_(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})\.csv', imu_file.name)
        if not timestamp_match:
            continue

        timestamp = timestamp_match.group(1)

        # 构建对应的sEMG文件路径
        imu_relative_path = imu_file.relative_to(imu_base)
        semg_relative_path_str = str(imu_relative_path).replace("IMU_data", "sEMG_data")
        semg_relative_path_str = semg_relative_path_str.replace("IMU_", "sEMG_")

        semg_dir = Path(semg_base) / Path(semg_relative_path_str).parent
        semg_file = semg_dir / f"processed_data_{timestamp}.csv"

        if semg_file.exists():
            file_pairs.append({
                'imu_path': imu_file,
                'semg_path': semg_file,
                'timestamp': timestamp
            })

    if sample_count:
        file_pairs = file_pairs[:sample_count]

    print(f"分析 {len(file_pairs)} 个文件对...")

    # 统计变量
    imu_first_count = 0
    semg_first_count = 0
    simultaneous_count = 0
    alignment_details = []

    def convert_timestamp(timestamp_str):
        """转换时间戳字符串为datetime对象"""
        try:
            if '.' in timestamp_str:
                main_part, millis_part = timestamp_str.split('.')
                millis_part = millis_part.ljust(3, '0')[:3]
                timestamp_str = f"{main_part}.{millis_part}"
            return pd.to_datetime(timestamp_str, format='%Y-%m-%d %H:%M:%S.%f')
        except:
            return None

    for i, pair in enumerate(file_pairs):
        try:
            # 读取IMU数据
            imu_data = pd.read_csv(pair['imu_path'])
            semg_data = pd.read_csv(pair['semg_path'])

            if imu_data.empty or semg_data.empty:
                continue

            # 获取第一个时间戳
            imu_first_time = convert_timestamp(imu_data.iloc[0, 0])
            semg_first_time = convert_timestamp(semg_data.iloc[0, 0])

            if imu_first_time is None or semg_first_time is None:
                continue

            # 计算时间差（毫秒）
            time_diff = (imu_first_time - semg_first_time).total_seconds() * 1000

            # 判断谁先开始
            if abs(time_diff) < 10:  # 10毫秒内认为是同时开始
                start_type = "同时开始"
                simultaneous_count += 1
            elif time_diff > 0:  # IMU先开始
                start_type = "IMU先开始"
                imu_first_count += 1
            else:  # sEMG先开始
                start_type = "sEMG先开始"
                semg_first_count += 1

            # 获取数据长度信息
            imu_duration = (convert_timestamp(imu_data.iloc[-1, 0]) - imu_first_time).total_seconds()
            semg_duration = (convert_timestamp(semg_data.iloc[-1, 0]) - semg_first_time).total_seconds()

            alignment_details.append({
                'timestamp': pair['timestamp'],
                'start_type': start_type,
                'time_diff_ms': time_diff,
                'imu_duration': imu_duration,
                'semg_duration': semg_duration,
                'imu_points': len(imu_data),
                'semg_points': len(semg_data),
                'imu_start': imu_first_time,
                'semg_start': semg_first_time
            })

            if i < 10:  # 显示前10个的详细信息
                print(f"{i + 1}. {pair['timestamp']}: {start_type} (差值: {time_diff:.1f}ms)")
                print(f"   IMU: {imu_first_time}, 时长: {imu_duration:.2f}s, 点数: {len(imu_data)}")
                print(f"   sEMG: {semg_first_time}, 时长: {semg_duration:.2f}s, 点数: {len(semg_data)}")

        except Exception as e:
            print(f"处理 {pair['timestamp']} 时出错: {e}")
            continue

    # 统计结果
    print(f"\n" + "=" * 50)
    print("统计结果:")
    print(f"IMU先开始的次数: {imu_first_count}")
    print(f"sEMG先开始的次数: {semg_first_count}")
    print(f"同时开始的次数: {simultaneous_count}")
    print(f"总文件对数: {len(alignment_details)}")

    # 详细分析
    if alignment_details:
        df_details = pd.DataFrame(alignment_details)

        print(f"\n时间差统计 (毫秒):")
        print(f"  平均时间差: {df_details['time_diff_ms'].mean():.2f}ms")
        print(f"  最大时间差: {df_details['time_diff_ms'].max():.2f}ms")
        print(f"  最小时间差: {df_details['time_diff_ms'].min():.2f}ms")
        print(f"  标准差: {df_details['time_diff_ms'].std():.2f}ms")

        print(f"\n数据时长统计:")
        print(f"  IMU平均时长: {df_details['imu_duration'].mean():.2f}s")
        print(f"  sEMG平均时长: {df_details['semg_duration'].mean():.2f}s")
        print(f"  IMU平均采样率: {df_details['imu_points'].mean() / df_details['imu_duration'].mean():.1f}Hz")
        print(f"  sEMG平均采样率: {df_details['semg_points'].mean() / df_details['semg_duration'].mean():.1f}Hz")

    return imu_first_count, semg_first_count, simultaneous_count, alignment_details


def recommend_alignment_strategy(alignment_details):
    """
    根据分析结果推荐对齐策略
    """
    print(f"\n" + "=" * 50)
    print("对齐策略建议")
    print("=" * 50)

    df_details = pd.DataFrame(alignment_details)

    # 分析时间差分布
    imu_first = df_details[df_details['time_diff_ms'] > 10]
    semg_first = df_details[df_details['time_diff_ms'] < -10]
    simultaneous = df_details[abs(df_details['time_diff_ms']) <= 10]

    print("基于您的数据特征，建议采用以下对齐策略:")
    print("\n1. 对齐基准: 以IMU时间为基准")
    print("   - 原因: IMU数据通常更稳定，作为运动捕捉的主要参考")
    print("   - 方法: 将sEMG数据对齐到IMU的时间范围")

    print("\n2. 非对齐数据处理: 丢弃非重叠部分")
    print("   - 原因:")
    print("     * 确保两个信号的时间完全同步")
    print("     * 避免时间不同步导致的特征提取误差")
    print("     * 运动分析和机器学习需要严格的时间对齐")

    print("\n3. 具体操作:")
    print("   - 对于IMU先开始的数据: 丢弃sEMG开始前的时间段")
    print("   - 对于sEMG先开始的数据: 丢弃IMU开始前的时间段")
    print("   - 保留两个信号完全重叠的时间段")

    print(f"\n4. 预期影响:")
    total_points_lost = 0
    for detail in alignment_details:
        time_diff = abs(detail['time_diff_ms']) / 1000  # 转换为秒
        if detail['start_type'] == "IMU先开始":
            # 假设sEMG采样率约1000Hz
            points_lost = time_diff * 1000
        elif detail['start_type'] == "sEMG先开始":
            # 假设IMU采样率约200Hz
            points_lost = time_diff * 200
        else:
            points_lost = 0
        total_points_lost += points_lost

    print(f"   预计平均每个文件会丢失 {total_points_lost / len(alignment_details):.1f} 个数据点")
    print(f"   但能确保时间同步精度")


if __name__ == "__main__":
    base_data_path = "data/interim"

    print("开始分析时间戳对齐情况...")
    imu_first, semg_first, simultaneous, details = analyze_timestamp_alignment(base_data_path)

    # 验证您的统计
    print(f"\n" + "=" * 50)
    print("验证您的统计:")
    print(f"您提到的: IMU先开始 = 58, sEMG先开始 = 163, 同时开始 = 6")
    print(f"实际统计: IMU先开始 = {imu_first}, sEMG先开始 = {semg_first}, 同时开始 = {simultaneous}")

    if imu_first == 58 and semg_first == 163 and simultaneous == 6:
        print("✓ 您的统计完全正确！")
    else:
        print("⚠ 统计有差异，请检查数据")

    # 提供对齐建议
    if details:
        recommend_alignment_strategy(details)
