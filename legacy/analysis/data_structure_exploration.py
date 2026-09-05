import pandas as pd
import os
from pathlib import Path
import numpy as np


def explore_data_structure(base_path, sample_count=5):
    """
    探索IMU和sEMG数据的结构
    """
    base_path = Path(base_path)
    imu_base = base_path / "IMU_data_copy"
    semg_base = base_path / "sEMG_data_copy"

    print("=" * 80)
    print("数据格式探查报告")
    print("=" * 80)

    # 查找样本文件
    imu_files = list(imu_base.rglob("IMU_*.csv"))[:sample_count]
    semg_files = list(semg_base.rglob("processed_data_*.csv"))[:sample_count]

    print(f"\n找到 {len(imu_files)} 个IMU样本文件")
    print(f"找到 {len(semg_files)} 个sEMG样本文件")

    # 分析IMU数据
    print("\n" + "=" * 50)
    print("IMU 数据分析")
    print("=" * 50)

    for i, imu_file in enumerate(imu_files):
        print(f"\n--- IMU样本 {i + 1}: {imu_file.name} ---")
        try:
            imu_data = pd.read_csv(imu_file)
            print(f"数据形状: {imu_data.shape}")
            print(f"列名: {list(imu_data.columns)}")
            print(f"数据类型:")
            print(imu_data.dtypes)
            print(f"前3行数据:")
            print(imu_data.head(3))
            print(f"时间戳列信息:")
            time_col = imu_data.columns[0]
            print(f"  列名: {time_col}")
            print(f"  类型: {type(imu_data[time_col].iloc[0])}")
            print(f"  示例值: {imu_data[time_col].iloc[:3].tolist()}")
            print(f"  唯一值数量: {imu_data[time_col].nunique()}")
            print(f"  时间范围: {imu_data[time_col].iloc[0]} 到 {imu_data[time_col].iloc[-1]}")
        except Exception as e:
            print(f"读取错误: {e}")

    # 分析sEMG数据
    print("\n" + "=" * 50)
    print("sEMG 数据分析")
    print("=" * 50)

    for i, semg_file in enumerate(semg_files):
        print(f"\n--- sEMG样本 {i + 1}: {semg_file.name} ---")
        try:
            semg_data = pd.read_csv(semg_file)
            print(f"数据形状: {semg_data.shape}")
            print(f"列名: {list(semg_data.columns)}")
            print(f"数据类型:")
            print(semg_data.dtypes)
            print(f"前3行数据:")
            print(semg_data.head(3))
            print(f"时间戳列信息:")
            time_col = semg_data.columns[0]
            print(f"  列名: {time_col}")
            print(f"  类型: {type(semg_data[time_col].iloc[0])}")
            print(f"  示例值: {semg_data[time_col].iloc[:3].tolist()}")
            print(f"  唯一值数量: {semg_data[time_col].nunique()}")
            print(f"  时间范围: {semg_data[time_col].iloc[0]} 到 {semg_data[time_col].iloc[-1]}")
        except Exception as e:
            print(f"读取错误: {e}")

    # 统计信息
    print("\n" + "=" * 50)
    print("数据统计摘要")
    print("=" * 50)

    if imu_files and semg_files:
        try:
            imu_sample = pd.read_csv(imu_files[0])
            semg_sample = pd.read_csv(semg_files[0])

            print(f"IMU样本:")
            print(f"  总列数: {len(imu_sample.columns)}")
            print(f"  总行数: {len(imu_sample)}")
            print(f"  时间戳格式: {type(imu_sample.iloc[0, 0])}")

            print(f"\nsEMG样本:")
            print(f"  总列数: {len(semg_sample.columns)}")
            print(f"  总行数: {len(semg_sample)}")
            print(f"  时间戳格式: {type(semg_sample.iloc[0, 0])}")

        except Exception as e:
            print(f"统计错误: {e}")


def check_time_format_compatibility(base_path):
    """
    检查时间戳格式兼容性
    """
    base_path = Path(base_path)
    imu_base = base_path / "IMU_data_copy"
    semg_base = base_path / "sEMG_data_copy"

    print("\n" + "=" * 50)
    print("时间戳格式兼容性检查")
    print("=" * 50)

    # 随机选择几对文件检查时间戳格式
    imu_files = list(imu_base.rglob("IMU_*.csv"))[:3]

    for imu_file in imu_files:
        # 找到对应的sEMG文件
        timestamp = imu_file.stem.replace("IMU_", "")
        semg_parent = imu_file.parent
        semg_parent_str = str(semg_parent).replace("IMU_data", "sEMG_data").replace("IMU_", "sEMG_")
        semg_file = Path(semg_parent_str) / f"processed_data_{timestamp}.csv"

        if semg_file.exists():
            print(f"\n检查文件对: {imu_file.name}")
            try:
                imu_data = pd.read_csv(imu_file)
                semg_data = pd.read_csv(semg_file)

                imu_time = imu_data.iloc[:, 0]
                semg_time = semg_data.iloc[:, 0]

                print(f"  IMU时间戳类型: {type(imu_time.iloc[0])}")
                print(f"  sEMG时间戳类型: {type(semg_time.iloc[0])}")
                print(f"  IMU时间示例: {imu_time.iloc[0]} (类型: {type(imu_time.iloc[0])})")
                print(f"  sEMG时间示例: {semg_time.iloc[0]} (类型: {type(semg_time.iloc[0])})")

                # 检查是否可以转换为数值
                try:
                    imu_numeric = pd.to_numeric(imu_time, errors='coerce')
                    semg_numeric = pd.to_numeric(semg_time, errors='coerce')
                    print(f"  可转换为数值: IMU-{not imu_numeric.isna().all()}, sEMG-{not semg_numeric.isna().all()}")
                except:
                    print(f"  数值转换检查失败")

            except Exception as e:
                print(f"  检查错误: {e}")


if __name__ == "__main__":
    base_data_path = "data/interim"

    print("开始数据格式探查...")
    explore_data_structure(base_data_path, sample_count=3)
    check_time_format_compatibility(base_data_path)

    print("\n探查完成！请查看上面的报告了解数据格式。")
