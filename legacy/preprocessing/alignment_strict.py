import pandas as pd
import os
import numpy as np
from pathlib import Path
import re
from datetime import datetime, timedelta


def find_all_data_pairs(base_path):
    """
    查找所有IMU和sEMG文件对
    """
    base_path = Path(base_path)
    imu_base = base_path / "IMU_data_copy"
    semg_base = base_path / "sEMG_data_copy"

    file_pairs = []

    # 遍历所有IMU文件
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
                'timestamp': timestamp,
                'subject': imu_file.parent.parent.name,
                'activity': imu_file.parent.name
            })

    return file_pairs


def convert_timestamp_to_datetime(timestamp_str):
    """
    将时间戳字符串转换为datetime对象
    """
    try:
        if '.' in timestamp_str:
            main_part, millis_part = timestamp_str.split('.')
            millis_part = millis_part.ljust(3, '0')[:3]
            timestamp_str = f"{main_part}.{millis_part}"
        return pd.to_datetime(timestamp_str, format='%Y-%m-%d %H:%M:%S.%f')
    except Exception as e:
        print(f"时间戳转换错误: {timestamp_str}, 错误: {e}")
        return None


def parse_log_timestamp(log_timestamp_str):
    """
    解析第二列日志时间戳格式: 时:分:秒:毫秒
    例如: 3:5:52:291
    """
    try:
        parts = log_timestamp_str.split(':')
        if len(parts) == 4:
            hours = int(parts[0])
            minutes = int(parts[1])
            seconds = int(parts[2])
            milliseconds = int(parts[3])

            # 转换为总毫秒数
            total_ms = (hours * 3600 + minutes * 60 + seconds) * 1000 + milliseconds
            return total_ms
        else:
            return None
    except:
        return None


def align_first_row_by_truncation(imu_data, semg_data):
    """
    通过删除前面数据来对齐第一行时间戳
    """
    imu_first_time = convert_timestamp_to_datetime(imu_data.iloc[0, 0])
    semg_first_time = convert_timestamp_to_datetime(semg_data.iloc[0, 0])

    if imu_first_time is None or semg_first_time is None:
        return None, None, "时间戳转换失败"

    # 计算时间差（毫秒）
    time_diff_ms = (imu_first_time - semg_first_time).total_seconds() * 1000

    aligned_imu = imu_data.copy()
    aligned_semg = semg_data.copy()

    if time_diff_ms > 0:  # IMU开始时间晚于sEMG
        # 删除sEMG前面的数据
        delete_rows = int(time_diff_ms)
        if delete_rows < len(semg_data):
            aligned_semg = semg_data.iloc[delete_rows:].reset_index(drop=True)
            print(f"  IMU开始时间晚于sEMG {time_diff_ms:.1f}ms，删除sEMG前{delete_rows}行")
            print(f"  sEMG: {len(semg_data)} → {len(aligned_semg)} 行")
        else:
            return None, None, f"sEMG数据不足，需要删除{delete_rows}行但只有{len(semg_data)}行"

    elif time_diff_ms < 0:  # IMU开始时间早于sEMG
        # 删除IMU前面的数据
        delete_rows = int(-time_diff_ms)
        if delete_rows < len(imu_data):
            aligned_imu = imu_data.iloc[delete_rows:].reset_index(drop=True)
            print(f"  IMU开始时间早于sEMG {-time_diff_ms:.1f}ms，删除IMU前{delete_rows}行")
            print(f"  IMU: {len(imu_data)} → {len(aligned_imu)} 行")
        else:
            return None, None, f"IMU数据不足，需要删除{delete_rows}行但只有{len(imu_data)}行"

    else:  # 开始时间相同
        print("  ✅ 第一行时间戳已对齐")

    # 验证对齐结果
    new_imu_first = convert_timestamp_to_datetime(aligned_imu.iloc[0, 0])
    new_semg_first = convert_timestamp_to_datetime(aligned_semg.iloc[0, 0])

    if new_imu_first and new_semg_first:
        new_diff = abs((new_imu_first - new_semg_first).total_seconds() * 1000)
        print(f"  对齐后第一行时间差: {new_diff:.1f}ms")
        if new_diff > 1:
            print(f"  ⚠ 第一行时间差仍然较大")

    return aligned_imu, aligned_semg, f"原始时间差: {time_diff_ms:.1f}ms"


def detect_and_fill_imu_gaps_precisely(imu_data, semg_first_time):
    """
    基于第二列日志时间戳精确检测断点并插入空白行
    使用sEMG的第一列时间戳作为基准
    """
    if len(imu_data) < 2:
        return imu_data, 0

    # 使用第二列日志时间戳检测断点
    log_time_col = imu_data.columns[1]  # 第二列是日志时间戳
    first_time_col = imu_data.columns[0]  # 第一列是时间戳

    expanded_rows = []
    total_gap_rows = 0

    # 第一行使用sEMG的第一列时间戳
    first_row = imu_data.iloc[0].copy()
    first_row[first_time_col] = semg_first_time.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
    expanded_rows.append(first_row.tolist())

    previous_log_ms = parse_log_timestamp(imu_data[log_time_col].iloc[0])
    current_time = semg_first_time

    for i in range(1, len(imu_data)):
        current_log_str = imu_data[log_time_col].iloc[i]
        current_log_ms = parse_log_timestamp(current_log_str)

        if previous_log_ms is None or current_log_ms is None:
            # 如果时间戳解析失败，直接添加当前行
            current_time = current_time + timedelta(milliseconds=1)
            current_row = imu_data.iloc[i].copy()
            current_row[first_time_col] = current_time.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
            expanded_rows.append(current_row.tolist())
            previous_log_ms = current_log_ms
            continue

        # 计算时间差（毫秒）
        time_diff_ms = current_log_ms - previous_log_ms

        # 如果时间差大于1ms，认为是断点
        if time_diff_ms > 1:
            gap_rows = time_diff_ms - 1  # 减去正常的1ms间隔

            print(f"    检测到断点: 位置 {i}")
            print(f"      前一行: {imu_data[log_time_col].iloc[i - 1]} ({previous_log_ms}ms)")
            print(f"      当前行: {current_log_str} ({current_log_ms}ms)")
            print(f"      时间差: {time_diff_ms}ms, 需要插入 {gap_rows} 行")

            total_gap_rows += gap_rows

            # 在断点处插入空白行，时间戳严格递增
            for j in range(gap_rows):
                blank_row = [None] * len(imu_data.columns)
                current_time = current_time + timedelta(milliseconds=1)
                blank_row[0] = current_time.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
                expanded_rows.append(blank_row)

        # 添加当前行，时间戳严格递增
        current_time = current_time + timedelta(milliseconds=1)
        current_row = imu_data.iloc[i].copy()
        current_row[first_time_col] = current_time.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        expanded_rows.append(current_row.tolist())
        previous_log_ms = current_log_ms

    expanded_imu = pd.DataFrame(expanded_rows, columns=imu_data.columns)
    return expanded_imu, total_gap_rows


def match_data_length_strict(imu_data, semg_data):
    """
    严格匹配IMU和sEMG的数据长度，使用sEMG的时间戳
    """
    if len(imu_data) == len(semg_data):
        return imu_data, semg_data, "长度已匹配"

    print(f"  长度不匹配: IMU={len(imu_data)}, sEMG={len(semg_data)}")

    # 以sEMG的时间戳为基准
    aligned_imu = imu_data.copy()

    if len(imu_data) < len(semg_data):
        # IMU较短，在后面补空白行，使用sEMG对应位置的时间戳
        additional_rows = len(semg_data) - len(imu_data)
        print(f"  IMU数据较短，需要在后面添加 {additional_rows} 行空白数据")

        additional_data = []
        for i in range(additional_rows):
            blank_row = [None] * len(imu_data.columns)
            # 使用sEMG对应位置的时间戳
            semg_time_str = semg_data.iloc[len(imu_data) + i, 0]
            blank_row[0] = semg_time_str
            additional_data.append(blank_row)

        additional_df = pd.DataFrame(additional_data, columns=imu_data.columns)
        final_imu = pd.concat([aligned_imu, additional_df], ignore_index=True)
        final_semg = semg_data

    else:
        # IMU较长，截断IMU数据
        print(f"  IMU数据较长，需要截断 {len(imu_data) - len(semg_data)} 行")
        final_imu = aligned_imu.head(len(semg_data))
        final_semg = semg_data

    print(f"  最终长度: IMU={len(final_imu)}, sEMG={len(final_semg)}")

    # 验证时间戳对齐
    for i in range(min(3, len(final_imu))):  # 检查前3行
        imu_time = final_imu.iloc[i, 0]
        semg_time = final_semg.iloc[i, 0]
        if imu_time != semg_time:
            print(f"  ⚠ 第{i + 1}行时间戳不匹配: IMU={imu_time}, sEMG={semg_time}")

    return final_imu, final_semg, f"调整了{abs(len(imu_data) - len(semg_data))}行"


def process_imu_semg_with_strict_alignment(imu_path, semg_path, output_dir):
    """
    严格时间戳对齐的完整流程
    """
    print(f"处理: {os.path.basename(imu_path)}")

    try:
        # 读取数据
        imu_data = pd.read_csv(imu_path)
        semg_data = pd.read_csv(semg_path)

        if imu_data.empty or semg_data.empty:
            print("  ⚠ 数据为空，跳过")
            return False

        print(f"  原始IMU数据: {len(imu_data)} 行")
        print(f"  原始sEMG数据: {len(semg_data)} 行")

        # 步骤1: 通过删除前面数据对齐第一行时间戳
        print("  步骤1: 对齐第一行时间戳...")
        aligned_imu, aligned_semg, align_info = align_first_row_by_truncation(imu_data, semg_data)

        if aligned_imu is None:
            print(f"  ❌ 第一行对齐失败: {align_info}")
            return False

        print(f"  第一行对齐后:")
        print(f"    IMU: {len(aligned_imu)} 行")
        print(f"    sEMG: {len(aligned_semg)} 行")

        # 获取sEMG的第一列时间戳作为基准
        semg_first_time = convert_timestamp_to_datetime(aligned_semg.iloc[0, 0])
        if semg_first_time is None:
            print("  ❌ 无法获取sEMG基准时间戳")
            return False

        # 步骤2: 精确检测并插入IMU断点，使用sEMG时间戳基准
        print("  步骤2: 精确检测IMU断点并插入空白行...")
        imu_with_gaps_filled, total_gap_rows = detect_and_fill_imu_gaps_precisely(aligned_imu, semg_first_time)
        print(f"    断点插入完成: 插入了 {total_gap_rows} 行空白数据")
        print(f"    插入后IMU: {len(imu_with_gaps_filled)} 行")

        # 步骤3: 严格匹配数据长度，使用sEMG时间戳
        print("  步骤3: 严格匹配数据长度...")
        final_imu, final_semg, length_info = match_data_length_strict(imu_with_gaps_filled, aligned_semg)

        print(f"  最终数据:")
        print(f"    IMU: {len(final_imu)} 行")
        print(f"    sEMG: {len(final_semg)} 行")
        print(f"    长度匹配: {'✅' if len(final_imu) == len(final_semg) else '❌'}")

        # 验证最终时间戳对齐
        perfect_alignment = True
        for i in range(min(5, len(final_imu))):  # 检查前5行
            imu_time = final_imu.iloc[i, 0]
            semg_time = final_semg.iloc[i, 0]
            if imu_time != semg_time:
                print(f"    ⚠ 第{i + 1}行时间戳不匹配: IMU={imu_time}, sEMG={semg_time}")
                perfect_alignment = False

        if perfect_alignment:
            print(f"    ✅ 前5行时间戳严格对齐")

        # 保存对齐后的数据
        output_path_imu = output_dir / "aligned_IMU" / imu_path.relative_to(imu_path.parent.parent.parent.parent)
        output_path_semg = output_dir / "aligned_sEMG" / semg_path.relative_to(semg_path.parent.parent.parent.parent)

        # 确保输出目录存在
        output_path_imu.parent.mkdir(parents=True, exist_ok=True)
        output_path_semg.parent.mkdir(parents=True, exist_ok=True)

        # 保存数据
        final_imu.to_csv(output_path_imu, index=False)
        final_semg.to_csv(output_path_semg, index=False)

        print(f"  ✅ 处理完成")
        print(f"    IMU保存至: {output_path_imu}")
        print(f"    sEMG保存至: {output_path_semg}")

        return True

    except Exception as e:
        print(f"  ❌ 处理失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def process_all_pairs_with_strict_alignment(base_path):
    """
    处理所有数据对，采用严格时间戳对齐策略
    """
    base_path = Path(base_path)
    output_dir = base_path / "aligned_data_strict"

    print("=" * 80)
    print("严格时间戳对齐流程")
    print("=" + "=" * 79)

    # 查找所有文件对
    file_pairs = find_all_data_pairs(base_path)
    print(f"找到 {len(file_pairs)} 个文件对需要处理")

    # 统计结果
    results = {
        'success': 0,
        'failed': 0,
        'total_gap_rows': 0,
        'details': []
    }

    print(f"\n开始处理...")
    print("=" * 60)

    for i, pair in enumerate(file_pairs, 1):
        print(f"\n[{i}/{len(file_pairs)}] {pair['subject']} - {pair['activity']}")

        success = process_imu_semg_with_strict_alignment(
            pair['imu_path'],
            pair['semg_path'],
            output_dir
        )

        if success:
            results['success'] += 1
        else:
            results['failed'] += 1

        results['details'].append({
            'pair': pair,
            'success': success
        })

    # 生成报告
    print(f"\n" + "=" * 60)
    print("处理完成统计")
    print("=" * 60)

    total = len(file_pairs)
    print(f"总处理文件对: {total}")
    print(f"✅ 成功处理: {results['success']} ({results['success'] / total * 100:.1f}%)")
    print(f"❌ 处理失败: {results['failed']} ({results['failed'] / total * 100:.1f}%)")

    return results


if __name__ == "__main__":
    base_data_path = "data/interim"

    print("开始严格时间戳对齐流程...")
    results = process_all_pairs_with_strict_alignment(base_data_path)

    print(f"\n" + "=" * 80)
    print("处理完成！")
    print("=" + "=" * 79)
    print(f"对齐后的数据保存在: {base_data_path}/aligned_data_strict/")
    print(f"包含:")
    print(f"  - aligned_IMU/: 严格时间戳对齐的IMU数据")
    print(f"  - aligned_sEMG/: 对应的时间戳对齐sEMG数据")
