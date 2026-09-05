import pandas as pd
import os
import numpy as np
from pathlib import Path
import re
from datetime import datetime
import shutil


def find_all_aligned_pairs(base_path):
    """
    查找所有对齐的文件对
    """
    base_path = Path(base_path)
    aligned_base = base_path / "aligned_data"

    aligned_pairs = []

    # 查找所有IMU文件
    for imu_file in aligned_base.rglob("aligned_IMU_*.csv"):
        timestamp_match = re.search(r'aligned_IMU_(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})\.csv', imu_file.name)
        if not timestamp_match:
            continue

        timestamp = timestamp_match.group(1)

        # 构建对应的sEMG文件路径
        semg_file = imu_file.parent / f"aligned_processed_data_{timestamp}.csv"

        if semg_file.exists():
            aligned_pairs.append({
                'imu_path': imu_file,
                'semg_path': semg_file,
                'timestamp': timestamp,
                'subject': imu_file.parent.parent.name,
                'activity': imu_file.parent.name,
                'folder': imu_file.parent
            })

    return aligned_pairs


def convert_timestamp(timestamp_str):
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


def check_time_alignment(imu_data, semg_data):
    """
    检查时间对齐情况
    """
    imu_time_col = imu_data.columns[0]  # First_Timestamp_ms
    semg_time_col = semg_data.columns[0]  # Timestamp

    # 转换时间戳
    imu_start = convert_timestamp(imu_data[imu_time_col].iloc[0])
    imu_end = convert_timestamp(imu_data[imu_time_col].iloc[-1])
    semg_start = convert_timestamp(semg_data[semg_time_col].iloc[0])
    semg_end = convert_timestamp(semg_data[semg_time_col].iloc[-1])

    if any(x is None for x in [imu_start, imu_end, semg_start, semg_end]):
        return None, None, None, None, "时间戳转换失败"

    # 计算时间差
    start_diff = (imu_start - semg_start).total_seconds()
    end_diff = (imu_end - semg_end).total_seconds()

    return imu_start, imu_end, semg_start, semg_end, start_diff, end_diff


def realign_semg_to_imu(imu_data, semg_data):
    """
    重新对齐sEMG数据到IMU时间范围
    """
    imu_time_col = imu_data.columns[0]
    semg_time_col = semg_data.columns[0]

    # 添加datetime列
    imu_data = imu_data.copy()
    semg_data = semg_data.copy()

    imu_data['datetime'] = imu_data[imu_time_col].apply(convert_timestamp)
    semg_data['datetime'] = semg_data[semg_time_col].apply(convert_timestamp)

    # 检查转换结果
    if imu_data['datetime'].isna().any() or semg_data['datetime'].isna().any():
        return None, "时间戳转换失败"

    # 获取IMU时间范围
    imu_start = imu_data['datetime'].iloc[0]
    imu_end = imu_data['datetime'].iloc[-1]

    # 对齐sEMG数据
    aligned_semg = semg_data[
        (semg_data['datetime'] >= imu_start) &
        (semg_data['datetime'] <= imu_end)
        ].copy()

    if len(aligned_semg) == 0:
        return None, "对齐后无数据"

    # 移除临时列
    aligned_semg = aligned_semg.drop('datetime', axis=1)

    return aligned_semg, f"成功对齐: {len(aligned_semg)}/{len(semg_data)} 点保留"


def check_and_fix_all_alignment(base_path, tolerance_ms=50):
    """
    检查并修复所有文件的时间对齐
    """
    base_path = Path(base_path)
    aligned_base = base_path / "aligned_data"
    backup_base = base_path / f"aligned_data_backup_{datetime.now():%Y%m%d_%H%M%S}"

    print("=" * 80)
    print("自动时间对齐检查与修复")
    print("=" * 80)

    # 创建备份
    print("创建数据备份...")
    shutil.copytree(aligned_base, backup_base)
    print(f"备份已创建: {backup_base}")

    # 查找所有文件对
    aligned_pairs = find_all_aligned_pairs(base_path)
    print(f"找到 {len(aligned_pairs)} 个文件对需要检查")

    # 统计结果
    results = {
        'perfect': 0,
        'fixed': 0,
        'failed': 0,
        'skipped': 0,
        'details': []
    }

    print(f"\n开始检查并修复时间对齐 (容忍度: {tolerance_ms}ms)...")
    print("=" * 60)

    for i, pair in enumerate(aligned_pairs, 1):
        print(f"\n[{i}/{len(aligned_pairs)}] 处理: {pair['subject']} - {pair['activity']}")
        print(f"时间戳: {pair['timestamp']}")

        try:
            # 读取数据
            imu_data = pd.read_csv(pair['imu_path'])
            semg_data = pd.read_csv(pair['semg_path'])

            if imu_data.empty or semg_data.empty:
                print("  ⚠ 数据为空，跳过")
                results['skipped'] += 1
                results['details'].append({
                    'pair': pair,
                    'status': 'skipped',
                    'reason': '空数据'
                })
                continue

            # 检查时间对齐
            imu_start, imu_end, semg_start, semg_end, start_diff, end_diff = check_time_alignment(imu_data, semg_data)

            if imu_start is None:
                print("  ❌ 时间戳检查失败")
                results['failed'] += 1
                results['details'].append({
                    'pair': pair,
                    'status': 'failed',
                    'reason': '时间戳转换失败'
                })
                continue

            start_diff_ms = abs(start_diff * 1000)
            end_diff_ms = abs(end_diff * 1000)

            print(f"  IMU: {imu_start} → {imu_end} ({len(imu_data)}点)")
            print(f"  sEMG: {semg_start} → {semg_end} ({len(semg_data)}点)")
            print(f"  开始时间差: {start_diff_ms:.1f}ms, 结束时间差: {end_diff_ms:.1f}ms")

            # 判断是否需要修复
            if start_diff_ms <= tolerance_ms and end_diff_ms <= tolerance_ms:
                print("  ✅ 时间对齐良好，无需修复")
                results['perfect'] += 1
                results['details'].append({
                    'pair': pair,
                    'status': 'perfect',
                    'start_diff_ms': start_diff_ms,
                    'end_diff_ms': end_diff_ms
                })
            else:
                print(f"  🔧 时间对齐偏差较大，开始修复...")

                # 重新对齐
                aligned_semg, message = realign_semg_to_imu(imu_data, semg_data)

                if aligned_semg is not None:
                    # 保存修复后的数据
                    aligned_semg.to_csv(pair['semg_path'], index=False)
                    print(f"  ✅ 修复成功: {message}")
                    print(f"     原始: {len(semg_data)}点 → 修复后: {len(aligned_semg)}点")

                    results['fixed'] += 1
                    results['details'].append({
                        'pair': pair,
                        'status': 'fixed',
                        'start_diff_ms': start_diff_ms,
                        'end_diff_ms': end_diff_ms,
                        'original_points': len(semg_data),
                        'aligned_points': len(aligned_semg),
                        'message': message
                    })
                else:
                    print(f"  ❌ 修复失败: {message}")
                    results['failed'] += 1
                    results['details'].append({
                        'pair': pair,
                        'status': 'failed',
                        'reason': message
                    })

        except Exception as e:
            print(f"  ❌ 处理异常: {e}")
            results['failed'] += 1
            results['details'].append({
                'pair': pair,
                'status': 'failed',
                'reason': f'异常: {str(e)}'
            })

    # 生成统计报告
    print(f"\n" + "=" * 60)
    print("修复完成统计")
    print("=" * 60)

    total = len(aligned_pairs)
    print(f"总处理文件对: {total}")
    print(f"✅ 完美对齐: {results['perfect']} ({results['perfect'] / total * 100:.1f}%)")
    print(f"🔧 成功修复: {results['fixed']} ({results['fixed'] / total * 100:.1f}%)")
    print(f"❌ 修复失败: {results['failed']} ({results['failed'] / total * 100:.1f}%)")
    print(f"⚠️  跳过: {results['skipped']} ({results['skipped'] / total * 100:.1f}%)")

    # 显示需要关注的修复情况
    fixed_details = [d for d in results['details'] if d['status'] == 'fixed']
    if fixed_details:
        print(f"\n修复详情 (前10个):")
        for i, detail in enumerate(fixed_details[:10]):
            pair = detail['pair']
            print(f"  {i + 1}. {pair['subject']} - {pair['activity']}")
            print(f"     时间差: {detail['start_diff_ms']:.1f}ms → {detail['end_diff_ms']:.1f}ms")
            print(f"     数据点: {detail['original_points']} → {detail['aligned_points']}")

    # 生成详细报告
    generate_alignment_report(results, base_path)

    return results


def generate_alignment_report(results, base_path):
    """
    生成详细的对齐报告
    """
    report_file = base_path / "time_alignment_report.txt"

    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("时间对齐检查与修复报告\n")
        f.write("=" * 50 + "\n\n")

        f.write(f"总处理文件对: {len(results['details'])}\n")
        f.write(f"完美对齐: {results['perfect']}\n")
        f.write(f"成功修复: {results['fixed']}\n")
        f.write(f"修复失败: {results['failed']}\n")
        f.write(f"跳过: {results['skipped']}\n\n")

        # 详细列表
        f.write("详细处理结果:\n")
        f.write("-" * 50 + "\n")

        for detail in results['details']:
            pair = detail['pair']
            f.write(f"{pair['subject']} - {pair['activity']} - {pair['timestamp']}\n")
            f.write(f"  状态: {detail['status']}\n")

            if detail['status'] == 'perfect':
                f.write(f"  时间差: {detail['start_diff_ms']:.1f}ms, {detail['end_diff_ms']:.1f}ms\n")
            elif detail['status'] == 'fixed':
                f.write(f"  修复前时间差: {detail['start_diff_ms']:.1f}ms, {detail['end_diff_ms']:.1f}ms\n")
                f.write(f"  数据点: {detail['original_points']} → {detail['aligned_points']}\n")
                f.write(f"  消息: {detail['message']}\n")
            elif detail['status'] in ['failed', 'skipped']:
                f.write(f"  原因: {detail['reason']}\n")

            f.write("\n")

    print(f"\n详细报告已生成: {report_file}")


def verify_final_alignment(base_path, tolerance_ms=10):
    """
    验证最终对齐结果
    """
    print(f"\n" + "=" * 60)
    print("最终对齐结果验证")
    print("=" * 60)

    aligned_pairs = find_all_aligned_pairs(base_path)
    perfectly_aligned = 0
    within_tolerance = 0

    for pair in aligned_pairs:
        try:
            imu_data = pd.read_csv(pair['imu_path'])
            semg_data = pd.read_csv(pair['semg_path'])

            imu_start, imu_end, semg_start, semg_end, start_diff, end_diff = check_time_alignment(imu_data, semg_data)

            if imu_start is None:
                continue

            start_diff_ms = abs(start_diff * 1000)
            end_diff_ms = abs(end_diff * 1000)

            if start_diff_ms <= tolerance_ms and end_diff_ms <= tolerance_ms:
                perfectly_aligned += 1
            if start_diff_ms <= 50 and end_diff_ms <= 50:  # 宽松标准
                within_tolerance += 1

        except:
            continue

    print(f"总文件对: {len(aligned_pairs)}")
    print(f"完美对齐 (<{tolerance_ms}ms): {perfectly_aligned} ({perfectly_aligned / len(aligned_pairs) * 100:.1f}%)")
    print(f"容忍范围内 (<50ms): {within_tolerance} ({within_tolerance / len(aligned_pairs) * 100:.1f}%)")

    if perfectly_aligned == len(aligned_pairs):
        print("🎉 所有数据完美对齐！")
    else:
        print("⚠ 仍有部分数据需要手动检查")


if __name__ == "__main__":
    base_data_path = "data"

    print("开始自动时间对齐检查与修复...")

    # 设置容忍度（毫秒）
    tolerance_ms = 50  # 50毫秒内的偏差认为可接受

    # 执行检查与修复
    results = check_and_fix_all_alignment(base_data_path, tolerance_ms)

    # 验证最终结果
    verify_final_alignment(base_data_path)

    print(f"\n" + "=" * 80)
    print("处理完成！")
    print("=" * 80)
    print(f"原始数据备份在: {base_data_path}/aligned_data_backup/")
    print(f"详细报告: {base_data_path}/time_alignment_report.txt")
