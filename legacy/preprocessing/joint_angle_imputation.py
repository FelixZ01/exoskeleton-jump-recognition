import pandas as pd
import numpy as np
from pathlib import Path
import os
from scipy import interpolate
from scipy.signal import savgol_filter


class JointAngleImputer:
    """
    关节角度数据缺失值补充器 - 基于传感器融合和物理约束
    """

    def __init__(self):
        self.joint_angle_cols = []  # 关节角度列

    def detect_joint_angle_columns(self, df):
        """检测关节角度数据列（排除第一列和第二列）"""
        all_cols = df.columns.tolist()

        # 排除第一列（时间戳）和第二列（日志时间戳）
        if len(all_cols) > 2:
            self.joint_angle_cols = all_cols[2:]  # 从第三列开始都是关节角度数据

        print(f"检测到关节角度列: {self.joint_angle_cols}")
        print(f"总关节角度列数: {len(self.joint_angle_cols)}")

    def madgwick_attitude_estimation(self, joint_angles, dt=0.001):
        """
        基于Madgwick滤波器思想的姿态平滑
        用于关节角度的平滑和预测
        """
        n = len(joint_angles)
        if n < 2:
            return joint_angles

        # 简化的互补滤波器实现
        smoothed_angles = np.zeros_like(joint_angles)
        alpha = 0.95  # 互补滤波器系数

        # 初始化
        smoothed_angles[0] = joint_angles[0] if not np.isnan(joint_angles[0]) else 0

        for i in range(1, n):
            if np.isnan(joint_angles[i]):
                # 缺失值，基于运动模型预测
                if i >= 2:
                    # 使用前两点的速度和加速度进行预测
                    v = smoothed_angles[i - 1] - smoothed_angles[i - 2]
                    predicted = smoothed_angles[i - 1] + v
                else:
                    predicted = smoothed_angles[i - 1]
                smoothed_angles[i] = predicted
            else:
                # 有效值，使用互补滤波器
                measured = joint_angles[i]
                predicted = smoothed_angles[i - 1]
                smoothed_angles[i] = alpha * predicted + (1 - alpha) * measured

        return smoothed_angles

    def kalman_smoother_joints(self, data, process_noise=0.01, measurement_noise=0.1):
        """
        专门针对关节角度的卡尔曼平滑器
        关节角度变化相对缓慢，使用较小的噪声参数
        """
        n = len(data)
        if n < 2:
            return data

        # 简化的卡尔曼滤波器实现
        x = data[0] if not np.isnan(data[0]) else 0  # 初始状态
        P = 1.0  # 初始协方差
        Q = process_noise  # 过程噪声（关节角度变化较慢）
        R = measurement_noise  # 测量噪声

        # 前向滤波
        x_forward = np.zeros_like(data)
        P_forward = np.zeros_like(data)
        x_forward[0] = x
        P_forward[0] = P

        for i in range(1, n):
            if np.isnan(data[i]):
                # 缺失值，只进行预测（基于关节运动的连续性）
                x_pred = x
                P_pred = P + Q
                x = x_pred
                P = P_pred
            else:
                # 预测步骤
                x_pred = x
                P_pred = P + Q

                # 更新步骤
                K = P_pred / (P_pred + R)
                x = x_pred + K * (data[i] - x_pred)
                P = (1 - K) * P_pred

            x_forward[i] = x
            P_forward[i] = P

        # 后向平滑 (Rauch–Tung–Striebel smoother)
        x_smooth = np.zeros_like(data)
        x_smooth[-1] = x_forward[-1]

        for i in range(n - 2, -1, -1):
            C = P_forward[i] / (P_forward[i] + Q)
            x_smooth[i] = x_forward[i] + C * (x_smooth[i + 1] - x_forward[i])

        return x_smooth

    def physical_constrained_interpolation(self, data, mask, max_angular_velocity=2.0, max_angular_accel=5.0):
        """
        基于物理约束的关节角度插值
        考虑人体关节运动限制
        """
        n = len(data)
        indices = np.arange(n)
        valid_indices = indices[~mask]
        missing_indices = indices[mask]

        if len(valid_indices) < 2:
            return data

        # 使用样条插值
        if len(valid_indices) >= 4:
            try:
                # 使用三次样条插值，提供更平滑的关节运动
                spline = interpolate.CubicSpline(valid_indices, data[valid_indices])
                interpolated = spline(missing_indices)
            except:
                # 回退到线性插值
                interpolated = np.interp(missing_indices, valid_indices, data[valid_indices])
        else:
            # 使用线性插值
            interpolated = np.interp(missing_indices, valid_indices, data[valid_indices])

        # 应用物理约束
        result = data.copy()
        result[missing_indices] = interpolated

        # 角速度约束 (人体关节角速度限制)
        angular_velocity = np.diff(result)
        if np.any(np.abs(angular_velocity) > max_angular_velocity):
            print(f"    应用角速度约束: 最大角速度 {np.max(np.abs(angular_velocity)):.2f} rad/s")
            angular_velocity_clipped = np.clip(angular_velocity, -max_angular_velocity, max_angular_velocity)
            # 重新积分
            result[1:] = result[0] + np.cumsum(angular_velocity_clipped)

        # 角加速度约束
        angular_acceleration = np.diff(angular_velocity)
        if np.any(np.abs(angular_acceleration) > max_angular_accel):
            print(f"    应用角加速度约束: 最大角加速度 {np.max(np.abs(angular_acceleration)):.2f} rad/s²")
            # 应用Savitzky-Golay滤波器平滑
            window_length = min(9, len(result))
            if window_length % 2 == 0:
                window_length -= 1
            if window_length >= 3:
                try:
                    result = savgol_filter(result, window_length, 3)  # 使用3阶多项式
                except:
                    pass

        return result

    def kinematic_model_extrapolation(self, data_before, data_after, gap_length, dt=0.001):
        """
        基于运动学模型的外推
        使用匀角加速度模型
        """
        if len(data_before) < 2 or len(data_after) < 2:
            return np.linspace(data_before[-1] if len(data_before) > 0 else 0,
                               data_after[0] if len(data_after) > 0 else 0, gap_length)

        # 计算前后的角速度和角加速度
        omega_before = (data_before[-1] - data_before[-2]) / dt  # 角速度
        omega_after = (data_after[1] - data_after[0]) / dt  # 角速度

        # 使用匀角加速度模型
        alpha = (omega_after - omega_before) / ((len(data_before) + gap_length + len(data_after)) * dt)

        # 生成外推序列
        extrapolated = []
        current_angle = data_before[-1]
        current_omega = omega_before

        for i in range(gap_length):
            # 匀角加速度运动方程
            current_omega = current_omega + alpha * dt
            current_angle = current_angle + current_omega * dt
            extrapolated.append(current_angle)

        return np.array(extrapolated)

    def impute_joint_angles(self, df, sampling_rate=1000):
        """
        主函数：补充关节角度缺失数据
        """
        print("开始关节角度数据缺失值补充...")

        # 检测关节角度列（排除第一列和第二列）
        self.detect_joint_angle_columns(df)

        if not self.joint_angle_cols:
            print("⚠ 未检测到关节角度列")
            return df

        # 创建结果DataFrame
        result_df = df.copy()
        dt = 1.0 / sampling_rate

        print(f"\n处理关节角度数据...")

        for col in self.joint_angle_cols:
            if col not in df.columns:
                continue

            data = df[col].values

            # 检测缺失值
            mask = df[col].isna()
            missing_count = mask.sum()

            if missing_count == 0:
                print(f"  {col}: 无缺失值")
                continue

            print(f"  {col}: 缺失 {missing_count}/{len(data)} 个值 ({missing_count / len(data) * 100:.1f}%)")

            # 根据缺失比例选择方法
            missing_ratio = missing_count / len(data)

            if missing_ratio < 0.1:  # 少量缺失
                # 使用Madgwick思想的姿态平滑
                print(f"    使用传感器融合平滑...")
                imputed_data = self.madgwick_attitude_estimation(data, dt)

            elif missing_ratio < 0.3:  # 中等缺失
                # 使用卡尔曼平滑器
                print(f"    使用卡尔曼平滑器...")
                imputed_data = self.kalman_smoother_joints(data)

            elif missing_ratio < 0.6:  # 较多缺失
                # 使用物理约束插值
                print(f"    使用物理约束插值...")
                imputed_data = self.physical_constrained_interpolation(data, mask.values)

            else:  # 大量缺失
                # 使用基于运动学模型的外推
                print(f"    使用运动学模型外推...")

                # 找到连续缺失段
                missing_regions = []
                in_region = False
                start_idx = 0

                for i, is_missing in enumerate(mask):
                    if is_missing and not in_region:
                        start_idx = i
                        in_region = True
                    elif not is_missing and in_region:
                        missing_regions.append((start_idx, i - 1))
                        in_region = False

                if in_region:
                    missing_regions.append((start_idx, len(mask) - 1))

                imputed_data = data.copy()
                for start, end in missing_regions:
                    gap_length = end - start + 1
                    data_before = data[max(0, start - 5):start]
                    data_after = data[end + 1:min(len(data), end + 6)]

                    if len(data_before) > 0 and len(data_after) > 0:
                        # 移除NaN值
                        data_before = data_before[~np.isnan(data_before)]
                        data_after = data_after[~np.isnan(data_after)]

                        if len(data_before) >= 2 and len(data_after) >= 2:
                            extrapolated = self.kinematic_model_extrapolation(data_before, data_after, gap_length, dt)
                            imputed_data[start:end + 1] = extrapolated
                            print(f"      区域 [{start}:{end}] 外推完成")

            result_df[col] = imputed_data

        return result_df

    def validate_imputation(self, original_df, imputed_df):
        """
        验证插值结果
        """
        print("\n" + "=" * 50)
        print("关节角度插值结果验证")
        print("=" * 50)

        # 只统计关节角度列的缺失值
        joint_cols = original_df.columns[2:] if len(original_df.columns) > 2 else []

        original_missing = original_df[joint_cols].isna().sum().sum()
        imputed_missing = imputed_df[joint_cols].isna().sum().sum()

        print(f"原始缺失值数量: {original_missing}")
        print(f"插值后缺失值数量: {imputed_missing}")

        if original_missing > 0:
            completion_rate = (1 - imputed_missing / original_missing) * 100
            print(f"插值完成率: {completion_rate:.1f}%")

            if completion_rate < 100:
                print("⚠ 仍有部分缺失值未填充")

        # 统计每列的插值情况
        for col in joint_cols:
            if original_df[col].isna().any():
                orig_missing = original_df[col].isna().sum()
                imp_missing = imputed_df[col].isna().sum()
                print(f"  {col}: {orig_missing} → {imp_missing} 缺失值")


def process_joint_angle_data(base_path):
    """
    处理所有对齐后的关节角度数据
    """
    base_path = Path(base_path)

    print("=" * 80)
    print("关节角度数据缺失值补充处理")
    print("=" * 80)

    # 查找IMU文件
    imu_files = []
    search_paths = [
        base_path / "aligned_IMU",
        base_path,
        base_path.parent / "aligned_data_strict" / "aligned_IMU"
    ]

    for path in search_paths:
        if path.exists():
            found_files = list(path.rglob("*.csv"))
            print(f"在 {path} 找到 {len(found_files)} 个CSV文件")
            imu_files.extend(found_files)

    # 去重
    imu_files = list(set(imu_files))

    if not imu_files:
        print("❌ 未找到任何CSV文件")
        return None

    print(f"找到 {len(imu_files)} 个IMU文件需要处理")

    # 创建输出目录
    output_dir = base_path.parent / "imputed_joint_angles"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 初始化插值器
    imputer = JointAngleImputer()

    results = {
        'success': 0,
        'failed': 0,
        'skipped': 0,
        'total_files': len(imu_files),
        'details': []
    }

    print(f"\n开始处理关节角度数据...")
    print("=" * 60)

    for i, imu_file in enumerate(imu_files, 1):
        print(f"\n[{i}/{len(imu_files)}] 处理: {imu_file.name}")

        try:
            # 读取数据
            df = pd.read_csv(imu_file)
            print(f"  数据形状: {df.shape}")
            print(f"  列名: {list(df.columns)}")

            # 检查关节角度列是否有缺失值
            joint_cols = df.columns[2:] if len(df.columns) > 2 else []
            joint_missing = df[joint_cols].isna().sum().sum()

            if joint_missing == 0:
                print("  ⚠ 关节角度无缺失值，跳过")
                results['skipped'] += 1
                results['details'].append({
                    'file': imu_file.name,
                    'status': 'skipped',
                    'reason': '关节角度无缺失值'
                })
                continue

            print(f"  关节角度缺失值: {joint_missing}")

            # 补充关节角度缺失值
            imputed_df = imputer.impute_joint_angles(df)

            # 验证结果
            imputer.validate_imputation(df, imputed_df)

            # 保存结果（保持第一列和第二列不变）
            relative_path = imu_file.relative_to(
                imu_file.parent.parent.parent) if imu_file.parent.parent.parent.exists() else imu_file.name
            output_path = output_dir / relative_path
            output_path.parent.mkdir(parents=True, exist_ok=True)
            imputed_df.to_csv(output_path, index=False)

            print(f"  ✅ 处理完成，保存至: {output_path}")
            results['success'] += 1
            results['details'].append({
                'file': imu_file.name,
                'status': 'success',
                'original_missing': joint_missing,
                'final_missing': imputed_df[joint_cols].isna().sum().sum()
            })

        except Exception as e:
            print(f"  ❌ 处理失败: {e}")
            results['failed'] += 1
            results['details'].append({
                'file': imu_file.name,
                'status': 'failed',
                'error': str(e)
            })

    # 生成报告
    print(f"\n" + "=" * 60)
    print("处理完成统计")
    print("=" * 60)

    total = results['total_files']
    print(f"总处理文件: {total}")
    print(f"✅ 成功处理: {results['success']}")
    print(f"⚠️  跳过: {results['skipped']}")
    print(f"❌ 处理失败: {results['failed']}")

    if total > 0:
        success_rate = results['success'] / total * 100
        print(f"成功率: {success_rate:.1f}%")

    # 保存详细报告
    generate_joint_imputation_report(results, base_path.parent)

    return results


def generate_joint_imputation_report(results, base_path):
    """生成关节角度插值报告"""
    report_file = base_path / "joint_angle_imputation_report.txt"

    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("关节角度数据缺失值补充报告\n")
        f.write("=" * 50 + "\n\n")

        f.write(f"总处理文件: {results['total_files']}\n")
        f.write(f"成功处理: {results['success']}\n")
        f.write(f"跳过: {results['skipped']}\n")
        f.write(f"处理失败: {results['failed']}\n\n")

        f.write("详细处理结果:\n")
        f.write("-" * 50 + "\n")

        for detail in results['details']:
            f.write(f"文件: {detail['file']}\n")
            f.write(f"状态: {detail['status']}\n")
            if detail['status'] == 'success':
                f.write(f"缺失值: {detail['original_missing']} → {detail['final_missing']}\n")
            elif detail['status'] == 'skipped':
                f.write(f"原因: {detail['reason']}\n")
            else:
                f.write(f"错误: {detail['error']}\n")
            f.write("\n")

    print(f"详细报告已生成: {report_file}")


if __name__ == "__main__":
    base_data_path = "data/interim/aligned_data_strict"

    print("开始关节角度数据缺失值补充处理...")
    results = process_joint_angle_data(base_data_path)

    if results:
        print(f"\n" + "=" * 80)
        print("处理完成！")
        print("=" + "=" * 79)
        print(f"补充后的关节角度数据保存在: {Path(base_data_path).parent}/imputed_joint_angles/")
        print(f"报告文件: {Path(base_data_path).parent}/joint_angle_imputation_report.txt")
    else:
        print("\n❌ 处理失败，请检查路径和文件")
