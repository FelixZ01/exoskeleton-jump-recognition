import pandas as pd
import numpy as np
from pathlib import Path
import os
from scipy import interpolate
from scipy.signal import savgol_filter
from sklearn.linear_model import RANSACRegressor
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')


class JointAngleImputer:
    """
    关节角度数据缺失值补充器 - 专门针对关节角度数据的优化版本
    """

    def __init__(self, sampling_rate=1000):
        self.joint_angle_cols = []
        self.sampling_rate = sampling_rate
        self.dt = 1.0 / sampling_rate

        # 关节运动生物力学约束
        self.joint_limits = {
            'R1_Roll': (-30, 30),  # 髋关节内外旋
            'R1_Pitch': (-20, 120),  # 髋关节屈伸
            'R1_Yaw': (-30, 30),  # 髋关节内收外展
            'R2_Roll': (-10, 10),  # 膝关节内外旋
            'R2_Pitch': (0, 140),  # 膝关节屈伸
            'R2_Yaw': (-10, 10),  # 膝关节内外翻
            'R3_Roll': (-20, 20),  # 踝关节内外翻
            'R3_Pitch': (-30, 50),  # 踝关节背屈跖屈
            'R3_Yaw': (-10, 10),
            'R4_Roll': (-20, 20),
            'R4_Pitch': (-30, 50),
            'R4_Yaw': (-10, 10)
        }

        # 关节最大角速度和角加速度（基于生物力学研究）
        self.max_angular_velocity = 6.0  # rad/s
        self.max_angular_acceleration = 50.0  # rad/s²

    def detect_joint_angle_columns(self, df):
        """检测关节角度数据列"""
        all_cols = df.columns.tolist()

        if len(all_cols) > 2:
            self.joint_angle_cols = all_cols[2:]

        return self.joint_angle_cols

    def biomechanical_constrained_interpolation(self, data, mask):
        """
        基于生物力学约束的关节角度插值
        专门针对人体关节运动特性优化
        """
        n = len(data)
        indices = np.arange(n)
        valid_indices = indices[~mask]
        missing_indices = indices[mask]

        if len(valid_indices) < 2:
            return data

        # 1. 使用Akima样条插值（更适合关节角度数据）
        try:
            akima_spline = interpolate.Akima1DInterpolator(valid_indices, data[valid_indices])
            interpolated = akima_spline(missing_indices)
        except:
            # 回退到三次样条
            try:
                cubic_spline = interpolate.CubicSpline(valid_indices, data[valid_indices])
                interpolated = cubic_spline(missing_indices)
            except:
                # 最终回退到线性插值
                interpolated = np.interp(missing_indices, valid_indices, data[valid_indices])

        result = data.copy()
        result[missing_indices] = interpolated

        # 2. 应用生物力学约束
        result = self.apply_biomechanical_constraints(result)

        return result

    def apply_biomechanical_constraints(self, data):
        """
        应用生物力学约束
        """
        # 角速度约束
        angular_velocity = np.diff(data) / self.dt

        if np.any(np.abs(angular_velocity) > self.max_angular_velocity):
            # 使用限幅滤波
            angular_velocity_clipped = np.clip(angular_velocity,
                                               -self.max_angular_velocity,
                                               self.max_angular_velocity)
            # 重新积分
            data_constrained = np.zeros_like(data)
            data_constrained[0] = data[0]
            for i in range(1, len(data)):
                data_constrained[i] = data_constrained[i - 1] + angular_velocity_clipped[i - 1] * self.dt
            data = data_constrained

        # 角加速度约束
        angular_acceleration = np.diff(angular_velocity) / self.dt

        if np.any(np.abs(angular_acceleration) > self.max_angular_acceleration):
            # 使用Savitzky-Golay滤波器平滑
            window_length = min(11, len(data))
            if window_length % 2 == 0:
                window_length -= 1
            if window_length >= 5:
                try:
                    data = savgol_filter(data, window_length, 3)
                except:
                    pass

        return data

    def kinematic_model_prediction(self, data_before, data_after, gap_length):
        """
        基于运动学模型预测缺失段
        使用匀角加速度模型 + RANSAC鲁棒回归
        """
        if len(data_before) < 3 or len(data_after) < 3:
            # 数据不足，使用线性插值
            return np.linspace(data_before[-1], data_after[0], gap_length)

        # 1. 使用RANSAC回归估计运动趋势
        X_before = np.arange(len(data_before)).reshape(-1, 1)
        X_after = np.arange(len(data_after)).reshape(-1, 1) + len(data_before) + gap_length

        # 合并前后数据用于RANSAC
        X_combined = np.vstack([X_before, X_after])
        y_combined = np.hstack([data_before, data_after])

        try:
            ransac = RANSACRegressor(random_state=42)
            ransac.fit(X_combined, y_combined)

            # 预测缺失段
            X_missing = np.arange(len(data_before), len(data_before) + gap_length).reshape(-1, 1)
            predicted = ransac.predict(X_missing)

        except:
            # RANSAC失败，使用多项式拟合
            poly_degree = min(2, len(data_before) - 1, len(data_after) - 1)
            coeffs_before = np.polyfit(np.arange(len(data_before)), data_before, poly_degree)
            coeffs_after = np.polyfit(np.arange(len(data_after)), data_after, poly_degree)

            poly_before = np.poly1d(coeffs_before)
            poly_after = np.poly1d(coeffs_after)

            # 前后预测的平均
            pred_before = poly_before(np.arange(len(data_before), len(data_before) + gap_length))
            pred_after = poly_after(np.arange(-gap_length, 0))

            # 加权平均（靠近哪边权重高）
            weights = np.linspace(1, 0, gap_length)
            predicted = weights * pred_before + (1 - weights) * pred_after

        return predicted

    def adaptive_kalman_smoothing(self, data, mask):
        """
        自适应卡尔曼平滑 - 专门针对关节角度数据优化
        """
        n = len(data)
        if n < 3:
            return data

        # 自适应噪声参数
        valid_data = data[~mask]
        if len(valid_data) > 1:
            measurement_noise = np.std(valid_data) * 0.1
        else:
            measurement_noise = 1.0

        process_noise = 0.01  # 关节角度变化相对缓慢

        # 简化的卡尔曼滤波器实现
        x = data[0] if not np.isnan(data[0]) else 0
        P = 1.0
        Q = process_noise
        R = measurement_noise

        # 前向滤波
        x_forward = np.zeros_like(data)
        P_forward = np.zeros_like(data)
        x_forward[0] = x
        P_forward[0] = P

        for i in range(1, n):
            if np.isnan(data[i]):
                # 缺失值，只进行预测
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

        # 后向平滑
        x_smooth = np.zeros_like(data)
        x_smooth[-1] = x_forward[-1]

        for i in range(n - 2, -1, -1):
            C = P_forward[i] / (P_forward[i] + Q)
            x_smooth[i] = x_forward[i] + C * (x_smooth[i + 1] - x_forward[i])

        return x_smooth

    def find_continuous_gaps(self, mask):
        """找到连续的缺失段"""
        gaps = []
        in_gap = False
        start_idx = 0

        for i, is_missing in enumerate(mask):
            if is_missing and not in_gap:
                start_idx = i
                in_gap = True
            elif not is_missing and in_gap:
                gaps.append((start_idx, i - 1))
                in_gap = False

        if in_gap:
            gaps.append((start_idx, len(mask) - 1))

        return gaps

    def handle_continuous_gaps(self, data, mask):
        """处理连续缺失段"""
        gaps = self.find_continuous_gaps(mask)
        result = data.copy()

        for start, end in gaps:
            gap_length = end - start + 1

            # 获取缺失段前后的有效数据
            before_start = max(0, start - min(10, start))
            after_end = min(len(data), end + min(10, len(data) - end - 1))

            data_before = data[before_start:start]
            data_after = data[end + 1:after_end + 1]

            # 移除NaN值
            data_before = data_before[~np.isnan(data_before)]
            data_after = data_after[~np.isnan(data_after)]

            if len(data_before) >= 2 and len(data_after) >= 2:
                # 使用运动学模型预测
                predicted = self.kinematic_model_prediction(data_before, data_after, gap_length)
                result[start:end + 1] = predicted
            else:
                # 数据不足，使用样条插值
                valid_indices = np.where(~mask)[0]
                if len(valid_indices) >= 2:
                    interpolated = np.interp(range(start, end + 1), valid_indices, data[valid_indices])
                    result[start:end + 1] = interpolated

        return result

    def impute_joint_angles(self, df):
        """
        主函数：补充关节角度缺失数据
        """
        # 检测关节角度列
        self.detect_joint_angle_columns(df)

        if not self.joint_angle_cols:
            return df

        result_df = df.copy()

        for col in self.joint_angle_cols:
            if col not in df.columns:
                continue

            data = df[col].values
            mask = df[col].isna()
            missing_count = mask.sum()

            if missing_count == 0:
                continue

            missing_ratio = missing_count / len(data)

            # 根据缺失比例选择最佳方法
            if missing_ratio < 0.1:
                # 少量缺失：使用生物力学约束插值
                imputed_data = self.biomechanical_constrained_interpolation(data, mask.values)

            elif missing_ratio < 0.3:
                # 中等缺失：使用自适应卡尔曼平滑
                imputed_data = self.adaptive_kalman_smoothing(data, mask.values)

            elif missing_ratio < 0.6:
                # 较多缺失：处理连续缺失段 + 生物力学约束
                imputed_data = self.handle_continuous_gaps(data, mask.values)
                imputed_data = self.biomechanical_constrained_interpolation(imputed_data, mask.values)

            else:
                # 大量缺失：综合方法
                # 先处理连续缺失段
                imputed_data = self.handle_continuous_gaps(data, mask.values)
                # 再用卡尔曼平滑
                imputed_data = self.adaptive_kalman_smoothing(imputed_data, mask.values)
                # 最后应用生物力学约束
                imputed_data = self.apply_biomechanical_constraints(imputed_data)

            result_df[col] = imputed_data

        return result_df


def extend_imu_to_match_semg(imu_df, semg_length, sampling_rate=1000):
    """
    将IMU数据扩展到与sEMG相同的长度
    基于时间戳进行外推
    """
    current_length = len(imu_df)
    if current_length >= semg_length:
        return imu_df

    # 计算时间间隔
    if current_length > 1:
        time_interval = imu_df.iloc[1, 0] - imu_df.iloc[0, 0]
    else:
        time_interval = 1.0 / sampling_rate

    # 创建扩展的时间戳
    last_timestamp = imu_df.iloc[-1, 0]
    extended_timestamps = [last_timestamp + time_interval * (i + 1) for i in range(semg_length - current_length)]

    # 创建扩展的数据框
    extended_rows = []

    for i, new_timestamp in enumerate(extended_timestamps):
        new_row = [new_timestamp]  # 时间戳

        # 硬件时间戳 - 使用最后的值或递增
        if len(imu_df.columns) > 1:
            last_hw_timestamp = imu_df.iloc[-1, 1]
            if isinstance(last_hw_timestamp, (int, float)):
                new_hw_timestamp = last_hw_timestamp + (i + 1) * 1000  # 假设1ms间隔
            else:
                new_hw_timestamp = last_hw_timestamp
            new_row.append(new_hw_timestamp)

        # 关节角度数据 - 使用最后的值（保持稳定）
        for col_idx in range(2, len(imu_df.columns)):
            last_value = imu_df.iloc[-1, col_idx]
            new_row.append(last_value)

        extended_rows.append(new_row)

    # 合并数据
    extended_df = pd.DataFrame(extended_rows, columns=imu_df.columns)
    result_df = pd.concat([imu_df, extended_df], ignore_index=True)

    return result_df


def batch_process_imu_semg_alignment(base_path, output_base):
    """
    自动匹配所有人的 IMU + sEMG 数据，进行关节角度插值并确保IMU长度与sEMG一致
    base_path: 原始对齐数据根目录
    output_base: 填充结果输出路径
    """
    base_path = Path(base_path)
    output_base = Path(output_base)
    output_base.mkdir(parents=True, exist_ok=True)

    imu_root = base_path / "aligned_IMU"
    semg_root = base_path / "aligned_sEMG"

    imputer = JointAngleImputer()

    print("🔍 开始扫描所有 IMU 文件...")
    imu_files = list(imu_root.rglob("IMU_*.csv"))
    print(f"共找到 {len(imu_files)} 个 IMU 文件。")

    results = {
        'success': 0,
        'failed': 0,
        'extended': 0,
        'imputed_only': 0,
        'details': []
    }

    for imu_file in tqdm(imu_files, desc="处理IMU文件"):
        try:
            # 路径解析
            rel_path = imu_file.relative_to(imu_root)
            person_dir = rel_path.parts[0]  # Participant-level IMU directory
            motion_dir = rel_path.parts[1]  # 例如 IMU_ZZF_tiaogao
            file_name = imu_file.name  # IMU_2025-10-24_16-34-02.csv

            # 构造 sEMG 路径
            semg_person_dir = person_dir.replace("IMU", "sEMG")
            semg_motion_dir = motion_dir.replace("IMU", "sEMG")
            semg_file_name = f"processed_data_{file_name.split('_', 1)[1]}"

            semg_path = semg_root / semg_person_dir / semg_motion_dir / file_name.parent.name / semg_file_name

            # 读取 IMU 数据
            imu_df = pd.read_csv(imu_file)
            original_imu_length = len(imu_df)

            # 填充关节角度
            filled_imu_df = imputer.impute_joint_angles(imu_df)

            # 检查对应的 sEMG 文件并扩展长度
            if semg_path.exists():
                semg_df = pd.read_csv(semg_path)
                semg_length = len(semg_df)

                if semg_length > original_imu_length:
                    # 需要扩展IMU数据
                    extended_imu_df = extend_imu_to_match_semg(filled_imu_df, semg_length)
                    results['extended'] += 1
                    status = 'extended'
                    print(f"✅ {imu_file.name}: 已插值并扩展至{semg_length}行 (原{original_imu_length}行)")
                else:
                    extended_imu_df = filled_imu_df
                    results['imputed_only'] += 1
                    status = 'imputed_only'
                    if semg_length < original_imu_length:
                        print(f"⚠️ {imu_file.name}: 已插值 (IMU比sEMG长{original_imu_length - semg_length}行)")
                    else:
                        print(f"✅ {imu_file.name}: 已插值 (长度一致)")
            else:
                # sEMG文件不存在，只进行插值
                extended_imu_df = filled_imu_df
                results['imputed_only'] += 1
                status = 'imputed_only'
                print(f"⚠️ {imu_file.name}: 已插值 (未找到匹配的sEMG文件)")

            # 输出路径
            output_path = output_base / person_dir / motion_dir / file_name.parent.name
            output_path.mkdir(parents=True, exist_ok=True)
            save_file = output_path / f"aligned_{file_name}"

            extended_imu_df.to_csv(save_file, index=False)
            results['success'] += 1

            # 记录详细信息
            semg_length = len(semg_df) if semg_path.exists() else 0
            results['details'].append({
                'imu_file': str(imu_file),
                'semg_file': str(semg_path) if semg_path.exists() else 'not_found',
                'status': status,
                'original_imu_length': original_imu_length,
                'semg_length': semg_length,
                'final_imu_length': len(extended_imu_df)
            })

        except Exception as e:
            print(f"❌ 处理失败: {imu_file}，错误: {e}")
            results['failed'] += 1
            results['details'].append({
                'imu_file': str(imu_file),
                'status': 'failed',
                'error': str(e)
            })

    # 生成报告
    print(f"\n" + "=" * 60)
    print("处理完成统计")
    print("=" * 60)

    total = len(imu_files)
    print(f"总处理文件: {total}")
    print(f"✅ 成功处理: {results['success']}")
    print(f"📈 需要扩展: {results['extended']}")
    print(f"🔧 只插值: {results['imputed_only']}")
    print(f"❌ 处理失败: {results['failed']}")

    if total > 0:
        success_rate = results['success'] / total * 100
        print(f"成功率: {success_rate:.1f}%")

    # 保存详细报告
    generate_comprehensive_report(results, output_base)

    return results


def generate_comprehensive_report(results, output_base):
    """生成综合处理报告"""
    report_file = output_base / "imu_semg_alignment_comprehensive_report.txt"

    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("IMU-sEMG数据对齐综合处理报告\n")
        f.write("=" * 60 + "\n\n")

        f.write("处理概述:\n")
        f.write(f"总处理文件: {len(results['details'])}\n")
        f.write(f"成功处理: {results['success']}\n")
        f.write(f"需要扩展: {results['extended']}\n")
        f.write(f"只插值: {results['imputed_only']}\n")
        f.write(f"处理失败: {results['failed']}\n\n")

        f.write("详细处理结果:\n")
        f.write("-" * 60 + "\n")

        for detail in results['details']:
            f.write(f"IMU文件: {detail['imu_file']}\n")
            if 'semg_file' in detail and detail['semg_file'] != 'not_found':
                f.write(f"sEMG文件: {detail['semg_file']}\n")
            f.write(f"状态: {detail['status']}\n")
            if detail['status'] != 'failed':
                f.write(f"原始IMU长度: {detail['original_imu_length']}\n")
                if 'semg_length' in detail and detail['semg_length'] > 0:
                    f.write(f"sEMG长度: {detail['semg_length']}\n")
                f.write(f"最终IMU长度: {detail['final_imu_length']}\n")
            else:
                f.write(f"错误: {detail['error']}\n")
            f.write("\n")

    print(f"详细报告已生成: {report_file}")


if __name__ == "__main__":
    base_path = "data/interim/aligned_data_strict/"
    output_base = "data/processed/aligned_IMU_extended/"

    print("开始IMU-sEMG数据长度对齐处理...")
    results = batch_process_imu_semg_alignment(base_path, output_base)

    if results:
        print(f"\n" + "=" * 80)
        print("处理完成！")
        print("=" * 80)
        print(f"处理后的数据保存在: {output_base}")
        print(f"报告文件: {output_base}/imu_semg_alignment_comprehensive_report.txt")
    else:
        print("\n❌ 处理失败，请检查路径和文件")
