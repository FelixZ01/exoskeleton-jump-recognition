import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.signal import find_peaks, butter, filtfilt
from sklearn.feature_selection import mutual_info_regression
import warnings
import os
import shutil
import time

warnings.filterwarnings('ignore')


class SEMG_IMU_Motion_Selector:
    def __init__(self):
        self.semg_data = None
        self.imu_data = None
        self.base_dir = "data/processed/aligned"
        self.output_base_dir = "outputs/quality_selection"

    def load_complete_data(self, semg_path, imu_path):
        """加载完整数据"""
        try:
            semg_data = pd.read_csv(semg_path)
            imu_data = pd.read_csv(imu_path)

            print(f"原始数据长度 - sEMG: {len(semg_data)}, IMU: {len(imu_data)}")

            # 识别数值列
            semg_numeric_cols = []
            for col in semg_data.columns:
                try:
                    pd.to_numeric(semg_data[col], errors='raise')
                    semg_numeric_cols.append(col)
                except:
                    continue

            imu_numeric_cols = []
            for col in imu_data.columns:
                try:
                    pd.to_numeric(imu_data[col], errors='raise')
                    imu_numeric_cols.append(col)
                except:
                    continue

            if not semg_numeric_cols or not imu_numeric_cols:
                return False

            self.semg_data = semg_data[semg_numeric_cols].apply(pd.to_numeric, errors='coerce')
            self.imu_data = imu_data[imu_numeric_cols].apply(pd.to_numeric, errors='coerce')

            # 清理NaN
            self.semg_data = self.semg_data.dropna(how='all')
            self.imu_data = self.imu_data.dropna(how='all')

            if len(self.semg_data) < 500 or len(self.imu_data) < 500:
                print("数据太短，跳过")
                return False

            return True

        except Exception as e:
            print(f"数据加载错误: {e}")
            return False

    def detect_motion_cycles(self):
        """检测运动周期（起跳-落地）"""
        try:
            # 使用IMU数据检测运动周期（通常膝关节角度变化最明显）
            imu_columns = self.imu_data.columns

            # 寻找可能的关节角度列（包含'knee', 'hip', 'angle'等关键词）
            angle_columns = [col for col in imu_columns if any(keyword in col.lower()
                                                               for keyword in ['knee', 'hip', 'angle', 'joint'])]

            if not angle_columns:
                # 如果没有明确的角度列，使用第一列
                motion_signal = self.imu_data.iloc[:, 0].values
            else:
                # 使用第一个找到的角度列
                motion_signal = self.imu_data[angle_columns[0]].values

            # 平滑信号以便更好地检测峰值
            window_size = min(50, len(motion_signal) // 10)
            if window_size % 2 == 0:
                window_size += 1

            smoothed_signal = np.convolve(motion_signal, np.ones(window_size) / window_size, mode='same')

            # 寻找峰值（起跳点）
            peaks, properties = find_peaks(smoothed_signal,
                                           height=np.median(smoothed_signal) + np.std(smoothed_signal),
                                           distance=100,  # 最小间隔
                                           prominence=np.std(smoothed_signal))

            # 寻找谷值（落地点）
            valleys, _ = find_peaks(-smoothed_signal,
                                    distance=100,
                                    prominence=np.std(smoothed_signal))

            print(f"检测到 {len(peaks)} 个峰值, {len(valleys)} 个谷值")

            # 构建运动周期：每个峰值到下一个谷值
            motion_cycles = []
            for i, peak in enumerate(peaks):
                # 找到峰值后的第一个谷值
                subsequent_valleys = valleys[valleys > peak]
                if len(subsequent_valleys) > 0:
                    valley = subsequent_valleys[0]
                    cycle_length = valley - peak

                    # 只保留合理长度的周期（0.5-3秒，假设采样率1000Hz）
                    if 500 <= cycle_length <= 3000:
                        motion_cycles.append({
                            'start': peak,
                            'end': valley,
                            'length': cycle_length
                        })

            print(f"找到 {len(motion_cycles)} 个有效运动周期")
            return motion_cycles

        except Exception as e:
            print(f"运动周期检测错误: {e}")
            return []

    def extract_motion_features(self, motion_cycles):
        """提取运动周期特征"""
        if not motion_cycles:
            return None

        cycle_features = []

        for cycle in motion_cycles[:3]:  # 只分析前3个周期以节省时间
            start, end = cycle['start'], cycle['end']

            # 提取该周期的sEMG和IMU数据
            cycle_semg = self.semg_data.iloc[start:end]
            cycle_imu = self.imu_data.iloc[start:end]

            if len(cycle_semg) < 100 or len(cycle_imu) < 100:
                continue

            features = {}

            # 1. 运动强度特征
            semg_intensity = np.mean(np.abs(cycle_semg.values))
            imu_variation = np.std(cycle_imu.values)
            features['intensity'] = semg_intensity * imu_variation

            # 2. 相关性特征（在运动周期内）
            correlations = []
            for semg_col in cycle_semg.columns[:3]:  # 前3个sEMG通道
                for imu_col in cycle_imu.columns[:3]:  # 前3个IMU维度
                    try:
                        corr, _ = stats.pearsonr(cycle_semg[semg_col].values,
                                                 cycle_imu[imu_col].values)
                        if not np.isnan(corr):
                            correlations.append(abs(corr))
                    except:
                        continue

            features['correlation'] = np.mean(correlations) if correlations else 0

            # 3. 同步性特征（峰值对齐）
            try:
                # 寻找sEMG和IMU的峰值时间差
                semg_peak_channel = cycle_semg.columns[0]  # 使用第一个sEMG通道
                imu_peak_channel = cycle_imu.columns[0]  # 使用第一个IMU维度

                semg_peaks, _ = find_peaks(cycle_semg[semg_peak_channel].values,
                                           height=np.median(cycle_semg[semg_peak_channel]) + np.std(
                                               cycle_semg[semg_peak_channel]))
                imu_peaks, _ = find_peaks(cycle_imu[imu_peak_channel].values,
                                          height=np.median(cycle_imu[imu_peak_channel]) + np.std(
                                              cycle_imu[imu_peak_channel]))

                if len(semg_peaks) > 0 and len(imu_peaks) > 0:
                    # 计算第一个峰值的时间差
                    time_diff = abs(semg_peaks[0] - imu_peaks[0])
                    features['synchronization'] = 1.0 / (1.0 + time_diff / 100)  # 时间差越小，同步性越高
                else:
                    features['synchronization'] = 0
            except:
                features['synchronization'] = 0

            cycle_features.append(features)

        if not cycle_features:
            return None

        # 聚合所有周期的特征
        aggregated = {
            'avg_intensity': np.mean([f['intensity'] for f in cycle_features]),
            'avg_correlation': np.mean([f['correlation'] for f in cycle_features]),
            'avg_synchronization': np.mean([f['synchronization'] for f in cycle_features]),
            'num_cycles': len(cycle_features)
        }

        # 计算综合得分
        aggregated['total_score'] = (
                                            aggregated['avg_intensity'] * 0.3 +
                                            aggregated['avg_correlation'] * 0.4 +
                                            aggregated['avg_synchronization'] * 0.3
                                    ) * min(aggregated['num_cycles'] / 3.0, 1.0)  # 周期数量也影响得分

        return aggregated

    def comprehensive_motion_evaluation(self, semg_path, imu_path):
        """基于运动周期的综合评估"""
        start_time = time.time()

        # 加载完整数据
        if not self.load_complete_data(semg_path, imu_path):
            return None

        # 检测运动周期
        motion_cycles = self.detect_motion_cycles()

        if not motion_cycles:
            print("    未检测到明显的运动周期")
            return None

        # 提取运动特征
        features = self.extract_motion_features(motion_cycles)

        if not features:
            return None

        eval_time = time.time() - start_time
        print(
            f"    评估完成: {eval_time:.1f}s, 运动周期: {features['num_cycles']}, 得分: {features['total_score']:.3f}")

        return features

    def find_all_data_pairs(self):
        """查找所有数据对"""
        pairs = []

        imu_base = os.path.join(self.base_dir, "aligned_IMU")
        semg_base = os.path.join(self.base_dir, "aligned_sEMG")

        if not os.path.exists(imu_base) or not os.path.exists(semg_base):
            return pairs

        print("扫描目录结构...")

        # 处理所有受试者，但限制每个运动类型的数据量
        max_per_movement = 8

        for subject_dir in os.listdir(imu_base):
            if subject_dir.endswith("_IMU_data"):
                subject_id = subject_dir.replace("_IMU_data", "")
                print(f"处理受试者: {subject_id}")

                subject_imu_dir = os.path.join(imu_base, subject_dir)
                subject_semg_dir = os.path.join(semg_base, f"{subject_id}_sEMG_data")

                if not os.path.exists(subject_semg_dir):
                    continue

                imu_movement_dirs = [d for d in os.listdir(subject_imu_dir)
                                     if os.path.isdir(os.path.join(subject_imu_dir, d)) and not d.startswith('.')]

                semg_movement_dirs = [d for d in os.listdir(subject_semg_dir)
                                      if os.path.isdir(os.path.join(subject_semg_dir, d)) and not d.startswith('.')]

                for imu_movement in imu_movement_dirs:
                    movement_type = imu_movement.split('_')[-1]

                    matching_semg_dir = None
                    for semg_dir in semg_movement_dirs:
                        if movement_type in semg_dir:
                            matching_semg_dir = semg_dir
                            break

                    if not matching_semg_dir:
                        continue

                    imu_movement_path = os.path.join(subject_imu_dir, imu_movement)
                    semg_movement_path = os.path.join(subject_semg_dir, matching_semg_dir)

                    imu_timestamps = [d for d in os.listdir(imu_movement_path)
                                      if os.path.isdir(os.path.join(imu_movement_path, d)) and not d.startswith('.')]
                    semg_timestamps = [d for d in os.listdir(semg_movement_path)
                                       if os.path.isdir(os.path.join(semg_movement_path, d)) and not d.startswith('.')]

                    common_timestamps = list(set(imu_timestamps) & set(semg_timestamps))[:max_per_movement]

                    for timestamp in common_timestamps:
                        imu_timestamp_path = os.path.join(imu_movement_path, timestamp)
                        semg_timestamp_path = os.path.join(semg_movement_path, timestamp)

                        imu_files = [f for f in os.listdir(imu_timestamp_path)
                                     if f.startswith("IMU_") and f.endswith(".csv") and not f.startswith('.')]

                        semg_files = [f for f in os.listdir(semg_timestamp_path)
                                      if
                                      f.startswith("processed_data_") and f.endswith(".csv") and not f.startswith('.')]

                        if imu_files and semg_files:
                            imu_file = os.path.join(imu_timestamp_path, imu_files[0])
                            semg_file = os.path.join(semg_timestamp_path, semg_files[0])

                            imu_time = imu_files[0].replace("IMU_", "").replace(".csv", "")
                            semg_time = semg_files[0].replace("processed_data_", "").replace(".csv", "")

                            if imu_time == semg_time:
                                pairs.append({
                                    'semg_path': semg_file,
                                    'imu_path': imu_file,
                                    'subject': subject_id,
                                    'movement': movement_type,
                                    'timestamp': timestamp
                                })

        return pairs

    def select_optimal_motion_datasets(self, top_k=10):
        """选择最优的运动数据集"""
        print("扫描数据对...")
        data_pairs = self.find_all_data_pairs()
        print(f"找到 {len(data_pairs)} 个数据对")

        if not data_pairs:
            return []

        # 评估所有数据对
        evaluations = []

        for i, pair in enumerate(data_pairs):
            print(f"评估 {i + 1}/{len(data_pairs)}: {pair['subject']} - {pair['movement']} - {pair['timestamp']}")

            features = self.comprehensive_motion_evaluation(pair['semg_path'], pair['imu_path'])

            if features and features['total_score'] > 0.1:
                evaluation = {
                    'subject': pair['subject'],
                    'movement': pair['movement'],
                    'timestamp': pair['timestamp'],
                    'semg_path': pair['semg_path'],
                    'imu_path': pair['imu_path'],
                    'features': features
                }
                evaluations.append(evaluation)

        # 按综合得分排序
        evaluations.sort(key=lambda x: x['features']['total_score'], reverse=True)

        # 选择前top_k个
        selected = evaluations[:top_k]

        # 保存选中的数据集
        self.save_selected_datasets(selected)

        # 输出结果
        self.print_selection_results(selected)

        return selected

    def save_selected_datasets(self, selected_datasets):
        """保存选中的数据集"""
        output_dir = os.path.join(self.output_base_dir, "selected_motion_data")
        os.makedirs(output_dir, exist_ok=True)

        for i, dataset in enumerate(selected_datasets):
            dataset_dir = os.path.join(output_dir, f"motion_dataset_{i + 1:02d}")
            os.makedirs(dataset_dir, exist_ok=True)

            # 复制完整文件
            shutil.copy2(dataset['semg_path'], os.path.join(dataset_dir, "sEMG.csv"))
            shutil.copy2(dataset['imu_path'], os.path.join(dataset_dir, "IMU.csv"))

            # 保存运动特征
            features_df = pd.DataFrame([dataset['features']])
            features_df.to_csv(os.path.join(dataset_dir, "motion_features.csv"), index=False)

            # 保存元数据
            metadata = {
                'subject': dataset['subject'],
                'movement': dataset['movement'],
                'timestamp': dataset['timestamp'],
                'dataset_rank': i + 1,
                'total_score': dataset['features']['total_score'],
                'num_cycles': dataset['features']['num_cycles']
            }
            metadata_df = pd.DataFrame([metadata])
            metadata_df.to_csv(os.path.join(dataset_dir, "metadata.csv"), index=False)

    def print_selection_results(self, selected_datasets):
        """输出选择结果"""
        print("\n" + "=" * 80)
        print("最优运动数据集选择结果 (用于TFCNN)")
        print("=" * 80)

        for i, dataset in enumerate(selected_datasets):
            features = dataset['features']
            print(f"\n{i + 1:2d}. {dataset['subject']} - {dataset['movement']} - {dataset['timestamp']}")
            print(f"    综合得分: {features['total_score']:.3f}")
            print(f"    运动周期数: {features['num_cycles']}")
            print(f"    运动强度: {features['avg_intensity']:.3f}")
            print(f"    相关性: {features['avg_correlation']:.3f}")
            print(f"    同步性: {features['avg_synchronization']:.3f}")


# 使用示例
def main():
    selector = SEMG_IMU_Motion_Selector()

    print("开始基于运动周期选择最优数据集...")
    selected_datasets = selector.select_optimal_motion_datasets(top_k=10)

    if selected_datasets:
        print(f"\n成功选择了 {len(selected_datasets)} 个最优运动数据集")
        print(f"数据集已保存到: {selector.output_base_dir}/selected_motion_data")

        # 显示统计信息
        movements = [d['movement'] for d in selected_datasets]
        subjects = [d['subject'] for d in selected_datasets]

        print(f"\n运动类型分布: {pd.Series(movements).value_counts().to_dict()}")
        print(f"受试者分布: {pd.Series(subjects).value_counts().to_dict()}")
    else:
        print("未找到合适的运动数据集")


if __name__ == "__main__":
    main()
