import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.signal import butter, filtfilt
import warnings

warnings.filterwarnings('ignore')


class SEMG_IMU_Analyzer:
    def __init__(self):
        self.semg_data = None
        self.imu_data = None

    def load_data(self, semg_path, imu_path):
        """加载sEMG和IMU数据"""
        try:
            # 读取数据，跳过时间戳列
            self.semg_data = pd.read_csv(semg_path).iloc[:, 2:]  # 跳过前两列时间戳
            self.imu_data = pd.read_csv(imu_path).iloc[:, 2:]  # 跳过前两列时间戳

            print(f"sEMG数据形状: {self.semg_data.shape}")
            print(f"IMU数据形状: {self.imu_data.shape}")
            print(f"sEMG列名: {list(self.semg_data.columns)}")
            print(f"IMU列名: {list(self.imu_data.columns)}")

            return True
        except Exception as e:
            print(f"数据加载错误: {e}")
            return False

    def preprocess_semg(self):
        """预处理sEMG信号"""
        if self.semg_data is None:
            return

        # 带通滤波 (20-450 Hz)
        def bandpass_filter(data, lowcut=20, highcut=450, fs=1000, order=4):
            nyq = 0.5 * fs
            low = lowcut / nyq
            high = highcut / nyq
            b, a = butter(order, [low, high], btype='band')
            return filtfilt(b, a, data)

        # 对每个sEMG通道应用滤波
        semg_filtered = self.semg_data.copy()
        for col in semg_filtered.columns:
            semg_filtered[col] = bandpass_filter(semg_filtered[col])

        # 全波整流
        semg_rectified = np.abs(semg_filtered)

        # 低通滤波得到包络线 (5 Hz)
        def lowpass_filter(data, cutoff=5, fs=1000, order=4):
            nyq = 0.5 * fs
            normal_cutoff = cutoff / nyq
            b, a = butter(order, normal_cutoff, btype='low')
            return filtfilt(b, a, data)

        semg_envelope = semg_rectified.copy()
        for col in semg_envelope.columns:
            semg_envelope[col] = lowpass_filter(semg_envelope[col])

        return semg_envelope

    def calculate_correlation(self, semg_processed):
        """计算sEMG和IMU的相关性"""
        correlations = {}

        # 确保数据长度一致
        min_len = min(len(semg_processed), len(self.imu_data))
        semg_processed = semg_processed.iloc[:min_len]
        imu_data = self.imu_data.iloc[:min_len]

        # 计算每个sEMG通道与每个IMU维度的相关性
        for semg_col in semg_processed.columns:
            semg_corrs = {}
            for imu_col in imu_data.columns:
                correlation, p_value = stats.pearsonr(semg_processed[semg_col], imu_data[imu_col])
                semg_corrs[imu_col] = {
                    'correlation': correlation,
                    'p_value': p_value,
                    'significant': p_value < 0.05
                }
            correlations[semg_col] = semg_corrs

        return correlations

    def analyze_data_quality(self, semg_processed):
        """分析数据质量"""
        print("\n=== 数据质量分析 ===")

        # sEMG数据质量
        print("\n1. sEMG数据质量:")
        for col in semg_processed.columns:
            data = semg_processed[col]
            print(f"   {col}: 均值={data.mean():.4f}, 标准差={data.std():.4f}, "
                  f"动态范围={data.max() - data.min():.4f}")

        # IMU数据质量
        print("\n2. IMU数据质量:")
        for col in self.imu_data.columns:
            data = self.imu_data[col]
            print(f"   {col}: 均值={data.mean():.4f}, 标准差={data.std():.4f}, "
                  f"动态范围={data.max() - data.min():.4f}")

    def visualize_correlations(self, correlations):
        """可视化相关性结果"""
        # 提取相关性矩阵
        semg_cols = list(correlations.keys())
        imu_cols = list(correlations[semg_cols[0]].keys())

        corr_matrix = np.zeros((len(semg_cols), len(imu_cols)))
        p_value_matrix = np.zeros((len(semg_cols), len(imu_cols)))

        for i, semg_col in enumerate(semg_cols):
            for j, imu_col in enumerate(imu_cols):
                corr_matrix[i, j] = correlations[semg_col][imu_col]['correlation']
                p_value_matrix[i, j] = correlations[semg_col][imu_col]['p_value']

        # 绘制热图
        plt.figure(figsize=(12, 8))
        mask = p_value_matrix >= 0.05  # 标记不显著的相关性
        sns.heatmap(corr_matrix,
                    xticklabels=imu_cols,
                    yticklabels=semg_cols,
                    annot=True, fmt=".3f",
                    cmap='RdBu_r', center=0,
                    mask=mask,
                    cbar_kws={'label': 'Pearson Correlation Coefficient'})
        plt.title('sEMG-IMU Correlation Matrix (Only Significant: p < 0.05)')
        plt.xlabel('IMU Features')
        plt.ylabel('sEMG Channels')
        plt.tight_layout()
        plt.show()

        return corr_matrix, p_value_matrix

    def plot_sample_data(self, semg_processed, start=0, end=1000):
        """绘制样本数据对比"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))

        # 绘制sEMG数据
        for col in semg_processed.columns:
            ax1.plot(semg_processed[col].iloc[start:end], label=col, alpha=0.7)
        ax1.set_title('Processed sEMG Signals (Envelope)')
        ax1.set_ylabel('Amplitude')
        ax1.legend()
        ax1.grid(True)

        # 绘制IMU数据
        for col in self.imu_data.columns:
            ax2.plot(self.imu_data[col].iloc[start:end], label=col, alpha=0.7)
        ax2.set_title('IMU Joint Angles')
        ax2.set_ylabel('Angle (degrees)')
        ax2.set_xlabel('Sample Index')
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        plt.show()

    def comprehensive_analysis(self, semg_path, imu_path):
        """执行综合分析"""
        print("开始sEMG-IMU数据相关性分析...")

        # 1. 加载数据
        if not self.load_data(semg_path, imu_path):
            return

        # 2. 预处理sEMG
        print("\n预处理sEMG信号...")
        semg_processed = self.preprocess_semg()

        # 3. 数据质量分析
        self.analyze_data_quality(semg_processed)

        # 4. 计算相关性
        print("\n计算sEMG-IMU相关性...")
        correlations = self.calculate_correlation(semg_processed)

        # 5. 输出相关性结果
        print("\n=== 相关性分析结果 ===")
        significant_correlations = []

        for semg_col in correlations:
            print(f"\nsEMG通道 {semg_col}:")
            for imu_col in correlations[semg_col]:
                corr_info = correlations[semg_col][imu_col]
                star = " ***" if corr_info['significant'] else ""
                print(f"  {imu_col}: r = {corr_info['correlation']:.3f}, "
                      f"p = {corr_info['p_value']:.3e}{star}")

                if corr_info['significant'] and abs(corr_info['correlation']) > 0.3:
                    significant_correlations.append(
                        (semg_col, imu_col, corr_info['correlation'])
                    )

        # 6. 总结显著相关性
        print("\n=== 显著相关性总结 (|r| > 0.3) ===")
        if significant_correlations:
            significant_correlations.sort(key=lambda x: abs(x[2]), reverse=True)
            for semg_col, imu_col, corr in significant_correlations:
                print(f"sEMG {semg_col} - IMU {imu_col}: r = {corr:.3f}")
        else:
            print("未发现强相关性 (|r| > 0.3)")

        # 7. 可视化
        print("\n生成可视化图表...")
        self.visualize_correlations(correlations)
        self.plot_sample_data(semg_processed)

        return correlations


# 使用示例
def analyze_all_datasets():
    """分析所有数据集"""
    analyzer = SEMG_IMU_Analyzer()

    # 这里替换为你的实际文件路径
    datasets = [
        {
            'name': 'ZZF跳高',
            'semg_path': 'data/example/sEMG.csv',
            'imu_path': 'data/example/IMU.csv'
        },
        # 添加其他数据集...
        # {
        #     'name': '某人跳远',
        #     'semg_path': '路径/to/semg.csv',
        #     'imu_path': '路径/to/imu.csv'
        # }
    ]

    results = {}
    for dataset in datasets:
        print(f"\n{'=' * 50}")
        print(f"分析数据集: {dataset['name']}")
        print(f"{'=' * 50}")

        correlations = analyzer.comprehensive_analysis(
            dataset['semg_path'],
            dataset['imu_path']
        )
        results[dataset['name']] = correlations

    return results


# 运行分析
if __name__ == "__main__":
    # 分析单个文件
    analyzer = SEMG_IMU_Analyzer()

    # 替换为你的实际文件路径
    semg_path = "data/example/sEMG.csv"
    imu_path = "data/example/IMU.csv"

    results = analyzer.comprehensive_analysis(semg_path, imu_path)

    # 或者分析所有数据集
    # all_results = analyze_all_datasets()
