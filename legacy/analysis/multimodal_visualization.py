import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.signal import butter, filtfilt
import warnings
from matplotlib.patches import FancyBboxPatch
import matplotlib.gridspec as gridspec

warnings.filterwarnings('ignore')


class SEMG_IMU_Visualizer:
    def __init__(self):
        self.semg_data = None
        self.imu_data = None
        self.semg_processed = None
        self.correlations = None

    def load_and_preprocess_data(self, semg_path, imu_path):
        """加载并预处理数据"""
        print("正在加载数据...")

        try:
            # 加载数据，跳过时间戳列
            self.semg_data = pd.read_csv(semg_path).iloc[:, 2:]  # 跳过前两列时间戳
            self.imu_data = pd.read_csv(imu_path).iloc[:, 2:]  # 跳过前两列时间戳

            print(f"sEMG数据形状: {self.semg_data.shape}")
            print(f"IMU数据形状: {self.imu_data.shape}")

            # 预处理sEMG信号
            self.semg_processed = self.preprocess_semg()

            # 计算相关性
            self.correlations = self.calculate_correlation()

            return True

        except Exception as e:
            print(f"数据加载错误: {e}")
            return False

    def preprocess_semg(self):
        """预处理sEMG信号"""
        if self.semg_data is None:
            return None

        # 带通滤波 (20-450 Hz)
        def bandpass_filter(data, lowcut=20, highcut=450, fs=1000, order=4):
            nyq = 0.5 * fs
            low = lowcut / nyq
            high = highcut / nyq
            b, a = butter(order, [low, high], btype='band')
            return filtfilt(b, a, data)

        # 低通滤波
        def lowpass_filter(data, cutoff=5, fs=1000, order=4):
            nyq = 0.5 * fs
            normal_cutoff = cutoff / nyq
            b, a = butter(order, normal_cutoff, btype='low')
            return filtfilt(b, a, data)

        # 对每个sEMG通道应用滤波
        semg_filtered = self.semg_data.copy()
        for col in semg_filtered.columns:
            if semg_filtered[col].std() > 0:  # 跳过全零通道
                semg_filtered[col] = bandpass_filter(semg_filtered[col])

        # 全波整流
        semg_rectified = np.abs(semg_filtered)

        # 低通滤波得到包络线
        semg_envelope = semg_rectified.copy()
        for col in semg_envelope.columns:
            if semg_envelope[col].std() > 0:
                semg_envelope[col] = lowpass_filter(semg_envelope[col])

        return semg_envelope

    def calculate_correlation(self):
        """计算sEMG和IMU的相关性"""
        if self.semg_processed is None or self.imu_data is None:
            return {}

        correlations = {}

        # 确保数据长度一致
        min_len = min(len(self.semg_processed), len(self.imu_data))
        semg_processed = self.semg_processed.iloc[:min_len]
        imu_data = self.imu_data.iloc[:min_len]

        # 计算每个sEMG通道与每个IMU维度的相关性
        for semg_col in semg_processed.columns:
            semg_corrs = {}
            for imu_col in imu_data.columns:
                try:
                    correlation, p_value = stats.pearsonr(semg_processed[semg_col], imu_data[imu_col])
                    semg_corrs[imu_col] = {
                        'correlation': correlation,
                        'p_value': p_value,
                        'significant': p_value < 0.05
                    }
                except:
                    semg_corrs[imu_col] = {
                        'correlation': np.nan,
                        'p_value': np.nan,
                        'significant': False
                    }
            correlations[semg_col] = semg_corrs

        return correlations

    def plot_signal_quality_comparison(self, ax):
        """绘制信号质量对比图"""
        if self.semg_processed is None:
            return

        # sEMG信号质量指标
        semg_quality = []
        for col in self.semg_processed.columns:
            data = self.semg_processed[col]
            if data.std() > 0:  # 跳过无效通道
                snr = data.mean() / data.std()
                dynamic_range = data.max() - data.min()
                semg_quality.append({
                    'channel': col,
                    'snr': snr,
                    'dynamic_range': dynamic_range,
                    'std': data.std()
                })

        # 绘制前8个通道
        channels = [q['channel'] for q in semg_quality[:8]]
        snr_values = [q['snr'] for q in semg_quality[:8]]

        bars = ax.bar(channels, snr_values, color=['#2E86AB', '#A23B72', '#F18F01', '#C73E1D',
                                                   '#3B1F2B', '#DB5461', '#686963', '#E9D985'])
        ax.set_title('sEMG Signal Quality Comparison (SNR)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Signal-to-Noise Ratio')
        ax.tick_params(axis='x', rotation=45)

        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=8)

    def plot_correlation_network(self, ax):
        """绘制相关性网络图"""
        if self.correlations is None:
            return

        # 提取强相关性对 (|r| > 0.5)
        strong_pairs = []
        for semg_col in self.correlations:
            for imu_col in self.correlations[semg_col]:
                corr_info = self.correlations[semg_col][imu_col]
                if not np.isnan(corr_info['correlation']) and abs(corr_info['correlation']) > 0.5:
                    strong_pairs.append((semg_col, imu_col, corr_info['correlation']))

        # 简化显示，选择前15个最强相关对
        strong_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
        display_pairs = strong_pairs[:15]

        if not display_pairs:
            return

        # 创建节点位置
        semg_nodes = list(set([pair[0] for pair in display_pairs]))
        imu_nodes = list(set([pair[1] for pair in display_pairs]))

        # 绘制节点和边
        pos = {}
        node_colors = []

        # sEMG节点 (左侧)
        for i, node in enumerate(semg_nodes):
            pos[node] = (0, i - len(semg_nodes) / 2)
            node_colors.append('#E74C3C')  # 红色

        # IMU节点 (右侧)
        for i, node in enumerate(imu_nodes):
            pos[node] = (1, i - len(imu_nodes) / 2)
            node_colors.append('#3498DB')  # 蓝色

        # 绘制边
        for semg, imu, corr in display_pairs:
            color = 'green' if corr > 0 else 'red'
            width = abs(corr) * 3
            ax.plot([pos[semg][0], pos[imu][0]], [pos[semg][1], pos[imu][1]],
                    color=color, linewidth=width, alpha=0.6)

        # 绘制节点
        for node, (x, y) in pos.items():
            ax.scatter(x, y, s=300, c=node_colors[list(pos.keys()).index(node)],
                       alpha=0.7, edgecolors='black')
            # 简化节点标签显示
            label = node.replace('Channel_', 'Ch').replace('FP_CH', 'FP')
            ax.text(x, y, label, ha='center', va='center', fontsize=8, fontweight='bold')

        ax.set_xlim(-0.2, 1.2)
        ax.set_ylim(-8, 8)
        ax.set_title('sEMG-IMU Strong Correlation Network (|r| > 0.5)', fontsize=12, fontweight='bold')
        ax.axis('off')

    def plot_time_series_comparison(self, ax):
        """绘制时序信号对比图"""
        if self.semg_processed is None or self.imu_data is None:
            return

        # 选择最具代表性的通道
        time = np.arange(min(len(self.semg_processed), len(self.imu_data)))

        # 创建双y轴
        ax1 = ax
        ax2 = ax.twinx()

        # 绘制sEMG信号 (左侧)
        semg_channel = 'Channel_3'  # 选择相关性最强的sEMG通道
        if semg_channel in self.semg_processed.columns:
            color1 = '#E74C3C'
            ax1.plot(time[:1000], self.semg_processed[semg_channel].values[:1000],
                     color=color1, linewidth=1, label=f'sEMG {semg_channel}')
            ax1.set_ylabel('sEMG Amplitude', color=color1)
            ax1.tick_params(axis='y', labelcolor=color1)

        # 绘制IMU信号 (右侧)
        imu_channel = 'R2_Roll'  # 选择相关性最强的IMU通道
        if imu_channel in self.imu_data.columns:
            color2 = '#3498DB'
            ax2.plot(time[:1000], self.imu_data[imu_channel].values[:1000],
                     color=color2, linewidth=1, label=f'IMU {imu_channel}')
            ax2.set_ylabel('Joint Angle (degrees)', color=color2)
            ax2.tick_params(axis='y', labelcolor=color2)

        ax.set_xlabel('Time (samples)')
        ax.set_title('sEMG and IMU Signal Time Series Comparison', fontsize=12, fontweight='bold')

        # 合并图例
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    def plot_feature_importance(self, ax):
        """绘制特征重要性热图"""
        if self.correlations is None:
            return

        # 创建相关性矩阵
        semg_channels = ['Channel_2', 'Channel_3', 'Channel_4', 'Channel_6',
                         'FP_CH42', 'FP_CH46', 'FP_CH48', 'sum_foot']
        imu_channels = ['R1_Roll', 'R1_Pitch', 'R1_Yaw', 'R2_Roll', 'R2_Pitch',
                        'R3_Roll', 'R3_Yaw', 'R4_Roll']

        corr_matrix = np.zeros((len(semg_channels), len(imu_channels)))

        for i, semg in enumerate(semg_channels):
            for j, imu in enumerate(imu_channels):
                if semg in self.correlations and imu in self.correlations[semg]:
                    corr_info = self.correlations[semg][imu]
                    if not np.isnan(corr_info['correlation']):
                        corr_matrix[i, j] = corr_info['correlation']

        # 绘制热图
        im = ax.imshow(corr_matrix, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)

        # 设置标签
        ax.set_xticks(range(len(imu_channels)))
        ax.set_yticks(range(len(semg_channels)))
        ax.set_xticklabels(imu_channels, rotation=45, ha='right')
        ax.set_yticklabels(semg_channels)

        # 添加数值
        for i in range(len(semg_channels)):
            for j in range(len(imu_channels)):
                if abs(corr_matrix[i, j]) > 0.3:  # 只显示显著相关性
                    text = ax.text(j, i, f'{corr_matrix[i, j]:.2f}',
                                   ha="center", va="center", color="black", fontsize=8)

        ax.set_title('Feature Correlation Matrix', fontsize=12, fontweight='bold')
        plt.colorbar(im, ax=ax, shrink=0.6)

    def plot_muscle_synergy(self, ax):
        """绘制肌肉协同分析"""
        if self.semg_processed is None:
            return

        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler

        # 选择有效的sEMG通道
        valid_channels = [col for col in self.semg_processed.columns
                          if self.semg_processed[col].std() > 0]
        X = self.semg_processed[valid_channels].values

        # 标准化
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # PCA分析
        pca = PCA(n_components=3)
        principal_components = pca.fit_transform(X_scaled)

        # 绘制PCA结果
        colors = ['#E74C3C', '#3498DB', '#2ECC71']
        for i in range(3):
            ax.bar([f'PC{i + 1}'], [pca.explained_variance_ratio_[i]],
                   color=colors[i], alpha=0.7, label=f'PC{i + 1}')

        ax.set_ylabel('Explained Variance Ratio')
        ax.set_title('Muscle Synergy Analysis (PCA)', fontsize=12, fontweight='bold')
        ax.legend()

        # 添加累计方差
        cumulative_variance = pca.explained_variance_ratio_.cumsum()
        for i, cum_var in enumerate(cumulative_variance):
            ax.text(i, pca.explained_variance_ratio_[i] + 0.01,
                    f'Cum: {cum_var:.1%}', ha='center', fontsize=8)

    def plot_research_framework(self, ax):
        """绘制论文框架图"""
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 6)
        ax.axis('off')

        # 定义框样式
        box_style = dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.7)
        arrow_style = dict(arrowstyle="->", color='black', lw=1.5)

        # 绘制框架元素
        elements = [
            (1, 5, 1.5, 0.6, "IMU\nSignals", '#3498DB'),
            (1, 4, 1.5, 0.6, "sEMG\nSignals", '#E74C3C'),
            (3, 5, 2, 0.6, "IMU Feature Extraction\nBiLSTM + Attention", '#5DADE2'),
            (3, 4, 2, 0.6, "sEMG Feature Extraction\nTFCNN + Channel Attention", '#F1948A'),
            (6, 4.5, 2, 1.2, "Dual Autoencoder\nFeature Alignment", '#A569BD'),
            (8.5, 4.5, 1.5, 0.6, "Transformer-MLP", '#F1C40F'),
            (8.5, 3.5, 1.5, 0.6, "Action\nClassification", '#2ECC71'),
            (3, 2.5, 2, 0.6, "BiLSTM + Meta-learning", '#48C9B0'),
            (6, 2.5, 2, 0.6, "Model Interpretation\nTime/Shape/Grad-CAM", '#E67E22')
        ]

        # 绘制所有元素
        for x, y, w, h, text, color in elements:
            rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                                  facecolor=color, alpha=0.8)
            ax.add_patch(rect)
            ax.text(x + w / 2, y + h / 2, text, ha='center', va='center',
                    fontsize=8, fontweight='bold')

        # 绘制连接箭头
        connections = [
            (2.5, 5.3, 3, 5.3), (2.5, 4.3, 3, 4.3),  # 输入到特征提取
            (5, 5.3, 6, 4.8), (5, 4.3, 6, 4.2),  # 特征提取到特征对齐
            (8, 4.5, 8.5, 4.5),  # 特征对齐到Transformer
            (9.7, 4.5, 9.7, 3.5),  # Transformer到分类
            (5, 2.5, 6, 2.5),  # 元学习
            (8, 2.5, 8.5, 2.5)  # 解释性分析
        ]

        for x1, y1, x2, y2 in connections:
            ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=arrow_style)

        ax.set_title('Research Framework for Exoskeleton Action Recognition',
                     fontsize=14, fontweight='bold', pad=20)

    def plot_statistical_summary(self, ax):
        """绘制统计分析总结"""
        if self.correlations is None:
            return

        # 统计相关性强度分布
        corr_values = []
        for semg_col in self.correlations:
            for imu_col in self.correlations[semg_col]:
                corr_info = self.correlations[semg_col][imu_col]
                if not np.isnan(corr_info['correlation']):
                    corr_values.append(abs(corr_info['correlation']))

        if not corr_values:
            return

        # 分类统计
        strong = len([x for x in corr_values if x > 0.7])
        moderate = len([x for x in corr_values if 0.5 < x <= 0.7])
        weak = len([x for x in corr_values if 0.3 < x <= 0.5])
        very_weak = len([x for x in corr_values if x <= 0.3])

        categories = ['Strong\n(r>0.7)', 'Moderate\n(0.5<r≤0.7)', 'Weak\n(0.3<r≤0.5)', 'Very Weak\n(r≤0.3)']
        counts = [strong, moderate, weak, very_weak]
        colors = ['#2ECC71', '#3498DB', '#F39C12', '#E74C3C']

        # 绘制饼图
        wedges, texts, autotexts = ax.pie(counts, labels=categories, colors=colors,
                                          autopct='%1.1f%%', startangle=90)

        # 美化文本
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')

        ax.set_title('sEMG-IMU Correlation Strength Distribution', fontsize=12, fontweight='bold')

    def create_comprehensive_visualizations(self):
        """创建综合可视化图表"""
        if self.semg_processed is None or self.correlations is None:
            print("请先加载和处理数据!")
            return

        fig = plt.figure(figsize=(20, 25))

        # 创建网格布局
        gs = gridspec.GridSpec(5, 2, figure=fig, height_ratios=[1, 1, 1, 1, 1.5])

        # 1. 信号质量对比图
        ax1 = fig.add_subplot(gs[0, 0])
        self.plot_signal_quality_comparison(ax1)

        # 2. 强相关性网络图
        ax2 = fig.add_subplot(gs[0, 1])
        self.plot_correlation_network(ax2)

        # 3. 时序信号对比
        ax3 = fig.add_subplot(gs[1, :])
        self.plot_time_series_comparison(ax3)

        # 4. 特征重要性热图
        ax4 = fig.add_subplot(gs[2, 0])
        self.plot_feature_importance(ax4)

        # 5. 肌肉协同分析
        ax5 = fig.add_subplot(gs[2, 1])
        self.plot_muscle_synergy(ax5)

        # 6. 论文框架图
        ax6 = fig.add_subplot(gs[3, :])
        self.plot_research_framework(ax6)

        # 7. 统计总结
        ax7 = fig.add_subplot(gs[4, :])
        self.plot_statistical_summary(ax7)

        plt.tight_layout()
        plt.savefig('sEMG_IMU_Comprehensive_Analysis.png', dpi=300, bbox_inches='tight')
        plt.show()

        # 打印相关性总结
        self.print_correlation_summary()

    def print_correlation_summary(self):
        """打印相关性分析总结"""
        if self.correlations is None:
            return

        print("\n" + "=" * 60)
        print("CORRELATION ANALYSIS SUMMARY")
        print("=" * 60)

        # 提取强相关性对 (|r| > 0.5)
        strong_pairs = []
        for semg_col in self.correlations:
            for imu_col in self.correlations[semg_col]:
                corr_info = self.correlations[semg_col][imu_col]
                if not np.isnan(corr_info['correlation']) and abs(corr_info['correlation']) > 0.5:
                    strong_pairs.append((semg_col, imu_col, corr_info['correlation']))

        # 排序并显示前10个
        strong_pairs.sort(key=lambda x: abs(x[2]), reverse=True)

        print(f"\nTop 10 Strongest Correlations (|r| > 0.5):")
        print("-" * 50)
        for i, (semg, imu, corr) in enumerate(strong_pairs[:10], 1):
            print(f"{i:2d}. sEMG {semg:12s} - IMU {imu:10s}: r = {corr:6.3f}")

        # 统计信息
        total_pairs = sum(len(self.correlations[semg]) for semg in self.correlations)
        strong_count = len(strong_pairs)

        print(f"\nStatistical Summary:")
        print(f"- Total sEMG-IMU pairs analyzed: {total_pairs}")
        print(f"- Strong correlations (|r| > 0.5): {strong_count} ({strong_count / total_pairs * 100:.1f}%)")
        print(f"- Maximum correlation: {max([abs(x[2]) for x in strong_pairs]):.3f}")
        print(f"- Average strong correlation: {np.mean([abs(x[2]) for x in strong_pairs]):.3f}")


# 使用示例
def main():
    # 创建可视化器
    visualizer = SEMG_IMU_Visualizer()

    # 你的文件路径
    semg_path = "data/example/sEMG.csv"
    imu_path = "data/example/IMU.csv"

    # 加载和处理数据
    success = visualizer.load_and_preprocess_data(semg_path, imu_path)

    if success:
        # 生成可视化
        visualizer.create_comprehensive_visualizations()
        print("\n可视化完成! 图片已保存为 'sEMG_IMU_Comprehensive_Analysis.png'")
    else:
        print("数据加载失败，请检查文件路径")


if __name__ == "__main__":
    main()
