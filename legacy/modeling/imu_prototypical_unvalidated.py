# imu_angle_data_final.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import os
import glob
from sklearn.metrics import accuracy_score, classification_report
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns


# ==================== 配置参数 ====================
class Config:
    def __init__(self):
        self.input_dim = 12  # 4个传感器 × 3个角度
        self.hidden_dim = 32
        self.num_classes = 6
        self.seq_len = 200


# ==================== 改进的原型网络 ====================
class RobustPrototypicalNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(RobustPrototypicalNetwork, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv1d(input_dim, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.AdaptiveAvgPool1d(50),

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(25),
        )

        self.prototype_layer = nn.Sequential(
            nn.Linear(128 * 25, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        features = self.encoder(x)
        features = features.view(features.size(0), -1)
        embeddings = self.prototype_layer(features)
        return embeddings


# ==================== 原型学习分类器 ====================
class PrototypicalClassifier:
    def __init__(self, config):
        self.config = config
        self.model = RobustPrototypicalNetwork(
            config.input_dim, config.hidden_dim, config.num_classes
        )
        self.prototypes = {}

    def fit(self, data, labels):
        """训练原型分类器"""
        print("🎯 训练原型分类器...")

        self.model.eval()
        with torch.no_grad():
            embeddings = self.model(data)

        # 计算每个类别的原型
        self.prototypes = {}
        for class_id in range(self.config.num_classes):
            class_mask = (labels == class_id)
            if class_mask.sum() > 0:
                class_embeddings = embeddings[class_mask]
                self.prototypes[class_id] = class_embeddings.mean(dim=0)

        print(f"✅ 计算了 {len(self.prototypes)} 个类别的原型")
        return embeddings

    def predict(self, data):
        """预测新样本"""
        self.model.eval()
        with torch.no_grad():
            embeddings = self.model(data)

        predictions = []
        confidence_scores = []

        for i in range(len(data)):
            query_embedding = embeddings[i]

            # 计算与每个原型的距离
            distances = {}
            for class_id, prototype in self.prototypes.items():
                distance = torch.norm(query_embedding - prototype)
                distances[class_id] = distance.item()

            # 选择距离最近的类别
            predicted_class = min(distances, key=distances.get)
            predictions.append(predicted_class)

            # 计算置信度
            min_dist = min(distances.values())
            total_dist = sum(distances.values())
            confidence = 1 - (min_dist / total_dist * len(distances)) if total_dist > 0 else 1.0
            confidence_scores.append(confidence)

        return predictions, confidence_scores, embeddings


# ==================== 改进的数据加载器 ====================
class RobustAngleIMUDataLoader:
    def __init__(self, config):
        self.config = config

    def load_from_motion_datasets(self, base_path):
        """加载数据并进行严格的数据清洗"""
        data = []
        labels = []

        print(f"📂 从基础路径加载数据: {base_path}")

        if not os.path.exists(base_path):
            print(f"❌ 基础路径不存在: {base_path}")
            return None, None

        dataset_folders = sorted([f for f in os.listdir(base_path)
                                  if f.startswith('motion_dataset_') and
                                  os.path.isdir(os.path.join(base_path, f))])

        if not dataset_folders:
            print("❌ 未找到motion_dataset文件夹")
            return None, None

        print(f"找到 {len(dataset_folders)} 个motion_dataset文件夹")

        for dataset_folder in dataset_folders:
            try:
                dataset_num = int(dataset_folder.split('_')[-1])
                class_id = dataset_num - 1
            except:
                print(f"⚠️  无法从 {dataset_folder} 提取类别ID，跳过")
                continue

            imu_file_path = os.path.join(base_path, dataset_folder, 'IMU.csv')

            if not os.path.exists(imu_file_path):
                print(f"⚠️  在 {dataset_folder} 中未找到IMU.csv文件")
                continue

            try:
                print(f"📖 读取: {dataset_folder}/IMU.csv")
                df = pd.read_csv(imu_file_path)

                # 数据质量检查
                if self.check_data_quality(df, dataset_folder):
                    angle_columns = [col for col in df.columns if any(x in col for x in ['Roll', 'Pitch', 'Yaw'])]
                    angle_data = df[angle_columns].values

                    # 数据清洗：处理NaN和无穷大值
                    cleaned_data = self.clean_data(angle_data, dataset_folder)
                    if cleaned_data is not None:
                        sequence_data = self.process_sequence(cleaned_data)
                        if sequence_data is not None:
                            data.append(sequence_data)
                            labels.append(class_id)
                            print(f"  ✅ 成功加载 -> 类别 {class_id}")
                        else:
                            print(f"  ❌ 序列处理失败")
                    else:
                        print(f"  ❌ 数据清洗失败")
                else:
                    print(f"  ❌ 数据质量检查失败")

            except Exception as e:
                print(f"❌ 读取文件 {imu_file_path} 时出错: {e}")
                continue

        if not data:
            print("❌ 未成功加载任何数据")
            return None, None

        data_array = np.array(data)
        labels_array = np.array(labels)

        print(f"\n🎉 数据加载完成:")
        print(f"  总样本数: {len(data)}")
        print(f"  数据形状: {data_array.shape}")
        print(f"  标签分布: {np.bincount(labels_array)}")

        return torch.FloatTensor(data_array), torch.LongTensor(labels_array)

    def check_data_quality(self, df, dataset_name):
        """检查数据质量"""
        angle_columns = [col for col in df.columns if any(x in col for x in ['Roll', 'Pitch', 'Yaw'])]

        if len(angle_columns) != 12:
            print(f"  ⚠️  {dataset_name}: 角度列数量异常 ({len(angle_columns)}/12)")
            return False

        # 检查NaN值
        nan_count = df[angle_columns].isna().sum().sum()
        if nan_count > 0:
            print(f"  ⚠️  {dataset_name}: 发现 {nan_count} 个NaN值")

        # 检查无穷大值
        inf_count = np.isinf(df[angle_columns].values).sum()
        if inf_count > 0:
            print(f"  ⚠️  {dataset_name}: 发现 {inf_count} 个无穷大值")

        return True

    def clean_data(self, data, dataset_name):
        """清洗数据：处理NaN和异常值"""
        if data.size == 0:
            return None

        # 复制数据以避免修改原始数据
        cleaned = data.copy()

        # 处理NaN值：用前后值的平均值填充
        nan_mask = np.isnan(cleaned)
        if nan_mask.any():
            print(f"  🧹 {dataset_name}: 处理 {nan_mask.sum()} 个NaN值")
            for i in range(cleaned.shape[1]):
                col_data = cleaned[:, i]
                nan_indices = np.where(np.isnan(col_data))[0]
                for idx in nan_indices:
                    # 用前后非NaN值的平均值填充
                    prev_val = col_data[idx - 1] if idx > 0 and not np.isnan(col_data[idx - 1]) else 0
                    next_val = col_data[idx + 1] if idx < len(col_data) - 1 and not np.isnan(col_data[idx + 1]) else 0
                    cleaned[idx, i] = (prev_val + next_val) / 2 if prev_val != 0 or next_val != 0 else 0

        # 处理无穷大值
        inf_mask = np.isinf(cleaned)
        if inf_mask.any():
            print(f"  🧹 {dataset_name}: 处理 {inf_mask.sum()} 个无穷大值")
            cleaned[inf_mask] = 0

        # 去除异常值（使用IQR方法）
        for i in range(cleaned.shape[1]):
            col_data = cleaned[:, i]
            Q1 = np.percentile(col_data, 25)
            Q3 = np.percentile(col_data, 75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            outlier_mask = (col_data < lower_bound) | (col_data > upper_bound)
            if outlier_mask.any():
                print(f"  🧹 {dataset_name}: 通道{i}发现 {outlier_mask.sum()} 个异常值")
                # 用中位数替换异常值
                median_val = np.median(col_data[~outlier_mask])
                cleaned[outlier_mask, i] = median_val

        return cleaned

    def process_sequence(self, sequence):
        """处理序列数据"""
        if len(sequence) == 0:
            return None

        # 简单的均匀采样
        if len(sequence) > self.config.seq_len:
            indices = np.linspace(0, len(sequence) - 1, self.config.seq_len, dtype=int)
            processed = sequence[indices]
        elif len(sequence) < self.config.seq_len:
            # 重复最后一个值进行填充
            padding = np.tile(sequence[-1:], (self.config.seq_len - len(sequence), 1))
            processed = np.vstack([sequence, padding])
        else:
            processed = sequence

        return processed


# ==================== 改进的数据预处理 ====================
class RobustDataPreprocessor:
    @staticmethod
    def normalize_data(data):
        """稳健的数据标准化"""
        data_np = data.numpy() if isinstance(data, torch.Tensor) else data

        # 确保没有NaN或无穷大
        data_np = np.nan_to_num(data_np, nan=0.0, posinf=0.0, neginf=0.0)

        # 按每个样本单独标准化（避免样本间的影响）
        normalized = np.zeros_like(data_np)
        for i in range(data_np.shape[0]):  # 对每个样本
            sample = data_np[i]
            mean = np.mean(sample, axis=0)
            std = np.std(sample, axis=0)
            std = np.where(std == 0, 1, std)  # 避免除零
            normalized[i] = (sample - mean) / std

        return torch.FloatTensor(normalized)

    @staticmethod
    def analyze_data(data, labels):
        """改进的数据分析"""
        print("\n🔍 数据特性分析:")
        data_np = data.numpy()

        # 确保数据有效
        data_np = np.nan_to_num(data_np, nan=0.0, posinf=0.0, neginf=0.0)

        print(f"总体统计:")
        print(f"  数据范围: [{data_np.min():.2f}, {data_np.max():.2f}]")
        print(f"  均值: {data_np.mean():.2f} ± {data_np.std():.2f}")

        for class_id in np.unique(labels):
            class_data = data_np[labels == class_id]
            print(f"\n类别 {class_id}:")
            print(f"  样本数: {len(class_data)}")
            print(f"  数据范围: [{class_data.min():.2f}, {class_data.max():.2f}]")
            print(f"  均值: {class_data.mean():.2f} ± {class_data.std():.2f}")


# ==================== 改进的评估和可视化 ====================
class RobustEvaluator:
    @staticmethod
    def evaluate_performance(true_labels, predictions, confidence_scores=None):
        """评估模型性能"""
        accuracy = accuracy_score(true_labels, predictions)

        print(f"\n📊 模型性能评估:")
        print(f"准确率: {accuracy:.4f}")

        # 只在有预测样本的类别显示precision
        unique_predicted = np.unique(predictions)
        target_names = [f'Class {i}' for i in unique_predicted]

        print("\n详细分类报告:")
        print(classification_report(true_labels, predictions,
                                    labels=unique_predicted, target_names=target_names,
                                    zero_division=0))

        if confidence_scores:
            avg_confidence = np.mean(confidence_scores)
            print(f"平均置信度: {avg_confidence:.4f}")

        return accuracy

    @staticmethod
    def visualize_angle_data(data, labels, dataset_folders):
        """可视化角度数据"""
        plt.figure(figsize=(20, 12))

        sensors = ['R1', 'R2', 'R3', 'R4']
        angles = ['Roll', 'Pitch', 'Yaw']

        unique_classes = np.unique(labels)

        for i, class_id in enumerate(unique_classes):
            class_samples = np.where(labels == class_id)[0]
            if len(class_samples) > 0:
                sample_idx = class_samples[0]
                sample_data = data[sample_idx]

                for sensor_idx in range(4):
                    plt.subplot(4, len(unique_classes), sensor_idx * len(unique_classes) + i + 1)

                    for angle_idx in range(3):
                        channel_idx = sensor_idx * 3 + angle_idx
                        plt.plot(sample_data[:, channel_idx],
                                 label=angles[angle_idx], alpha=0.8, linewidth=1)

                    plt.title(f'{dataset_folders[class_id]}\n{sensors[sensor_idx]}')
                    plt.xlabel('Time')
                    plt.ylabel('Angle')
                    if i == 0:
                        plt.legend()

        plt.tight_layout()
        plt.savefig('robust_angle_data.png', dpi=300, bbox_inches='tight')
        plt.show()

    @staticmethod
    def visualize_results_safe(data, labels, predictions, embeddings, prototypes, dataset_folders):
        """安全的可视化（处理NaN问题）"""
        plt.figure(figsize=(18, 12))

        # 1. 简单的准确率展示
        plt.subplot(2, 3, 1)
        accuracy = accuracy_score(labels, predictions)
        colors_bar = ['green' if p == l else 'red' for p, l in zip(predictions, labels)]
        plt.bar(range(len(predictions)), [1] * len(predictions), color=colors_bar)
        plt.xticks(range(len(predictions)), [f'S{i}\nT:{l}' for i, l in enumerate(labels)])
        plt.title(f'Predictions (Accuracy: {accuracy:.2f})')
        plt.ylabel('Correct(Green)/Wrong(Red)')

        # 2. 类别分布
        plt.subplot(2, 3, 2)
        unique, counts = np.unique(labels, return_counts=True)
        plt.bar(unique, counts, alpha=0.7, color=plt.cm.Set1(np.linspace(0, 1, len(unique))))
        plt.xticks(unique, [f'DS{i + 1}' for i in unique])
        plt.title('Class Distribution')
        plt.ylabel('Sample Count')

        # 3. 数据统计
        plt.subplot(2, 3, 3)
        data_stats = []
        for class_id in unique:
            class_data = data[labels == class_id]
            data_stats.append({
                'min': class_data.min(),
                'max': class_data.max(),
                'mean': class_data.mean(),
                'std': class_data.std()
            })

        means = [stat['mean'] for stat in data_stats]
        stds = [stat['std'] for stat in data_stats]

        plt.bar(unique, means, yerr=stds, alpha=0.7, capsize=5,
                color=plt.cm.Set1(np.linspace(0, 1, len(unique))))
        plt.title('Data Statistics by Class')
        plt.xlabel('Class')
        plt.ylabel('Mean ± Std')

        # 4. 传感器数据示例
        plt.subplot(2, 3, 4)
        sample_data = data[0]  # 第一个样本
        for sensor in range(4):
            sensor_data = sample_data[:, sensor * 3:(sensor + 1) * 3]
            plt.plot(sensor_data[:, 0], label=f'R{sensor + 1}_Roll', alpha=0.7)
        plt.title('Sample Sensor Data (Roll)')
        plt.xlabel('Time')
        plt.ylabel('Angle')
        plt.legend()

        # 5. 预测分布
        plt.subplot(2, 3, 5)
        pred_counts = np.bincount(predictions, minlength=len(unique))
        true_counts = np.bincount(labels, minlength=len(unique))

        x = np.arange(len(unique))
        width = 0.35

        plt.bar(x - width / 2, true_counts, width, label='True', alpha=0.7)
        plt.bar(x + width / 2, pred_counts, width, label='Predicted', alpha=0.7)
        plt.xticks(x, [f'DS{i + 1}' for i in unique])
        plt.title('True vs Predicted Distribution')
        plt.legend()

        plt.tight_layout()
        plt.savefig('robust_results.png', dpi=300, bbox_inches='tight')
        plt.show()


# ==================== 主函数 ====================
def main():
    print("🚀 稳健的IMU角度数据动作识别")
    print("=" * 60)

    # 初始化配置
    config = Config()

    # 数据加载器
    data_loader = RobustAngleIMUDataLoader(config)

    # 加载真实数据
    print("\n1. 加载和清洗真实数据...")
    base_path = "data/processed/selected_motion_data"

    data, labels = data_loader.load_from_motion_datasets(base_path)

    if data is None:
        print("❌ 数据加载失败，退出程序")
        return

    # 获取数据集文件夹名称
    dataset_folders = sorted([f for f in os.listdir(base_path)
                              if f.startswith('motion_dataset_') and
                              os.path.isdir(os.path.join(base_path, f))])

    # 数据预处理
    print("\n2. 数据预处理...")
    data = RobustDataPreprocessor.normalize_data(data)
    print(f"预处理后数据形状: {data.shape}")
    print(f"标签分布: {np.bincount(labels.numpy())}")

    # 数据分析
    RobustDataPreprocessor.analyze_data(data, labels.numpy())

    # 可视化角度数据
    print("\n3. 生成角度数据可视化...")
    RobustEvaluator.visualize_angle_data(data.numpy(), labels.numpy(), dataset_folders)

    # 训练原型分类器
    print("\n4. 训练模型...")
    classifier = PrototypicalClassifier(config)
    embeddings = classifier.fit(data, labels)

    # 预测
    predictions, confidence_scores, test_embeddings = classifier.predict(data)

    # 评估性能
    accuracy = RobustEvaluator.evaluate_performance(labels.numpy(), predictions, confidence_scores)

    # 可视化完整结果（安全版本）
    print("\n5. 生成完整结果可视化...")
    RobustEvaluator.visualize_results_safe(data.numpy(), labels.numpy(), predictions,
                                           embeddings, classifier.prototypes, dataset_folders)

    # 保存模型
    print("\n6. 保存模型...")
    torch.save({
        'model_state_dict': classifier.model.state_dict(),
        'prototypes': classifier.prototypes,
        'config': config,
        'accuracy': accuracy,
        'dataset_folders': dataset_folders
    }, 'imu_robust_model.pth')
    print("💾 模型已保存: imu_robust_model.pth")

    # 总结和建议
    print("\n" + "=" * 60)
    print("📊 最终结果:")
    print(f"• 成功加载和清洗 {len(data)} 个样本")
    print(f"• 数据维度: {data.shape}")
    print(f"• 最终准确率: {accuracy:.4f}")

    print("\n💡 建议:")
    if accuracy < 0.5:
        print("• 当前数据量太少（每个类别只有1个样本）")
        print("• 建议采集更多数据（每个类别至少10-20个样本）")
        print("• 考虑数据增强来增加样本多样性")
    else:
        print("• 模型表现不错，可以继续优化")


if __name__ == "__main__":
    main()
