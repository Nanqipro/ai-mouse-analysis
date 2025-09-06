#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
import os
from matplotlib.colors import ListedColormap
import matplotlib.cm as cm
import argparse  # 导入命令行参数处理模块
import glob  # 导入用于文件路径模式匹配的模块
import logging
import datetime
import sys
from scipy import stats
from scipy.spatial.distance import pdist, squareform

# 初始化日志记录器
def setup_logger(output_dir=None, prefix="cluster-integrate", capture_all_output=True):
    """
    设置日志记录器，将日志消息输出到控制台和文件
    
    参数:
        output_dir: 日志文件输出目录，默认为输出到当前脚本所在目录的logs文件夹
        prefix: 日志文件名称前缀
        capture_all_output: 是否捕获所有标准输出到日志文件
    
    返回:
        logger: 已配置好的日志记录器
    """
    # 创建日志记录器
    logger = logging.getLogger(prefix)
    logger.setLevel(logging.INFO)
    
    # 清除现有的处理器，避免重复添加
    if logger.handlers:
        for handler in logger.handlers:
            logger.removeHandler(handler)
    
    # 创建格式化器
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # 添加控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 如果指定了输出目录，则添加文件处理器
    if output_dir is None:
        # 默认在当前脚本目录下创建 logs 文件夹
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建日志文件名，包含时间戳
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_file = os.path.join(output_dir, f"{prefix}_{timestamp}.log")
    
    # 添加文件处理器
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # 创建输出重定向类，捕获所有的标准输出和错误输出
    class OutputRedirector:
        def __init__(self, original_stream, logger, level=logging.INFO):
            self.original_stream = original_stream
            self.logger = logger
            self.level = level
            self.buffer = ''
        
        def write(self, buf):
            self.original_stream.write(buf)
            self.buffer += buf
            if '\n' in buf:
                self.flush()
        
        def flush(self):
            if self.buffer.strip():
                for line in self.buffer.rstrip().split('\n'):
                    if line.strip():  # 只记录非空行
                        self.logger.log(self.level, f"OUTPUT: {line.rstrip()}")
            self.buffer = ''
    
    # 重定向标准输出和错误输出到日志文件
    if capture_all_output:
        sys.stdout = OutputRedirector(sys.stdout, logger, logging.INFO)
        sys.stderr = OutputRedirector(sys.stderr, logger, logging.ERROR)
    
    logger.info(f"日志文件创建于: {log_file}")
    return logger

def load_data(file_path):
    """
    加载钙爆发数据
    
    参数
    ----------
    file_path : str
        数据文件路径
        
    返回
    -------
    df : pandas.DataFrame
        加载的数据
    """
    print(f"正在从{file_path}加载数据...")
    df = pd.read_excel(file_path)
    print(f"成功加载数据，共{len(df)}行")
    return df

def enhance_preprocess_data(df, feature_weights=None):
    """
    增强版预处理功能，支持子峰分析和更多特征，并支持特征权重调整
    
    参数
    ----------
    df : pandas.DataFrame
        原始数据
    feature_weights : dict, 可选
        特征权重字典，键为特征名称，值为权重值，默认为None（所有特征权重相等）
        
    返回
    -------
    features_scaled : numpy.ndarray
        标准化并应用权重后的特征数据
    feature_names : list
        特征名称列表
    """
    # 基础特征集
    feature_names = ['amplitude', 'duration', 'fwhm', 'rise_time', 'decay_time', 'auc', 'snr']
    
    # 检查是否包含波形类型信息，增加波形分类特征
    if 'wave_type' in df.columns:
        df['is_complex'] = df['wave_type'].apply(lambda x: 1 if x == 'complex' else 0)
        feature_names.append('is_complex')
    
    # 检查是否包含子峰信息
    if 'subpeaks_count' in df.columns:
        feature_names.append('subpeaks_count')
    
    # 将特征值转为数值类型
    for col in feature_names:
        if df[col].dtype == 'object':
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # 删除缺失值
    df = df.dropna(subset=feature_names)
    
    # 提取特征
    features = df[feature_names].values
    
    # 特征标准化
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # 应用特征权重
    if feature_weights is not None:
        weights_array = np.ones(len(feature_names))
        weight_info = []
        
        # 构建权重数组
        for i, feature in enumerate(feature_names):
            if feature in feature_weights:
                weights_array[i] = feature_weights[feature]
                weight_info.append(f"{feature}:{feature_weights[feature]:.2f}")
            else:
                weight_info.append(f"{feature}:1.00")
        
        # 应用权重
        features_scaled = features_scaled * weights_array.reshape(1, -1)
        print(f"应用特征权重: {', '.join(weight_info)}")
    else:
        print("未设置特征权重，所有特征权重相等")
    
    print(f"预处理完成，保留{len(df)}个有效样本，使用特征: {', '.join(feature_names)}")
    return features_scaled, feature_names, df

def determine_optimal_k(features_scaled, max_k=10, output_dir='../results'):
    """
    确定最佳聚类数（增强版，包含Gap Statistic方法）
    
    参数
    ----------
    features_scaled : numpy.ndarray
        标准化后的特征数据
    max_k : int, 可选
        最大测试聚类数，默认为10
    output_dir : str, 可选
        输出目录路径，默认为'../results'
        
    返回
    -------
    optimal_k : int
        最佳聚类数
    """
    print("正在确定最佳聚类数...")
    inertia = []
    silhouette_scores = []
    
    # 计算不同k值的肘部指标和轮廓系数
    for k in range(2, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(features_scaled)
        inertia.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(features_scaled, kmeans.labels_))
    
    # 计算Gap Statistic
    try:
        print("计算Gap Statistic...")
        gap_values, optimal_k_gap = calculate_gap_statistic(features_scaled, max_k=max_k)
        print(f"Gap Statistic建议的最佳聚类数: K = {optimal_k_gap}")
    except Exception as e:
        print(f"计算Gap Statistic时出错: {str(e)}")
        gap_values = None
        optimal_k_gap = None
    
    # 绘制三种方法的比较图
    fig_width = 15 if gap_values is not None else 12
    plt.figure(figsize=(fig_width, 5))
    
    n_subplots = 3 if gap_values is not None else 2
    
    # 肘部法则图
    plt.subplot(1, n_subplots, 1)
    plt.plot(range(2, max_k + 1), inertia, 'o-', color='blue', linewidth=2, markersize=6)
    plt.title('Elbow Method', fontsize=12)
    plt.xlabel('Number of Clusters (K)', fontsize=11)
    plt.ylabel('Inertia', fontsize=11)
    plt.grid(False)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    
    # 轮廓系数图
    plt.subplot(1, n_subplots, 2)
    plt.plot(range(2, max_k + 1), silhouette_scores, 'o-', color='green', linewidth=2, markersize=6)
    optimal_k_sil = silhouette_scores.index(max(silhouette_scores)) + 2
    plt.plot(optimal_k_sil, max(silhouette_scores), 'ro', markersize=8)
    plt.title('Silhouette Score', fontsize=12)
    plt.xlabel('Number of Clusters (K)', fontsize=11)
    plt.ylabel('Silhouette Score', fontsize=11)
    plt.grid(False)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    
    # Gap Statistic图（如果计算成功）
    if gap_values is not None:
        plt.subplot(1, n_subplots, 3)
        k_range = range(1, len(gap_values) + 1)
        plt.plot(k_range, gap_values, 'o-', color='red', linewidth=2, markersize=6)
        max_gap_idx = np.argmax(gap_values)
        plt.plot(max_gap_idx + 1, gap_values[max_gap_idx], 'ro', markersize=8)
        plt.title('Gap Statistic', fontsize=12)
        plt.xlabel('Number of Clusters (K)', fontsize=11)
        plt.ylabel('Gap Statistic', fontsize=11)
        plt.grid(False)
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f'{output_dir}/optimal_k_determination_enhanced.png', dpi=300)
    
    # 综合决策：优先使用Gap Statistic，其次是轮廓系数
    if optimal_k_gap is not None:
        optimal_k = optimal_k_gap
        print(f"采用Gap Statistic建议的最佳聚类数：K = {optimal_k}")
    else:
        optimal_k = silhouette_scores.index(max(silhouette_scores)) + 2
        print(f"采用轮廓系数建议的最佳聚类数：K = {optimal_k}")
    
    print(f"轮廓系数建议：K = {optimal_k_sil}")
    
    return optimal_k

def cluster_kmeans(features_scaled, n_clusters):
    """
    使用K均值聚类
    
    参数
    ----------
    features_scaled : numpy.ndarray
        标准化后的特征数据
    n_clusters : int
        聚类数
        
    返回
    -------
    labels : numpy.ndarray
        聚类标签
    """
    print(f"使用K均值聚类算法，聚类数={n_clusters}...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(features_scaled)
    return labels

def cluster_dbscan(features_scaled):
    """
    使用DBSCAN聚类
    
    参数
    ----------
    features_scaled : numpy.ndarray
        标准化后的特征数据
        
    返回
    -------
    labels : numpy.ndarray
        聚类标签
    """
    print("使用DBSCAN聚类算法...")
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    labels = dbscan.fit_predict(features_scaled)
    return labels

def visualize_clusters_2d(features_scaled, labels, feature_names, method='pca', output_dir='../results'):
    """
    使用降维方法可视化聚类结果
    
    参数
    ----------
    features_scaled : numpy.ndarray
        标准化后的特征数据
    labels : numpy.ndarray
        聚类标签
    feature_names : list
        特征名称列表
    method : str, 可选
        降维方法，'pca'或't-sne'，默认为'pca'
    output_dir : str, 可选
        输出目录路径，默认为'../results'
    """
    n_clusters = len(np.unique(labels))
    if -1 in np.unique(labels):  # DBSCAN可能有噪声点标记为-1
        n_clusters -= 1
    
    # 创建随机颜色映射
    # 使用统一的聚类颜色方案
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    if n_clusters > len(colors):
        colors = colors * (n_clusters // len(colors) + 1)
    cmap = ListedColormap(colors[:n_clusters])
    
    # 降维到2D
    if method == 'pca':
        print("使用PCA降维可视化...")
        reducer = PCA(n_components=2, random_state=42)
        embedding = reducer.fit_transform(features_scaled)
        title = 'PCA Dimensionality Reduction Cluster Visualization'
        filename = f'{output_dir}/cluster_visualization_pca.png'
    else:  # t-SNE
        print("使用t-SNE降维可视化...")
        reducer = TSNE(n_components=2, random_state=42)
        embedding = reducer.fit_transform(features_scaled)
        title = 't-SNE Dimensionality Reduction Cluster Visualization'
        filename = f'{output_dir}/cluster_visualization_tsne.png'
    
    # 创建散点图
    plt.figure(figsize=(10, 8))
    for i in np.unique(labels):
        if i == -1:  # DBSCAN noise points
            plt.scatter(embedding[labels==i, 0], embedding[labels==i, 1], 
                       c='black', marker='x', label='Noise')
        else:
            plt.scatter(embedding[labels==i, 0], embedding[labels==i, 1], 
                       c=[cmap(i)], marker='o', label=f'Cluster {i+1}')
    
    plt.title(title)
    plt.xlabel(f"{method.upper()} Dimension 1")
    plt.ylabel(f"{method.upper()} Dimension 2")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(False)
    plt.tight_layout()
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(filename, dpi=300)

def visualize_feature_distribution(df, labels, output_dir='../results'):
    """
    可视化各个簇的特征分布（学术风格改进版）
    
    参数
    ----------
    df : pandas.DataFrame
        原始数据
    labels : numpy.ndarray
        聚类标签
    output_dir : str, 可选
        输出目录路径，默认为'../results'
    """
    print("可视化各个簇的特征分布...")
    # 将标签添加到数据框
    df_cluster = df.copy()
    df_cluster['cluster'] = labels
    
    # 特征分布图
    features = ['amplitude', 'duration', 'fwhm', 'rise_time', 'decay_time', 'auc', 'snr']
    n_clusters = len(np.unique(labels))
    if -1 in np.unique(labels):  # DBSCAN可能有噪声点标记为-1
        n_clusters -= 1
    
    # 设置图形尺寸和学术风格
    fig, axes = plt.subplots(len(features), 1, figsize=(12, 4*len(features)))
    
    # 如果只有一个特征，确保axes是数组
    if len(features) == 1:
        axes = [axes]
    
    # 设置学术风格参数
    plt.rcParams.update({
        'font.size': 10,
        'axes.linewidth': 1,
        'xtick.major.width': 1,
        'ytick.major.width': 1
    })
    
    # 遍历每个特征并创建箱形图
    for i, feature in enumerate(features):
        # 使用学术风格的颜色方案
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        if n_clusters > len(colors):
            colors = colors * (n_clusters // len(colors) + 1)
        palette = {j: colors[j] for j in range(n_clusters)}
        
        # 创建箱形图，使用更学术的样式
        sns.boxplot(x='cluster', y=feature, hue='cluster', data=df_cluster, ax=axes[i], 
                   palette=palette, legend=False, linewidth=1.5, fliersize=3)
        
        # 设置标题和标签（学术风格）
        axes[i].set_title(f'{feature.title()} Distribution by Cluster', fontsize=12, fontweight='bold')
        axes[i].set_xlabel('Cluster', fontsize=11)
        axes[i].set_ylabel(f'{feature.title()}', fontsize=11)
        
        # 移除顶部和右侧边框（学术风格）
        axes[i].spines['top'].set_visible(False)
        axes[i].spines['right'].set_visible(False)
        axes[i].grid(False)
        
        # 添加轻微的背景色
        axes[i].set_facecolor('#f8f9fa')
    
    plt.tight_layout()
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f'{output_dir}/cluster_feature_distribution_academic.png', dpi=300, bbox_inches='tight')
    
    print(f"学术风格特征分布图已保存到: {output_dir}/cluster_feature_distribution_academic.png")

def analyze_clusters(df, labels, output_dir='../results'):
    """
    分析各个簇的特征统计信息
    
    参数
    ----------
    df : pandas.DataFrame
        原始数据
    labels : numpy.ndarray
        聚类标签
    output_dir : str, 可选
        输出目录路径，默认为'../results'
        
    返回
    -------
    cluster_stats : pandas.DataFrame
        每个簇的特征统计信息
    """
    print("分析各个簇的特征统计信息...")
    # 将标签添加到数据框
    df_cluster = df.copy()
    df_cluster['cluster'] = labels
    
    # 特征列表
    features = ['amplitude', 'duration', 'fwhm', 'rise_time', 'decay_time', 'auc', 'snr']
    
    # 计算每个簇的特征均值
    cluster_means = df_cluster.groupby('cluster')[features].mean()
    
    # 计算每个簇的标准差
    cluster_stds = df_cluster.groupby('cluster')[features].std()
    
    # 计算每个簇的样本数
    cluster_counts = df_cluster.groupby('cluster').size().rename('count')
    
    # 合并统计信息
    cluster_stats = pd.concat([cluster_means, cluster_stds.add_suffix('_std'), cluster_counts], axis=1)
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    # 保存到CSV
    cluster_stats.to_csv(f'{output_dir}/cluster_statistics.csv')
    
    print(f"聚类统计信息已保存到 '{output_dir}/cluster_statistics.csv'")
    return cluster_stats

def visualize_cluster_radar(cluster_stats, output_dir='../results'):
    """
    使用雷达图可视化各簇的特征
    
    参数
    ----------
    cluster_stats : pandas.DataFrame
        每个簇的特征统计信息
    output_dir : str, 可选
        输出目录路径，默认为'../results'
    """
    print("使用雷达图可视化各簇的特征...")
    features = ['amplitude', 'duration', 'fwhm', 'rise_time', 'decay_time', 'auc', 'snr']
    
    # 获取均值数据
    means = cluster_stats[features]
    
    # 标准化均值，使其适合雷达图
    scaler = StandardScaler()
    means_scaled = pd.DataFrame(scaler.fit_transform(means), 
                               index=means.index, columns=means.columns)
    
    # 准备绘图
    n_clusters = len(means_scaled)
    angles = np.linspace(0, 2*np.pi, len(features), endpoint=False).tolist()
    angles += angles[:1]  # 闭合雷达图
    
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(polar=True))
    
    for i, idx in enumerate(means_scaled.index):
        values = means_scaled.loc[idx].values.tolist()
        values += values[:1]  # 闭合雷达图
        ax.plot(angles, values, 'o-', linewidth=2, label=f'Cluster {idx+1}')
        ax.fill(angles, values, alpha=0.1)
    
    # 设置图的属性
    ax.set_thetagrids(np.degrees(angles[:-1]), features)
    ax.set_title('Comparison of Cluster Features using Radar Chart')
    # 移除网格线
    ax.grid(False)
    ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    plt.tight_layout()
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f'{output_dir}/cluster_radar.png', dpi=300)

def add_cluster_to_excel(input_file, output_file, labels, df=None):
    """
    将聚类标签添加到原始Excel文件
    
    参数
    ----------
    input_file : str
        输入文件路径，如果为"combined_data"则使用传入的df参数
    output_file : str
        输出文件路径
    labels : numpy.ndarray
        聚类标签
    df : pandas.DataFrame, 可选
        当input_file为"combined_data"时使用的数据框
    """
    print("将聚类标签添加到原始数据...")
    
    if input_file == "combined_data" and df is not None:
        # 使用已有的数据框
        df_output = df.copy()
    else:
        # 读取原始数据
        df_output = pd.read_excel(input_file)
    
    # 添加聚类列
    df_output['cluster'] = labels
    
    # 保存到新的Excel文件
    df_output.to_excel(output_file, index=False)
    print(f"聚类结果已保存到 {output_file}")

def visualize_neuron_cluster_distribution(df, labels, k_value=None, output_dir='../results'):
    """
    可视化不同神经元的聚类分布
    
    参数
    ----------
    df : pandas.DataFrame
        原始数据
    labels : numpy.ndarray
        聚类标签
    k_value : int, 可选
        当前使用的K值，用于文件命名
    output_dir : str, 可选
        输出目录路径，默认为'../results'
    """
    print("可视化不同神经元的聚类分布...")
    # 将标签添加到数据框
    df_cluster = df.copy()
    df_cluster['cluster'] = labels
    
    # 计算每个神经元不同簇的数量
    cluster_counts = df_cluster.groupby(['neuron', 'cluster']).size().unstack().fillna(0)
    
    # 修复神经元排序问题：按照数值顺序排序而不是字符串顺序
    def extract_neuron_number(neuron_name):
        """
        从神经元名称中提取数字用于排序
        
        参数
        ----------
        neuron_name : str or int
            神经元名称，可能是 'n1', '1', 1 等格式
            
        返回
        -------
        int
            提取的数字
        """
        if isinstance(neuron_name, (int, float)):
            return int(neuron_name)
        elif isinstance(neuron_name, str):
            # 移除前缀 'n' 并提取数字
            if neuron_name.startswith('n') and neuron_name[1:].isdigit():
                return int(neuron_name[1:])
            elif neuron_name.isdigit():
                return int(neuron_name)
            else:
                # 如果无法提取数字，则返回一个很大的数以确保排在最后
                return 999999
        else:
            return 999999
    
    # 获取所有神经元名称并按数值排序
    neuron_names = cluster_counts.index.tolist()
    neuron_names_sorted = sorted(neuron_names, key=extract_neuron_number)
    
    # 重新排序cluster_counts的行
    cluster_counts = cluster_counts.reindex(neuron_names_sorted)
    
    # 绘制堆叠条形图
    ax = cluster_counts.plot(kind='bar', stacked=True, figsize=(12, 6), colormap='tab10')
    ax.set_title(f'Cluster Distribution for Different Neurons (k={len(np.unique(labels))})')
    ax.set_xlabel('Neuron')
    ax.set_ylabel('Number of Calcium Transients')
    ax.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
    # 移除网格线
    ax.grid(False)
    
    plt.tight_layout()
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 根据k_value调整文件名
    if k_value:
        filename = f'{output_dir}/neuron_cluster_distribution_k{k_value}.png'
    else:
        filename = f'{output_dir}/neuron_cluster_distribution.png'
    
    plt.savefig(filename, dpi=300)
    print(f"神经元聚类分布图已保存到: {filename}")
    print(f"神经元排序已修正，按数值顺序排列: {neuron_names_sorted[:10]}...")  # 显示前10个用于确认

def visualize_wave_type_distribution(df, labels, output_dir='../results'):
    """
    可视化不同波形类型在各聚类中的分布
    
    参数
    ----------
    df : pandas.DataFrame
        包含wave_type信息的数据
    labels : numpy.ndarray
        聚类标签
    output_dir : str, 可选
        输出目录路径
    """
    if 'wave_type' not in df.columns:
        print("数据中没有wave_type信息，跳过波形类型分布可视化")
        return
        
    print("可视化不同波形类型在各聚类中的分布...")
    
    # 将标签添加到数据框
    df_cluster = df.copy()
    df_cluster['cluster'] = labels
    
    # 计算每个聚类中不同波形类型的分布
    wave_type_counts = df_cluster.groupby(['cluster', 'wave_type']).size().unstack().fillna(0)
    
    # 计算百分比
    wave_type_pcts = wave_type_counts.div(wave_type_counts.sum(axis=1), axis=0) * 100
    
    # 绘制堆叠条形图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 绝对数量图
    wave_type_counts.plot(kind='bar', stacked=True, ax=ax1, colormap='Set3')
    ax1.set_title('Wave Type Count in Each Cluster')
    ax1.set_xlabel('Cluster')
    ax1.set_ylabel('Count')
    ax1.legend(title='Wave Type')
    ax1.grid(True, alpha=0.3)
    
    # 百分比图
    wave_type_pcts.plot(kind='bar', stacked=True, ax=ax2, colormap='Set3')
    ax2.set_title('Wave Type Percentage in Each Cluster')
    ax2.set_xlabel('Cluster')
    ax2.set_ylabel('Percentage (%)')
    ax2.legend(title='Wave Type')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f'{output_dir}/wave_type_distribution.png', dpi=300)
    
    print(f"波形类型分布图已保存到: {output_dir}/wave_type_distribution.png")

def analyze_subpeaks(df, labels, output_dir='../results'):
    """
    分析各聚类中子峰特征
    
    参数
    ----------
    df : pandas.DataFrame
        包含subpeaks_count信息的数据
    labels : numpy.ndarray
        聚类标签
    output_dir : str, 可选
        输出目录路径
    """
    if 'subpeaks_count' not in df.columns:
        print("数据中没有subpeaks_count信息，跳过子峰分析")
        return
        
    print("分析各聚类中的子峰特征...")
    
    # 将标签添加到数据框
    df_cluster = df.copy()
    df_cluster['cluster'] = labels
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 绘制子峰数量箱线图 - 修复FutureWarning
    plt.figure(figsize=(10, 6))
    # 修改前: sns.boxplot(x='cluster', y='subpeaks_count', data=df_cluster, palette='Set2')
    # 修改后: 将x变量分配给hue，并设置legend=False
    # 使用统一的聚类颜色方案
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    n_clusters = len(df_cluster['cluster'].unique())
    if n_clusters > len(colors):
        colors = colors * (n_clusters // len(colors) + 1)
    palette = {i: colors[i] for i in range(n_clusters)}
    sns.boxplot(x='cluster', y='subpeaks_count', hue='cluster', data=df_cluster, palette=palette, legend=False)
    plt.title('Distribution of Subpeaks Count in Each Cluster')
    plt.xlabel('Cluster')
    plt.ylabel('Number of Subpeaks')
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/subpeaks_distribution.png', dpi=300)
    
    # 计算各聚类中子峰的统计信息
    subpeak_stats = df_cluster.groupby('cluster')['subpeaks_count'].agg(['mean', 'median', 'std', 'min', 'max'])
    subpeak_stats.to_csv(f'{output_dir}/subpeaks_statistics.csv')
    print(f"子峰统计信息已保存到 {output_dir}/subpeaks_statistics.csv")

def compare_multiple_k(features_scaled, feature_names, df_clean, k_values, input_file, output_dir='../results'):
    """
    比较不同K值的聚类效果
    
    参数
    ----------
    features_scaled : numpy.ndarray
        标准化后的特征数据
    feature_names : list
        特征名称列表
    df_clean : pandas.DataFrame
        清洗后的数据
    k_values : list
        要比较的K值列表
    input_file : str
        输入文件路径，用于生成输出文件名
    output_dir : str, 可选
        输出目录路径，默认为'../results'
    """
    print(f"正在比较不同K值的聚类效果: {k_values}...")
    
    # 确保主输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 计算每个K值的轮廓系数
    silhouette_scores_dict = {}
    
    # 创建比较图
    fig, axes = plt.subplots(1, len(k_values), figsize=(5*len(k_values), 5))
    if len(k_values) == 1:
        axes = [axes]
    
    for i, k in enumerate(k_values):
        # 为每个K值创建单独的输出目录
        k_output_dir = os.path.join(output_dir, f'k{k}')
        os.makedirs(k_output_dir, exist_ok=True)
        
        print(f"\n分析K={k}的聚类效果...")
        
        # 执行K-means聚类
        labels = cluster_kmeans(features_scaled, k)
        
        # 计算轮廓系数
        sil_score = silhouette_score(features_scaled, labels)
        silhouette_scores_dict[k] = sil_score
        print(f"K={k}的轮廓系数: {sil_score:.4f}")
        
        # 使用PCA降维可视化
        reducer = PCA(n_components=2, random_state=42)
        embedding = reducer.fit_transform(features_scaled)
        
        # 绘制聚类结果
        # 使用统一的聚类颜色方案
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        if k > len(colors):
            colors = colors * (k // len(colors) + 1)
        cmap = ListedColormap(colors[:k])
        
        # 在子图中绘制
        for j in range(k):
            axes[i].scatter(embedding[labels==j, 0], embedding[labels==j, 1], 
                         c=[cmap(j)], marker='o', label=f'Cluster {j+1}')
        
        axes[i].set_title(f'K={k}, Silhouette={sil_score:.3f}')
        axes[i].set_xlabel('PCA Dimension 1')
        axes[i].set_ylabel('PCA Dimension 2')
        # 移除网格线
        axes[i].grid(False)
        
        # 保存该K值的结果
        output_file = f'{output_dir}/transients_clustered_k{k}.xlsx'
        add_cluster_to_excel(input_file, output_file, labels, df=df_clean)
        
        # 生成该K值的特征分布图
        visualize_feature_distribution(df_clean, labels, output_dir=k_output_dir)
        
        # 神经元簇分布
        visualize_neuron_cluster_distribution(df_clean, labels, k_value=k, output_dir=k_output_dir)
        
        # 波形类型分析
        visualize_wave_type_distribution(df_clean, labels, output_dir=k_output_dir)
        
        # 子峰分析
        analyze_subpeaks(df_clean, labels, output_dir=k_output_dir)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/k_comparison.png', dpi=300)
    
    # 绘制轮廓系数比较图
    plt.figure(figsize=(8, 5))
    plt.bar(silhouette_scores_dict.keys(), silhouette_scores_dict.values(), color='skyblue')
    plt.title('Silhouette Score Comparison for Different K Values')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Silhouette Score')
    plt.grid(False)
    plt.xticks(list(silhouette_scores_dict.keys()))
    plt.tight_layout()
    plt.savefig(f'{output_dir}/silhouette_comparison.png', dpi=300)
    
    print("不同K值对比完成，结果已保存")
    
    # 返回轮廓系数最高的K值
    best_k = max(silhouette_scores_dict, key=silhouette_scores_dict.get)
    return best_k

def visualize_cluster_waveforms(df, labels, output_dir='../results', raw_data_path=None, raw_data_dir=None, sampling_freq=4.8):
    """
    可视化不同聚类类别的平均钙爆发波形，以钙波开始时间为原点，只展示X轴正半轴部分
    
    参数
    ----------
    df : pandas.DataFrame
        包含钙爆发特征的数据框，必须包含start_idx, peak_idx, end_idx和neuron字段
    labels : numpy.ndarray
        聚类标签
    output_dir : str, 可选
        输出目录路径，默认为'../results'
    raw_data_path : str, 可选
        单个原始数据文件路径
    raw_data_dir : str, 可选
        原始数据文件目录，用于查找多个数据文件
    sampling_freq : float, 可选
        采样频率，单位Hz，默认为4.8Hz，用于将数据点转换为以秒为单位
    """
    """
    可视化不同聚类类别的平均钙爆发波形
    
    参数
    ----------
    df : pandas.DataFrame
        包含钙爆发特征的数据框，必须包含start_idx, peak_idx, end_idx和neuron字段
    labels : numpy.ndarray
        聚类标签
    output_dir : str, 可选
        输出目录路径，默认为'../results'
    raw_data_path : str, 可选
        单个原始数据文件路径
    raw_data_dir : str, 可选
        原始数据文件目录，用于查找多个数据文件
    """
    print("正在可视化不同聚类类别的平均钙爆发波形...")
    
    # 设置时间窗口（采样点数）- 减小窗口大小以提高匹配成功率
    time_window = 200
    
    # 将标签添加到数据框
    df_cluster = df.copy()
    df_cluster['cluster'] = labels
    
    # 检查必要的字段
    required_fields = ['start_idx', 'peak_idx', 'end_idx', 'neuron']
    if not all(field in df_cluster.columns for field in required_fields):
        print("错误: 数据中缺少必要字段(start_idx, peak_idx, end_idx, neuron)，无法绘制波形")
        return
    
    # 获取聚类的数量
    n_clusters = len(np.unique(labels))
    if -1 in np.unique(labels):  # DBSCAN可能有噪声点标记为-1
        n_clusters -= 1
    
    # 尝试加载原始数据
    raw_data_dict = {}
    
    # 检查是否存在dataset列，表示合并了不同数据集
    has_dataset_column = 'dataset' in df_cluster.columns
    
    # 检查是否包含源文件信息
    has_source_info = all(col in df_cluster.columns for col in ['source_file', 'source_path', 'source_abs_path'])
    if has_source_info:
        print("检测到源文件路径信息，将优先使用这些信息加载原始数据")
    
    if raw_data_path:
        # 如果指定了单个原始数据文件路径
        try:
            print(f"加载原始数据从: {raw_data_path}")
            raw_data = pd.read_excel(raw_data_path)
            # 使用文件名作为数据集名称
            dataset_name = os.path.splitext(os.path.basename(raw_data_path))[0]
            raw_data_dict[dataset_name] = raw_data
            print(f"  已加载数据集: {dataset_name}, 形状: {raw_data.shape}")
        except Exception as e:
            print(f"无法加载原始数据: {str(e)}")
            return
    elif raw_data_dir:
        # 如果指定了原始数据目录，查找所有Excel文件
        try:
            excel_files = glob.glob(os.path.join(raw_data_dir, "**/*.xlsx"), recursive=True)
            print(f"在目录{raw_data_dir}下找到{len(excel_files)}个Excel文件")
            
            for file in excel_files:
                # 使用文件名作为数据集名称，而不是目录名
                dataset_name = os.path.splitext(os.path.basename(file))[0]
                try:
                    raw_data = pd.read_excel(file)
                    raw_data_dict[dataset_name] = raw_data
                    print(f"  已加载数据集: {dataset_name}, 形状: {raw_data.shape}")
                except Exception as e:
                    print(f"  加载数据集{dataset_name}失败: {str(e)}")
        except Exception as e:
            print(f"搜索原始数据文件时出错: {str(e)}")
            return
    else:
        # 使用源文件信息加载原始数据（如果可用）
        if has_source_info:
            # 获取不同的源文件
            unique_source_files = df_cluster['source_path'].unique()
            print(f"从事件数据中检测到 {len(unique_source_files)} 个不同的源文件")
            
            # 使用项目根目录（通常是工作目录的上一级）作为基础
            root_dir = os.path.abspath(os.path.join(os.getcwd(), '../..'))
            print(f"使用项目根目录: {root_dir}")
            
            # 加载每个源文件
            for source_path in unique_source_files:
                try:
                    # 构建绝对路径
                    if os.path.isabs(source_path):
                        abs_path = source_path
                    else:
                        abs_path = os.path.join(root_dir, source_path)
                    
                    if os.path.exists(abs_path):
                        print(f"加载源文件: {abs_path}")
                        raw_data = pd.read_excel(abs_path)
                        dataset_name = os.path.splitext(os.path.basename(abs_path))[0]
                        raw_data_dict[dataset_name] = raw_data
                        print(f"  成功加载数据集: {dataset_name}, 形状: {raw_data.shape}")
                    else:
                        print(f"  源文件不存在: {abs_path}")
                except Exception as e:
                    print(f"  加载源文件 {source_path} 失败: {str(e)}")
        
        # 如果无法使用源文件信息或未找到任何文件，尝试使用默认位置
        if not raw_data_dict:
            try:
                # 直接指定原始数据路径
                raw_data_path = "../datasets/processed_EMtrace.xlsx"
                print(f"尝试加载默认原始数据从: {raw_data_path}")
                
                # 加载原始数据
                raw_data = pd.read_excel(raw_data_path)
                dataset_name = os.path.splitext(os.path.basename(raw_data_path))[0]
                raw_data_dict[dataset_name] = raw_data
                print(f"成功加载原始数据，形状: {raw_data.shape}")
            except Exception as e:
                print(f"无法加载默认原始数据: {str(e)}")
                print("尝试在../datasets目录下搜索原始数据...")
                
                try:
                    # 尝试搜索datasets目录下的所有Excel文件
                    datasets_dir = "../datasets"
                    excel_files = glob.glob(os.path.join(datasets_dir, "*.xlsx"))
                    
                    if excel_files:
                        for file in excel_files:
                            dataset_name = os.path.splitext(os.path.basename(file))[0]
                            try:
                                raw_data = pd.read_excel(file)
                                raw_data_dict[dataset_name] = raw_data
                                print(f"  已加载数据集: {dataset_name}, 形状: {raw_data.shape}")
                            except Exception as e:
                                print(f"  加载数据集{dataset_name}失败: {str(e)}")
                    else:
                        print("在../datasets目录下未找到任何Excel文件")
                        return
                except Exception as e:
                    print(f"搜索原始数据时出错: {str(e)}")
                    return
    
    if not raw_data_dict:
        print("未能加载任何原始数据，无法可视化波形")
        return
    
    # 打印所有可用神经元列以供调试
    print("原始数据中的神经元列：")
    for dataset_name, data in raw_data_dict.items():
        neuron_cols = [col for col in data.columns if col.startswith('n') and col[1:].isdigit()]
        print(f"  数据集 {dataset_name}: {len(neuron_cols)} 个神经元列 - {neuron_cols[:5]}...")
    
    # 打印钙爆发数据中的神经元名称以供调试
    unique_neurons = df_cluster['neuron'].unique()
    print(f"钙爆发数据中的神经元: {len(unique_neurons)} 个 - {unique_neurons[:5]}...")
    
    # 创建神经元名称映射，处理可能的命名不一致问题
    neuron_mapping = {}
    for neuron_name in unique_neurons:
        # 检查神经元名称是否以'n'开头，并且第二个字符是数字
        if isinstance(neuron_name, str) and neuron_name.startswith('n') and neuron_name[1:].isdigit():
            # 保持原名
            neuron_mapping[neuron_name] = neuron_name
        elif isinstance(neuron_name, (int, float)) or (isinstance(neuron_name, str) and neuron_name.isdigit()):
            # 如果是纯数字，则转为"n数字"格式
            formatted_name = f"n{int(float(neuron_name))}"
            neuron_mapping[neuron_name] = formatted_name
    
    print(f"创建了 {len(neuron_mapping)} 个神经元名称映射")
    
    # 创建颜色映射 - 使用统一的聚类颜色方案
    # 为聚类定义固定颜色
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    # 确保颜色足够
    if n_clusters > len(colors):
        # 如果聚类数量超过预设颜色，则循环使用
        colors = colors * (n_clusters // len(colors) + 1)
    
    # 创建自定义颜色映射用于matplotlib
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(colors[:n_clusters])
    
    # 创建自定义颜色映射用于plotly
    cluster_colors = {i: colors[i] for i in range(n_clusters)}
    
    # 为每个聚类提取和平均波形
    plt.figure(figsize=(12, 8))
    
    # 记录不同聚类的平均波形数据，用于保存
    avg_waveforms = {}
    
    for cluster_id in range(n_clusters):
        # 获取当前聚类的所有钙爆发事件
        cluster_events = df_cluster[df_cluster['cluster'] == cluster_id]
        
        if len(cluster_events) == 0:
            continue
        
        # 收集所有波形，指定从start开始的固定长度
        all_waveforms = []
        fixed_length = time_window * 2  # 固定长度：足够长以显示完整钙波
        print(f"聚类 {cluster_id+1}: 使用从起始点开始的固定长度{fixed_length}处理波形...")
        
        # 对每个事件，提取波形
        for idx, event in cluster_events.iterrows():
            neuron_col = event['neuron']
            
            # 应用神经元名称映射
            if neuron_col in neuron_mapping:
                neuron_col = neuron_mapping[neuron_col]
            
            # 确定使用哪个原始数据集
            raw_data = None
            
            # 1. 优先使用源文件信息精确匹配
            if has_source_info and 'source_file' in event:
                source_file = event['source_file']
                # 提取不带扩展名的文件名作为数据集名称
                source_dataset = os.path.splitext(source_file)[0]
                if source_dataset in raw_data_dict:
                    raw_data = raw_data_dict[source_dataset]
                    # 只对每个聚类的第一个事件显示此消息，避免输出过多
                    if idx == cluster_events.index[0]:
                        print(f"聚类 {cluster_id+1}: 使用源文件信息匹配原始数据")
            
            # 2. 如果无法通过源文件匹配，尝试使用dataset列
            if raw_data is None and has_dataset_column and 'dataset' in event and event['dataset'] in raw_data_dict:
                # 如果事件有数据集标识且该数据集已加载
                raw_data = raw_data_dict[event['dataset']]
            
            # 3. 如果前两种方法都失败，尝试所有数据集进行列名匹配
            if raw_data is None or neuron_col not in raw_data.columns:
                # 尝试所有数据集，查找包含此神经元的数据集
                for dataset_name, dataset_raw_data in raw_data_dict.items():
                    if neuron_col in dataset_raw_data.columns:
                        raw_data = dataset_raw_data
                        break
            
            if raw_data is None or neuron_col not in raw_data.columns:
                # 如果还找不到，尝试其他命名方式
                for dataset_name, dataset_raw_data in raw_data_dict.items():
                    # 尝试格式如 "n3" 或 "3" 等
                    if neuron_col.lstrip('n') in dataset_raw_data.columns:
                        neuron_col = neuron_col.lstrip('n')
                        raw_data = dataset_raw_data
                        break
                    elif f"n{neuron_col}" in dataset_raw_data.columns:
                        neuron_col = f"n{neuron_col}"
                        raw_data = dataset_raw_data
                        break
                
                # 如果仍找不到，则跳过此事件
                if raw_data is None or neuron_col not in raw_data.columns:
                    continue
            
            # 提取以start_idx为起点的时间窗口数据
            try:
                # 获取起始点和峰值点
                start_idx = int(event['start_idx'])
                peak_idx = int(event['peak_idx'])
                
                # 计算从起始点到峰值点的距离
                peak_offset = peak_idx - start_idx
                
                # 设置新的窗口大小，以起始点为原点，只展示正半轴
                window_end = time_window * 2  # 扩大窗口以确保能看到完整的钙波
                
                # 确定提取的起始点和结束点
                start = max(0, start_idx)  # 从起始点开始
                end = min(len(raw_data), start_idx + window_end + 1)  # 到足够长的时间展示完整波形
                
                # 如果提取的窗口不够长，进行调整
                if end - start < window_end:
                    # 如果窗口太小则跳过
                    if end - start < 20:
                        continue
                    window_end = end - start
                
                # 提取波形
                waveform = raw_data[neuron_col].values[start:end]
                
                # 创建相对于start_idx的时间点数组（单位为秒）
                time_points = np.arange(0, len(waveform)) / sampling_freq
                
                # 计算peak位置相对于start的偏移（用于后续标记）
                peak_relative_pos = peak_idx - start
                
                # 确保所有波形长度统一，便于后续平均计算
                fixed_length = window_end  # 使用固定长度
                if len(waveform) != fixed_length:
                    # 修剪或填充波形以匹配固定长度
                    if len(waveform) > fixed_length:
                        waveform = waveform[:fixed_length]
                    else:
                        # 填充不足部分
                        padding = np.full(fixed_length - len(waveform), np.nan)
                        waveform = np.concatenate([waveform, padding])
                    
                    # 重置时间点数组为固定长度（单位为秒）
                    time_points = np.arange(fixed_length) / sampling_freq
                
                # 归一化处理：减去基线并除以峰值振幅
                # 忽略NaN值
                valid_indices = ~np.isnan(waveform)
                if np.sum(valid_indices) > 10:  # 确保有足够的有效点
                    baseline = np.nanmin(waveform)
                    amplitude = np.nanmax(waveform) - baseline
                    if amplitude > 0:  # 避免除以零
                        norm_waveform = (waveform - baseline) / amplitude
                        all_waveforms.append(norm_waveform)
            except Exception as e:
                print(f"处理事件 {idx} 时出错: {str(e)}")
                continue
        
        # 如果没有有效波形，跳过此聚类
        if len(all_waveforms) == 0:
            print(f"警告: 聚类 {cluster_id+1} 没有有效波形")
            continue
        
        # 预处理所有波形，确保长度一致
        # 转换为统一长度的波形数组之前，先确认所有波形长度是否已一致
        wave_lengths = [len(w) for w in all_waveforms]
        if len(set(wave_lengths)) > 1:
            # 存在长度不一致的情况，调整为统一长度
            max_len = max(wave_lengths)
            standardized_waveforms = []
            for w in all_waveforms:
                if len(w) < max_len:
                    padding = np.full(max_len - len(w), np.nan)
                    std_w = np.concatenate([w, padding])
                else:
                    std_w = w
                standardized_waveforms.append(std_w)
            all_waveforms = standardized_waveforms
            # 调整时间点以匹配，使用从0开始的时间点（单位为秒）
            time_points = np.arange(max_len) / sampling_freq
            
        # 计算平均波形（忽略NaN值）
        all_waveforms_array = np.array(all_waveforms)
        avg_waveform = np.nanmean(all_waveforms_array, axis=0)
        std_waveform = np.nanstd(all_waveforms_array, axis=0)
        
        # 存储平均波形
        avg_waveforms[f"Cluster_{cluster_id+1}"] = {
            "time": time_points,
            "mean": avg_waveform,
            "std": std_waveform,
            "n_samples": len(all_waveforms)
        }
        
        # 绘制平均波形 - 移除标准差范围，仅绘制平均曲线
        plt.plot(time_points, avg_waveform, 
                 color=cmap(cluster_id), 
                 linewidth=2.5,  # 增加线宽使曲线更突出
                 label=f'Cluster {cluster_id+1} (n={len(all_waveforms)})')
        
        # 移除标准差范围的绘制
        # plt.fill_between(time_points, 
        #                  avg_waveform - std_waveform, 
        #                  avg_waveform + std_waveform, 
        #                  color=cmap(cluster_id), 
        #                  alpha=0.2)
    
    # 检查是否有任何有效的聚类波形
    if not avg_waveforms:
        print("没有找到任何有效的波形数据，无法生成波形图")
        return
    
    # 设置图表属性
    # 标记峰值位置（如果有记录）- 使用平均峰值位置
    peak_positions = [np.argmax(avg_waveforms[f"Cluster_{i+1}"]["mean"]) for i in range(n_clusters) if f"Cluster_{i+1}" in avg_waveforms]
    if peak_positions:
        avg_peak_position = int(np.mean(peak_positions))
        # 将数据点转换为秒
        avg_peak_position_sec = avg_peak_position / sampling_freq
        plt.axvline(x=avg_peak_position_sec, color='grey', linestyle='--', alpha=0.7, label='Average Peak Position')
    
    plt.title('Typical Calcium Wave Morphology Comparison (Cluster Averages)', fontsize=14)
    plt.xlabel('Time Relative to Start Point (seconds)', fontsize=12)
    plt.ylabel('Normalized Fluorescence Intensity (F/F0)', fontsize=12)
    plt.grid(False)
    plt.legend(loc='upper right')
    # 设置X轴只显示正半轴部分，并限制最大范围为50秒
    plt.xlim(left=0, right=50)  # 从0开始显示X轴，最大显示50秒
    plt.grid(False)
    
    # 添加额外标注说明X轴起点为钙波起始位置
    # 添加合适的标注位置，考虑到现在横坐标是秒
    annotation_x = 0.2 # 秒
    plt.annotate('Calcium Wave Start Point', xy=(0, 0), xytext=(annotation_x, 0.1), 
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8, alpha=0.7))
    
    # 保存图表
    os.makedirs(output_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/cluster_average_waveforms.png', dpi=300)
    # 保存第二个版本，文件名表明是以起始点为起点的可视化
    # plt.savefig(f'{output_dir}/cluster_average_waveforms_from_start.png', dpi=300)
    
    # 保存平均波形数据 - 修复append方法已弃用的问题
    waveform_data = []
    for cluster_name, waveform_data_dict in avg_waveforms.items():
        for i, t in enumerate(waveform_data_dict["time"]):
            waveform_data.append({
                "cluster": cluster_name,
                "time_point": t,
                "mean_intensity": waveform_data_dict["mean"][i],
                "std_intensity": waveform_data_dict["std"][i],
                "n_samples": waveform_data_dict["n_samples"]
            })
    
    # 创建DataFrame
    waveform_df = pd.DataFrame(waveform_data)
    waveform_df.to_csv(f'{output_dir}/cluster_average_waveforms.csv', index=False)
    
    print(f"平均钙爆发波形可视化已保存到 {output_dir}/cluster_average_waveforms.png")
    print(f"波形数据已保存到 {output_dir}/cluster_average_waveforms.csv")

def visualize_integrated_neuron_timeline(df, labels, neuron_map_path=None, output_dir='../results', use_timestamp=False, interactive=False, sampling_freq=4.8):
    """
    基于神经元ID对应表可视化不同神经元的钙爆发时间线，将不同数据集的神经元数据整合到一个时间线图上
    
    参数
    ----------
    df : pandas.DataFrame
        包含钙爆发特征的数据框，必须包含start_idx, end_idx, neuron, duration和dataset字段
    labels : numpy.ndarray
        聚类标签
    neuron_map_path : str, 可选
        神经元对应表路径，默认为None，会自动查找'../datasets/神经元对应表.xlsx'
    output_dir : str, 可选
        输出目录路径，默认为'../results'
    use_timestamp : bool, 可选
        是否使用时间戳模式，如果为True且数据中有timestamp字段，则使用时间戳作为X轴；否则使用start_idx
    interactive : bool, 可选
        是否创建交互式图表（使用plotly），默认为False
    sampling_freq : float, 可选
        采样频率，单位Hz，默认为4.8Hz，用于将帧索引转换为时间
    """
    print("开始生成整合神经元钙波活动时间线图...")
    
    # 将标签添加到数据框
    df_cluster = df.copy()
    df_cluster['cluster'] = labels
    
    # 检查必要的字段
    required_fields = ['start_idx', 'end_idx', 'neuron', 'duration']
    if not all(field in df_cluster.columns for field in required_fields):
        print("错误: 数据中缺少必要字段(start_idx, end_idx, neuron, duration)，无法绘制时间线")
        return
    
    # 检查是否有dataset字段，如果没有，尝试添加
    if 'dataset' not in df_cluster.columns:
        if 'source_file' in df_cluster.columns:
            # 从source_file提取数据集信息
            df_cluster['dataset'] = df_cluster['source_file'].apply(lambda x: os.path.splitext(os.path.basename(x))[0] if isinstance(x, str) else "unknown")
        else:
            # 如果没有数据集信息，默认为同一数据集
            print("警告: 数据中没有dataset字段或source_file字段，假设所有数据来自同一数据集")
            df_cluster['dataset'] = "default_dataset"
    
    # 加载神经元对应表
    if neuron_map_path is None:
        # 尝试自动查找神经元对应表
        default_map_path = '../datasets/神经元对应表.xlsx'
        if os.path.exists(default_map_path):
            neuron_map_path = default_map_path
        else:
            print("错误: 未指定神经元对应表路径，且未找到默认位置的对应表")
            return
    
    try:
        print(f"加载神经元对应表: {neuron_map_path}")
        neuron_map_df = pd.read_excel(neuron_map_path)
        print(f"成功加载神经元对应表，形状: {neuron_map_df.shape}")
        
        # 获取对应表中的所有列名（数据集名称）
        dataset_cols = neuron_map_df.columns.tolist()
        print(f"对应表中的数据集: {dataset_cols}")
    except Exception as e:
        print(f"加载神经元对应表失败: {str(e)}")
        return
    
    # 创建神经元ID映射字典
    neuron_id_mapping = {}
    
    # 为每个数据集创建神经元ID映射
    for col in dataset_cols:
        neuron_id_mapping[col] = {}
        # 遍历每一行，建立该数据集中神经元ID与统一ID的映射
        for idx, row in neuron_map_df.iterrows():
            # 检查值是否为NaN或None
            if pd.notna(row[col]) and row[col] is not None:
                # 使用行索引作为统一ID，将各数据集的神经元ID映射到这个统一ID
                unified_id = idx  # 行索引作为统一ID
                orig_id = row[col]
                # 处理可能的数据类型问题
                if isinstance(orig_id, (float, int)):
                    orig_id = str(int(orig_id))
                elif isinstance(orig_id, str):
                    # 将'n1'格式转为'1'，以便统一处理
                    if orig_id.startswith('n') and orig_id[1:].isdigit():
                        orig_id = orig_id[1:]
                else:
                    continue  # 跳过无法处理的值
                
                # 进行两种格式的映射以增加匹配成功率
                neuron_id_mapping[col][orig_id] = unified_id          # '1' -> unified_id
                neuron_id_mapping[col][f"n{orig_id}"] = unified_id    # 'n1' -> unified_id
    
    print(f"创建了 {len(neuron_id_mapping)} 个数据集的神经元ID映射")
    
    # 添加统一神经元ID列
    df_cluster['unified_neuron_id'] = None
    
    # 根据数据集和原始神经元ID确定统一ID
    for idx, row in df_cluster.iterrows():
        dataset = row['dataset']
        orig_neuron = row['neuron']
        
        # 获取该数据集的映射字典
        if dataset in neuron_id_mapping:
            mapping = neuron_id_mapping[dataset]
            
            # 处理不同格式的神经元ID
            if isinstance(orig_neuron, (float, int)):
                orig_neuron = str(int(orig_neuron))
            elif isinstance(orig_neuron, str):
                if orig_neuron.startswith('n') and orig_neuron[1:].isdigit():
                    # 如果是'n1'格式，也尝试使用'1'格式查找
                    stripped_neuron = orig_neuron[1:]
                    if stripped_neuron in mapping:
                        df_cluster.at[idx, 'unified_neuron_id'] = mapping[stripped_neuron]
                        continue
            
            # 查找统一ID
            if orig_neuron in mapping:
                df_cluster.at[idx, 'unified_neuron_id'] = mapping[orig_neuron]
            elif f"n{orig_neuron}" in mapping:
                df_cluster.at[idx, 'unified_neuron_id'] = mapping[f"n{orig_neuron}"]
    
    # 检查映射结果
    mapped_count = df_cluster['unified_neuron_id'].notna().sum()
    total_count = len(df_cluster)
    mapping_rate = mapped_count / total_count * 100 if total_count > 0 else 0
    
    print(f"神经元ID映射完成: 成功映射 {mapped_count}/{total_count} 条记录 ({mapping_rate:.2f}%)")
    
    # 如果映射率太低，发出警告但继续处理
    if mapping_rate < 50:
        print(f"警告: 神经元ID映射率低于50%，可能导致时间线图不完整")
    
    # 过滤掉未能映射的记录
    df_mapped = df_cluster.dropna(subset=['unified_neuron_id']).copy()
    
    # 如果没有足够的映射记录，返回
    if len(df_mapped) < 10:
        print("错误: 映射后的记录数量不足，无法生成有意义的时间线图")
        return
    
    # 设置统一线条粗细
    line_width = 5  # 使用统一的线条粗细
    df_mapped['line_width'] = line_width
    
    # 检查是否使用时间戳模式
    has_timestamp = 'timestamp' in df_mapped.columns
    if use_timestamp and has_timestamp:
        time_mode = "timestamp"
        x_label = "Time (Timestamp)"
        print("使用时间戳作为X轴...")
    else:
        time_mode = "frame"
        x_label = "Time (Seconds)"
        if use_timestamp and not has_timestamp:
            print("数据中没有timestamp字段，将使用帧索引作为替代...")
            
    # 为不同数据集设置X轴上的偏移量，使数据集在时间轴上依次排列而不是重叠
    datasets = sorted(df_mapped['dataset'].unique())
    print(f"检测到 {len(datasets)} 个不同的数据集: {datasets}")
    
    # 计算每个数据集的最大持续时间，用于设置偏移量
    dataset_max_times = {}
    for dataset in datasets:
        dataset_data = df_mapped[df_mapped['dataset'] == dataset]
        if time_mode == "timestamp" and has_timestamp:
            max_time = dataset_data['timestamp'].max() + dataset_data['duration'].max()
        else:
            # 将最大帧索引转换为秒
            max_time = (dataset_data['start_idx'].max() + dataset_data['duration'].max()) / sampling_freq
        dataset_max_times[dataset] = max_time
        print(f"数据集 {dataset} 的最大时间为 {max_time:.2f} {x_label.lower()}")
    
    # 为每个数据集设置偏移量
    dataset_to_offset = {}
    current_offset = 0
    padding = 10  # 数据集之间的间隔（以秒为单位）
    
    for dataset in datasets:
        dataset_to_offset[dataset] = current_offset
        print(f"数据集 {dataset} 的时间偏移量设置为 {current_offset:.2f} {x_label.lower()}")
        current_offset += dataset_max_times[dataset] + padding
    
    # 获取所有唯一的统一神经元ID并排序
    unified_neurons = sorted(df_mapped['unified_neuron_id'].unique())
    
    # 创建统一神经元ID到Y轴位置的映射
    neuron_to_y = {neuron: i for i, neuron in enumerate(unified_neurons)}
    
    # 获取聚类的数量
    n_clusters = len(np.unique(labels))
    if -1 in np.unique(labels):  # DBSCAN可能有噪声点标记为-1
        n_clusters -= 1
    
    # 创建颜色映射 - 使用统一的聚类颜色方案
    # 为聚类定义固定颜色
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    # 确保颜色足够
    if n_clusters > len(colors):
        # 如果聚类数量超过预设颜色，则循环使用
        colors = colors * (n_clusters // len(colors) + 1)
    
    # 创建自定义颜色映射用于matplotlib
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(colors[:n_clusters])
    
    # 创建自定义颜色映射用于plotly
    cluster_colors = {i: colors[i] for i in range(n_clusters)}
    
    # 为matplotlib创建colormap
    try:
        # 使用新的推荐方式替代被弃用的get_cmap函数
        cmap = plt.colormaps['tab10']
        # 获取颜色映射实例
        cmap_instance = lambda i: cmap(i % 10)  # 确保索引在范围内
    except (AttributeError, ValueError):
        # 兼容旧版本
        cmap = plt.cm.tab10
        cmap_instance = lambda i: cmap(i % 10)  # 确保索引在范围内
    
    # 根据是否使用交互式模式选择不同的绘图方法
    if interactive:
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            
            # 创建plotly图形
            fig = make_subplots(specs=[[{"secondary_y": False}]])
            
            # 按聚类绘制时间线
            for cluster_id in range(n_clusters):
                # 提取属于该聚类的钙爆发事件
                cluster_events = df_mapped[df_mapped['cluster'] == cluster_id]
                
                # 如果该簇没有事件，则跳过
                if len(cluster_events) == 0:
                    continue
                
                # 使用预定义的颜色方案
                cluster_color = cluster_colors[cluster_id]
                
                # 准备绘图数据
                for _, event in cluster_events.iterrows():
                    unified_neuron = event['unified_neuron_id']
                    y_position = neuron_to_y[unified_neuron]
                    
                    # 根据模式选择时间轴数据
                    if time_mode == "timestamp" and has_timestamp:
                        start_time = event['timestamp']
                        end_time = start_time + event['duration']
                    else:
                        # 将帧索引转换为秒
                        start_time = event['start_idx'] / sampling_freq
                        # 注意：duration在帧索引下是帧数，需要转换为秒
                        end_time = start_time + (event['duration'] / sampling_freq)
                    
                    # 准备悬停信息
                    hover_text = f"统一神经元ID: {unified_neuron}<br>原始神经元: {event['neuron']}<br>数据集: {event['dataset']}<br>聚类: {cluster_id+1}<br>开始: {start_time:.2f}<br>结束: {end_time:.2f}"
                    if 'amplitude' in event:
                        hover_text += f"<br>振幅: {event['amplitude']:.2f}"
                    
                    # 绘制水平线段，使用统一线宽
                    fig.add_trace(
                        go.Scatter(
                            x=[start_time, end_time],
                            y=[y_position, y_position],
                            mode='lines',
                            line=dict(
                                color=cluster_color,
                                width=line_width
                            ),
                            name=f'Cluster {cluster_id+1}',
                            legendgroup=f'Cluster {cluster_id+1}',
                            showlegend=(y_position == neuron_to_y[unified_neurons[0]]),  # 只在第一次出现时显示图例
                            hoverinfo='text',
                            hovertext=hover_text
                        )
                    )
            
            # 添加数据集分隔标记
            shapes = []
            annotations = []
            for dataset, offset in dataset_to_offset.items():
                # 添加垂直分隔线
                shapes.append(dict(
                    type="line",
                    x0=offset, y0=0,
                    x1=offset, y1=1,
                    yref="paper",
                    line=dict(color="gray", width=1, dash="dash")
                ))
                
                # 添加数据集名称标签
                annotations.append(dict(
                    x=offset + 2,
                    y=1.05,
                    xref="x",
                    yref="paper",
                    text=dataset,
                    showarrow=False,
                    font=dict(size=10),
                    bgcolor="white",
                    bordercolor="gray",
                    borderwidth=1,
                    borderpad=3,
                    opacity=0.8
                ))
            
            # 设置布局
            fig.update_layout(
                title='Integrated Neuron Calcium Activity Timeline (Arranged by Dataset)',
                xaxis_title=f"{x_label} (Including Dataset Offset)",
                yaxis_title='Unified Neuron ID',
                yaxis=dict(
                    tickmode='array',
                    tickvals=list(neuron_to_y.values()),
                    ticktext=[f"Neuron {i}" for i in neuron_to_y.keys()]
                ),
                hoverlabel=dict(
                    bgcolor="white",
                    font_size=12
                ),
                height=800,   # 调整高度
                width=1500,   # 调整宽度为15:8比例
                shapes=shapes,
                annotations=annotations
            )
            
            # 保存交互式图表为HTML文件
            suffix = "_timestamp" if time_mode == "timestamp" else ""
            os.makedirs(output_dir, exist_ok=True)
            html_path = f'{output_dir}/integrated_neuron_timeline{suffix}_interactive.html'
            fig.write_html(html_path)
            
            print(f"交互式整合神经元钙波活动时间线图已保存到 {html_path}")
                
        except ImportError:
            print("无法导入plotly，回退到使用matplotlib生成静态图表...")
            interactive = False
    
    # 如果不使用交互式或者plotly导入失败，使用matplotlib绘制
    if not interactive:
        # 创建图形 - 调整比例为15:8
        plt.figure(figsize=(20, 8))
        
        # 按聚类绘制时间线，确保颜色对应聚类
        for cluster_id in range(n_clusters):
            # 提取属于该聚类的钙爆发事件
            cluster_events = df_mapped[df_mapped['cluster'] == cluster_id]
            
            # 如果该簇没有事件，则跳过
            if len(cluster_events) == 0:
                continue
            
            # 使用预定义的颜色方案
            cluster_color = cluster_colors[cluster_id]
            
            # 为每个事件绘制条线
            for _, event in cluster_events.iterrows():
                unified_neuron = event['unified_neuron_id']
                y_position = neuron_to_y[unified_neuron]
                
                # 根据模式选择时间轴数据
                dataset = event['dataset']
                # 获取该数据集的时间偏移量
                offset = dataset_to_offset[dataset]
                
                if time_mode == "timestamp" and has_timestamp:
                    start_time = event['timestamp'] + offset  # 添加偏移量
                    end_time = start_time + event['duration']
                else:
                    # 将帧索引转换为秒并添加偏移量
                    start_time = (event['start_idx'] / sampling_freq) + offset
                    # 注意：duration在帧索引下是帧数，需要转换为秒
                    end_time = start_time + (event['duration'] / sampling_freq)
                
                # 在对应神经元的位置上绘制代表钙波事件的水平线段，使用统一线宽
                plt.hlines(y=y_position, xmin=start_time, xmax=end_time, 
                          linewidth=line_width, color=cluster_color, alpha=0.7)
        
        # 设置Y轴刻度和标签
        plt.yticks(list(neuron_to_y.values()), [f"Neuron {i}" for i in neuron_to_y.keys()])
        
        # 为每个数据集添加分隔线和标签
        for dataset, offset in dataset_to_offset.items():
            # 添加垂直分隔线
            plt.axvline(x=offset, color='gray', linestyle='--', alpha=0.5)
            # 添加数据集名称标签
            plt.text(offset + 2, len(unified_neurons) + 0.5, dataset, 
                    fontsize=10, rotation=0, ha='left', va='bottom',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
        
        # 设置图表属性
        plt.title('Integrated Neuron Ca2+ Activity Timeline (Arranged by Dataset)', fontsize=14)
        plt.xlabel(f"{x_label} (Including Dataset Offset)", fontsize=12)
        plt.ylabel('Unified Neuron ID', fontsize=12)
        # 移除网格线
        plt.grid(False)
        
        # 添加聚类标签到图例，使用统一颜色方案
        legend_elements = [plt.Line2D([0], [0], color=cluster_colors[i], lw=4, label=f'Cluster {i+1}')
                           for i in range(n_clusters)]
        plt.legend(handles=legend_elements, loc='upper right')
        
        # 保存图表
        os.makedirs(output_dir, exist_ok=True)
        plt.tight_layout()
        
        # 根据时间模式添加后缀
        suffix = "_timestamp" if time_mode == "timestamp" else ""
        plt.savefig(f'{output_dir}/integrated_neuron_timeline{suffix}.png', dpi=300)
        
        print(f"整合神经元钙波活动时间线图已保存到 {output_dir}/integrated_neuron_timeline{suffix}.png")

def calculate_gap_statistic(features_scaled, max_k=10, n_refs=10, random_state=42):
    """
    计算Gap Statistic以确定最佳聚类数
    
    参数
    ----------
    features_scaled : numpy.ndarray
        标准化后的特征数据
    max_k : int, 可选
        最大测试聚类数，默认为10
    n_refs : int, 可选
        参考数据集数量，默认为10
    random_state : int, 可选
        随机种子，默认为42
        
    返回
    -------
    gap_values : list
        各K值的Gap Statistic值
    optimal_k : int
        最佳聚类数
    """
    np.random.seed(random_state)
    
    # 计算数据的边界
    mins = features_scaled.min(axis=0)
    maxs = features_scaled.max(axis=0)
    
    gaps = []
    
    for k in range(1, max_k + 1):
        # 计算原始数据的聚类内平方和
        if k == 1:
            wk = np.sum(pdist(features_scaled) ** 2) / (2 * len(features_scaled))
        else:
            kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
            labels = kmeans.fit_predict(features_scaled)
            
            wk = 0
            for i in range(k):
                cluster_data = features_scaled[labels == i]
                if len(cluster_data) > 0:
                    # 计算簇内点对距离平方和
                    if len(cluster_data) > 1:
                        wk += np.sum(pdist(cluster_data) ** 2) / (2 * len(cluster_data))
        
        # 生成参考数据集并计算期望值
        ref_wks = []
        for _ in range(n_refs):
            # 生成均匀分布的参考数据
            ref_data = np.random.uniform(mins, maxs, features_scaled.shape)
            
            if k == 1:
                ref_wk = np.sum(pdist(ref_data) ** 2) / (2 * len(ref_data))
            else:
                ref_kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
                ref_labels = ref_kmeans.fit_predict(ref_data)
                
                ref_wk = 0
                for i in range(k):
                    ref_cluster_data = ref_data[ref_labels == i]
                    if len(ref_cluster_data) > 1:
                        ref_wk += np.sum(pdist(ref_cluster_data) ** 2) / (2 * len(ref_cluster_data))
            
            ref_wks.append(np.log(ref_wk) if ref_wk > 0 else 0)
        
        # 计算Gap值
        gap = np.mean(ref_wks) - np.log(wk) if wk > 0 else 0
        gaps.append(gap)
    
    # 找到最佳K值（Gap Statistic最大值）
    optimal_k = np.argmax(gaps) + 1
    
    return gaps, optimal_k

def calculate_burst_intervals(df):
    """
    计算钙爆发间隔时间
    
    参数
    ----------
    df : pandas.DataFrame
        包含钙爆发数据的数据框，需要包含neuron和start_idx字段
        
    返回
    -------
    intervals : list
        所有钙爆发间隔时间列表
    """
    intervals = []
    
    # 按神经元分组计算间隔
    for neuron in df['neuron'].unique():
        neuron_data = df[df['neuron'] == neuron].sort_values('start_idx')
        
        if len(neuron_data) > 1:
            # 计算连续钙爆发之间的间隔
            start_times = neuron_data['start_idx'].values
            intervals.extend(np.diff(start_times))
    
    return intervals

def calculate_burst_frequency(df, total_time_seconds=None, sampling_freq=4.8):
    """
    计算每个神经元的钙爆发频率
    
    参数
    ----------
    df : pandas.DataFrame
        包含钙爆发数据的数据框
    total_time_seconds : float, 可选
        总记录时间（秒），如果未提供则根据数据自动计算
    sampling_freq : float, 可选
        采样频率，默认4.8Hz
        
    返回
    -------
    frequencies : list
        每个神经元的钙爆发频率（Hz）
    """
    frequencies = []
    
    # 如果没有提供总时间，根据数据计算
    if total_time_seconds is None:
        max_time_frames = df['start_idx'].max() + df['duration'].max()
        total_time_seconds = max_time_frames / sampling_freq
    
    # 按神经元计算频率
    for neuron in df['neuron'].unique():
        neuron_data = df[df['neuron'] == neuron]
        burst_count = len(neuron_data)
        frequency = burst_count / total_time_seconds
        frequencies.append(frequency)
    
    return frequencies

def visualize_burst_attributes_academic(df, labels=None, output_dir='../results', features_scaled=None, max_k=10):
    """
    以学术论文风格可视化钙爆发特征分布，类似于参考图片的a-e子图
    
    参数
    ----------
    df : pandas.DataFrame
        钙爆发数据
    labels : numpy.ndarray, 可选
        聚类标签，如果提供则会在图中显示聚类信息
    output_dir : str, 可选
        输出目录
    features_scaled : numpy.ndarray, 可选
        用于Gap Statistic计算的标准化特征数据
    max_k : int, 可选
        Gap Statistic计算的最大K值
    """
    print("正在生成学术风格的钙爆发特征可视化...")
    
    # 设置学术风格的参数
    plt.style.use('default')  # 使用默认样式
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 设置全局字体和样式
    plt.rcParams.update({
        'font.size': 10,
        'axes.linewidth': 1,
        'xtick.major.width': 1,
        'ytick.major.width': 1,
        'xtick.minor.width': 0.5,
        'ytick.minor.width': 0.5
    })
    
    # 颜色设置 - 使用学术论文常用的红色
    academic_red = '#D62728'  # 学术红色
    academic_blue = '#1F77B4'  # 学术蓝色
    
    # 子图a: Gap Statistic (如果提供了features_scaled)
    if features_scaled is not None:
        try:
            print("计算Gap Statistic...")
            gap_values, optimal_k_gap = calculate_gap_statistic(features_scaled, max_k=max_k)
            
            k_range = range(1, len(gap_values) + 1)
            axes[0, 0].plot(k_range, gap_values, 'o-', color=academic_blue, linewidth=2, markersize=6)
            
            # 标记最佳K值
            max_gap_idx = np.argmax(gap_values)
            axes[0, 0].plot(max_gap_idx + 1, gap_values[max_gap_idx], 'o', markersize=8, color=academic_red)
            axes[0, 0].annotate(f'K = {max_gap_idx + 1}', 
                              xy=(max_gap_idx + 1, gap_values[max_gap_idx]),
                              xytext=(max_gap_idx + 1 + 0.5, gap_values[max_gap_idx] + 0.01),
                              fontsize=10, ha='left')
            
            axes[0, 0].set_xlabel('Number of Clusters (K)', fontsize=12)
            axes[0, 0].set_ylabel('Gap Statistic', fontsize=12)
            axes[0, 0].set_title('a', fontsize=14, fontweight='bold', loc='left')
            axes[0, 0].grid(False)
            axes[0, 0].spines['top'].set_visible(False)
            axes[0, 0].spines['right'].set_visible(False)
            
            print(f"Gap Statistic建议的最佳聚类数: K = {optimal_k_gap}")
        except Exception as e:
            print(f"计算Gap Statistic时出错: {str(e)}")
            axes[0, 0].text(0.5, 0.5, 'Gap Statistic\nCalculation Failed', 
                          ha='center', va='center', transform=axes[0, 0].transAxes)
            axes[0, 0].set_title('a', fontsize=14, fontweight='bold', loc='left')
    else:
        axes[0, 0].text(0.5, 0.5, 'Gap Statistic\n(Requires feature data)', 
                      ha='center', va='center', transform=axes[0, 0].transAxes)
        axes[0, 0].set_title('a', fontsize=14, fontweight='bold', loc='left')
    
    # 子图b: Duration分布
    if 'duration' in df.columns:
        # 过滤异常值（大于300秒的数据）
        duration_data = df['duration'][df['duration'] <= 300]
        axes[0, 1].hist(duration_data, bins=30, color=academic_red, alpha=0.8, edgecolor='black', linewidth=0.5)
        axes[0, 1].set_xlabel('Duration (s)', fontsize=12)
        axes[0, 1].set_ylabel('Number of Observations', fontsize=12)
        axes[0, 1].set_title('b', fontsize=14, fontweight='bold', loc='left')
        axes[0, 1].grid(False)
        axes[0, 1].spines['top'].set_visible(False)
        axes[0, 1].spines['right'].set_visible(False)
        
        # 添加统计信息
        n_excluded = len(df) - len(duration_data)
        if n_excluded > 0:
            axes[0, 1].text(0.98, 0.98, f'Excluded: {n_excluded} obs > 300s', 
                          ha='right', va='top', transform=axes[0, 1].transAxes, 
                          fontsize=8, bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    # 子图c: Amplitude分布
    if 'amplitude' in df.columns:
        # 过滤异常值（大于3的数据）
        amplitude_data = df['amplitude'][df['amplitude'] <= 3]
        axes[0, 2].hist(amplitude_data, bins=30, color=academic_red, alpha=0.8, edgecolor='black', linewidth=0.5)
        axes[0, 2].set_xlabel('Amplitude', fontsize=12)
        axes[0, 2].set_ylabel('Number of Observations', fontsize=12)
        axes[0, 2].set_title('c', fontsize=14, fontweight='bold', loc='left')
        axes[0, 2].grid(False)
        axes[0, 2].spines['top'].set_visible(False)
        axes[0, 2].spines['right'].set_visible(False)
        
        # 添加统计信息
        n_excluded = len(df) - len(amplitude_data)
        if n_excluded > 0:
            axes[0, 2].text(0.98, 0.98, f'Excluded: {n_excluded} obs > 3 ΔF/F', 
                          ha='right', va='top', transform=axes[0, 2].transAxes, 
                          fontsize=8, bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    # 子图d: 钙爆发间隔时间分布
    try:
        intervals = calculate_burst_intervals(df)
        if intervals:
            # 过滤异常值（大于3000秒的数据）
            intervals_filtered = [x for x in intervals if x <= 3000]
            axes[1, 0].hist(intervals_filtered, bins=30, color=academic_red, alpha=0.8, edgecolor='black', linewidth=0.5)
            axes[1, 0].set_xlabel('Inter-Ca²⁺ burst interval (s)', fontsize=12)
            axes[1, 0].set_ylabel('Number of Observations', fontsize=12)
            axes[1, 0].set_title('d', fontsize=14, fontweight='bold', loc='left')
            axes[1, 0].grid(False)
            axes[1, 0].spines['top'].set_visible(False)
            axes[1, 0].spines['right'].set_visible(False)
            
            # 添加统计信息
            n_excluded = len(intervals) - len(intervals_filtered)
            if n_excluded > 0:
                axes[1, 0].text(0.98, 0.98, f'Excluded: {n_excluded} obs > 3000s', 
                              ha='right', va='top', transform=axes[1, 0].transAxes, 
                              fontsize=8, bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
            
            print(f"计算了 {len(intervals)} 个钙爆发间隔，其中 {len(intervals_filtered)} 个在显示范围内")
        else:
            axes[1, 0].text(0.5, 0.5, 'No Intervals\nCalculated', 
                          ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 0].set_title('d', fontsize=14, fontweight='bold', loc='left')
    except Exception as e:
        print(f"计算钙爆发间隔时出错: {str(e)}")
        axes[1, 0].text(0.5, 0.5, 'Interval Calculation\nFailed', 
                      ha='center', va='center', transform=axes[1, 0].transAxes)
        axes[1, 0].set_title('d', fontsize=14, fontweight='bold', loc='left')
    
    # 子图e: 钙爆发频率分布
    try:
        frequencies = calculate_burst_frequency(df)
        if frequencies:
            axes[1, 1].hist(frequencies, bins=20, color=academic_red, alpha=0.8, edgecolor='black', linewidth=0.5)
            axes[1, 1].set_xlabel('Frequency (Hz)', fontsize=12)
            axes[1, 1].set_ylabel('Number of Neurons', fontsize=12)
            axes[1, 1].set_title('e', fontsize=14, fontweight='bold', loc='left')
            axes[1, 1].grid(False)
            axes[1, 1].spines['top'].set_visible(False)
            axes[1, 1].spines['right'].set_visible(False)
            
            print(f"计算了 {len(frequencies)} 个神经元的钙爆发频率")
        else:
            axes[1, 1].text(0.5, 0.5, 'No Frequencies\nCalculated', 
                          ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('e', fontsize=14, fontweight='bold', loc='left')
    except Exception as e:
        print(f"计算钙爆发频率时出错: {str(e)}")
        axes[1, 1].text(0.5, 0.5, 'Frequency Calculation\nFailed', 
                      ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('e', fontsize=14, fontweight='bold', loc='left')
    
    # 子图f: 如果有聚类信息，显示聚类统计
    if labels is not None:
        n_clusters = len(np.unique(labels))
        cluster_counts = np.bincount(labels)
        
        bars = axes[1, 2].bar(range(n_clusters), cluster_counts, color=academic_blue, alpha=0.8, edgecolor='black', linewidth=0.5)
        axes[1, 2].set_xlabel('Cluster', fontsize=12)
        axes[1, 2].set_ylabel('Number of Events', fontsize=12)
        axes[1, 2].set_title('f', fontsize=14, fontweight='bold', loc='left')
        axes[1, 2].grid(False)
        axes[1, 2].spines['top'].set_visible(False)
        axes[1, 2].spines['right'].set_visible(False)
        
        # 添加数值标签
        for i, bar in enumerate(bars):
            height = bar.get_height()
            axes[1, 2].text(bar.get_x() + bar.get_width()/2., height + 0.01*max(cluster_counts),
                          f'{int(height)}', ha='center', va='bottom', fontsize=9)
    else:
        axes[1, 2].text(0.5, 0.5, 'Cluster Distribution\n(Requires clustering)', 
                      ha='center', va='center', transform=axes[1, 2].transAxes)
        axes[1, 2].set_title('f', fontsize=14, fontweight='bold', loc='left')
    
    # 调整布局并保存
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f'{output_dir}/burst_attributes_academic_style.png', dpi=300, bbox_inches='tight')
    
    print(f"学术风格钙爆发特征图已保存到: {output_dir}/burst_attributes_academic_style.png")

def generate_individual_academic_figures(df, labels=None, output_dir='../results', features_scaled=None, max_k=10):
    """
    生成学术风格的单独子图文件，每个子图保存为独立的PNG文件
    
    参数
    ----------
    df : pandas.DataFrame
        钙爆发数据
    labels : numpy.ndarray, 可选
        聚类标签
    output_dir : str, 可选
        输出目录
    features_scaled : numpy.ndarray, 可选
        用于Gap Statistic计算的标准化特征数据
    max_k : int, 可选
        Gap Statistic计算的最大K值
    """
    print("正在生成学术风格的单独子图文件...")
    
    # 确保输出目录存在
    individual_dir = os.path.join(output_dir, 'individual_figures')
    os.makedirs(individual_dir, exist_ok=True)
    
    # 设置学术风格的参数
    plt.rcParams.update({
        'font.size': 12,
        'axes.linewidth': 1.5,
        'xtick.major.width': 1.5,
        'ytick.major.width': 1.5,
        'figure.dpi': 300
    })
    
    # 颜色设置
    academic_red = '#D62728'
    academic_blue = '#1F77B4'
    
    # 子图a: Gap Statistic
    if features_scaled is not None:
        try:
            print("生成Gap Statistic单独图...")
            plt.figure(figsize=(8, 6))
            gap_values, optimal_k_gap = calculate_gap_statistic(features_scaled, max_k=max_k)
            
            k_range = range(1, len(gap_values) + 1)
            plt.plot(k_range, gap_values, 'o-', color=academic_blue, linewidth=3, markersize=8)
            
            # 标记最佳K值
            max_gap_idx = np.argmax(gap_values)
            plt.plot(max_gap_idx + 1, gap_values[max_gap_idx], 'o', markersize=12, color=academic_red)
            plt.annotate(f'K = {max_gap_idx + 1}', 
                        xy=(max_gap_idx + 1, gap_values[max_gap_idx]),
                        xytext=(max_gap_idx + 1 + 0.5, gap_values[max_gap_idx] + 0.01),
                        fontsize=12, ha='left', fontweight='bold')
            
            plt.xlabel('Number of Clusters (K)', fontsize=14, fontweight='bold')
            plt.ylabel('Gap Statistic', fontsize=14, fontweight='bold')
            plt.title('Determining Number of Ca²⁺ Burst Clusters', fontsize=16, fontweight='bold', pad=20)
            plt.grid(False)
            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['right'].set_visible(False)
            plt.tight_layout()
            plt.savefig(f'{individual_dir}/figure_a_gap_statistic.png', dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"生成Gap Statistic图时出错: {str(e)}")
    
    # 子图b: Duration分布
    if 'duration' in df.columns:
        print("生成Duration分布单独图...")
        plt.figure(figsize=(8, 6))
        duration_data = df['duration'][df['duration'] <= 300]
        
        plt.hist(duration_data, bins=30, color=academic_red, alpha=0.8, edgecolor='black', linewidth=0.8)
        plt.xlabel('Duration (s)', fontsize=14, fontweight='bold')
        plt.ylabel('Number of Observations', fontsize=14, fontweight='bold')
        plt.title('Duration of Ca²⁺ Bursts', fontsize=16, fontweight='bold', pad=20)
        plt.grid(False)
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        
        # 添加统计信息
        n_excluded = len(df) - len(duration_data)
        if n_excluded > 0:
            plt.text(0.98, 0.98, f'{n_excluded} observations with\nduration > 300 s omitted', 
                    ha='right', va='top', transform=plt.gca().transAxes, 
                    fontsize=10, bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9))
        
        plt.tight_layout()
        plt.savefig(f'{individual_dir}/figure_b_duration_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 子图c: Amplitude分布
    if 'amplitude' in df.columns:
        print("生成Amplitude分布单独图...")
        plt.figure(figsize=(8, 6))
        amplitude_data = df['amplitude'][df['amplitude'] <= 3]
        
        plt.hist(amplitude_data, bins=30, color=academic_red, alpha=0.8, edgecolor='black', linewidth=0.8)
        plt.xlabel('Amplitude (ΔF/F)', fontsize=14, fontweight='bold')
        plt.ylabel('Number of Observations', fontsize=14, fontweight='bold')
        plt.title('Amplitude of Ca²⁺ Bursts', fontsize=16, fontweight='bold', pad=20)
        plt.grid(False)
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        
        # 添加统计信息
        n_excluded = len(df) - len(amplitude_data)
        if n_excluded > 0:
            plt.text(0.98, 0.98, f'{n_excluded} observations with\namplitude > 3 ΔF/F omitted', 
                    ha='right', va='top', transform=plt.gca().transAxes, 
                    fontsize=10, bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9))
        
        plt.tight_layout()
        plt.savefig(f'{individual_dir}/figure_c_amplitude_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 子图d: 钙爆发间隔时间分布
    try:
        print("生成间隔时间分布单独图...")
        plt.figure(figsize=(8, 6))
        intervals = calculate_burst_intervals(df)
        
        if intervals:
            intervals_filtered = [x for x in intervals if x <= 3000]
            plt.hist(intervals_filtered, bins=30, color=academic_red, alpha=0.8, edgecolor='black', linewidth=0.8)
            plt.xlabel('Inter-Ca²⁺ Burst Interval (s)', fontsize=14, fontweight='bold')
            plt.ylabel('Number of Observations', fontsize=14, fontweight='bold')
            plt.title('Intervals Between Consecutive Ca²⁺ Bursts', fontsize=16, fontweight='bold', pad=20)
            plt.grid(False)
            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['right'].set_visible(False)
            
            # 添加统计信息
            n_excluded = len(intervals) - len(intervals_filtered)
            if n_excluded > 0:
                plt.text(0.98, 0.98, f'{n_excluded} observations with\ninterval > 3000 s omitted', 
                        ha='right', va='top', transform=plt.gca().transAxes, 
                        fontsize=10, bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9))
            
            # 添加数据信息
            plt.text(0.02, 0.98, f'n = {len(intervals_filtered)} intervals from {len(df["neuron"].unique())} neurons', 
                    ha='left', va='top', transform=plt.gca().transAxes, 
                    fontsize=10, bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.7))
        
        plt.tight_layout()
        plt.savefig(f'{individual_dir}/figure_d_interval_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        print(f"生成间隔时间分布图时出错: {str(e)}")
    
    # 子图e: 钙爆发频率分布
    try:
        print("生成频率分布单独图...")
        plt.figure(figsize=(8, 6))
        frequencies = calculate_burst_frequency(df)
        
        if frequencies:
            plt.hist(frequencies, bins=20, color=academic_red, alpha=0.8, edgecolor='black', linewidth=0.8)
            plt.xlabel('Frequency (Hz)', fontsize=14, fontweight='bold')
            plt.ylabel('Number of Neurons', fontsize=14, fontweight='bold')
            plt.title('Frequency of Ca²⁺ Bursts in Individual Neurons', fontsize=16, fontweight='bold', pad=20)
            plt.grid(False)
            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['right'].set_visible(False)
            
            # 添加统计信息
            mean_freq = np.mean(frequencies)
            plt.text(0.98, 0.98, f'Mean frequency: {mean_freq:.3f} Hz\nn = {len(frequencies)} neurons', 
                    ha='right', va='top', transform=plt.gca().transAxes, 
                    fontsize=10, bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.7))
        
        plt.tight_layout()
        plt.savefig(f'{individual_dir}/figure_e_frequency_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        print(f"生成频率分布图时出错: {str(e)}")
    
    # 子图f: 聚类分布统计
    if labels is not None:
        print("生成聚类分布单独图...")
        plt.figure(figsize=(8, 6))
        n_clusters = len(np.unique(labels))
        cluster_counts = np.bincount(labels)
        cluster_names = [f'Cluster {i+1}' for i in range(n_clusters)]
        
        bars = plt.bar(cluster_names, cluster_counts, color=academic_blue, alpha=0.8, edgecolor='black', linewidth=0.8)
        plt.xlabel('Cluster', fontsize=14, fontweight='bold')
        plt.ylabel('Number of Events', fontsize=14, fontweight='bold')
        plt.title('Distribution of Ca²⁺ Burst Events Across Clusters', fontsize=16, fontweight='bold', pad=20)
        plt.grid(False)
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        
        # 添加数值标签
        for i, bar in enumerate(bars):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + max(cluster_counts)*0.01,
                    f'{int(height)}', ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        # 旋转X轴标签以避免重叠
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(f'{individual_dir}/figure_f_cluster_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 重置matplotlib参数
    plt.rcParams.update(plt.rcParamsDefault)
    
    print(f"单独图表已保存到: {individual_dir}/")
    print("生成的文件:")
    print("  - figure_a_gap_statistic.png")
    print("  - figure_b_duration_distribution.png") 
    print("  - figure_c_amplitude_distribution.png")
    print("  - figure_d_interval_distribution.png")
    print("  - figure_e_frequency_distribution.png")
    print("  - figure_f_cluster_distribution.png")

def main():
    """
    主函数
    """
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='钙爆发事件聚类分析工具')
    parser.add_argument('--k', type=int, help='指定聚类数K，不指定则自动确定最佳值')
    parser.add_argument('--compare', type=str, help='比较多个K值的效果，格式如"2,3,4,5"')
    parser.add_argument('--input', type=str, default='../results/all_datasets_transients/all_datasets_transients.xlsx', 
                       help='输入数据文件路径，可以是单个文件或合并后的文件')
    parser.add_argument('--input_dir', type=str, help='输入数据目录，会处理该目录下所有的transients.xlsx文件')
    parser.add_argument('--combine', action='store_true', 
                       help='是否合并指定目录下的所有钙爆发数据再进行聚类（与--input_dir一起使用）')
    parser.add_argument('--output', type=str, default=None,
                       help='输出目录，不指定则根据数据集名称自动生成')
    parser.add_argument('--raw_data_dir', type=str, help='原始数据文件所在目录，用于波形可视化')
    parser.add_argument('--raw_data_path', type=str, help='单个原始数据文件路径，用于波形可视化')
    parser.add_argument('--skip_waveform', action='store_true', help='跳过波形可视化步骤')
    parser.add_argument('--weights', type=str, help='特征权重，格式如"amplitude:1.2,duration:0.8"')
    parser.add_argument('--log_dir', type=str, default=None, help='日志文件保存目录，默认为输出目录下的logs文件夹')
    # 添加新的命令行参数
    parser.add_argument('--neuron_map_path', type=str, default=None, 
                        help='神经元对应表路径，默认会查找"../datasets/神经元对应表.xlsx"')
    parser.add_argument('--skip_timeline', action='store_true', help='跳过神经元时间线可视化步骤')
    parser.add_argument('--interactive', action='store_true', help='使用交互式绘图模式(需要安装plotly)')
    parser.add_argument('--use_timestamp', action='store_true', help='使用时间戳模式，如果数据中有timestamp字段')
    parser.add_argument('--sampling_freq', type=float, default=4.8, help='采样频率，默认为4.8Hz，用于帧索引转换为时间')
    parser.add_argument('--academic_style', action='store_true', help='使用学术论文风格的可视化（参考图片风格）')
    args = parser.parse_args()
    
    # 设置输出目录
    output_dir = args.output
    if output_dir is None:
        # 自动生成输出目录
        if args.input_dir and args.combine:
            # 如果是合并多个文件，使用输入目录名
            input_dir_name = os.path.basename(os.path.normpath(args.input_dir))
            output_dir = f"../results/cluster_results_{input_dir_name}_combined"
        else:
            # 否则使用输入文件名
            input_basename = os.path.basename(args.input)
            output_dir = f"../results/cluster_results_{os.path.splitext(input_basename)[0]}"
    
    # 设置日志目录
    log_dir = args.log_dir
    if log_dir is None:
        log_dir = os.path.join(output_dir, 'logs')
    
    # 初始化日志记录器
    logger = setup_logger(log_dir, "cluster-integrate")
    logger.info(f"开始钙爆发聚类分析，输出目录: {output_dir}")
    
    # 处理输入文件路径
    input_files = []
    
    if args.input_dir:
        # 如果提供了输入目录，搜索该目录下所有的transients.xlsx文件
        pattern = os.path.join(args.input_dir, "**", "*transients.xlsx")
        input_files = glob.glob(pattern, recursive=True)
        if not input_files:
            logger.error(f"错误: 在目录{args.input_dir}下未找到任何匹配的transients.xlsx文件")
            return
        logger.info(f"在目录{args.input_dir}下找到{len(input_files)}个钙爆发数据文件")
    else:
        # 否则使用单个输入文件
        if not os.path.exists(args.input):
            logger.error(f"错误: 输入文件 {args.input} 不存在")
            return
        input_files = [args.input]
        logger.info(f"使用输入文件: {args.input}")
    
    # 如果需要合并多个输入文件
    if args.input_dir and args.combine and len(input_files) > 1:
        logger.info("正在合并多个钙爆发数据文件...")
        
        all_data = []
        for file in input_files:
            try:
                df = load_data(file)
                # 添加数据源标识，使用文件名而非目录名
                dataset_name = os.path.splitext(os.path.basename(file))[0]
                df['dataset'] = dataset_name
                all_data.append(df)
            except Exception as e:
                print(f"处理文件{file}时出错: {str(e)}")
        
        if all_data:
            # 合并所有数据
            df = pd.concat(all_data, ignore_index=True)
            print(f"成功合并{len(all_data)}个数据文件，总共{len(df)}行数据")
            
            # 设置输出目录
            if args.output is None:
                output_dir = "../results/combined_transients_clustering"
            else:
                output_dir = args.output
        else:
            print("未能加载任何有效数据，请检查输入路径")
            return
    else:
        # 使用单个输入文件
        input_file = input_files[0]
        try:
            df = load_data(input_file)
        except Exception as e:
            print(f"加载文件{input_file}时出错: {str(e)}")
            return
        
        # 根据输入文件名生成输出目录
        if args.output is None:
            # 提取输入文件目录
            input_dir = os.path.dirname(input_file)
            # 提取数据文件名（不含扩展名）
            data_basename = os.path.basename(input_file)
            dataset_name = os.path.splitext(data_basename)[0]
            
            # 如果是all_datasets_transients.xlsx这种合并文件，使用专门的输出目录
            if dataset_name == "all_datasets_transients":
                output_dir = "../results/all_datasets_clustering"
            else:
                output_dir = f"../results/{dataset_name}_clustering"
        else:
            output_dir = args.output
    
    print(f"输出目录设置为: {output_dir}")
    
    # 确保结果目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 使用增强版预处理函数
    try:
        # 添加特征权重设置 - 修改为无条件应用默认权重
        feature_weights = {
            'amplitude': 2,  # 振幅权重更高
            'duration': 2,   # 持续时间权重更高
            'rise_time': 1.5,  # 上升时间权重较低
            'decay_time': 1.5, # 衰减时间权重较低
            'snr': 0.5,        # 信噪比正常权重
            'fwhm': 0.5,       # 半高宽正常权重
            'auc': 0.5         # 曲线下面积正常权重
        }
        
        # 如果用户指定了权重，则覆盖默认值
        if args.weights:
            # 解析用户指定的权重
            for pair in args.weights.split(','):
                feature, weight = pair.split(':')
                feature_weights[feature] = float(weight)
        
        print("使用权重设置进行聚类分析")
        features_scaled, feature_names, df_clean = enhance_preprocess_data(df, feature_weights=feature_weights)
    except Exception as e:
        print(f"预处理数据时出错: {str(e)}")
        return
    
    # 检查数据是否足够进行聚类
    if len(df_clean) < 10:
        print(f"错误: 有效数据不足(只有{len(df_clean)}行)，无法进行聚类分析")
        return
    
    # 处理聚类数K
    if args.compare:
        # 如果需要比较多个K值
        try:
            k_values = [int(k) for k in args.compare.split(',')]
            best_k = compare_multiple_k(features_scaled, feature_names, df_clean, k_values, input_file, output_dir=output_dir)
            print(f"在比较的K值中，K={best_k}的轮廓系数最高")
            # 使用最佳K值进行后续分析
            optimal_k = best_k
        except Exception as e:
            print(f"比较K值时出错: {str(e)}")
            return
    else:
        # 如果指定了K值，使用指定值
        if args.k:
            optimal_k = args.k
            print(f"使用指定的聚类数: K={optimal_k}")
        else:
            # 自动确定最佳聚类数
            try:
                optimal_k = determine_optimal_k(features_scaled, output_dir=output_dir)
            except Exception as e:
                print(f"确定最佳聚类数时出错: {str(e)}")
                print("使用默认聚类数K=5")
                optimal_k = 5
    
    # K均值聚类
    try:
        kmeans_labels = cluster_kmeans(features_scaled, optimal_k)
    except Exception as e:
        print(f"执行K-means聚类时出错: {str(e)}")
        return
    
    # 生成学术风格的钙爆发特征可视化（参考图片风格）
    try:
        visualize_burst_attributes_academic(df_clean, labels=kmeans_labels, output_dir=output_dir, 
                                          features_scaled=features_scaled, max_k=10)
    except Exception as e:
        print(f"生成学术风格特征可视化时出错: {str(e)}")
    
    # 生成单独的学术风格子图文件
    try:
        generate_individual_academic_figures(df_clean, labels=kmeans_labels, output_dir=output_dir,
                                           features_scaled=features_scaled, max_k=10)
    except Exception as e:
        print(f"生成单独学术风格图表时出错: {str(e)}")
    
    # 可视化聚类结果
    try:
        visualize_clusters_2d(features_scaled, kmeans_labels, feature_names, method='pca', output_dir=output_dir)
        visualize_clusters_2d(features_scaled, kmeans_labels, feature_names, method='t-sne', output_dir=output_dir)
    except Exception as e:
        print(f"可视化聚类结果时出错: {str(e)}")
    
    # 特征分布可视化
    try:
        visualize_feature_distribution(df_clean, kmeans_labels, output_dir=output_dir)
    except Exception as e:
        print(f"可视化特征分布时出错: {str(e)}")
    
    # 分析聚类结果
    try:
        cluster_stats = analyze_clusters(df_clean, kmeans_labels, output_dir=output_dir)
    except Exception as e:
        print(f"分析聚类结果时出错: {str(e)}")
        cluster_stats = None
    
    # 如果有统计结果，则生成雷达图
    if cluster_stats is not None:
        try:
            visualize_cluster_radar(cluster_stats, output_dir=output_dir)
        except Exception as e:
            print(f"生成雷达图时出错: {str(e)}")
    
    # 神经元簇分布
    try:
        visualize_neuron_cluster_distribution(df_clean, kmeans_labels, output_dir=output_dir)
    except Exception as e:
        print(f"可视化神经元簇分布时出错: {str(e)}")
    
    # 添加波形类型分析
    try:
        visualize_wave_type_distribution(df_clean, kmeans_labels, output_dir=output_dir)
    except Exception as e:
        print(f"可视化波形类型分布时出错: {str(e)}")
    
    # 添加子峰分析
    try:
        analyze_subpeaks(df_clean, kmeans_labels, output_dir=output_dir)
    except Exception as e:
        print(f"分析子峰时出错: {str(e)}")
    
    # 尝试可视化不同聚类的平均钙爆发波形（如果原始数据可用且未指定跳过）
    if not args.skip_waveform:
        try:
            visualize_cluster_waveforms(df_clean, kmeans_labels, output_dir=output_dir, 
                                      raw_data_path=args.raw_data_path, raw_data_dir=args.raw_data_dir)
        except Exception as e:
            print(f"可视化波形时出错: {str(e)}")
    else:
        print("根据参数设置，跳过波形可视化")
    
    # 在main函数的最后，添加调用新函数的代码
    try:
        # 尝试生成整合神经元时间线图
        if not args.skip_timeline:
            try:
                visualize_integrated_neuron_timeline(
                    df_clean, kmeans_labels, 
                    neuron_map_path=args.neuron_map_path,
                    output_dir=output_dir, 
                    use_timestamp=args.use_timestamp,
                    interactive=args.interactive,
                    sampling_freq=args.sampling_freq
                )
            except Exception as e:
                print(f"生成整合神经元时间线图时出错: {str(e)}")
    except Exception as e:
        print(f"可视化波形时出错: {str(e)}")
    
    # 将聚类标签添加到Excel
    try:
        output_file = f'{output_dir}/transients_clustered_k{optimal_k}.xlsx'
        if args.input_dir and args.combine and len(input_files) > 1:
            add_cluster_to_excel("combined_data", output_file, kmeans_labels, df=df_clean)
        else:
            add_cluster_to_excel(input_file, output_file, kmeans_labels, df=df_clean)
        print(f"聚类结果已保存到: {output_file}")
    except Exception as e:
        print(f"保存聚类结果时出错: {str(e)}")
    
    print("聚类分析完成!")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        # 设置应急日志
        logger = setup_logger(None, "cluster-integrate-error")
        logger.error(f"程序运行时出错: {str(e)}", exc_info=True)
        print(f"程序运行时出错: {str(e)}")
        raise