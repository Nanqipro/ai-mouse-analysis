import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import os

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
        if col in df.columns and df[col].dtype == 'object':
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # 删除缺失值
    df_clean = df.dropna(subset=feature_names).copy()
    
    # 提取特征
    features = df_clean[feature_names].values
    
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
    
    print(f"预处理完成，保留{len(df_clean)}个有效样本，使用特征: {', '.join(feature_names)}")
    return features_scaled, feature_names, df_clean

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
        每个样本的聚类标签
    """
    print(f"开始K-Means聚类，K={n_clusters}...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(features_scaled)
    labels = kmeans.labels_
    print("K-Means聚类完成")
    return labels

def visualize_clusters_2d(features_scaled, labels, feature_names, method='pca', output_dir='../results'):
    """
    使用PCA或t-SNE将聚类结果降维至2D并可视化
    
    参数
    ----------
    features_scaled : numpy.ndarray
        标准化后的特征数据
    labels : numpy.ndarray
        聚类标签
    feature_names : list
        特征名称列表
    method : str, 可选
        降维方法 ('pca' 或 'tsne')，默认为'pca'
    output_dir : str, 可选
        输出目录路径，默认为'../results'
        
    返回
    -------
    fig : matplotlib.figure.Figure
        绘图对象
    """
    print(f"开始使用{method.upper()}进行2D可视化...")
    n_clusters = len(np.unique(labels))
    
    # 降维
    if method == 'pca':
        reducer = PCA(n_components=2)
        title = 'PCA of Clusters'
    elif method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(features_scaled)-1))
        title = 't-SNE of Clusters'
    else:
        raise ValueError("方法必须是 'pca' 或 'tsne'")
    
    features_2d = reducer.fit_transform(features_scaled)
    
    # 绘图
    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(features_2d[:, 0], features_2d[:, 1], c=labels, cmap='viridis', alpha=0.7)
    
    # 添加图例
    legend1 = ax.legend(*scatter.legend_elements(), title="Clusters")
    ax.add_artist(legend1)
    
    ax.set_title(title)
    ax.set_xlabel(f'{method.upper()} Component 1')
    ax.set_ylabel(f'{method.upper()} Component 2')
    ax.grid(True, alpha=0.3)
    
    print(f"{method.upper()}可视化完成")
    return fig

def visualize_feature_distribution(df, labels, output_dir='../results'):
    """
    可视化每个聚类中特征的分布（箱形图）
    
    参数
    ----------
    df : pandas.DataFrame
        包含特征和聚类标签的数据
    labels : numpy.ndarray
        聚类标签
    output_dir : str, 可选
        输出目录路径，默认为'../results'
        
    返回
    -------
    fig : matplotlib.figure.Figure
        绘图对象
    """
    print("开始生成特征分布图...")
    df_vis = df.copy()
    df_vis['cluster'] = labels
    
    # 选取数值型特征用于绘图
    features_to_plot = df_vis.select_dtypes(include=np.number).columns.tolist()
    features_to_plot.remove('cluster')
    if 'neuron_id' in features_to_plot:
        features_to_plot.remove('neuron_id')
    if 'event_id' in features_to_plot:
        features_to_plot.remove('event_id')
    
    # 限制特征数量以避免图表过于拥挤
    if len(features_to_plot) > 8:
        features_to_plot = features_to_plot[:8]
    
    # 计算子图布局
    n_features = len(features_to_plot)
    n_cols = min(4, n_features)
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    if n_features == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    
    # 为每个特征绘制箱形图
    for i, feature in enumerate(features_to_plot):
        row = i // n_cols
        col = i % n_cols
        ax = axes[row, col] if n_rows > 1 else axes[col]
        
        # 准备数据
        data_for_plot = []
        cluster_labels = []
        for cluster_id in sorted(df_vis['cluster'].unique()):
            cluster_data = df_vis[df_vis['cluster'] == cluster_id][feature]
            data_for_plot.append(cluster_data)
            cluster_labels.append(f'Cluster {cluster_id}')
        
        # 绘制箱形图
        bp = ax.boxplot(data_for_plot, labels=cluster_labels, patch_artist=True)
        
        # 设置颜色
        colors = plt.cm.viridis(np.linspace(0, 1, len(data_for_plot)))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.set_title(f'{feature}')
        ax.set_ylabel('Value')
        ax.grid(True, alpha=0.3)
        
        # 旋转x轴标签以避免重叠
        ax.tick_params(axis='x', rotation=45)
    
    # 隐藏多余的子图
    for i in range(n_features, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        if n_rows > 1:
            axes[row, col].set_visible(False)
        else:
            axes[col].set_visible(False)
    
    plt.tight_layout()
    print("特征分布图生成完成")
    return fig

def analyze_clusters(df, labels):
    """
    分析聚类结果，生成统计摘要
    
    参数
    ----------
    df : pandas.DataFrame
        特征数据
    labels : numpy.ndarray
        聚类标签
        
    返回
    -------
    summary : pandas.DataFrame
        聚类统计摘要
    """
    print("开始分析聚类结果...")
    df_with_labels = df.copy()
    df_with_labels['cluster'] = labels
    
    # 选择数值型特征
    numeric_features = df.select_dtypes(include=np.number).columns.tolist()
    if 'neuron_id' in numeric_features:
        numeric_features.remove('neuron_id')
    if 'event_id' in numeric_features:
        numeric_features.remove('event_id')
    
    summary_data = []
    for cluster_id in sorted(np.unique(labels)):
        cluster_data = df_with_labels[df_with_labels['cluster'] == cluster_id]
        cluster_summary = {
            'Cluster': cluster_id,
            'Count': len(cluster_data),
            'Percentage': f"{len(cluster_data)/len(df_with_labels)*100:.1f}%"
        }
        
        # 计算每个特征的均值
        for feature in numeric_features:
            cluster_summary[f'{feature}_mean'] = cluster_data[feature].mean()
        
        summary_data.append(cluster_summary)
    
    summary_df = pd.DataFrame(summary_data)
    print("聚类结果分析完成")
    return summary_df

def add_cluster_to_excel(input_file, output_file, labels):
    """
    将聚类标签添加到Excel文件中
    
    参数
    ----------
    input_file : str
        输入文件路径
    output_file : str
        输出文件路径
    labels : numpy.ndarray
        聚类标签
    """
    df = pd.read_excel(input_file)
    df['cluster'] = labels
    df.to_excel(output_file, index=False)
    print(f"已将聚类结果保存到 {output_file}")