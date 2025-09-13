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
import logging
import datetime
import base64
from io import BytesIO
from pathlib import Path

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

def determine_optimal_k(features_scaled, max_k=10):
    """
    确定最佳聚类数
    
    参数
    ----------
    features_scaled : numpy.ndarray
        标准化后的特征数据
    max_k : int, 可选
        最大测试聚类数，默认为10
        
    返回
    -------
    optimal_k : int
        最佳聚类数
    inertia_values : list
        各K值对应的惯性值
    silhouette_scores : list
        各K值对应的轮廓系数
    """
    print("正在确定最佳聚类数...")
    inertia = []
    silhouette_scores = []
    k_range = range(2, max_k + 1)
    
    # 计算不同k值的肘部指标和轮廓系数
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(features_scaled)
        inertia.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(features_scaled, kmeans.labels_))
    
    # 找到轮廓系数最高的k值
    optimal_k = silhouette_scores.index(max(silhouette_scores)) + 2
    print(f"基于轮廓系数，最佳聚类数为{optimal_k}")
    
    return optimal_k, inertia, silhouette_scores

def cluster_dbscan(features_scaled, eps=0.5, min_samples=5):
    """
    使用DBSCAN聚类
    
    参数
    ----------
    features_scaled : numpy.ndarray
        标准化后的特征数据
    eps : float, 可选
        邻域半径，默认为0.5
    min_samples : int, 可选
        核心点最小邻居数，默认为5
        
    返回
    -------
    labels : numpy.ndarray
        聚类标签
    """
    print(f"使用DBSCAN聚类算法，eps={eps}, min_samples={min_samples}...")
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(features_scaled)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    print(f"DBSCAN聚类完成，共{n_clusters}个簇，{n_noise}个噪声点")
    return labels

def visualize_cluster_radar(cluster_stats):
    """
    使用雷达图可视化各簇的特征
    
    参数
    ----------
    cluster_stats : pandas.DataFrame
        每个簇的特征统计信息
        
    返回
    -------
    fig : matplotlib.figure.Figure
        绘图对象
    """
    print("使用雷达图可视化各簇的特征...")
    features = ['amplitude', 'duration', 'fwhm', 'rise_time', 'decay_time', 'auc', 'snr']
    
    # 过滤存在的特征
    available_features = [f for f in features if f + '_mean' in cluster_stats.columns]
    
    if not available_features:
        print("警告：没有可用的特征用于雷达图")
        return None
    
    # 获取均值数据
    means_columns = [f + '_mean' for f in available_features]
    means = cluster_stats[means_columns]
    means.columns = available_features  # 重命名列
    
    # 标准化均值，使其适合雷达图
    scaler = StandardScaler()
    means_scaled = pd.DataFrame(scaler.fit_transform(means), 
                               index=means.index, columns=means.columns)
    
    # 准备绘图
    n_clusters = len(means_scaled)
    angles = np.linspace(0, 2*np.pi, len(available_features), endpoint=False).tolist()
    angles += angles[:1]  # 闭合雷达图
    
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(polar=True))
    
    for i, idx in enumerate(means_scaled.index):
        values = means_scaled.loc[idx].values.tolist()
        values += values[:1]  # 闭合雷达图
        ax.plot(angles, values, 'o-', linewidth=2, label=f'Cluster {idx}')
        ax.fill(angles, values, alpha=0.1)
    
    # 设置图的属性
    ax.set_thetagrids(np.degrees(angles[:-1]), available_features)
    ax.set_title('Comparison of Cluster Features using Radar Chart')
    ax.grid(True)
    ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    plt.tight_layout()
    print("雷达图生成完成")
    return fig

def plot_to_base64(fig):
    """
    将matplotlib图形转换为base64编码字符串
    
    参数
    ----------
    fig : matplotlib.figure.Figure
        matplotlib图形对象
        
    返回
    -------
    str
        base64编码的图形字符串
    """
    buffer = BytesIO()
    fig.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    buffer.close()
    plt.close(fig)  # 关闭图形以释放内存
    return f"data:image/png;base64,{image_base64}"

def create_k_comparison_plot(features_scaled, k_values):
    """
    创建不同K值的比较图
    
    参数
    ----------
    features_scaled : numpy.ndarray
        标准化后的特征数据
    k_values : list
        要比较的K值列表
        
    返回
    -------
    fig : matplotlib.figure.Figure
        比较图对象
    silhouette_scores_dict : dict
        各K值对应的轮廓系数
    """
    print(f"正在比较不同K值的聚类效果: {k_values}...")
    
    # 计算每个K值的轮廓系数
    silhouette_scores_dict = {}
    
    # 创建比较图
    fig, axes = plt.subplots(1, len(k_values), figsize=(5*len(k_values), 5))
    if len(k_values) == 1:
        axes = [axes]
    
    for i, k in enumerate(k_values):
        # 执行K-means聚类
        labels = cluster_kmeans(features_scaled, k)
        
        # 计算轮廓系数
        sil_score = silhouette_score(features_scaled, labels)
        silhouette_scores_dict[k] = sil_score
        
        # 使用PCA降维可视化
        reducer = PCA(n_components=2, random_state=42)
        embedding = reducer.fit_transform(features_scaled)
        
        # 绘制聚类结果
        cmap = ListedColormap(plt.cm.tab10(np.linspace(0, 1, k+1)))
        
        # 在子图中绘制
        for j in range(k):
            axes[i].scatter(embedding[labels==j, 0], embedding[labels==j, 1], 
                         c=[cmap(j)], marker='o', label=f'Cluster {j+1}')
        
        axes[i].set_title(f'K={k}, Silhouette={sil_score:.3f}')
        axes[i].set_xlabel('PCA Dimension 1')
        axes[i].set_ylabel('PCA Dimension 2')
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    return fig, silhouette_scores_dict

def create_optimal_k_plot(inertia_values, silhouette_scores, k_range):
    """
    创建最佳K值确定图（肘部法则和轮廓系数）
    
    参数
    ----------
    inertia_values : list
        惯性值列表
    silhouette_scores : list
        轮廓系数列表
    k_range : range
        K值范围
        
    返回
    -------
    fig : matplotlib.figure.Figure
        绘图对象
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # 肘部法则图
    ax1.plot(k_range, inertia_values, 'o-', color='blue')
    ax1.set_title('Elbow Method')
    ax1.set_xlabel('Number of Clusters (k)')
    ax1.set_ylabel('Inertia')
    ax1.grid(True, alpha=0.3)
    
    # 轮廓系数图
    ax2.plot(k_range, silhouette_scores, 'o-', color='green')
    ax2.set_title('Silhouette Score')
    ax2.set_xlabel('Number of Clusters (k)')
    ax2.set_ylabel('Silhouette Score')
    ax2.grid(True, alpha=0.3)
    
    # 标记最佳K值
    best_k_idx = np.argmax(silhouette_scores)
    best_k = k_range[best_k_idx]
    ax2.scatter([best_k], [silhouette_scores[best_k_idx]], color='red', s=100, zorder=5)
    ax2.annotate(f'Best K={best_k}', 
                xy=(best_k, silhouette_scores[best_k_idx]),
                xytext=(best_k+1, silhouette_scores[best_k_idx]+0.02),
                arrowprops=dict(arrowstyle='->', color='red'))
    
    plt.tight_layout()
    return fig

def generate_comprehensive_cluster_analysis(file_path, k=None, algorithm='kmeans', 
                                          feature_weights=None, reduction_method='pca',
                                          auto_k_range=(2, 10), dbscan_eps=0.5, 
                                          dbscan_min_samples=5):
    """
    生成综合的聚类分析结果
    
    参数
    ----------
    file_path : str
        数据文件路径
    k : int, 可选
        聚类数，如果为None则自动确定最佳值
    algorithm : str, 可选
        聚类算法，'kmeans'或'dbscan'，默认为'kmeans'
    feature_weights : dict, 可选
        特征权重字典
    reduction_method : str, 可选
        降维方法，'pca'或'tsne'，默认为'pca'
    auto_k_range : tuple, 可选
        自动确定K值时的搜索范围，默认为(2, 10)
    dbscan_eps : float, 可选
        DBSCAN的eps参数，默认为0.5
    dbscan_min_samples : int, 可选
        DBSCAN的min_samples参数，默认为5
        
    返回
    -------
    dict
        包含所有分析结果的字典
    """
    print("开始综合聚类分析...")
    
    # 1. 加载和预处理数据
    df = load_data(file_path)
    features_scaled, feature_names, df_clean = enhance_preprocess_data(df, feature_weights)
    
    result = {
        'data_info': {
            'total_samples': len(df),
            'valid_samples': len(df_clean),
            'features_used': feature_names
        }
    }
    
    # 2. 如果使用K-means且未指定K值，则自动确定最佳K
    if algorithm == 'kmeans' and k is None:
        print("自动确定最佳K值...")
        optimal_k, inertia_values, silhouette_scores = determine_optimal_k(
            features_scaled, max_k=auto_k_range[1]
        )
        k = optimal_k
        k_range = list(range(auto_k_range[0], auto_k_range[1] + 1))
        
        # 生成最佳K值确定图
        k_plot = create_optimal_k_plot(inertia_values, silhouette_scores, k_range)
        result['optimal_k_plot'] = plot_to_base64(k_plot)
        result['k_analysis'] = {
            'optimal_k': optimal_k,
            'inertia_values': inertia_values,
            'silhouette_scores': silhouette_scores,
            'k_range': k_range
        }
    
    # 3. 执行聚类
    if algorithm == 'kmeans':
        labels = cluster_kmeans(features_scaled, k)
        result['algorithm'] = 'K-means'
        result['k'] = k
    elif algorithm == 'dbscan':
        labels = cluster_dbscan(features_scaled, dbscan_eps, dbscan_min_samples)
        k = len(set(labels)) - (1 if -1 in labels else 0)  # 实际簇数
        result['algorithm'] = 'DBSCAN'
        result['eps'] = dbscan_eps
        result['min_samples'] = dbscan_min_samples
    else:
        raise ValueError("算法必须是 'kmeans' 或 'dbscan'")
    
    result['n_clusters'] = k
    result['labels'] = labels.tolist()
    
    # 4. 生成可视化图表
    print("生成可视化图表...")
    
    # 2D聚类可视化
    cluster_plot = visualize_clusters_2d(features_scaled, labels, feature_names, 
                                        method=reduction_method)
    result['cluster_plot'] = plot_to_base64(cluster_plot)
    
    # 特征分布图
    feature_plot = visualize_feature_distribution(df_clean, labels)
    result['feature_distribution_plot'] = plot_to_base64(feature_plot)
    
    # 5. 聚类统计分析
    cluster_stats = analyze_clusters(df_clean, labels)
    result['cluster_summary'] = cluster_stats.to_dict('records')
    
    # 雷达图（如果有足够的特征）
    if len(feature_names) >= 3:
        radar_plot = visualize_cluster_radar(cluster_stats)
        if radar_plot:
            result['radar_plot'] = plot_to_base64(radar_plot)
    
    # 6. 保存带聚类标签的结果文件
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_filename = Path(file_path).stem
    output_filename = f"{base_filename}_clustered_{algorithm}_k{k}_{timestamp}.xlsx"
    output_path = output_dir / output_filename
    
    add_cluster_to_excel(file_path, str(output_path), labels)
    result['output_file'] = output_filename
    
    print("综合聚类分析完成!")
    return result