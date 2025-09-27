import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 导入并应用matplotlib样式配置
try:
    from matplotlib_config import setup_matplotlib_style
    setup_matplotlib_style()
except ImportError:
    print("警告: 无法导入matplotlib_config，使用默认字体设置")
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

class TemporalPatternAnalyzer:
    """
    分析小鼠活动前后神经元活动时间模式的类
    """
    
    def __init__(self, data_path):
        """
        初始化分析器
        
        Args:
            data_path: 数据文件路径
        """
        self.data_path = data_path
        self.df = None
        self.behaviors = None
        self.neurons = None
        
    def load_and_preprocess_data(self):
        """加载并预处理数据"""
        print("正在加载数据...")
        self.df = pd.read_csv(self.data_path)
        
        # 获取行为类型和神经元列名
        self.behaviors = self.df['Behavior'].tolist()
        self.neurons = [col for col in self.df.columns if col.startswith('Neuron_')]
        
        print(f"数据加载完成:")
        print(f"- 行为类型数量: {len(self.behaviors)}")
        print(f"- 神经元数量: {len(self.neurons)}")
        print(f"- 行为类型: {list(set(self.behaviors))}")
        
        return self.df
    
    def analyze_sequential_patterns(self):
        """
        分析行为序列中的时间模式
        寻找活动前后的神经元活动变化
        """
        print("\n=== 时间序列模式分析 ===")
        
        # 创建行为转换矩阵
        behavior_transitions = self._analyze_behavior_transitions()
        
        # 分析每个转换的神经元活动变化
        transition_patterns = self._analyze_transition_patterns()
        
        # 分析神经元活动的时间动力学
        temporal_dynamics = self._analyze_temporal_dynamics()
        
        return {
            'behavior_transitions': behavior_transitions,
            'transition_patterns': transition_patterns,
            'temporal_dynamics': temporal_dynamics
        }
    
    def _analyze_behavior_transitions(self):
        """分析行为转换模式"""
        print("\n分析行为转换模式...")
        
        transitions = []
        for i in range(len(self.behaviors) - 1):
            current_behavior = self.behaviors[i]
            next_behavior = self.behaviors[i + 1]
            transitions.append((current_behavior, next_behavior))
        
        transition_counts = {}
        for transition in transitions:
            transition_counts[transition] = transition_counts.get(transition, 0) + 1
        
        print("行为转换频率:")
        for transition, count in sorted(transition_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {transition[0]} -> {transition[1]}: {count}次")
        
        return transition_counts
    
    def _analyze_transition_patterns(self):
        """分析转换期间的神经元活动模式"""
        print("\n分析转换期间神经元活动模式...")
        
        transition_data = []
        
        for i in range(len(self.behaviors) - 1):
            current_row = self.df.iloc[i]
            next_row = self.df.iloc[i + 1]
            
            current_behavior = current_row['Behavior']
            next_behavior = next_row['Behavior']
            
            # 计算神经元活动变化
            activity_changes = {}
            for neuron in self.neurons:
                current_activity = current_row[neuron]
                next_activity = next_row[neuron]
                change = next_activity - current_activity
                activity_changes[neuron] = change
            
            transition_data.append({
                'from_behavior': current_behavior,
                'to_behavior': next_behavior,
                'transition_index': i,
                **activity_changes
            })
        
        transition_df = pd.DataFrame(transition_data)
        
        # 分析不同转换类型的神经元活动模式
        unique_transitions = transition_df[['from_behavior', 'to_behavior']].drop_duplicates()
        
        patterns = {}
        for _, row in unique_transitions.iterrows():
            from_behavior = row['from_behavior']
            to_behavior = row['to_behavior']
            
            mask = (transition_df['from_behavior'] == from_behavior) & (transition_df['to_behavior'] == to_behavior)
            subset = transition_df[mask]
            
            if len(subset) > 0:
                # 计算该转换类型的平均神经元活动变化
                neuron_changes = subset[self.neurons].mean()
                patterns[f"{from_behavior} -> {to_behavior}"] = neuron_changes
        
        return patterns
    
    def _analyze_temporal_dynamics(self):
        """分析神经元活动的时间动力学"""
        print("\n分析神经元活动时间动力学...")
        
        dynamics = {}
        
        # 为每个神经元分析时间序列特征
        for neuron in self.neurons:
            activity_series = self.df[neuron].values
            
            # 计算基本统计特征
            mean_activity = np.mean(activity_series)
            std_activity = np.std(activity_series)
            
            # 计算活动变化率
            activity_changes = np.diff(activity_series)
            mean_change_rate = np.mean(np.abs(activity_changes))
            
            # 计算活动峰值检测
            peaks = self._detect_activity_peaks(activity_series)
            
            dynamics[neuron] = {
                'mean_activity': mean_activity,
                'std_activity': std_activity,
                'mean_change_rate': mean_change_rate,
                'num_peaks': len(peaks),
                'peak_indices': peaks
            }
        
        return dynamics
    
    def _detect_activity_peaks(self, activity_series, threshold_factor=1.5):
        """检测神经元活动峰值"""
        mean_activity = np.mean(activity_series)
        std_activity = np.std(activity_series)
        threshold = mean_activity + threshold_factor * std_activity
        
        peaks = []
        for i in range(1, len(activity_series) - 1):
            if (activity_series[i] > threshold and 
                activity_series[i] > activity_series[i-1] and 
                activity_series[i] > activity_series[i+1]):
                peaks.append(i)
        
        return peaks
    
    def cluster_neuron_patterns(self, method='hierarchical'):
        """
        对神经元活动模式进行聚类分析
        
        Args:
            method: 聚类方法 ('hierarchical', 'kmeans', 'pca')
        """
        print(f"\n=== 神经元模式聚类分析 (方法: {method}) ===")
        
        # 准备数据矩阵 (神经元 x 行为状态)
        activity_matrix = self.df[self.neurons].T  # 转置，行为神经元，列为行为状态
        
        if method == 'hierarchical':
            return self._hierarchical_clustering(activity_matrix)
        elif method == 'kmeans':
            return self._kmeans_clustering(activity_matrix)
        elif method == 'pca':
            return self._pca_analysis(activity_matrix)
        else:
            raise ValueError(f"未知的聚类方法: {method}")
    
    def _hierarchical_clustering(self, activity_matrix):
        """层次聚类分析"""
        print("执行层次聚类分析...")
        
        # 计算距离矩阵和聚类
        linkage_matrix = linkage(activity_matrix, method='ward')
        
        # 获取聚类标签
        n_clusters = 5  # 可以调整聚类数量
        cluster_labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
        
        # 创建聚类结果
        clusters = {}
        for i, neuron in enumerate(self.neurons):
            cluster_id = cluster_labels[i]
            if cluster_id not in clusters:
                clusters[cluster_id] = []
            clusters[cluster_id].append(neuron)
        
        print(f"识别出 {len(clusters)} 个神经元聚类:")
        for cluster_id, neurons in clusters.items():
            print(f"  聚类 {cluster_id}: {len(neurons)} 个神经元 - {neurons[:5]}{'...' if len(neurons) > 5 else ''}")
        
        return {
            'linkage_matrix': linkage_matrix,
            'cluster_labels': cluster_labels,
            'clusters': clusters,
            'activity_matrix': activity_matrix
        }
    
    def _kmeans_clustering(self, activity_matrix):
        """K-means聚类分析"""
        print("执行K-means聚类分析...")
        
        # 标准化数据
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(activity_matrix)
        
        # K-means聚类
        n_clusters = 5
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(scaled_data)
        
        # 创建聚类结果
        clusters = {}
        for i, neuron in enumerate(self.neurons):
            cluster_id = cluster_labels[i]
            if cluster_id not in clusters:
                clusters[cluster_id] = []
            clusters[cluster_id].append(neuron)
        
        print(f"K-means识别出 {len(clusters)} 个神经元聚类:")
        for cluster_id, neurons in clusters.items():
            print(f"  聚类 {cluster_id}: {len(neurons)} 个神经元")
        
        return {
            'kmeans_model': kmeans,
            'cluster_labels': cluster_labels,
            'clusters': clusters,
            'scaled_data': scaled_data
        }
    
    def _pca_analysis(self, activity_matrix):
        """主成分分析"""
        print("执行主成分分析...")
        
        # 标准化数据
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(activity_matrix)
        
        # PCA分析
        pca = PCA()
        pca_result = pca.fit_transform(scaled_data)
        
        # 计算方差解释比例
        explained_variance_ratio = pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance_ratio)
        
        print("主成分方差解释:")
        for i in range(min(5, len(explained_variance_ratio))):
            print(f"  PC{i+1}: {explained_variance_ratio[i]:.3f} ({cumulative_variance[i]:.3f} 累积)")
        
        return {
            'pca_model': pca,
            'pca_result': pca_result,
            'explained_variance_ratio': explained_variance_ratio,
            'cumulative_variance': cumulative_variance,
            'scaled_data': scaled_data
        }
    
    def identify_response_patterns(self):
        """识别神经元响应模式"""
        print("\n=== 神经元响应模式识别 ===")
        
        response_patterns = {}
        
        # 分析每个行为类型的神经元响应特征
        unique_behaviors = list(set(self.behaviors))
        
        for behavior in unique_behaviors:
            behavior_indices = [i for i, b in enumerate(self.behaviors) if b == behavior]
            
            if len(behavior_indices) > 0:
                # 计算该行为类型的神经元活动特征
                behavior_activity = self.df.iloc[behavior_indices][self.neurons]
                
                pattern = {
                    'mean_activity': behavior_activity.mean(),
                    'std_activity': behavior_activity.std(),
                    'max_activity': behavior_activity.max(),
                    'min_activity': behavior_activity.min(),
                    'occurrence_count': len(behavior_indices)
                }
                
                response_patterns[behavior] = pattern
        
        return response_patterns
    
    def suggest_analysis_models(self, patterns_data):
        """根据发现的模式推荐分析模型"""
        print("\n=== 推荐分析模型 ===")
        
        recommendations = []
        
        # 基于数据特征推荐模型
        print("基于您的数据特征，推荐以下分析模型:")
        
        # 1. 时间序列分析模型
        recommendations.append({
            'model_type': '时间序列分析',
            'specific_models': [
                'ARIMA模型 - 用于分析神经元活动的时间依赖性',
                '状态空间模型 - 用于跟踪神经元状态变化',
                '隐马尔可夫模型(HMM) - 用于识别隐藏的行为状态'
            ],
            'use_case': '分析神经元活动的时间动态和预测未来活动',
            'libraries': ['statsmodels', 'pykalman', 'hmmlearn']
        })
        
        # 2. 聚类和降维模型
        recommendations.append({
            'model_type': '模式识别与聚类',
            'specific_models': [
                '层次聚类 - 识别具有相似活动模式的神经元群',
                'K-means聚类 - 将神经元分组为功能模块',
                '主成分分析(PCA) - 降维并识别主要变化模式',
                't-SNE/UMAP - 可视化高维神经元活动模式'
            ],
            'use_case': '发现神经元功能模块和活动模式',
            'libraries': ['scikit-learn', 'umap-learn', 'seaborn']
        })
        
        # 3. 因果分析模型
        recommendations.append({
            'model_type': '因果关系分析',
            'specific_models': [
                'Granger因果检验 - 分析神经元间的因果关系',
                '结构方程模型(SEM) - 建模神经元网络结构',
                '动态因果模型(DCM) - 分析神经网络的有效连接'
            ],
            'use_case': '理解神经元间的因果关系和信息流',
            'libraries': ['statsmodels', 'semopy', 'mne']
        })
        
        # 4. 机器学习模型
        recommendations.append({
            'model_type': '预测性机器学习',
            'specific_models': [
                '随机森林 - 预测行为类型基于神经元活动',
                '支持向量机(SVM) - 分类不同的行为状态',
                '神经网络 - 深度学习模型用于复杂模式识别',
                'LSTM/GRU - 处理时间序列的深度学习模型'
            ],
            'use_case': '基于神经元活动预测行为或分类状态',
            'libraries': ['scikit-learn', 'tensorflow', 'pytorch']
        })
        
        # 5. 网络分析模型
        recommendations.append({
            'model_type': '网络分析',
            'specific_models': [
                '功能连接分析 - 基于相关性的网络构建',
                '图论分析 - 分析神经网络的拓扑特征',
                '社区检测算法 - 识别神经元群落结构'
            ],
            'use_case': '分析神经元网络的结构和功能特性',
            'libraries': ['networkx', 'igraph', 'brain-connectivity-toolbox']
        })
        
        # 打印推荐
        for i, rec in enumerate(recommendations, 1):
            print(f"\n{i}. {rec['model_type']}")
            print(f"   用途: {rec['use_case']}")
            print("   具体模型:")
            for model in rec['specific_models']:
                print(f"     - {model}")
            print(f"   推荐库: {', '.join(rec['libraries'])}")
        
        return recommendations
    
    def visualize_patterns(self, clustering_result=None, save_plots=True):
        """可视化分析结果"""
        print("\n=== 生成可视化图表 ===")
        
        if save_plots:
            import os
            plot_dir = "temporal_analysis_plots"
            if not os.path.exists(plot_dir):
                os.makedirs(plot_dir)
        
        # 1. 神经元活动热图
        plt.figure(figsize=(15, 8))
        activity_data = self.df[self.neurons].T
        sns.heatmap(activity_data, cmap='viridis', cbar=True)
        plt.title('神经元活动热图 (神经元 x 时间点)', fontsize=16, fontweight='bold')
        plt.xlabel('时间点 (行为序列)', fontsize=14, fontweight='bold')
        plt.ylabel('神经元', fontsize=14, fontweight='bold')
        plt.xticks(fontsize=12, fontweight='bold')
        plt.yticks(fontsize=12, fontweight='bold')
        if save_plots:
            plt.savefig(f"{plot_dir}/neuron_activity_heatmap.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        # 2. 行为转换可视化
        behaviors_numeric = pd.Categorical(self.behaviors).codes
        plt.figure(figsize=(12, 4))
        plt.plot(behaviors_numeric, marker='o', linewidth=2, markersize=6)
        plt.title('行为序列时间线', fontsize=16, fontweight='bold')
        plt.xlabel('时间点', fontsize=14, fontweight='bold')
        plt.ylabel('行为类型 (编码)', fontsize=14, fontweight='bold')
        plt.tick_params(axis='both', labelsize=12, labelweight='bold')
        unique_behaviors = list(set(self.behaviors))
        plt.yticks(range(len(unique_behaviors)), unique_behaviors)
        plt.grid(True, alpha=0.3)
        if save_plots:
            plt.savefig(f"{plot_dir}/behavior_timeline.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        # 3. 如果有聚类结果，绘制聚类树状图
        if clustering_result and 'linkage_matrix' in clustering_result:
            plt.figure(figsize=(12, 8))
            dendrogram(clustering_result['linkage_matrix'], 
                      labels=self.neurons, 
                      leaf_rotation=90)
            plt.title('神经元层次聚类树状图', fontsize=16, fontweight='bold')
            plt.xlabel('神经元', fontsize=14, fontweight='bold')
            plt.ylabel('距离', fontsize=14, fontweight='bold')
            plt.tick_params(axis='both', labelsize=12, labelweight='bold')
            if save_plots:
                plt.savefig(f"{plot_dir}/neuron_clustering_dendrogram.png", dpi=300, bbox_inches='tight')
            plt.show()
        
        print("可视化完成!")

def main():
    """主分析流程"""
    print("=== 小鼠神经元活动时间模式分析 ===")
    
    # 初始化分析器
    data_path = 'data/EMtrace01-多标签版.csv'
    analyzer = TemporalPatternAnalyzer(data_path)
    
    # 加载数据
    analyzer.load_and_preprocess_data()
    
    # 分析时间序列模式
    temporal_patterns = analyzer.analyze_sequential_patterns()
    
    # 聚类分析
    clustering_result = analyzer.cluster_neuron_patterns(method='hierarchical')
    
    # 识别响应模式
    response_patterns = analyzer.identify_response_patterns()
    
    # 推荐分析模型
    model_recommendations = analyzer.suggest_analysis_models(temporal_patterns)
    
    # 可视化结果
    analyzer.visualize_patterns(clustering_result)
    
    print("\n=== 分析总结 ===")
    print("1. 已完成时间序列模式分析")
    print("2. 已完成神经元聚类分析") 
    print("3. 已完成响应模式识别")
    print("4. 已生成模型推荐")
    print("5. 已生成可视化图表")
    
    return {
        'temporal_patterns': temporal_patterns,
        'clustering_result': clustering_result,
        'response_patterns': response_patterns,
        'model_recommendations': model_recommendations
    }

if __name__ == "__main__":
    results = main()