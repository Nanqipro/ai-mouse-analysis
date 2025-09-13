import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

# 导入并应用matplotlib样式配置
try:
    from matplotlib_config import setup_matplotlib_style
    setup_matplotlib_style()
except ImportError:
    print("警告: 无法导入matplotlib_config，使用默认字体设置")
from scipy.stats import pearsonr, spearmanr
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, SpectralClustering
import community as community_louvain
from collections import defaultdict, Counter
import warnings
warnings.filterwarnings('ignore')

class KeyNeuronCommunityAnalyzer:
    """
    关键神经元社区结构分析器
    """
    
    def __init__(self, data_path, effect_size_path=None):
        """
        初始化分析器
        
        Args:
            data_path: 神经元活动数据路径
            effect_size_path: 效应量数据路径（可选）
        """
        self.data_path = data_path
        self.effect_size_path = effect_size_path
        self.df = None
        self.effect_df = None
        self.key_neurons = None
        self.networks = {}
        self.communities = {}
        
    def load_data(self):
        """加载数据"""
        print("=== 加载数据 ===")
        
        # 加载神经元活动数据
        self.df = pd.read_csv(self.data_path)
        neuron_cols = [col for col in self.df.columns if col.startswith('Neuron_')]
        
        print(f"神经元活动数据:")
        print(f"  - 行为状态: {len(self.df)} 个")
        print(f"  - 神经元总数: {len(neuron_cols)} 个")
        print(f"  - 行为类型: {list(self.df['Behavior'].unique())}")
        
        # 如果有效应量数据，加载并识别关键神经元
        if self.effect_size_path:
            try:
                self.effect_df = pd.read_csv(self.effect_size_path)
                print(f"\n效应量数据加载成功")
                self.identify_key_neurons()
            except:
                print(f"\n效应量数据加载失败，将分析所有神经元")
                self.key_neurons = neuron_cols
        else:
            print(f"\n未提供效应量数据，将分析所有神经元")
            self.key_neurons = neuron_cols
            
    def identify_key_neurons(self, effect_threshold=0.5):
        """
        基于效应量识别关键神经元
        
        Args:
            effect_threshold: 效应量阈值
        """
        print(f"\n=== 识别关键神经元 (阈值: {effect_threshold}) ===")
        
        key_neurons_set = set()
        
        # 遍历每个行为的效应量数据
        behaviors = [col for col in self.effect_df.columns if col != 'Neuron']
        
        for behavior in behaviors:
            if behavior in self.effect_df.columns:
                # 找出该行为下效应量大于阈值的神经元
                mask = self.effect_df[behavior].abs() >= effect_threshold
                behavior_key_neurons = self.effect_df[mask]['Neuron'].tolist()
                key_neurons_set.update(behavior_key_neurons)
                print(f"  {behavior}: {len(behavior_key_neurons)} 个关键神经元")
        
        self.key_neurons = list(key_neurons_set)
        print(f"\n总关键神经元数量: {len(self.key_neurons)}")
        print(f"关键神经元占比: {len(self.key_neurons)/len([col for col in self.df.columns if col.startswith('Neuron_')])*100:.1f}%")
        
        return self.key_neurons
    
    def build_functional_networks(self, correlation_method='pearson', threshold=0.3):
        """
        构建功能连接网络
        
        Args:
            correlation_method: 相关性计算方法 ('pearson' 或 'spearman')
            threshold: 连接强度阈值
        """
        print(f"\n=== 构建功能连接网络 ===")
        print(f"相关性方法: {correlation_method}")
        print(f"连接阈值: {threshold}")
        
        behaviors = self.df['Behavior'].unique()
        
        for behavior in behaviors:
            print(f"\n构建 {behavior} 行为的网络...")
            
            # 获取该行为状态下的神经元活动数据
            behavior_data = self.df[self.df['Behavior'] == behavior]
            if len(behavior_data) == 0:
                continue
                
            # 提取关键神经元的活动
            key_neuron_data = behavior_data[self.key_neurons].T  # 转置，行为神经元，列为样本
            
            # 计算相关性矩阵
            n_neurons = len(self.key_neurons)
            correlation_matrix = np.zeros((n_neurons, n_neurons))
            p_values = np.zeros((n_neurons, n_neurons))
            
            for i in range(n_neurons):
                for j in range(i, n_neurons):
                    if i == j:
                        correlation_matrix[i, j] = 1.0
                        p_values[i, j] = 0.0
                    else:
                        data_i = key_neuron_data.iloc[i].values
                        data_j = key_neuron_data.iloc[j].values
                        
                        if correlation_method == 'pearson':
                            corr, p_val = pearsonr(data_i, data_j)
                        else:
                            corr, p_val = spearmanr(data_i, data_j)
                        
                        correlation_matrix[i, j] = corr
                        correlation_matrix[j, i] = corr
                        p_values[i, j] = p_val
                        p_values[j, i] = p_val
            
            # 构建网络图
            G = nx.Graph()
            
            # 添加节点
            for neuron in self.key_neurons:
                G.add_node(neuron)
            
            # 添加边（基于阈值）
            edge_count = 0
            for i in range(n_neurons):
                for j in range(i+1, n_neurons):
                    if abs(correlation_matrix[i, j]) >= threshold:
                        G.add_edge(
                            self.key_neurons[i], 
                            self.key_neurons[j], 
                            weight=abs(correlation_matrix[i, j]),
                            correlation=correlation_matrix[i, j],
                            p_value=p_values[i, j]
                        )
                        edge_count += 1
            
            self.networks[behavior] = {
                'graph': G,
                'correlation_matrix': correlation_matrix,
                'p_values': p_values,
                'edge_count': edge_count,
                'density': nx.density(G)
            }
            
            print(f"  节点数: {G.number_of_nodes()}")
            print(f"  边数: {edge_count}")
            print(f"  网络密度: {nx.density(G):.3f}")
    
    def detect_communities(self, methods=['louvain', 'spectral', 'hierarchical']):
        """
        使用多种方法检测社区结构
        
        Args:
            methods: 社区检测方法列表
        """
        print(f"\n=== 社区检测 ===")
        
        for behavior, network_data in self.networks.items():
            print(f"\n分析 {behavior} 行为的社区结构:")
            G = network_data['graph']
            
            behavior_communities = {}
            
            # 1. Louvain算法
            if 'louvain' in methods:
                try:
                    louvain_communities = community_louvain.best_partition(G)
                    n_communities = len(set(louvain_communities.values()))
                    modularity = community_louvain.modularity(louvain_communities, G)
                    
                    behavior_communities['louvain'] = {
                        'partition': louvain_communities,
                        'n_communities': n_communities,
                        'modularity': modularity
                    }
                    print(f"  Louvain: {n_communities} 个社区, 模块度 = {modularity:.3f}")
                except:
                    print(f"  Louvain算法失败")
            
            # 2. 谱聚类
            if 'spectral' in methods and G.number_of_edges() > 0:
                try:
                    # 获取邻接矩阵
                    adj_matrix = nx.adjacency_matrix(G).todense()
                    
                    # 尝试不同的聚类数
                    best_spectral = None
                    best_silhouette = -1
                    
                    for n_clusters in range(2, min(8, G.number_of_nodes())):
                        spectral = SpectralClustering(n_clusters=n_clusters, random_state=42)
                        try:
                            labels = spectral.fit_predict(adj_matrix)
                            # 这里可以计算轮廓系数，但简化处理
                            spectral_communities = {node: label for node, label in zip(G.nodes(), labels)}
                            
                            if best_spectral is None:
                                best_spectral = {
                                    'partition': spectral_communities,
                                    'n_communities': n_clusters
                                }
                        except:
                            continue
                    
                    if best_spectral:
                        behavior_communities['spectral'] = best_spectral
                        print(f"  谱聚类: {best_spectral['n_communities']} 个社区")
                except:
                    print(f"  谱聚类失败")
            
            # 3. 层次聚类
            if 'hierarchical' in methods and G.number_of_edges() > 0:
                try:
                    # 使用相关性矩阵进行层次聚类
                    correlation_matrix = network_data['correlation_matrix']
                    distance_matrix = 1 - np.abs(correlation_matrix)
                    
                    # 执行层次聚类
                    linkage_matrix = linkage(distance_matrix, method='ward')
                    
                    # 确定最优聚类数（简化处理，使用固定数目）
                    n_clusters = min(5, len(self.key_neurons) // 3)
                    if n_clusters >= 2:
                        cluster_labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
                        
                        hierarchical_communities = {
                            neuron: label for neuron, label in zip(self.key_neurons, cluster_labels)
                        }
                        
                        behavior_communities['hierarchical'] = {
                            'partition': hierarchical_communities,
                            'n_communities': n_clusters,
                            'linkage_matrix': linkage_matrix
                        }
                        print(f"  层次聚类: {n_clusters} 个社区")
                except:
                    print(f"  层次聚类失败")
            
            self.communities[behavior] = behavior_communities
    
    def analyze_community_characteristics(self):
        """分析社区特征"""
        print(f"\n=== 社区特征分析 ===")
        
        community_analysis = {}
        
        for behavior, communities in self.communities.items():
            print(f"\n{behavior} 行为的社区特征:")
            behavior_analysis = {}
            
            G = self.networks[behavior]['graph']
            
            for method, community_data in communities.items():
                partition = community_data['partition']
                n_communities = community_data['n_communities']
                
                print(f"\n  {method} 方法:")
                print(f"    社区数量: {n_communities}")
                
                # 分析每个社区的特征
                community_stats = {}
                for community_id in range(n_communities):
                    # 获取该社区的神经元
                    community_neurons = [node for node, comm in partition.items() if comm == community_id]
                    
                    if len(community_neurons) > 0:
                        # 社区大小
                        size = len(community_neurons)
                        
                        # 社区内连接
                        subgraph = G.subgraph(community_neurons)
                        internal_edges = subgraph.number_of_edges()
                        
                        # 计算社区内连接密度
                        max_internal_edges = size * (size - 1) / 2
                        internal_density = internal_edges / max_internal_edges if max_internal_edges > 0 else 0
                        
                        community_stats[community_id] = {
                            'neurons': community_neurons,
                            'size': size,
                            'internal_edges': internal_edges,
                            'internal_density': internal_density
                        }
                        
                        print(f"      社区 {community_id}: {size} 个神经元, 内部密度 = {internal_density:.3f}")
                
                behavior_analysis[method] = community_stats
            
            community_analysis[behavior] = behavior_analysis
        
        return community_analysis
    
    def compare_communities_across_behaviors(self):
        """比较不同行为状态下的社区结构"""
        print(f"\n=== 跨行为社区结构比较 ===")
        
        # 使用Louvain方法的结果进行比较
        louvain_communities = {}
        for behavior, communities in self.communities.items():
            if 'louvain' in communities:
                louvain_communities[behavior] = communities['louvain']['partition']
        
        if len(louvain_communities) < 2:
            print("需要至少两个行为状态才能进行比较")
            return
        
        behaviors = list(louvain_communities.keys())
        
        print(f"比较行为: {behaviors}")
        
        # 计算社区结构相似性
        for i in range(len(behaviors)):
            for j in range(i+1, len(behaviors)):
                behavior1, behavior2 = behaviors[i], behaviors[j]
                partition1 = louvain_communities[behavior1]
                partition2 = louvain_communities[behavior2]
                
                # 计算调整兰德指数（简化版本）
                common_neurons = set(partition1.keys()) & set(partition2.keys())
                if len(common_neurons) > 0:
                    # 简化的相似性计算：相同社区分配的神经元比例
                    same_assignment = sum(1 for neuron in common_neurons 
                                        if partition1[neuron] == partition2[neuron])
                    similarity = same_assignment / len(common_neurons)
                    
                    print(f"  {behavior1} vs {behavior2}: 相似度 = {similarity:.3f}")
        
        return louvain_communities
    
    def visualize_networks_and_communities(self, save_plots=True):
        """可视化网络和社区结构"""
        print(f"\n=== 网络和社区可视化 ===")
        
        if save_plots:
            import os
            plot_dir = "community_analysis_plots"
            if not os.path.exists(plot_dir):
                os.makedirs(plot_dir)
        
        behaviors = list(self.networks.keys())
        n_behaviors = len(behaviors)
        
        # 为每个行为创建网络可视化
        for behavior in behaviors:
            if behavior not in self.communities or 'louvain' not in self.communities[behavior]:
                continue
                
            G = self.networks[behavior]['graph']
            partition = self.communities[behavior]['louvain']['partition']
            
            # 创建图形
            plt.figure(figsize=(12, 10))
            
            # 设置布局
            if G.number_of_nodes() > 0:
                try:
                    pos = nx.spring_layout(G, k=1, iterations=50)
                except:
                    pos = nx.random_layout(G)
                
                # 绘制边
                nx.draw_networkx_edges(G, pos, alpha=0.5, width=0.5, edge_color='gray')
                
                # 为不同社区设置不同颜色
                unique_communities = set(partition.values())
                colors = plt.cm.Set3(np.linspace(0, 1, len(unique_communities)))
                
                for i, community in enumerate(unique_communities):
                    community_nodes = [node for node, comm in partition.items() if comm == community]
                    nx.draw_networkx_nodes(G, pos, nodelist=community_nodes, 
                                         node_color=[colors[i]], node_size=300, alpha=0.8)
                
                # 绘制标签
                nx.draw_networkx_labels(G, pos, font_size=8)
            
            plt.title(f'{behavior} 行为的神经元网络和社区结构', fontsize=16, fontweight='bold')
            plt.axis('off')
            
            if save_plots:
                plt.savefig(f"{plot_dir}/network_{behavior}.png", dpi=300, bbox_inches='tight')
            plt.show()
        
        # 创建社区统计汇总图
        self._plot_community_statistics(save_plots, plot_dir if save_plots else None)
    
    def _plot_community_statistics(self, save_plots=True, plot_dir=None):
        """绘制社区统计图表"""
        
        # 收集统计数据
        stats_data = []
        for behavior, communities in self.communities.items():
            if 'louvain' in communities:
                stats_data.append({
                    'Behavior': behavior,
                    'N_Communities': communities['louvain']['n_communities'],
                    'Modularity': communities['louvain']['modularity'],
                    'Network_Density': self.networks[behavior]['density']
                })
        
        if not stats_data:
            return
        
        stats_df = pd.DataFrame(stats_data)
        
        # 创建子图
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('社区结构统计分析', fontsize=16)
        
        # 1. 社区数量
        ax1 = axes[0, 0]
        bars1 = ax1.bar(stats_df['Behavior'], stats_df['N_Communities'], 
                       color='skyblue', alpha=0.7)
        ax1.set_title('各行为状态的社区数量', fontsize=14, fontweight='bold')
        ax1.set_ylabel('社区数量', fontsize=12, fontweight='bold')
        ax1.tick_params(axis='x', rotation=45, labelsize=10, labelweight='bold')
        ax1.tick_params(axis='y', labelsize=10, labelweight='bold')
        
        # 添加数值标签
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom')
        
        # 2. 模块度
        ax2 = axes[0, 1]
        bars2 = ax2.bar(stats_df['Behavior'], stats_df['Modularity'], 
                       color='lightgreen', alpha=0.7)
        ax2.set_title('各行为状态的模块度', fontsize=14, fontweight='bold')
        ax2.set_ylabel('模块度', fontsize=12, fontweight='bold')
        ax2.tick_params(axis='x', rotation=45, labelsize=10, labelweight='bold')
        ax2.tick_params(axis='y', labelsize=10, labelweight='bold')
        
        # 添加数值标签
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom')
        
        # 3. 网络密度
        ax3 = axes[1, 0]
        bars3 = ax3.bar(stats_df['Behavior'], stats_df['Network_Density'], 
                       color='orange', alpha=0.7)
        ax3.set_title('各行为状态的网络密度', fontsize=14, fontweight='bold')
        ax3.set_ylabel('网络密度', fontsize=12, fontweight='bold')
        ax3.tick_params(axis='x', rotation=45, labelsize=10, labelweight='bold')
        ax3.tick_params(axis='y', labelsize=10, labelweight='bold')
        
        # 添加数值标签
        for bar in bars3:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom')
        
        # 4. 关系散点图
        ax4 = axes[1, 1]
        scatter = ax4.scatter(stats_df['Network_Density'], stats_df['Modularity'], 
                            c=stats_df['N_Communities'], cmap='viridis', s=100, alpha=0.7)
        ax4.set_xlabel('网络密度', fontsize=12, fontweight='bold')
        ax4.set_ylabel('模块度', fontsize=12, fontweight='bold')
        ax4.set_title('网络密度 vs 模块度', fontsize=14, fontweight='bold')
        ax4.tick_params(axis='both', labelsize=10, labelweight='bold')
        
        # 添加颜色条
        cbar = plt.colorbar(scatter, ax=ax4)
        cbar.set_label('社区数量')
        
        # 添加行为标签
        for i, row in stats_df.iterrows():
            ax4.annotate(row['Behavior'], 
                        (row['Network_Density'], row['Modularity']),
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        plt.tight_layout()
        
        if save_plots and plot_dir:
            plt.savefig(f"{plot_dir}/community_statistics.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_community_report(self):
        """生成社区分析报告"""
        print(f"\n=== 生成社区分析报告 ===")
        
        report = []
        report.append("# 关键神经元社区结构分析报告\n")
        
        # 数据概览
        report.append("## 📊 数据概览\n")
        report.append(f"- **关键神经元数量**: {len(self.key_neurons)}\n")
        report.append(f"- **分析行为状态**: {list(self.networks.keys())}\n")
        report.append(f"- **网络构建方法**: 功能连接（相关性）\n")
        report.append(f"- **社区检测方法**: Louvain、谱聚类、层次聚类\n\n")
        
        # 网络特征
        report.append("## 🕸️ 网络特征\n")
        for behavior, network_data in self.networks.items():
            G = network_data['graph']
            report.append(f"### {behavior} 行为网络\n")
            report.append(f"- **节点数**: {G.number_of_nodes()}\n")
            report.append(f"- **边数**: {network_data['edge_count']}\n")
            report.append(f"- **网络密度**: {network_data['density']:.3f}\n")
            
            if behavior in self.communities and 'louvain' in self.communities[behavior]:
                louvain_data = self.communities[behavior]['louvain']
                report.append(f"- **社区数量**: {louvain_data['n_communities']}\n")
                report.append(f"- **模块度**: {louvain_data['modularity']:.3f}\n")
            report.append("\n")
        
        # 社区分析结果
        report.append("## 🏘️ 社区结构分析\n")
        
        for behavior, communities in self.communities.items():
            report.append(f"### {behavior} 行为社区\n")
            
            for method, community_data in communities.items():
                report.append(f"#### {method.title()} 方法\n")
                report.append(f"- **社区数量**: {community_data['n_communities']}\n")
                
                if 'modularity' in community_data:
                    report.append(f"- **模块度**: {community_data['modularity']:.3f}\n")
                
                # 列出各社区的神经元
                partition = community_data['partition']
                community_neurons = defaultdict(list)
                for neuron, comm_id in partition.items():
                    community_neurons[comm_id].append(neuron)
                
                for comm_id, neurons in community_neurons.items():
                    report.append(f"  - **社区 {comm_id}**: {neurons}\n")
                
                report.append("\n")
        
        # 保存报告
        report_text = "".join(report)
        with open('community_analysis_report.md', 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        print("报告已保存到 'community_analysis_report.md'")
        
        return report_text

def main():
    """主分析流程"""
    print("=== 关键神经元社区结构分析 ===")
    
    # 初始化分析器
    analyzer = KeyNeuronCommunityAnalyzer(
        data_path='data/EMtrace02-3标签版.csv',
        effect_size_path=None  # 如果有效应量数据，请提供路径
    )
    
    # 加载数据
    analyzer.load_data()
    
    # 构建功能连接网络
    analyzer.build_functional_networks(
        correlation_method='pearson',
        threshold=0.3
    )
    
    # 检测社区结构
    analyzer.detect_communities(methods=['louvain', 'spectral', 'hierarchical'])
    
    # 分析社区特征
    community_characteristics = analyzer.analyze_community_characteristics()
    
    # 比较不同行为的社区结构
    community_comparison = analyzer.compare_communities_across_behaviors()
    
    # 可视化结果
    analyzer.visualize_networks_and_communities()
    
    # 生成报告
    report = analyzer.generate_community_report()
    
    print("\n=== 分析完成 ===")
    print("1. 已构建功能连接网络")
    print("2. 已检测社区结构")
    print("3. 已分析社区特征")
    print("4. 已生成可视化图表")
    print("5. 已生成分析报告")
    
    return {
        'analyzer': analyzer,
        'characteristics': community_characteristics,
        'comparison': community_comparison,
        'report': report
    }

if __name__ == "__main__":
    results = main()