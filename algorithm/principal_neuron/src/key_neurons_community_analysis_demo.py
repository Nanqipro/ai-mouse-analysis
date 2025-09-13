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

class KeyNeuronCommunityDemo:
    """
    关键神经元社区结构分析演示器
    """
    
    def __init__(self, data_path):
        """
        初始化演示器
        
        Args:
            data_path: 神经元活动数据路径
        """
        self.data_path = data_path
        self.df = None
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
        
        # 识别关键神经元（基于活动水平）
        self.identify_key_neurons_by_activity()
        
    def identify_key_neurons_by_activity(self, top_n=20):
        """
        基于活动水平识别关键神经元
        
        Args:
            top_n: 选择的关键神经元数量
        """
        print(f"\n=== 基于活动水平识别关键神经元 ===")
        
        neuron_cols = [col for col in self.df.columns if col.startswith('Neuron_')]
        
        # 计算每个神经元的平均活动水平和变异性
        neuron_stats = {}
        for neuron in neuron_cols:
            activity_values = self.df[neuron].values
            mean_activity = np.mean(activity_values)
            std_activity = np.std(activity_values)
            max_activity = np.max(activity_values)
            
            # 综合评分：考虑平均活动水平和变异性
            score = mean_activity + std_activity * 0.5
            
            neuron_stats[neuron] = {
                'mean': mean_activity,
                'std': std_activity,
                'max': max_activity,
                'score': score
            }
        
        # 选择得分最高的神经元作为关键神经元
        sorted_neurons = sorted(neuron_stats.items(), key=lambda x: x[1]['score'], reverse=True)
        self.key_neurons = [neuron for neuron, stats in sorted_neurons[:top_n]]
        
        print(f"选择了 {len(self.key_neurons)} 个关键神经元")
        print("前10个关键神经元:")
        for i, (neuron, stats) in enumerate(sorted_neurons[:10]):
            print(f"  {i+1}. {neuron}: 平均={stats['mean']:.3f}, 标准差={stats['std']:.3f}, 得分={stats['score']:.3f}")
        
        return self.key_neurons
    
    def build_activity_similarity_networks(self, similarity_method='euclidean'):
        """
        基于活动模式相似性构建网络
        
        Args:
            similarity_method: 相似性计算方法
        """
        print(f"\n=== 构建活动相似性网络 ===")
        print(f"相似性方法: {similarity_method}")
        
        behaviors = self.df['Behavior'].unique()
        
        # 获取关键神经元的活动数据
        key_neuron_data = self.df[self.key_neurons].values  # shape: (n_behaviors, n_key_neurons)
        
        # 计算神经元间的相似性矩阵
        n_neurons = len(self.key_neurons)
        similarity_matrix = np.zeros((n_neurons, n_neurons))
        
        for i in range(n_neurons):
            for j in range(n_neurons):
                if i == j:
                    similarity_matrix[i, j] = 1.0
                else:
                    # 计算神经元i和j在所有行为状态下的活动相似性
                    activity_i = key_neuron_data[:, i]
                    activity_j = key_neuron_data[:, j]
                    
                    if similarity_method == 'euclidean':
                        # 欧氏距离的相似性
                        distance = np.sqrt(np.sum((activity_i - activity_j) ** 2))
                        similarity = 1 / (1 + distance)  # 转换为相似性
                    elif similarity_method == 'cosine':
                        # 余弦相似性
                        norm_i = np.linalg.norm(activity_i)
                        norm_j = np.linalg.norm(activity_j)
                        if norm_i > 0 and norm_j > 0:
                            similarity = np.dot(activity_i, activity_j) / (norm_i * norm_j)
                        else:
                            similarity = 0
                    elif similarity_method == 'correlation':
                        # 皮尔逊相关系数的绝对值
                        if len(activity_i) > 1:
                            similarity = abs(np.corrcoef(activity_i, activity_j)[0, 1])
                            if np.isnan(similarity):
                                similarity = 0
                        else:
                            similarity = 0
                    else:
                        similarity = 0
                    
                    similarity_matrix[i, j] = similarity
        
        # 构建网络图
        G = nx.Graph()
        
        # 添加节点
        for neuron in self.key_neurons:
            G.add_node(neuron)
        
        # 设置连接阈值（选择相似性前30%的连接）
        threshold = np.percentile(similarity_matrix[similarity_matrix < 1.0], 70)
        
        # 添加边
        edge_count = 0
        for i in range(n_neurons):
            for j in range(i+1, n_neurons):
                if similarity_matrix[i, j] >= threshold:
                    G.add_edge(
                        self.key_neurons[i], 
                        self.key_neurons[j], 
                        weight=similarity_matrix[i, j],
                        similarity=similarity_matrix[i, j]
                    )
                    edge_count += 1
        
        self.networks['combined'] = {
            'graph': G,
            'similarity_matrix': similarity_matrix,
            'edge_count': edge_count,
            'density': nx.density(G),
            'threshold': threshold
        }
        
        print(f"网络构建完成:")
        print(f"  节点数: {G.number_of_nodes()}")
        print(f"  边数: {edge_count}")
        print(f"  网络密度: {nx.density(G):.3f}")
        print(f"  相似性阈值: {threshold:.3f}")
        
        return G
    
    def detect_communities(self, methods=['louvain', 'spectral', 'hierarchical']):
        """
        检测社区结构
        
        Args:
            methods: 社区检测方法列表
        """
        print(f"\n=== 社区检测 ===")
        
        G = self.networks['combined']['graph']
        
        if G.number_of_nodes() == 0:
            print("网络中没有节点，无法进行社区检测")
            return
        
        communities = {}
        
        # 1. Louvain算法
        if 'louvain' in methods and G.number_of_edges() > 0:
            try:
                louvain_communities = community_louvain.best_partition(G)
                n_communities = len(set(louvain_communities.values()))
                modularity = community_louvain.modularity(louvain_communities, G)
                
                communities['louvain'] = {
                    'partition': louvain_communities,
                    'n_communities': n_communities,
                    'modularity': modularity
                }
                print(f"Louvain算法: {n_communities} 个社区, 模块度 = {modularity:.3f}")
                
                # 显示各社区的神经元
                community_neurons = defaultdict(list)
                for neuron, comm_id in louvain_communities.items():
                    community_neurons[comm_id].append(neuron)
                
                for comm_id, neurons in community_neurons.items():
                    print(f"  社区 {comm_id}: {neurons}")
                    
            except Exception as e:
                print(f"Louvain算法失败: {e}")
        
        # 2. 基于相似性矩阵的层次聚类
        if 'hierarchical' in methods:
            try:
                similarity_matrix = self.networks['combined']['similarity_matrix']
                distance_matrix = 1 - similarity_matrix
                
                # 只使用上三角矩阵进行聚类
                condensed_distances = []
                n = len(self.key_neurons)
                for i in range(n):
                    for j in range(i+1, n):
                        condensed_distances.append(distance_matrix[i, j])
                
                # 执行层次聚类
                linkage_matrix = linkage(condensed_distances, method='ward')
                
                # 确定聚类数（基于网络密度）
                density = self.networks['combined']['density']
                if density > 0.3:
                    n_clusters = max(2, min(6, len(self.key_neurons) // 4))
                else:
                    n_clusters = max(2, min(4, len(self.key_neurons) // 5))
                
                cluster_labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
                
                hierarchical_communities = {
                    neuron: label-1 for neuron, label in zip(self.key_neurons, cluster_labels)
                }
                
                communities['hierarchical'] = {
                    'partition': hierarchical_communities,
                    'n_communities': n_clusters,
                    'linkage_matrix': linkage_matrix
                }
                
                print(f"层次聚类: {n_clusters} 个社区")
                
                # 显示各社区的神经元
                community_neurons = defaultdict(list)
                for neuron, comm_id in hierarchical_communities.items():
                    community_neurons[comm_id].append(neuron)
                
                for comm_id, neurons in community_neurons.items():
                    print(f"  社区 {comm_id}: {neurons}")
                    
            except Exception as e:
                print(f"层次聚类失败: {e}")
        
        # 3. K-means聚类（基于神经元活动模式）
        if 'kmeans' in methods:
            try:
                # 使用神经元的活动数据进行K-means聚类
                neuron_activity = self.df[self.key_neurons].T.values  # shape: (n_neurons, n_behaviors)
                
                # 标准化数据
                scaler = StandardScaler()
                scaled_activity = scaler.fit_transform(neuron_activity)
                
                # 确定聚类数
                n_clusters = max(2, min(5, len(self.key_neurons) // 4))
                
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                cluster_labels = kmeans.fit_predict(scaled_activity)
                
                kmeans_communities = {
                    neuron: label for neuron, label in zip(self.key_neurons, cluster_labels)
                }
                
                communities['kmeans'] = {
                    'partition': kmeans_communities,
                    'n_communities': n_clusters,
                    'model': kmeans
                }
                
                print(f"K-means聚类: {n_clusters} 个社区")
                
                # 显示各社区的神经元
                community_neurons = defaultdict(list)
                for neuron, comm_id in kmeans_communities.items():
                    community_neurons[comm_id].append(neuron)
                
                for comm_id, neurons in community_neurons.items():
                    print(f"  社区 {comm_id}: {neurons}")
                    
            except Exception as e:
                print(f"K-means聚类失败: {e}")
        
        self.communities['combined'] = communities
        return communities
    
    def analyze_behavior_specific_patterns(self):
        """分析不同行为状态下的神经元活动模式"""
        print(f"\n=== 行为特异性模式分析 ===")
        
        behaviors = self.df['Behavior'].unique()
        behavior_patterns = {}
        
        for behavior in behaviors:
            behavior_data = self.df[self.df['Behavior'] == behavior]
            
            if len(behavior_data) > 0:
                # 获取该行为状态下关键神经元的活动
                activity_values = behavior_data[self.key_neurons].iloc[0].values
                
                # 识别高活动和低活动的神经元
                mean_activity = np.mean(activity_values)
                std_activity = np.std(activity_values)
                
                high_threshold = mean_activity + 0.5 * std_activity
                low_threshold = mean_activity - 0.5 * std_activity
                
                high_activity_neurons = [neuron for neuron, activity in zip(self.key_neurons, activity_values) 
                                       if activity >= high_threshold]
                low_activity_neurons = [neuron for neuron, activity in zip(self.key_neurons, activity_values) 
                                      if activity <= low_threshold]
                
                behavior_patterns[behavior] = {
                    'high_activity': high_activity_neurons,
                    'low_activity': low_activity_neurons,
                    'mean_activity': mean_activity,
                    'activity_values': dict(zip(self.key_neurons, activity_values))
                }
                
                print(f"{behavior} 行为:")
                print(f"  平均活动水平: {mean_activity:.3f}")
                print(f"  高活动神经元 ({len(high_activity_neurons)}个): {high_activity_neurons[:5]}{'...' if len(high_activity_neurons) > 5 else ''}")
                print(f"  低活动神经元 ({len(low_activity_neurons)}个): {low_activity_neurons[:5]}{'...' if len(low_activity_neurons) > 5 else ''}")
        
        return behavior_patterns
    
    def visualize_community_networks(self, save_plots=True):
        """可视化社区网络结构"""
        print(f"\n=== 网络和社区可视化 ===")
        
        if save_plots:
            import os
            plot_dir = "community_demo_plots"
            if not os.path.exists(plot_dir):
                os.makedirs(plot_dir)
        
        G = self.networks['combined']['graph']
        
        if 'combined' not in self.communities or 'louvain' not in self.communities['combined']:
            print("没有可用的社区检测结果")
            return
        
        partition = self.communities['combined']['louvain']['partition']
        
        # 创建网络可视化
        plt.figure(figsize=(14, 10))
        
        if G.number_of_nodes() > 0:
            # 设置布局
            pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
            
            # 绘制边
            edges = G.edges()
            if len(edges) > 0:
                edge_weights = [G[u][v]['weight'] for u, v in edges]
                nx.draw_networkx_edges(G, pos, alpha=0.3, width=[w*3 for w in edge_weights], 
                                     edge_color='gray')
            
            # 为不同社区设置不同颜色
            unique_communities = set(partition.values())
            colors = plt.cm.Set3(np.linspace(0, 1, len(unique_communities)))
            color_map = {comm: colors[i] for i, comm in enumerate(unique_communities)}
            
            # 绘制节点
            for community in unique_communities:
                community_nodes = [node for node, comm in partition.items() if comm == community]
                nx.draw_networkx_nodes(G, pos, nodelist=community_nodes, 
                                     node_color=[color_map[community]], 
                                     node_size=500, alpha=0.8,
                                     label=f'社区 {community}')
            
            # 绘制标签
            nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold')
            
            plt.title('关键神经元网络和社区结构', fontsize=18, fontweight='bold')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=14)
            plt.axis('off')
            
            if save_plots:
                plt.savefig(f"{plot_dir}/community_network.png", dpi=300, bbox_inches='tight')
            plt.show()
        
        # 创建社区活动热图
        self._plot_community_activity_heatmap(save_plots, plot_dir if save_plots else None)
    
    def _plot_community_activity_heatmap(self, save_plots=True, plot_dir=None):
        """绘制社区活动热图"""
        
        if 'combined' not in self.communities or 'louvain' not in self.communities['combined']:
            return
        
        partition = self.communities['combined']['louvain']['partition']
        
        # 按社区组织神经元数据
        community_data = defaultdict(list)
        community_neurons = defaultdict(list)
        
        for neuron, comm_id in partition.items():
            community_data[comm_id].extend(self.df[neuron].values)
            community_neurons[comm_id].append(neuron)
        
        # 创建热图数据
        behaviors = self.df['Behavior'].unique()
        communities = sorted(partition.values())
        
        heatmap_data = []
        row_labels = []
        
        for comm_id in communities:
            neurons = community_neurons[comm_id]
            for neuron in neurons:
                row_data = []
                for behavior in behaviors:
                    activity = self.df[self.df['Behavior'] == behavior][neuron].iloc[0]
                    row_data.append(activity)
                heatmap_data.append(row_data)
                row_labels.append(f"C{comm_id}_{neuron}")
        
        # 绘制热图
        plt.figure(figsize=(8, max(6, len(row_labels) * 0.3)))
        
        sns.heatmap(heatmap_data, 
                   xticklabels=behaviors,
                   yticklabels=row_labels,
                   cmap='viridis',
                   cbar_kws={'label': '神经元活动'})
        
        plt.title('各社区神经元在不同行为状态下的活动', fontsize=16, fontweight='bold')
        plt.xlabel('行为状态', fontsize=14, fontweight='bold')
        plt.ylabel('神经元 (按社区分组)', fontsize=14, fontweight='bold')
        plt.xticks(fontsize=12, fontweight='bold')
        plt.yticks(fontsize=10, fontweight='bold')
        plt.tight_layout()
        
        if save_plots and plot_dir:
            plt.savefig(f"{plot_dir}/community_activity_heatmap.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_community_insights(self):
        """生成社区分析洞察"""
        print(f"\n=== 社区分析洞察 ===")
        
        insights = []
        
        if 'combined' in self.communities and 'louvain' in self.communities['combined']:
            louvain_data = self.communities['combined']['louvain']
            partition = louvain_data['partition']
            n_communities = louvain_data['n_communities']
            modularity = louvain_data['modularity']
            
            insights.append(f"🏘️ **社区结构发现**:")
            insights.append(f"- 识别出 **{n_communities}** 个神经元社区")
            insights.append(f"- 网络模块度: **{modularity:.3f}** (>0.3表示良好的社区结构)")
            
            # 分析各社区特征
            community_sizes = defaultdict(int)
            for neuron, comm_id in partition.items():
                community_sizes[comm_id] += 1
            
            insights.append(f"\n📊 **社区大小分布**:")
            for comm_id, size in sorted(community_sizes.items()):
                insights.append(f"- 社区 {comm_id}: {size} 个神经元")
            
            # 分析社区的功能特征
            behavior_patterns = self.analyze_behavior_specific_patterns()
            
            insights.append(f"\n🧠 **功能特征分析**:")
            for behavior, pattern in behavior_patterns.items():
                insights.append(f"- **{behavior}** 行为:")
                insights.append(f"  - 高活动神经元: {len(pattern['high_activity'])} 个")
                insights.append(f"  - 低活动神经元: {len(pattern['low_activity'])} 个")
                insights.append(f"  - 平均活动水平: {pattern['mean_activity']:.3f}")
        
        # 网络特征分析
        if 'combined' in self.networks:
            network_data = self.networks['combined']
            insights.append(f"\n🕸️ **网络拓扑特征**:")
            insights.append(f"- 网络密度: **{network_data['density']:.3f}**")
            insights.append(f"- 连接边数: **{network_data['edge_count']}**")
            insights.append(f"- 连接阈值: **{network_data['threshold']:.3f}**")
        
        # 研究意义
        insights.append(f"\n💡 **研究意义**:")
        insights.append(f"- 揭示了关键神经元的功能模块化组织")
        insights.append(f"- 不同社区可能承担不同的功能角色")
        insights.append(f"- 为理解神经网络的信息处理机制提供线索")
        insights.append(f"- 可指导后续的靶向研究和干预策略")
        
        # 保存洞察报告
        insights_text = "\n".join(insights)
        with open('community_insights.md', 'w', encoding='utf-8') as f:
            f.write("# 关键神经元社区结构分析洞察\n\n")
            f.write(insights_text)
        
        print("分析洞察:")
        for insight in insights:
            print(insight)
        
        print(f"\n洞察报告已保存到 'community_insights.md'")
        
        return insights

def main():
    """主分析流程"""
    print("=== 关键神经元社区结构分析演示 ===")
    
    # 初始化演示器
    demo = KeyNeuronCommunityDemo(data_path='data/EMtrace02-3标签版.csv')
    
    # 加载数据
    demo.load_data()
    
    # 构建活动相似性网络
    demo.build_activity_similarity_networks(similarity_method='euclidean')
    
    # 检测社区结构
    demo.detect_communities(methods=['louvain', 'hierarchical', 'kmeans'])
    
    # 分析行为特异性模式
    behavior_patterns = demo.analyze_behavior_specific_patterns()
    
    # 可视化结果
    demo.visualize_community_networks()
    
    # 生成分析洞察
    insights = demo.generate_community_insights()
    
    print("\n=== 演示完成 ===")
    print("1. ✅ 已识别关键神经元")
    print("2. ✅ 已构建功能网络")
    print("3. ✅ 已检测社区结构")
    print("4. ✅ 已分析行为模式")
    print("5. ✅ 已生成可视化")
    print("6. ✅ 已生成分析洞察")
    
    return {
        'demo': demo,
        'behavior_patterns': behavior_patterns,
        'insights': insights
    }

if __name__ == "__main__":
    results = main()