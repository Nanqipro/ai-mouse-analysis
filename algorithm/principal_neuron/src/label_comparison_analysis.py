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
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

class LabelComparisonAnalyzer:
    """
    比较多标签版本和简化标签版本的数据分析器
    """
    
    def __init__(self, multi_label_path, simple_label_path):
        """
        初始化分析器
        
        Args:
            multi_label_path: 多标签版本数据路径
            simple_label_path: 简化标签版本数据路径
        """
        self.multi_label_path = multi_label_path
        self.simple_label_path = simple_label_path
        self.multi_df = None
        self.simple_df = None
        
    def load_data(self):
        """加载两个数据集"""
        print("=== 加载数据集 ===")
        
        # 加载多标签版本
        self.multi_df = pd.read_csv(self.multi_label_path)
        print(f"多标签版本 ({self.multi_label_path}):")
        print(f"  - 数据行数: {len(self.multi_df)}")
        print(f"  - 神经元数量: {len([col for col in self.multi_df.columns if col.startswith('Neuron_')])}")
        print(f"  - 行为标签: {list(self.multi_df['Behavior'].unique())}")
        
        # 加载简化标签版本
        self.simple_df = pd.read_csv(self.simple_label_path)
        print(f"\n简化标签版本 ({self.simple_label_path}):")
        print(f"  - 数据行数: {len(self.simple_df)}")
        print(f"  - 神经元数量: {len([col for col in self.simple_df.columns if col.startswith('Neuron_')])}")
        print(f"  - 行为标签: {list(self.simple_df['Behavior'].unique())}")
        
    def analyze_label_distribution(self):
        """分析标签分布"""
        print("\n=== 标签分布分析 ===")
        
        # 多标签版本分析
        multi_label_counts = self.multi_df['Behavior'].value_counts()
        print("多标签版本标签分布:")
        for label, count in multi_label_counts.items():
            print(f"  {label}: {count}次")
        
        # 简化标签版本分析
        simple_label_counts = self.simple_df['Behavior'].value_counts()
        print("\n简化标签版本标签分布:")
        for label, count in simple_label_counts.items():
            print(f"  {label}: {count}次")
        
        return multi_label_counts, simple_label_counts
    
    def assess_statistical_power(self, multi_label_counts):
        """评估统计效力"""
        print("\n=== 统计效力评估 ===")
        
        # 统计学建议的最小样本量
        min_samples_descriptive = 3  # 描述性统计的最小值
        min_samples_inferential = 5  # 推断统计的建议值
        min_samples_ml = 10  # 机器学习的建议值
        
        print("各标签的统计效力评估:")
        print("标签                     | 样本数 | 描述统计 | 推断统计 | 机器学习")
        print("-" * 70)
        
        insufficient_labels = []
        marginal_labels = []
        sufficient_labels = []
        
        for label, count in multi_label_counts.items():
            descriptive_ok = "✓" if count >= min_samples_descriptive else "✗"
            inferential_ok = "✓" if count >= min_samples_inferential else "✗"
            ml_ok = "✓" if count >= min_samples_ml else "✗"
            
            print(f"{label:<25} | {count:>4}   | {descriptive_ok:>8} | {inferential_ok:>8} | {ml_ok:>8}")
            
            if count < min_samples_descriptive:
                insufficient_labels.append(label)
            elif count < min_samples_inferential:
                marginal_labels.append(label)
            else:
                sufficient_labels.append(label)
        
        print(f"\n统计效力总结:")
        print(f"  数据不足标签 (< {min_samples_descriptive}): {len(insufficient_labels)}个")
        print(f"  数据边缘标签 ({min_samples_descriptive}-{min_samples_inferential-1}): {len(marginal_labels)}个")
        print(f"  数据充足标签 (≥ {min_samples_inferential}): {len(sufficient_labels)}个")
        
        return {
            'insufficient': insufficient_labels,
            'marginal': marginal_labels,
            'sufficient': sufficient_labels
        }
    
    def suggest_label_mapping(self):
        """建议标签映射策略"""
        print("\n=== 标签简化映射建议 ===")
        
        # 基于语义相似性的映射建议
        mapping_strategies = {
            'strategy_1': {
                'name': '三分类映射 (Close-Middle-Open)',
                'mapping': {
                    # Close相关
                    'Closed-Armed-Exp': 'Close',
                    'Closed-arm': 'Close',
                    'Closed-arm-stiff': 'Close',
                    'middle-exp-close-arm': 'Close',
                    
                    # Middle相关
                    'Middle-Zone': 'Middle',
                    'Middle-Zone-stiff': 'Middle',
                    'Middle-Zone-to-Close-arm': 'Middle',
                    'Middle-Zone-to-Open-arm': 'Middle',
                    
                    # Open相关
                    'Open-Armed-Exp': 'Open',
                    'Open-arm': 'Open',
                    'Open-arm-exp-middle': 'Open',
                    'Open-arm-probe': 'Open',
                    'middle-exp-open-arm': 'Open',
                    
                    # 运动状态
                    'Move': 'Middle',  # 可以归类为过渡状态
                    'stiff': 'Middle'  # 可以归类为中间状态
                }
            },
            
            'strategy_2': {
                'name': '五分类映射 (Close-Middle-Open-Transition-Special)',
                'mapping': {
                    # Close相关
                    'Closed-Armed-Exp': 'Close',
                    'Closed-arm': 'Close',
                    'Closed-arm-stiff': 'Close',
                    'middle-exp-close-arm': 'Close',
                    
                    # Middle相关
                    'Middle-Zone': 'Middle',
                    'Middle-Zone-stiff': 'Middle',
                    
                    # Open相关
                    'Open-Armed-Exp': 'Open',
                    'Open-arm': 'Open',
                    'Open-arm-exp-middle': 'Open',
                    'Open-arm-probe': 'Open',
                    'middle-exp-open-arm': 'Open',
                    
                    # 转换状态
                    'Middle-Zone-to-Close-arm': 'Transition',
                    'Middle-Zone-to-Open-arm': 'Transition',
                    'Move': 'Transition',
                    
                    # 特殊状态
                    'stiff': 'Special'
                }
            }
        }
        
        for strategy_key, strategy in mapping_strategies.items():
            print(f"\n策略: {strategy['name']}")
            
            # 计算映射后的标签分布
            mapped_counts = Counter()
            for original_label in self.multi_df['Behavior']:
                mapped_label = strategy['mapping'].get(original_label, original_label)
                mapped_counts[mapped_label] += 1
            
            print("映射规则:")
            current_mapped = None
            for original, mapped in strategy['mapping'].items():
                if mapped != current_mapped:
                    if current_mapped is not None:
                        print()
                    print(f"  {mapped}:")
                    current_mapped = mapped
                original_count = (self.multi_df['Behavior'] == original).sum()
                print(f"    {original} ({original_count}次)")
            
            print(f"\n映射后标签分布:")
            for mapped_label, count in mapped_counts.most_common():
                print(f"  {mapped_label}: {count}次")
        
        return mapping_strategies
    
    def apply_mapping_and_analyze(self, mapping_strategy):
        """应用映射策略并分析结果"""
        print(f"\n=== 应用映射策略分析 ===")
        
        # 创建映射后的数据集
        mapped_df = self.multi_df.copy()
        mapped_df['Original_Behavior'] = mapped_df['Behavior']
        mapped_df['Behavior'] = mapped_df['Original_Behavior'].map(mapping_strategy['mapping'])
        
        # 分析映射后的数据分布
        mapped_counts = mapped_df['Behavior'].value_counts()
        print(f"映射后标签分布:")
        for label, count in mapped_counts.items():
            print(f"  {label}: {count}次")
        
        # 计算神经元活动的统计特征
        neuron_cols = [col for col in mapped_df.columns if col.startswith('Neuron_')]
        
        print(f"\n各标签的神经元活动统计:")
        for label in mapped_counts.index:
            label_data = mapped_df[mapped_df['Behavior'] == label][neuron_cols]
            mean_activity = label_data.mean().mean()
            std_activity = label_data.std().mean()
            print(f"  {label}: 平均活动={mean_activity:.3f}, 标准差={std_activity:.3f}")
        
        return mapped_df, mapped_counts
    
    def compare_with_existing_simple_version(self, mapped_df):
        """与现有简化版本比较"""
        print(f"\n=== 与现有简化版本比较 ===")
        
        # 比较标签
        mapped_labels = set(mapped_df['Behavior'].unique())
        simple_labels = set(self.simple_df['Behavior'].unique())
        
        print(f"映射后标签: {mapped_labels}")
        print(f"现有简化版标签: {simple_labels}")
        print(f"标签重叠: {mapped_labels.intersection(simple_labels)}")
        print(f"映射版独有: {mapped_labels - simple_labels}")
        print(f"简化版独有: {simple_labels - mapped_labels}")
        
        # 比较神经元数量
        mapped_neurons = len([col for col in mapped_df.columns if col.startswith('Neuron_')])
        simple_neurons = len([col for col in self.simple_df.columns if col.startswith('Neuron_')])
        
        print(f"\n神经元数量比较:")
        print(f"  映射版: {mapped_neurons}个神经元")
        print(f"  简化版: {simple_neurons}个神经元")
        
    def recommend_best_approach(self, power_assessment, mapping_strategies):
        """推荐最佳方法"""
        print(f"\n=== 最佳方法推荐 ===")
        
        n_insufficient = len(power_assessment['insufficient'])
        n_marginal = len(power_assessment['marginal'])
        total_labels = len(self.multi_df['Behavior'].unique())
        
        print(f"问题严重程度评估:")
        print(f"  总标签数: {total_labels}")
        print(f"  数据不足标签: {n_insufficient}个 ({n_insufficient/total_labels*100:.1f}%)")
        print(f"  数据边缘标签: {n_marginal}个 ({n_marginal/total_labels*100:.1f}%)")
        
        if n_insufficient > total_labels * 0.3:  # 超过30%标签数据不足
            recommendation = "强烈建议使用标签简化"
            reason = "大量标签数据不足，统计分析不可靠"
            suggested_strategy = "strategy_1"  # 三分类
        elif n_insufficient + n_marginal > total_labels * 0.5:  # 超过50%标签有问题
            recommendation = "建议考虑标签简化"
            reason = "较多标签数据量偏少，可能影响分析质量"
            suggested_strategy = "strategy_2"  # 五分类
        else:
            recommendation = "可以保持现有标签结构"
            reason = "大部分标签有足够数据支撑分析"
            suggested_strategy = None
        
        print(f"\n推荐结果: {recommendation}")
        print(f"推荐理由: {reason}")
        
        if suggested_strategy:
            strategy = mapping_strategies[suggested_strategy]
            print(f"建议策略: {strategy['name']}")
            
            # 应用推荐策略并分析
            mapped_df, mapped_counts = self.apply_mapping_and_analyze(strategy)
            
            print(f"\n使用建议策略的优势:")
            print(f"  1. 增加每个标签的样本量")
            print(f"  2. 提高统计分析的可靠性")
            print(f"  3. 减少过拟合风险")
            print(f"  4. 改善模型泛化能力")
            
            return {
                'recommendation': recommendation,
                'strategy': strategy,
                'mapped_df': mapped_df,
                'mapped_counts': mapped_counts
            }
        
        return {
            'recommendation': recommendation,
            'strategy': None,
            'mapped_df': None,
            'mapped_counts': None
        }
    
    def visualize_label_comparison(self, multi_label_counts, simple_label_counts, mapped_counts=None):
        """可视化标签分布比较"""
        print(f"\n=== 生成可视化图表 ===")
        
        # 设置图表样式
        plt.style.use('default')
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('标签分布比较分析', fontsize=18, fontweight='bold')
        
        # 1. 多标签版本分布
        ax1 = axes[0, 0]
        bars1 = ax1.bar(range(len(multi_label_counts)), multi_label_counts.values, 
                       color='skyblue', alpha=0.7)
        ax1.set_title('多标签版本分布', fontsize=14, fontweight='bold')
        ax1.set_xlabel('标签', fontsize=12, fontweight='bold')
        ax1.set_ylabel('样本数量', fontsize=12, fontweight='bold')
        ax1.tick_params(axis='both', labelsize=10, labelweight='bold')
        ax1.set_xticks(range(len(multi_label_counts)))
        ax1.set_xticklabels(multi_label_counts.index, rotation=45, ha='right')
        
        # 添加数值标签
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom')
        
        # 2. 简化版本分布
        ax2 = axes[0, 1]
        bars2 = ax2.bar(range(len(simple_label_counts)), simple_label_counts.values, 
                       color='lightgreen', alpha=0.7)
        ax2.set_title('简化标签版本分布', fontsize=14, fontweight='bold')
        ax2.set_xlabel('标签', fontsize=12, fontweight='bold')
        ax2.set_ylabel('样本数量', fontsize=12, fontweight='bold')
        ax2.tick_params(axis='both', labelsize=10, labelweight='bold')
        ax2.set_xticks(range(len(simple_label_counts)))
        ax2.set_xticklabels(simple_label_counts.index, rotation=45, ha='right')
        
        # 添加数值标签
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom')
        
        # 3. 映射后分布（如果提供）
        if mapped_counts is not None:
            ax3 = axes[1, 0]
            bars3 = ax3.bar(range(len(mapped_counts)), mapped_counts.values, 
                           color='orange', alpha=0.7)
            ax3.set_title('映射后标签分布', fontsize=14, fontweight='bold')
            ax3.set_xlabel('标签', fontsize=12, fontweight='bold')
            ax3.set_ylabel('样本数量', fontsize=12, fontweight='bold')
            ax3.tick_params(axis='both', labelsize=10, labelweight='bold')
            ax3.set_xticks(range(len(mapped_counts)))
            ax3.set_xticklabels(mapped_counts.index, rotation=45, ha='right')
            
            # 添加数值标签
            for bar in bars3:
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height)}', ha='center', va='bottom')
        else:
            axes[1, 0].text(0.5, 0.5, '无映射数据', ha='center', va='center', 
                           transform=axes[1, 0].transAxes, fontsize=12)
            axes[1, 0].set_title('映射后标签分布', fontsize=14, fontweight='bold')
        
        # 4. 统计效力对比
        ax4 = axes[1, 1]
        
        # 计算统计效力指标
        multi_sufficient = sum(1 for count in multi_label_counts.values if count >= 5)
        multi_marginal = sum(1 for count in multi_label_counts.values if 3 <= count < 5)
        multi_insufficient = sum(1 for count in multi_label_counts.values if count < 3)
        
        simple_sufficient = sum(1 for count in simple_label_counts.values if count >= 5)
        simple_marginal = sum(1 for count in simple_label_counts.values if 3 <= count < 5)
        simple_insufficient = sum(1 for count in simple_label_counts.values if count < 3)
        
        categories = ['充足(≥5)', '边缘(3-4)', '不足(<3)']
        multi_data = [multi_sufficient, multi_marginal, multi_insufficient]
        simple_data = [simple_sufficient, simple_marginal, simple_insufficient]
        
        x = np.arange(len(categories))
        width = 0.35
        
        ax4.bar(x - width/2, multi_data, width, label='多标签版', alpha=0.7, color='skyblue')
        ax4.bar(x + width/2, simple_data, width, label='简化版', alpha=0.7, color='lightgreen')
        
        ax4.set_title('统计效力对比', fontsize=14, fontweight='bold')
        ax4.set_xlabel('数据量分类', fontsize=12, fontweight='bold')
        ax4.set_ylabel('标签数量', fontsize=12, fontweight='bold')
        ax4.set_xticks(x)
        ax4.set_xticklabels(categories)
        ax4.tick_params(axis='both', labelsize=10, labelweight='bold')
        ax4.legend(fontsize=10)
        
        plt.tight_layout()
        plt.savefig('label_comparison_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()

def main():
    """主分析流程"""
    print("=== 标签版本比较分析 ===")
    
    # 初始化分析器
    analyzer = LabelComparisonAnalyzer(
        'data/EMtrace01-多标签版.csv',
        'data/EMtrace02-3标签版.csv'
    )
    
    # 加载数据
    analyzer.load_data()
    
    # 分析标签分布
    multi_counts, simple_counts = analyzer.analyze_label_distribution()
    
    # 评估统计效力
    power_assessment = analyzer.assess_statistical_power(multi_counts)
    
    # 建议标签映射
    mapping_strategies = analyzer.suggest_label_mapping()
    
    # 推荐最佳方法
    recommendation = analyzer.recommend_best_approach(power_assessment, mapping_strategies)
    
    # 与现有简化版本比较
    if recommendation['mapped_df'] is not None:
        analyzer.compare_with_existing_simple_version(recommendation['mapped_df'])
    
    # 可视化比较
    mapped_counts = recommendation['mapped_counts'] if recommendation['mapped_counts'] is not None else None
    analyzer.visualize_label_comparison(multi_counts, simple_counts, mapped_counts)
    
    print("\n=== 分析完成 ===")
    print("已生成标签比较分析图表: label_comparison_analysis.png")
    
    return recommendation

if __name__ == "__main__":
    results = main()