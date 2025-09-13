#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Principal Neuron 使用示例

本脚本展示了如何使用新集成的效应量计算功能，
从原始神经元数据到最终的可视化分析的完整流程。

作者：AI Assistant
日期：2024年
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# 添加src目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.effect_size_calculator import EffectSizeCalculator, load_and_calculate_effect_sizes
from src.main_emtrace01_analysis import create_effect_sizes_workflow, get_key_neurons


def generate_sample_data(n_samples: int = 1000, n_neurons: int = 50, 
                        output_path: str = "data/sample_raw_data.xlsx") -> str:
    """
    生成示例神经元数据用于演示
    
    参数
    ----------
    n_samples : int
        样本数量
    n_neurons : int
        神经元数量
    output_path : str
        输出文件路径
        
    返回
    ----------
    str
        生成的数据文件路径
    """
    print(f"生成示例数据: {n_samples} 个样本, {n_neurons} 个神经元")
    
    # 设置随机种子确保可重现性
    np.random.seed(42)
    
    # 生成行为标签
    behaviors = ['Close', 'Middle', 'Open']
    behavior_data = np.random.choice(behaviors, n_samples)
    
    # 生成神经元活动数据
    neuron_data = np.random.randn(n_samples, n_neurons) * 0.5
    
    # 为不同行为添加特定的神经活动模式
    behavior_indices = {behavior: np.where(behavior_data == behavior)[0] for behavior in behaviors}
    
    # Close行为：神经元1-15有较高活动
    close_indices = behavior_indices['Close']
    neuron_data[close_indices, 0:15] += np.random.normal(1.5, 0.3, (len(close_indices), 15))
    
    # Middle行为：神经元16-30有较高活动
    middle_indices = behavior_indices['Middle']
    neuron_data[middle_indices, 15:30] += np.random.normal(1.2, 0.3, (len(middle_indices), 15))
    
    # Open行为：神经元31-45有较高活动
    open_indices = behavior_indices['Open']
    neuron_data[open_indices, 30:45] += np.random.normal(1.8, 0.3, (len(open_indices), 15))
    
    # 创建DataFrame
    columns = [f'Neuron_{i+1}' for i in range(n_neurons)] + ['Behavior']
    data = np.column_stack([neuron_data, behavior_data])
    df = pd.DataFrame(data, columns=columns)
    
    # 确保神经元数据为数值型
    for col in columns[:-1]:
        df[col] = pd.to_numeric(df[col])
    
    # 保存数据
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_excel(output_path, index=False)
    
    print(f"示例数据已保存到: {output_path}")
    print(f"数据形状: {df.shape}")
    print(f"行为分布:")
    print(df['Behavior'].value_counts())
    
    return output_path


def example_basic_usage():
    """
    基本使用示例：从原始数据到效应量计算
    """
    print("\n" + "="*60)
    print("示例1: 基本使用流程")
    print("="*60)
    
    # 1. 生成示例数据
    sample_data_path = generate_sample_data()
    
    # 2. 使用便捷函数计算效应量
    print("\n使用便捷函数计算效应量...")
    results = load_and_calculate_effect_sizes(
        neuron_data_path=sample_data_path,
        behavior_col="Behavior",
        output_dir="output/example1"
    )
    
    # 3. 查看结果
    print("\n效应量计算结果:")
    print(f"行为类别: {results['behavior_labels']}")
    print(f"效应量数据保存在: {results['output_files']['effect_sizes_csv']}")
    
    # 4. 显示top神经元
    print("\n各行为的top-5关键神经元:")
    for behavior, info in results['top_neurons'].items():
        print(f"\n{behavior}:")
        for i, (neuron_id, effect_size) in enumerate(zip(info['neuron_ids'][:5], info['effect_sizes'][:5])):
            print(f"  {i+1}. 神经元 {neuron_id}: 效应量 = {effect_size:.4f}")


def example_advanced_usage():
    """
    高级使用示例：自定义效应量计算和分析
    """
    print("\n" + "="*60)
    print("示例2: 高级自定义分析")
    print("="*60)
    
    # 1. 创建效应量计算器
    calculator = EffectSizeCalculator()
    
    # 2. 加载数据
    data_path = "data/sample_raw_data.xlsx"
    if not os.path.exists(data_path):
        data_path = generate_sample_data()
    
    data = pd.read_excel(data_path)
    neuron_data = data.iloc[:, :-1].values
    behavior_data = data.iloc[:, -1].values
    
    print(f"加载数据: {neuron_data.shape[0]} 个样本, {neuron_data.shape[1]} 个神经元")
    
    # 3. 计算效应量
    effect_sizes, X_scaled, y_encoded = calculator.calculate_effect_sizes_from_raw_data(
        neuron_data, behavior_data
    )
    
    # 4. 多个阈值的关键神经元分析
    thresholds = [0.3, 0.4, 0.5, 0.6]
    print(f"\n不同阈值下的关键神经元数量:")
    for threshold in thresholds:
        key_neurons = calculator.identify_key_neurons(effect_sizes, threshold=threshold)
        total_neurons = sum(len(neurons) for neurons in key_neurons.values())
        print(f"阈值 {threshold}: 总计 {total_neurons} 个关键神经元")
        for behavior, neurons in key_neurons.items():
            print(f"  {behavior}: {len(neurons)} 个")
    
    # 5. 效应量分布可视化
    print("\n生成效应量分布图...")
    plt.figure(figsize=(15, 5))
    
    for i, (behavior, effect_array) in enumerate(effect_sizes.items()):
        plt.subplot(1, 3, i+1)
        plt.hist(effect_array, bins=20, alpha=0.7, label=behavior)
        plt.xlabel('Effect Size')
        plt.ylabel('Frequency')
        plt.title(f'{behavior} Behavior Effect Size Distribution')
        plt.axvline(x=0.4, color='red', linestyle='--', label='Threshold=0.4')
        plt.legend()
    
    plt.tight_layout()
    output_dir = "output/example2"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/effect_size_distributions.png", dpi=150, bbox_inches='tight')
    print(f"效应量分布图已保存到: {output_dir}/effect_size_distributions.png")
    plt.close()


def example_workflow_integration():
    """
    工作流集成示例：与现有分析流程的整合
    """
    print("\n" + "="*60)
    print("示例3: 完整工作流集成")
    print("="*60)
    
    # 1. 使用工作流函数（模拟主分析脚本的使用方式）
    raw_data_file = "data/sample_raw_data.xlsx"
    precomputed_file = "output/example1/effect_sizes.csv"
    
    # 如果原始数据不存在，生成它
    if not os.path.exists(raw_data_file):
        generate_sample_data(output_path=raw_data_file)
    
    # 使用工作流函数
    print("\n使用效应量计算工作流...")
    df_effect_sizes_long = create_effect_sizes_workflow(
        raw_data_file=raw_data_file,
        precomputed_file=precomputed_file,
        recalculate=True  # 强制重新计算以展示功能
    )
    
    print(f"工作流完成，得到效应量数据: {df_effect_sizes_long.shape}")
    
    # 2. 识别关键神经元
    key_neurons = get_key_neurons(df_effect_sizes_long, threshold=0.4)
    
    # 3. 生成分析报告
    print("\n生成分析报告...")
    output_dir = "output/example3"
    os.makedirs(output_dir, exist_ok=True)
    
    report_content = f"""
# 神经元效应量分析报告

生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 数据概况
- 总样本数: {len(df_effect_sizes_long) // len(df_effect_sizes_long['Behavior'].unique())}
- 神经元数量: {len(df_effect_sizes_long['NeuronID'].unique())}
- 行为类别: {list(df_effect_sizes_long['Behavior'].unique())}

## 关键神经元统计 (阈值 = 0.4)
"""
    
    for behavior, neurons in key_neurons.items():
        report_content += f"\n### {behavior} 行为\n"
        report_content += f"- 关键神经元数量: {len(neurons)}\n"
        report_content += f"- 神经元ID: {neurons}\n"
        
        # 计算该行为的效应量统计
        behavior_effects = df_effect_sizes_long[df_effect_sizes_long['Behavior'] == behavior]['EffectSize']
        report_content += f"- 平均效应量: {behavior_effects.mean():.4f}\n"
        report_content += f"- 最大效应量: {behavior_effects.max():.4f}\n"
        report_content += f"- 超过阈值的神经元比例: {(behavior_effects >= 0.4).mean()*100:.1f}%\n"
    
    # 保存报告
    report_path = f"{output_dir}/analysis_report.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    print(f"分析报告已保存到: {report_path}")
    
    # 4. 保存关键结果
    results_summary = {
        'timestamp': datetime.now().isoformat(),
        'total_neurons': len(df_effect_sizes_long['NeuronID'].unique()),
        'behaviors': list(df_effect_sizes_long['Behavior'].unique()),
        'threshold': 0.4,
        'key_neurons_count': {behavior: len(neurons) for behavior, neurons in key_neurons.items()},
        'key_neurons': key_neurons
    }
    
    import json
    with open(f"{output_dir}/results_summary.json", 'w', encoding='utf-8') as f:
        json.dump(results_summary, f, indent=2, ensure_ascii=False)
    
    print(f"结果摘要已保存到: {output_dir}/results_summary.json")


def main():
    """
    主函数：运行所有示例
    """
    print("Principal Neuron 使用示例")
    print("="*60)
    print("本脚本将演示如何使用新集成的效应量计算功能")
    
    try:
        # 运行基本使用示例
        example_basic_usage()
        
        # 运行高级使用示例  
        example_advanced_usage()
        
        # 运行工作流集成示例
        example_workflow_integration()
        
        print("\n" + "="*60)
        print("所有示例运行完成！")
        print("="*60)
        print("\n生成的文件:")
        print("- data/sample_raw_data.xlsx: 示例原始数据")
        print("- output/example1/: 基本使用示例输出")
        print("- output/example2/: 高级使用示例输出")
        print("- output/example3/: 工作流集成示例输出")
        print("\n请查看输出目录中的文件和图表。")
        
    except Exception as e:
        print(f"\n运行示例时发生错误: {str(e)}")
        print("请检查错误信息并确保所有依赖包已正确安装。")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 