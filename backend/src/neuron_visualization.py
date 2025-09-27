"""
神经元可视化模块
集成principal_neuron-1模块的功能，提供神经元分析和可视化服务
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any, Optional
import os
from pathlib import Path
from matplotlib.figure import Figure
import base64
from io import BytesIO


class NeuronVisualizer:
    """
    神经元可视化器类：用于生成神经元分析的可视化图表
    """
    
    def __init__(self, effect_size_threshold: float = 0.5):
        """
        初始化神经元可视化器
        
        参数
        ----------
        effect_size_threshold : float
            效应量阈值，用于筛选关键神经元
        """
        self.effect_size_threshold = effect_size_threshold
        self.behavior_colors = {
            'Close': 'red',
            'Middle': 'green', 
            'Open': 'blue',
            'Explore': 'orange',
            'Water': 'purple',
            'Groom': 'brown',
            'Sleep': 'gray',
            'Active': 'pink'
        }
        
    def get_key_neurons(self, effect_sizes: pd.DataFrame, behavior: str) -> List[str]:
        """
        获取指定行为的关键神经元
        
        参数
        ----------
        effect_sizes : pd.DataFrame
            效应量数据
        behavior : str
            行为名称
            
        返回
        ----------
        List[str]
            关键神经元列表
        """
        if behavior not in effect_sizes.columns:
            return []
        
        # 找出效应量绝对值超过阈值的神经元
        key_neurons = effect_sizes[
            effect_sizes[behavior].abs() > self.effect_size_threshold
        ].index.tolist()
        
        return key_neurons
    
    def get_shared_neurons(self, effect_sizes: pd.DataFrame, behavior1: str, behavior2: str) -> List[str]:
        """
        获取两个行为共享的关键神经元
        
        参数
        ----------
        effect_sizes : pd.DataFrame
            效应量数据
        behavior1 : str
            第一个行为名称
        behavior2 : str
            第二个行为名称
            
        返回
        ----------
        List[str]
            共享神经元列表
        """
        key_neurons_1 = set(self.get_key_neurons(effect_sizes, behavior1))
        key_neurons_2 = set(self.get_key_neurons(effect_sizes, behavior2))
        
        return list(key_neurons_1.intersection(key_neurons_2))
    
    def get_unique_neurons(self, effect_sizes: pd.DataFrame, behavior: str) -> List[str]:
        """
        获取指定行为的特有神经元（不与其他行为共享）
        
        参数
        ----------
        effect_sizes : pd.DataFrame
            效应量数据
        behavior : str
            行为名称
            
        返回
        ----------
        List[str]
            特有神经元列表
        """
        key_neurons = set(self.get_key_neurons(effect_sizes, behavior))
        all_other_neurons = set()
        
        # 收集其他行为的关键神经元
        for other_behavior in effect_sizes.columns:
            if other_behavior != behavior:
                all_other_neurons.update(self.get_key_neurons(effect_sizes, other_behavior))
        
        # 找出特有神经元
        unique_neurons = key_neurons - all_other_neurons
        
        return list(unique_neurons)
    
    def create_single_behavior_plot(self, effect_sizes: pd.DataFrame, positions: pd.DataFrame, 
                                  behavior: str, figsize: Tuple[int, int] = (12, 8)) -> Figure:
        """
        创建单一行为的关键神经元分布图
        
        参数
        ----------
        effect_sizes : pd.DataFrame
            效应量数据
        positions : pd.DataFrame
            位置数据
        behavior : str
            行为名称
        figsize : Tuple[int, int]
            图形大小
            
        返回
        ----------
        Figure
            生成的图形对象
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # 获取关键神经元
        key_neurons = self.get_key_neurons(effect_sizes, behavior)
        
        if not key_neurons:
            ax.text(0.5, 0.5, f'没有找到行为 "{behavior}" 的关键神经元', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=16)
            ax.set_title(f'{behavior} 关键神经元分布', fontsize=16)
            return fig
        
        # 绘制所有神经元作为背景
        ax.scatter(positions['relative_x'], positions['relative_y'], 
                  c='lightgray', s=50, alpha=0.3, label='所有神经元')
        
        # 绘制关键神经元
        key_positions = positions[positions['number'].isin([int(n.split('_')[1]) for n in key_neurons])]
        if not key_positions.empty:
            ax.scatter(key_positions['relative_x'], key_positions['relative_y'],
                      c=self.behavior_colors.get(behavior, 'red'), s=100, alpha=0.8,
                      label=f'{behavior} 关键神经元')
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel('相对X坐标', fontsize=12)
        ax.set_ylabel('相对Y坐标', fontsize=12)
        ax.set_title(f'{behavior} 关键神经元分布 (阈值: {self.effect_size_threshold})', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return fig
    
    def create_shared_neurons_plot(self, effect_sizes: pd.DataFrame, positions: pd.DataFrame,
                                 behavior1: str, behavior2: str, figsize: Tuple[int, int] = (12, 8)) -> Figure:
        """
        创建两个行为共享神经元的分布图
        
        参数
        ----------
        effect_sizes : pd.DataFrame
            效应量数据
        positions : pd.DataFrame
            位置数据
        behavior1 : str
            第一个行为名称
        behavior2 : str
            第二个行为名称
        figsize : Tuple[int, int]
            图形大小
            
        返回
        ----------
        Figure
            生成的图形对象
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # 获取共享神经元
        shared_neurons = self.get_shared_neurons(effect_sizes, behavior1, behavior2)
        
        if not shared_neurons:
            ax.text(0.5, 0.5, f'没有找到行为 "{behavior1}" 和 "{behavior2}" 的共享神经元', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=16)
            ax.set_title(f'{behavior1} 和 {behavior2} 共享神经元分布', fontsize=16)
            return fig
        
        # 绘制所有神经元作为背景
        ax.scatter(positions['relative_x'], positions['relative_y'], 
                  c='lightgray', s=50, alpha=0.3, label='所有神经元')
        
        # 绘制共享神经元
        shared_positions = positions[positions['number'].isin([int(n.split('_')[1]) for n in shared_neurons])]
        if not shared_positions.empty:
            ax.scatter(shared_positions['relative_x'], shared_positions['relative_y'],
                      c='yellow', s=100, alpha=0.8, edgecolors='black', linewidth=1,
                      label=f'{behavior1}-{behavior2} 共享神经元')
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel('相对X坐标', fontsize=12)
        ax.set_ylabel('相对Y坐标', fontsize=12)
        ax.set_title(f'{behavior1} 和 {behavior2} 共享神经元分布', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return fig
    
    def create_unique_neurons_plot(self, effect_sizes: pd.DataFrame, positions: pd.DataFrame,
                                 behavior: str, figsize: Tuple[int, int] = (12, 8)) -> Figure:
        """
        创建指定行为特有神经元的分布图
        
        参数
        ----------
        effect_sizes : pd.DataFrame
            效应量数据
        positions : pd.DataFrame
            位置数据
        behavior : str
            行为名称
        figsize : Tuple[int, int]
            图形大小
            
        返回
        ----------
        Figure
            生成的图形对象
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # 获取特有神经元
        unique_neurons = self.get_unique_neurons(effect_sizes, behavior)
        
        if not unique_neurons:
            ax.text(0.5, 0.5, f'没有找到行为 "{behavior}" 的特有神经元', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=16)
            ax.set_title(f'{behavior} 特有神经元分布', fontsize=16)
            return fig
        
        # 绘制所有神经元作为背景
        ax.scatter(positions['relative_x'], positions['relative_y'], 
                  c='lightgray', s=50, alpha=0.3, label='所有神经元')
        
        # 绘制特有神经元
        unique_positions = positions[positions['number'].isin([int(n.split('_')[1]) for n in unique_neurons])]
        if not unique_positions.empty:
            ax.scatter(unique_positions['relative_x'], unique_positions['relative_y'],
                      c=self.behavior_colors.get(behavior, 'red'), s=100, alpha=0.8,
                      marker='^', label=f'{behavior} 特有神经元')
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel('相对X坐标', fontsize=12)
        ax.set_ylabel('相对Y坐标', fontsize=12)
        ax.set_title(f'{behavior} 特有神经元分布', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return fig
    
    def create_comprehensive_analysis(self, effect_sizes: pd.DataFrame, positions: pd.DataFrame) -> Dict[str, Any]:
        """
        创建综合分析结果
        
        参数
        ----------
        effect_sizes : pd.DataFrame
            效应量数据
        positions : pd.DataFrame
            位置数据
            
        返回
        ----------
        Dict[str, Any]
            综合分析结果
        """
        behaviors = effect_sizes.columns.tolist()
        
        analysis_result = {
            'total_neurons': len(effect_sizes),
            'behaviors': behaviors,
            'threshold': self.effect_size_threshold,
            'key_neurons': {},
            'shared_neurons': {},
            'unique_neurons': {},
            'statistics': {}
        }
        
        # 分析每种行为的关键神经元
        for behavior in behaviors:
            key_neurons = self.get_key_neurons(effect_sizes, behavior)
            unique_neurons = self.get_unique_neurons(effect_sizes, behavior)
            
            analysis_result['key_neurons'][behavior] = {
                'neurons': key_neurons,
                'count': len(key_neurons),
                'effect_sizes': effect_sizes.loc[key_neurons, behavior].to_dict() if key_neurons else {}
            }
            
            analysis_result['unique_neurons'][behavior] = {
                'neurons': unique_neurons,
                'count': len(unique_neurons)
            }
        
        # 分析行为间的共享神经元
        for i, behavior1 in enumerate(behaviors):
            for behavior2 in behaviors[i+1:]:
                shared_neurons = self.get_shared_neurons(effect_sizes, behavior1, behavior2)
                pair_key = f"{behavior1}-{behavior2}"
                
                analysis_result['shared_neurons'][pair_key] = {
                    'neurons': shared_neurons,
                    'count': len(shared_neurons)
                }
        
        # 统计信息
        total_key_neurons = set()
        for behavior_data in analysis_result['key_neurons'].values():
            total_key_neurons.update(behavior_data['neurons'])
        
        analysis_result['statistics'] = {
            'total_key_neurons': len(total_key_neurons),
            'key_neuron_percentage': len(total_key_neurons) / len(effect_sizes) * 100,
            'average_key_neurons_per_behavior': np.mean([len(data['neurons']) for data in analysis_result['key_neurons'].values()]),
            'total_shared_pairs': len(analysis_result['shared_neurons']),
            'average_shared_neurons_per_pair': np.mean([data['count'] for data in analysis_result['shared_neurons'].values()]) if analysis_result['shared_neurons'] else 0
        }
        
        return analysis_result


def analyze_neuron_visualization(effect_sizes: pd.DataFrame, positions: pd.DataFrame, 
                               threshold: float = 0.5) -> Dict[str, Any]:
    """
    执行神经元可视化分析
    
    参数
    ----------
    effect_sizes : pd.DataFrame
        效应量数据
    positions : pd.DataFrame
        位置数据
    threshold : float
        效应量阈值
        
    返回
    ----------
    Dict[str, Any]
        分析结果
    """
    # 创建可视化器
    visualizer = NeuronVisualizer(threshold)
    
    # 执行综合分析
    analysis_result = visualizer.create_comprehensive_analysis(effect_sizes, positions)
    
    # 生成可视化图表
    figures = {}
    behaviors = effect_sizes.columns.tolist()
    
    # 为每种行为生成关键神经元图
    for behavior in behaviors:
        fig = visualizer.create_single_behavior_plot(effect_sizes, positions, behavior)
        figures[f'{behavior}_key_neurons'] = fig
    
    # 为每种行为生成特有神经元图
    for behavior in behaviors:
        fig = visualizer.create_unique_neurons_plot(effect_sizes, positions, behavior)
        figures[f'{behavior}_unique_neurons'] = fig
    
    # 为行为对生成共享神经元图
    for i, behavior1 in enumerate(behaviors):
        for behavior2 in behaviors[i+1:]:
            fig = visualizer.create_shared_neurons_plot(effect_sizes, positions, behavior1, behavior2)
            figures[f'{behavior1}_{behavior2}_shared'] = fig
    
    return {
        'analysis_result': analysis_result,
        'figures': figures
    }

