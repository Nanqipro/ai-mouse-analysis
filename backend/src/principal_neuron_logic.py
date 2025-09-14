import os
import sys
import numpy as np
import pandas as pd
import tempfile
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import json
import base64
from io import BytesIO
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端

# 添加algorithm/principal_neuron路径到sys.path
project_root = Path(__file__).parent.parent.parent
principal_neuron_path = project_root / "algorithm" / "principal_neuron"
sys.path.insert(0, str(principal_neuron_path))
sys.path.insert(0, str(principal_neuron_path / "src"))

try:
    from effect_size_calculator import EffectSizeCalculator, load_and_calculate_effect_sizes
    from main_emtrace01_analysis import create_effect_sizes_workflow, get_key_neurons
    from data_loader import load_effect_sizes, load_neuron_positions
    from config import (
        EFFECT_SIZE_THRESHOLD, BEHAVIOR_COLORS, MIXED_BEHAVIOR_COLORS,
        SHOW_BACKGROUND_NEURONS, BACKGROUND_NEURON_COLOR, 
        BACKGROUND_NEURON_SIZE, BACKGROUND_NEURON_ALPHA,
        STANDARD_KEY_NEURON_ALPHA
    )
    from plotting_utils import (
        plot_single_behavior_activity_map, 
        plot_shared_neurons_map,
        plot_unique_neurons_map
    )
    from neuron_animation_generator import NeuronActivityAnimator
except ImportError as e:
    print(f"Warning: Could not import principal_neuron modules: {e}")
    # 定义默认值以防导入失败
    EFFECT_SIZE_THRESHOLD = 0.5
    BEHAVIOR_COLORS = {'Close-Arm': 'red', 'Middle-Zone': 'green', 'Open-Arm': 'blue'}

class PrincipalNeuronAnalyzer:
    """
    主神经元分析器 - 集成principal_neuron模块的核心功能
    """
    
    def __init__(self):
        self.temp_dir = Path(tempfile.gettempdir()) / "principal_neuron_analysis"
        self.temp_dir.mkdir(exist_ok=True)
        
    def analyze_effect_sizes(self, data_file_path: str, positions_file_path: str = None) -> Dict[str, Any]:
        """
        分析神经元效应量
        
        参数:
            data_file_path: 神经元数据文件路径
            positions_file_path: 神经元位置文件路径（可选）
            
        返回:
            包含效应量分析结果的字典
        """
        try:
            # 创建效应量计算器
            calculator = EffectSizeCalculator()
            
            # 加载数据并计算效应量
            results = load_and_calculate_effect_sizes(
                neuron_data_path=data_file_path,
                behavior_col="behavior",
                output_dir=str(self.temp_dir)
            )
            
            # 获取关键神经元
            key_neurons = calculator.identify_key_neurons(
                results['effect_sizes'], 
                threshold=EFFECT_SIZE_THRESHOLD
            )
            
            # 获取前10个关键神经元
            top_neurons = calculator.get_top_neurons_per_behavior(
                results['effect_sizes'], 
                top_n=10
            )
            
            # 转换numpy数组为列表以便JSON序列化
            serializable_effect_sizes = {}
            for behavior, effect_array in results['effect_sizes'].items():
                if isinstance(effect_array, np.ndarray):
                    serializable_effect_sizes[behavior] = effect_array.tolist()
                else:
                    serializable_effect_sizes[behavior] = effect_array
            
            # 转换key_neurons为可序列化格式
            serializable_key_neurons = {}
            for behavior, neurons in key_neurons.items():
                if isinstance(neurons, dict):
                    serializable_key_neurons[behavior] = {
                        'neuron_ids': neurons.get('neuron_ids', []),
                        'effect_sizes': [float(x) if isinstance(x, (np.floating, np.integer)) else x 
                                       for x in neurons.get('effect_sizes', [])]
                    }
                else:
                    serializable_key_neurons[behavior] = neurons
            
            # 转换top_neurons为可序列化格式
            serializable_top_neurons = {}
            for behavior, neurons in top_neurons.items():
                if isinstance(neurons, dict):
                    serializable_top_neurons[behavior] = {
                        'neuron_ids': neurons.get('neuron_ids', []),
                        'effect_sizes': [float(x) if isinstance(x, (np.floating, np.integer)) else x 
                                       for x in neurons.get('effect_sizes', [])]
                    }
                else:
                    serializable_top_neurons[behavior] = neurons
            
            return {
                'success': True,
                'total_neurons': len(serializable_effect_sizes.get(list(serializable_effect_sizes.keys())[0], [])) if serializable_effect_sizes else 0,
                'significant_neurons': sum(len(neurons.get('neuron_ids', [])) for neurons in serializable_key_neurons.values()),
                'behavior_labels': list(results['behavior_labels']) if hasattr(results['behavior_labels'], '__iter__') else results['behavior_labels'],
                'key_neurons': serializable_key_neurons,
                'top_neurons': serializable_top_neurons,
                'threshold': float(EFFECT_SIZE_THRESHOLD),
                'mean_effect_size': float(np.mean([np.mean(arr) for arr in results['effect_sizes'].values()])) if results['effect_sizes'] else 0.0,
                'max_effect_size': float(np.max([np.max(arr) for arr in results['effect_sizes'].values()])) if results['effect_sizes'] else 0.0,
                'csv_path': results.get('output_files', {}).get('csv_path', '')
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def generate_neuron_activity_maps(self, data_file_path: str, positions_file_path: str, 
                                    effect_sizes: Dict[str, np.ndarray] = None) -> Dict[str, Any]:
        """
        生成神经元活动图
        
        参数:
            data_file_path: 神经元数据文件路径
            positions_file_path: 神经元位置文件路径
            effect_sizes: 预计算的效应量数据（可选）
            
        返回:
            包含生成的图表信息的字典
        """
        try:
            # 如果没有提供效应量数据，先计算效应量
            if effect_sizes is None:
                effect_result = self.analyze_effect_sizes(data_file_path, positions_file_path)
                if not effect_result['success']:
                    return effect_result
                effect_sizes = effect_result['effect_sizes']
            
            # 加载神经元位置数据
            df_neuron_positions = load_neuron_positions(positions_file_path)
            if df_neuron_positions is None:
                return {
                    'success': False,
                    'error': 'Failed to load neuron positions'
                }
            
            # 获取关键神经元
            key_neurons_by_behavior = get_key_neurons(effect_sizes, EFFECT_SIZE_THRESHOLD)
            
            generated_plots = []
            
            # 为每个行为生成单独的活动图
            for behavior_name, neuron_ids in key_neurons_by_behavior.items():
                if not neuron_ids:
                    continue
                    
                # 获取关键神经元的位置数据
                key_neurons_df = df_neuron_positions[
                    df_neuron_positions['NeuronID'].isin(neuron_ids)
                ]
                
                # 生成图表
                plot_base64 = self._create_single_behavior_plot(
                    key_neurons_df, behavior_name, df_neuron_positions
                )
                
                if plot_base64:
                    generated_plots.append({
                        'behavior': behavior_name,
                        'type': 'single_behavior',
                        'neuron_count': len(neuron_ids),
                        'plot_data': plot_base64
                    })
            
            return {
                'success': True,
                'plots': generated_plots,
                'key_neurons': key_neurons_by_behavior,
                'total_behaviors': len(key_neurons_by_behavior)
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def _create_single_behavior_plot(self, key_neurons_df: pd.DataFrame, 
                                   behavior_name: str, all_positions_df: pd.DataFrame) -> str:
        """
        创建单个行为的神经元活动图并返回base64编码
        """
        try:
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # 使用plotting_utils中的函数
            plot_single_behavior_activity_map(
                key_neurons_df=key_neurons_df,
                behavior_name=behavior_name,
                behavior_color=BEHAVIOR_COLORS.get(behavior_name, 'gray'),
                title=f'{behavior_name} Key Neurons',
                output_path=None,  # 不保存文件
                all_neuron_positions_df=all_positions_df,
                show_background_neurons=SHOW_BACKGROUND_NEURONS,
                background_neuron_color=BACKGROUND_NEURON_COLOR,
                background_neuron_size=BACKGROUND_NEURON_SIZE,
                background_neuron_alpha=BACKGROUND_NEURON_ALPHA,
                key_neuron_size=300,
                key_neuron_alpha=STANDARD_KEY_NEURON_ALPHA,
                show_title=True,
                ax=ax
            )
            
            # 转换为base64
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
            buffer.seek(0)
            plot_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            plt.close(fig)
            
            return plot_base64
            
        except Exception as e:
            print(f"Error creating plot for {behavior_name}: {e}")
            return None
    
    def get_neuron_animation_data(self, data_file_path: str, positions_file_path: str) -> Dict[str, Any]:
        """
        获取神经元动画数据
        
        参数:
            data_file_path: 神经元数据文件路径
            positions_file_path: 神经元位置文件路径
            
        返回:
            包含动画数据的字典
        """
        try:
            # 创建动画生成器
            animator = NeuronActivityAnimator(
                data_path=data_file_path,
                positions_path=positions_file_path
            )
            
            # 加载数据
            animator.load_data()
            animator._initialize_neuron_positions()
            
            # 获取Web端数据
            animation_data = animator.get_data_for_web()
            
            return {
                'success': True,
                'animation_data': animation_data
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def analyze_shared_neurons(self, data_file_path: str, positions_file_path: str) -> Dict[str, Any]:
        """
        分析行为间的共享神经元
        
        参数:
            data_file_path: 神经元数据文件路径
            positions_file_path: 神经元位置文件路径
            
        返回:
            包含共享神经元分析结果的字典
        """
        try:
            # 先获取效应量分析结果
            effect_result = self.analyze_effect_sizes(data_file_path, positions_file_path)
            if not effect_result['success']:
                return effect_result
            
            key_neurons = effect_result['key_neurons']
            behaviors = list(key_neurons.keys())
            
            shared_analysis = []
            
            # 分析每对行为的共享神经元
            from itertools import combinations
            for b1, b2 in combinations(behaviors, 2):
                ids1 = set(key_neurons.get(b1, []))
                ids2 = set(key_neurons.get(b2, []))
                shared_ids = list(ids1.intersection(ids2))
                
                if shared_ids:
                    shared_analysis.append({
                        'behavior1': b1,
                        'behavior2': b2,
                        'shared_neurons': shared_ids,
                        'shared_count': len(shared_ids),
                        'behavior1_total': len(ids1),
                        'behavior2_total': len(ids2),
                        'overlap_ratio': len(shared_ids) / min(len(ids1), len(ids2)) if min(len(ids1), len(ids2)) > 0 else 0
                    })
            
            return {
                'success': True,
                'shared_analysis': shared_analysis,
                'key_neurons': key_neurons,
                'behaviors': behaviors
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_analysis_summary(self, data_file_path: str, positions_file_path: str) -> Dict[str, Any]:
        """
        获取完整的分析摘要
        
        参数:
            data_file_path: 神经元数据文件路径
            positions_file_path: 神经元位置文件路径
            
        返回:
            包含完整分析摘要的字典
        """
        try:
            # 效应量分析
            effect_result = self.analyze_effect_sizes(data_file_path, positions_file_path)
            if not effect_result['success']:
                return effect_result
            
            # 共享神经元分析
            shared_result = self.analyze_shared_neurons(data_file_path, positions_file_path)
            
            # 生成活动图
            plots_result = self.generate_neuron_activity_maps(
                data_file_path, positions_file_path, effect_result['effect_sizes']
            )
            
            return {
                'success': True,
                'effect_analysis': {
                    'behavior_labels': effect_result['behavior_labels'],
                    'key_neurons': effect_result['key_neurons'],
                    'top_neurons': effect_result['top_neurons'],
                    'threshold': effect_result['threshold']
                },
                'shared_analysis': shared_result.get('shared_analysis', []) if shared_result['success'] else [],
                'activity_plots': plots_result.get('plots', []) if plots_result['success'] else [],
                'summary_stats': self._calculate_summary_stats(effect_result)
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def _calculate_summary_stats(self, effect_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        计算分析摘要统计信息
        """
        try:
            key_neurons = effect_result['key_neurons']
            
            total_key_neurons = sum(len(neurons) for neurons in key_neurons.values())
            unique_key_neurons = len(set().union(*key_neurons.values()))
            
            behavior_stats = {}
            for behavior, neurons in key_neurons.items():
                behavior_stats[behavior] = {
                    'key_neuron_count': len(neurons),
                    'neuron_ids': neurons
                }
            
            return {
                'total_behaviors': len(key_neurons),
                'total_key_neurons': total_key_neurons,
                'unique_key_neurons': unique_key_neurons,
                'behavior_stats': behavior_stats,
                'threshold_used': effect_result['threshold']
            }
            
        except Exception as e:
            return {
                'error': f'Failed to calculate summary stats: {str(e)}'
            }

# 创建全局分析器实例
principal_analyzer = PrincipalNeuronAnalyzer()