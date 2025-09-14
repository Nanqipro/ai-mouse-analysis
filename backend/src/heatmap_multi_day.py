"""
多天数据组合热力图分析模块

基于 heatmap_comb-sort.py 算法的功能实现，支持：
- 多天神经元数据的对齐和比较分析
- 基于神经元对应表的数据映射
- 支持峰值时间和钙波时间排序
- CD1行为事件标记
- 生成组合热力图和单独热力图
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from scipy.signal import find_peaks
from matplotlib.figure import Figure
from dataclasses import dataclass
import os
from pathlib import Path


@dataclass
class MultiDayHeatmapConfig:
    """
    多天热力图配置类
    
    Attributes
    ----------
    sort_method : str
        排序方式：'peak'（按峰值时间）或'calcium_wave'（按钙波时间）
    calcium_wave_threshold : float
        钙波检测阈值（标准差的倍数）
    min_prominence : float
        最小峰值突出度
    min_rise_rate : float
        最小上升速率
    max_fall_rate : float
        最大下降速率
    vmin : float
        热力图颜色范围最小值
    vmax : float
        热力图颜色范围最大值
    colormap : str
        热力图颜色映射
    figure_size_combo : Tuple[int, int]
        组合热力图的图形大小
    figure_size_single : Tuple[int, int]
        单独热力图的图形大小
    """
    
    sort_method: str = 'peak'
    calcium_wave_threshold: float = 1.5
    min_prominence: float = 1.0
    min_rise_rate: float = 0.1
    max_fall_rate: float = 0.05
    vmin: float = -2.0
    vmax: float = 2.0
    colormap: str = 'viridis'
    figure_size_combo: Tuple[int, int] = (45, 10)
    figure_size_single: Tuple[int, int] = (15, 10)


def detect_first_calcium_wave_multiday(neuron_data: pd.Series, config: MultiDayHeatmapConfig) -> float:
    """
    检测神经元第一次真实钙波发生的时间点（多天数据版本）
    
    Parameters
    ----------
    neuron_data : pd.Series
        神经元活动的时间序列数据（标准化后）
    config : MultiDayHeatmapConfig
        配置对象
        
    Returns
    -------
    float
        第一次真实钙波发生的时间点，如果没有检测到则返回数据最后一个时间点
    """
    # 使用find_peaks函数检测峰值
    peaks, properties = find_peaks(
        neuron_data, 
        height=config.calcium_wave_threshold, 
        prominence=config.min_prominence,
        distance=5  # 要求峰值之间至少间隔5个时间点
    )
    
    if len(peaks) == 0:
        return neuron_data.index[-1]
    
    # 对每个峰值进行验证，确认是否为真实钙波
    for peak_idx in peaks:
        if peak_idx <= 1 or peak_idx >= len(neuron_data) - 2:
            continue
            
        # 计算峰值前的上升速率
        pre_peak_idx = max(0, peak_idx - 5)
        rise_rate = (neuron_data.iloc[peak_idx] - neuron_data.iloc[pre_peak_idx]) / (peak_idx - pre_peak_idx)
        
        # 计算峰值后的下降速率
        post_peak_idx = min(len(neuron_data) - 1, peak_idx + 10)
        if post_peak_idx <= peak_idx:
            continue
        
        fall_rate = (neuron_data.iloc[peak_idx] - neuron_data.iloc[post_peak_idx]) / (post_peak_idx - peak_idx)
        
        # 确认是否符合钙波特征
        if rise_rate > config.min_rise_rate and 0 < fall_rate < config.max_fall_rate:
            return neuron_data.index[peak_idx]
    
    return neuron_data.index[-1]


def prepare_multiday_data(data_dict: Dict[str, pd.DataFrame], 
                         correspondence_table: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    """
    准备多天数据的对齐和排序
    
    Parameters
    ----------
    data_dict : Dict[str, pd.DataFrame]
        多天数据字典，键为天数标识（如'day0', 'day3'等），值为数据框
    correspondence_table : Optional[pd.DataFrame]
        神经元对应表（可选）
        
    Returns
    -------
    Dict[str, Any]
        包含对齐数据和映射信息的字典
    """
    available_days = list(data_dict.keys())
    result = {
        'available_days': available_days,
        'aligned_data': {},
        'neuron_labels': {},
        'correspondence_info': {}
    }
    
    # 检查数据格式并预处理
    for day, data in data_dict.items():
        # 确保 'stamp' 列设置为索引
        if 'stamp' in data.columns:
            data = data.set_index('stamp')
        
        # 分离行为数据
        if 'behavior' in data.columns:
            behavior_data = data['behavior']
            neural_data = data.drop(columns=['behavior'])
        else:
            neural_data = data.copy()
            behavior_data = None
        
        result['aligned_data'][day] = {
            'neural_data': neural_data,
            'behavior_data': behavior_data
        }
    
    # 如果有对应表，进行数据对齐
    if correspondence_table is not None:
        result['correspondence_info'] = align_data_with_correspondence_table(
            result['aligned_data'], correspondence_table
        )
    
    return result


def align_data_with_correspondence_table(aligned_data: Dict[str, Dict], 
                                       correspondence_table: pd.DataFrame) -> Dict[str, Any]:
    """
    基于对应表对齐多天数据
    
    Parameters
    ----------
    aligned_data : Dict[str, Dict]
        多天数据字典
    correspondence_table : pd.DataFrame
        神经元对应表
        
    Returns
    -------
    Dict[str, Any]
        对齐后的数据信息
    """
    # 确定要对齐的天数（基于对应表的列名）
    available_days = list(aligned_data.keys())
    
    # 查找三天都有数据的神经元
    aligned_neurons = {}
    valid_indices = []
    
    for idx, row in correspondence_table.iterrows():
        day_neurons = {}
        all_available = True
        
        for day in available_days:
            # 尝试找到对应表中与当前天相关的列
            possible_columns = [col for col in correspondence_table.columns if day.lower() in col.lower()]
            
            if possible_columns:
                col_name = possible_columns[0]
                neuron_id = row[col_name]
                
                if (pd.notna(neuron_id) and 
                    neuron_id in aligned_data[day]['neural_data'].columns):
                    day_neurons[day] = neuron_id
                else:
                    all_available = False
                    break
            else:
                all_available = False
                break
        
        if all_available and len(day_neurons) == len(available_days):
            aligned_neurons[idx] = day_neurons
            valid_indices.append(idx)
    
    return {
        'aligned_neurons': aligned_neurons,
        'valid_indices': valid_indices,
        'alignment_count': len(valid_indices)
    }


def calculate_sorting_for_multiday(reference_data: pd.DataFrame, 
                                 config: MultiDayHeatmapConfig) -> pd.Index:
    """
    计算多天数据的排序顺序（基于参考数据）
    
    Parameters
    ----------
    reference_data : pd.DataFrame
        参考数据（通常是某一天的数据）
    config : MultiDayHeatmapConfig
        配置对象
        
    Returns
    -------
    pd.Index
        排序后的神经元索引
    """
    # 标准化数据
    standardized_data = {}
    for col in reference_data.columns:
        if col != 'stamp' and col != 'behavior':
            series = reference_data[col]
            standardized_data[col] = (series - series.mean()) / series.std()
    
    if config.sort_method == 'peak':
        # 计算每个神经元的峰值时间
        peak_times = {}
        for neuron, data in standardized_data.items():
            peak_idx = data.idxmax()
            peak_times[neuron] = peak_idx
        
        # 按峰值时间排序
        peak_times_series = pd.Series(peak_times)
        sorted_neurons = peak_times_series.sort_values().index
        
    else:  # calcium_wave
        # 计算每个神经元的第一次钙波时间
        first_wave_times = {}
        for neuron, data in standardized_data.items():
            series = pd.Series(data.values, index=data.index, name=neuron)
            first_wave_times[neuron] = detect_first_calcium_wave_multiday(series, config)
        
        # 按第一次钙波时间排序
        first_wave_times_series = pd.Series(first_wave_times)
        sorted_neurons = first_wave_times_series.sort_values().index
    
    return sorted_neurons


def extract_cd1_behavior_events(behavior_data: Optional[pd.Series]) -> List[float]:
    """
    提取CD1行为事件的时间点
    
    Parameters
    ----------
    behavior_data : Optional[pd.Series]
        行为数据
        
    Returns
    -------
    List[float]
        CD1行为事件的时间戳列表
    """
    if behavior_data is None:
        return []
    
    cd1_events = behavior_data[behavior_data == 'CD1'].index.tolist()
    return cd1_events


def create_multiday_combination_heatmap(multiday_data: Dict[str, pd.DataFrame],
                                      config: MultiDayHeatmapConfig,
                                      reference_day: str = 'day3') -> Tuple[Figure, Dict[str, Any]]:
    """
    创建多天数据组合热力图
    
    Parameters
    ----------
    multiday_data : Dict[str, pd.DataFrame]
        多天数据字典
    config : MultiDayHeatmapConfig
        配置对象
    reference_day : str
        用于排序的参考天数
        
    Returns
    -------
    Tuple[Figure, Dict[str, Any]]
        生成的图形对象和分析信息
    """
    available_days = list(multiday_data.keys())
    
    # 预处理数据
    processed_data = {}
    behavior_data = {}
    
    for day, data in multiday_data.items():
        # 确保 'stamp' 列设置为索引
        if 'stamp' in data.columns:
            data = data.set_index('stamp')
        
        # 分离行为数据
        if 'behavior' in data.columns:
            behavior_data[day] = data['behavior']
            neural_data = data.drop(columns=['behavior'])
        else:
            neural_data = data.copy()
            behavior_data[day] = None
        
        # 标准化神经元数据
        processed_data[day] = (neural_data - neural_data.mean()) / neural_data.std()
    
    # 基于参考天数计算排序顺序
    if reference_day in processed_data:
        sorted_neurons = calculate_sorting_for_multiday(multiday_data[reference_day], config)
    else:
        # 如果参考天数不存在，使用第一个可用的天数
        first_day = available_days[0]
        sorted_neurons = calculate_sorting_for_multiday(multiday_data[first_day], config)
        reference_day = first_day
    
    # 创建组合图形
    fig, axes = plt.subplots(1, len(available_days), figsize=config.figure_size_combo)
    if len(available_days) == 1:
        axes = [axes]
    
    cd1_events_info = {}
    
    for i, day in enumerate(available_days):
        # 获取当前天的数据
        day_data = processed_data[day]
        
        # 确保神经元顺序一致（只保留排序中存在的神经元）
        available_neurons = [neuron for neuron in sorted_neurons if neuron in day_data.columns]
        if available_neurons:
            sorted_day_data = day_data[available_neurons]
        else:
            sorted_day_data = day_data
        
        # 绘制热力图
        sns.heatmap(
            sorted_day_data.T, 
            cmap=config.colormap, 
            cbar=True, 
            vmin=config.vmin, 
            vmax=config.vmax, 
            ax=axes[i]
        )
        
        # 设置标题和标签
        if day == reference_day:
            sort_method_str = "peak time" if config.sort_method == 'peak' else "calcium wave time"
            title = f'{day.upper()} (Sorted by {sort_method_str})'
        else:
            title = f'{day.upper()} (Using {reference_day.upper()} order)'
        
        axes[i].set_title(title, fontsize=25)
        axes[i].set_xlabel('Stamp', fontsize=25)
        
        if i == 0:
            axes[i].set_ylabel('Neuron', fontsize=25)
        else:
            axes[i].set_ylabel('')
        
        # 标记CD1行为事件
        cd1_events = extract_cd1_behavior_events(behavior_data.get(day))
        cd1_events_info[day] = cd1_events
        
        if cd1_events:
            for cd1_time in cd1_events:
                if cd1_time in sorted_day_data.index:
                    position = sorted_day_data.index.get_loc(cd1_time)
                    axes[i].axvline(x=position, color='white', linestyle='--', linewidth=2)
                    axes[i].text(position + 0.5, -5, 'CD1', 
                               color='black', rotation=90, verticalalignment='top', 
                               fontsize=20, fontweight='bold')
    
    plt.tight_layout()
    
    # 生成分析信息
    info = {
        'sort_method': config.sort_method,
        'reference_day': reference_day,
        'available_days': available_days,
        'total_neurons_by_day': {day: len(processed_data[day].columns) for day in available_days},
        'cd1_events': cd1_events_info,
        'sorted_neurons_count': len(sorted_neurons)
    }
    
    return fig, info


def create_single_day_heatmap(data: pd.DataFrame, 
                             day_name: str,
                             config: MultiDayHeatmapConfig,
                             neuron_order: Optional[pd.Index] = None,
                             separation_info: Optional[Dict] = None) -> Tuple[Figure, Dict[str, Any]]:
    """
    创建单天数据热力图
    
    Parameters
    ----------
    data : pd.DataFrame
        单天数据
    day_name : str
        天数名称
    config : MultiDayHeatmapConfig
        配置对象
    neuron_order : Optional[pd.Index]
        神经元排序顺序
    separation_info : Optional[Dict]
        分割信息（用于标记不同类型的神经元）
        
    Returns
    -------
    Tuple[Figure, Dict[str, Any]]
        生成的图形对象和分析信息
    """
    # 确保 'stamp' 列设置为索引
    if 'stamp' in data.columns:
        data = data.set_index('stamp')
    
    # 分离行为数据
    if 'behavior' in data.columns:
        behavior_data = data['behavior']
        neural_data = data.drop(columns=['behavior'])
    else:
        neural_data = data.copy()
        behavior_data = None
    
    # 标准化数据
    neural_data_standardized = (neural_data - neural_data.mean()) / neural_data.std()
    
    # 应用神经元排序（如果提供）
    if neuron_order is not None:
        available_neurons = [neuron for neuron in neuron_order if neuron in neural_data_standardized.columns]
        remaining_neurons = [neuron for neuron in neural_data_standardized.columns if neuron not in available_neurons]
        
        # 按ID排序剩余神经元
        remaining_neurons.sort(key=lambda x: float(x) if str(x).replace('.', '').isdigit() else float('inf'))
        
        final_order = available_neurons + remaining_neurons
        sorted_data = neural_data_standardized[final_order]
    else:
        sorted_data = neural_data_standardized
    
    # 创建图形
    fig, ax = plt.subplots(figsize=config.figure_size_single)
    
    # 绘制热力图
    sns.heatmap(
        sorted_data.T, 
        cmap=config.colormap, 
        cbar=True, 
        vmin=config.vmin, 
        vmax=config.vmax, 
        ax=ax
    )
    
    # 设置标题和标签
    if neuron_order is not None:
        title = f'{day_name.upper()} (Ordered by reference + Remaining by Neuron ID)'
    else:
        title = f'{day_name.upper()}'
    
    ax.set_title(title, fontsize=25)
    ax.set_xlabel('Stamp', fontsize=25)
    ax.set_ylabel('Neuron', fontsize=25)
    
    # 添加分割线（如果有分割信息）
    if separation_info and neuron_order is not None:
        corresponding_count = len([n for n in neuron_order if n in neural_data_standardized.columns])
        remaining_count = len(neural_data_standardized.columns) - corresponding_count
        
        if corresponding_count > 0 and remaining_count > 0:
            separation_line = corresponding_count
            ax.axhline(y=separation_line, color='red', linestyle='-', linewidth=3)
            
            # 添加文本标注
            ax.text(-10, separation_line/2, f'Reference neurons\n({corresponding_count})', 
                   color='red', fontsize=15, fontweight='bold', 
                   verticalalignment='center', rotation=90)
            ax.text(-10, separation_line + remaining_count/2, f'Remaining neurons\n({remaining_count})', 
                   color='red', fontsize=15, fontweight='bold', 
                   verticalalignment='center', rotation=90)
    
    # 标记CD1行为事件
    cd1_events = extract_cd1_behavior_events(behavior_data)
    if cd1_events:
        for cd1_time in cd1_events:
            if cd1_time in sorted_data.index:
                position = sorted_data.index.get_loc(cd1_time)
                ax.axvline(x=position, color='white', linestyle='--', linewidth=2)
                ax.text(position + 0.5, -5, 'CD1', 
                       color='black', rotation=90, verticalalignment='top', 
                       fontsize=20, fontweight='bold')
    
    plt.tight_layout()
    
    # 生成分析信息
    info = {
        'day_name': day_name,
        'total_neurons': len(sorted_data.columns),
        'cd1_events_count': len(cd1_events),
        'has_neuron_order': neuron_order is not None,
        'separation_applied': separation_info is not None and neuron_order is not None
    }
    
    return fig, info


def analyze_multiday_heatmap(data_dict: Dict[str, pd.DataFrame], 
                           config: MultiDayHeatmapConfig,
                           correspondence_table: Optional[pd.DataFrame] = None,
                           create_combination: bool = True,
                           create_individual: bool = True) -> Dict[str, Any]:
    """
    执行多天数据热力图分析
    
    Parameters
    ----------
    data_dict : Dict[str, pd.DataFrame]
        多天数据字典
    config : MultiDayHeatmapConfig
        配置对象
    correspondence_table : Optional[pd.DataFrame]
        神经元对应表
    create_combination : bool
        是否创建组合热力图
    create_individual : bool
        是否创建单独热力图
        
    Returns
    -------
    Dict[str, Any]
        分析结果字典，包含图形对象和分析信息
    """
    results = {
        'config': config,
        'combination_heatmap': None,
        'individual_heatmaps': {},
        'analysis_info': {}
    }
    
    available_days = list(data_dict.keys())
    
    # 创建组合热力图
    if create_combination and len(available_days) > 1:
        try:
            combo_fig, combo_info = create_multiday_combination_heatmap(
                data_dict, config, reference_day=available_days[0]
            )
            results['combination_heatmap'] = {
                'figure': combo_fig,
                'info': combo_info
            }
        except Exception as e:
             # 组合热力图创建失败
             pass
    
    # 创建单独热力图
    if create_individual:
        reference_order = None
        
        # 如果有参考天数，计算参考排序
        if available_days:
            try:
                reference_day = available_days[0]
                reference_order = calculate_sorting_for_multiday(data_dict[reference_day], config)
            except Exception as e:
                 # 参考排序计算失败
                 pass
        
        for day, data in data_dict.items():
            try:
                fig, info = create_single_day_heatmap(
                    data, day, config, reference_order
                )
                results['individual_heatmaps'][day] = {
                    'figure': fig,
                    'info': info
                }
            except Exception as e:
                 # 单独热力图创建失败
                 pass
    
    # 生成总体分析信息
    results['analysis_info'] = {
        'total_days': int(len(available_days)),
        'available_days': available_days,
        'sort_method': config.sort_method,
        'has_correspondence_table': bool(correspondence_table is not None),
        'combination_created': bool(results['combination_heatmap'] is not None),
        'individual_created': bool(len(results['individual_heatmaps']) > 0)
    }
    
    return results
