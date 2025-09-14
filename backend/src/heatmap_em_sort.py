"""
EM排序热力图分析模块

基于 heatmap_sort_EM.py 算法的功能实现，支持：
- 多种排序方式（峰值时间、钙波时间、自定义顺序）
- 时间区间筛选
- 行为标记和区间显示
- 高质量热力图生成
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from scipy.signal import find_peaks
from matplotlib.figure import Figure
from dataclasses import dataclass
import os
from pathlib import Path


@dataclass
class EMSortHeatmapConfig:
    """
    EM排序热力图配置类
    
    Attributes
    ----------
    stamp_min : Optional[float]
        最小时间戳值（None表示不限制）
    stamp_max : Optional[float]
        最大时间戳值（None表示不限制）
    sort_method : str
        排序方式：'peak'（按峰值时间）、'calcium_wave'（按钙波时间）或'custom'（自定义顺序）
    custom_neuron_order : List[str]
        自定义神经元排序顺序（仅在sort_method='custom'时使用）
    sampling_rate : float
        采样率（Hz）
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
    """
    
    stamp_min: Optional[float] = None
    stamp_max: Optional[float] = None
    sort_method: str = 'peak'
    custom_neuron_order: List[str] = None
    sampling_rate: float = 4.8
    calcium_wave_threshold: float = 1.5
    min_prominence: float = 1.0
    min_rise_rate: float = 0.1
    max_fall_rate: float = 0.05
    vmin: float = -2.0
    vmax: float = 2.0
    colormap: str = 'viridis'
    
    def __post_init__(self):
        """初始化后处理，设置默认的自定义神经元顺序"""
        if self.custom_neuron_order is None:
            self.custom_neuron_order = [
                'n53', 'n40', 'n29', 'n34', 'n4', 'n32', 'n25', 
                'n27', 'n22', 'n55', 'n21', 'n5', 'n19'
            ]


def load_and_preprocess_data(data: pd.DataFrame, config: EMSortHeatmapConfig) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
    """
    加载和预处理数据
    
    Parameters
    ----------
    data : pd.DataFrame
        输入数据，包含时间戳和神经元活动数据
    config : EMSortHeatmapConfig
        配置对象
        
    Returns
    -------
    Tuple[pd.DataFrame, Optional[pd.Series]]
        神经元数据和行为数据（如果存在）
    """
    # 确保 'stamp' 列设置为索引
    if 'stamp' in data.columns:
        data = data.set_index('stamp')
    
    # 检查是否存在 'behavior' 列
    has_behavior = 'behavior' in data.columns
    behavior_data = None
    
    if has_behavior:
        behavior_data = data['behavior'].copy()
        neural_data = data.drop(columns=['behavior'])
    else:
        neural_data = data.copy()
    
    # 时间区间筛选
    if config.stamp_min is not None or config.stamp_max is not None:
        min_stamp = config.stamp_min if config.stamp_min is not None else neural_data.index.min()
        max_stamp = config.stamp_max if config.stamp_max is not None else neural_data.index.max()
        
        neural_data = neural_data.loc[min_stamp:max_stamp]
        if has_behavior:
            behavior_data = behavior_data.loc[min_stamp:max_stamp]
    
    return neural_data, behavior_data


def detect_first_calcium_wave(neuron_data: pd.Series, config: EMSortHeatmapConfig) -> float:
    """
    检测神经元第一次真实钙波发生的时间点
    
    Parameters
    ----------
    neuron_data : pd.Series
        神经元活动的时间序列数据（标准化后）
    config : EMSortHeatmapConfig
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
    
    # 对每个峰值进行验证，确认是否为真实钙波（上升快，下降慢）
    for peak_idx in peaks:
        # 确保峰值不在时间序列的开始或结束处
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
        
        # 确认是否符合钙波特征：上升快，下降慢
        if rise_rate > config.min_rise_rate and 0 < fall_rate < config.max_fall_rate:
            return neuron_data.index[peak_idx]
    
    return neuron_data.index[-1]


def sort_neurons_by_custom_order(data_columns: pd.Index, custom_order: List[str]) -> List[str]:
    """
    按自定义神经元顺序排序
    
    Parameters
    ----------
    data_columns : pd.Index
        数据中的神经元列名
    custom_order : List[str]
        自定义的神经元顺序列表
        
    Returns
    -------
    List[str]
        按自定义顺序排列的神经元列表
    """
    available_neurons = set(data_columns)
    
    # 首先按照自定义顺序排列存在的神经元
    ordered_neurons = []
    for neuron in custom_order:
        if neuron in available_neurons:
            ordered_neurons.append(neuron)
    
    # 找出剩余的神经元，按字符串大小顺序排列
    remaining_neurons = sorted(list(available_neurons - set(ordered_neurons)))
    
    return ordered_neurons + remaining_neurons


def calculate_neuron_sorting(neural_data: pd.DataFrame, config: EMSortHeatmapConfig) -> pd.Index:
    """
    根据配置计算神经元排序顺序
    
    Parameters
    ----------
    neural_data : pd.DataFrame
        神经元数据
    config : EMSortHeatmapConfig
        配置对象
        
    Returns
    -------
    pd.Index
        排序后的神经元索引
    """
    # 标准化数据用于排序计算
    neural_data_standardized = (neural_data - neural_data.mean()) / neural_data.std()
    
    if config.sort_method == 'peak':
        # 按峰值时间排序
        peak_times = neural_data_standardized.idxmax()
        sorted_neurons = peak_times.sort_values().index
        
    elif config.sort_method == 'calcium_wave':
        # 按第一次真实钙波发生时间排序
        first_wave_times = {}
        
        for neuron in neural_data_standardized.columns:
            neuron_data = neural_data_standardized[neuron]
            first_wave_times[neuron] = detect_first_calcium_wave(neuron_data, config)
        
        first_wave_times_series = pd.Series(first_wave_times)
        sorted_neurons = first_wave_times_series.sort_values().index
        
    else:  # custom
        # 自定义排序
        sorted_neurons_list = sort_neurons_by_custom_order(
            neural_data_standardized.columns, 
            config.custom_neuron_order
        )
        sorted_neurons = pd.Index(sorted_neurons_list)
    
    return sorted_neurons


def extract_behavior_intervals(behavior_data: pd.Series) -> Dict[str, List[Tuple[float, float]]]:
    """
    提取行为区间信息
    
    Parameters
    ----------
    behavior_data : pd.Series
        行为标签时间序列
        
    Returns
    -------
    Dict[str, List[Tuple[float, float]]]
        每种行为的区间列表，格式为 {行为名: [(开始时间, 结束时间), ...]}
    """
    if behavior_data is None or behavior_data.empty:
        return {}
    
    # 获取所有不同的行为标签
    unique_behaviors = behavior_data.dropna().unique()
    behavior_intervals = {behavior: [] for behavior in unique_behaviors}
    
    # 对行为数据进行处理，找出每种行为的连续区间
    current_behavior = None
    start_time = None
    
    # 为了确保最后一个区间也被记录，将索引列表扩展一个元素
    extended_index = list(behavior_data.index) + [None]
    extended_values = list(behavior_data.values) + [None]
    
    for i, (timestamp, behavior) in enumerate(zip(extended_index, extended_values)):
        # 最后一个元素特殊处理
        if i == len(behavior_data):
            if start_time is not None and current_behavior is not None:
                behavior_intervals[current_behavior].append((start_time, extended_index[i-1]))
            break
        
        # 跳过空值
        if pd.isna(behavior):
            # 如果之前有行为，则结束当前区间
            if start_time is not None and current_behavior is not None:
                behavior_intervals[current_behavior].append((start_time, timestamp))
                start_time = None
                current_behavior = None
            continue
        
        # 如果是新的行为类型或第一个行为
        if behavior != current_behavior:
            # 如果之前有行为，先结束当前区间
            if start_time is not None and current_behavior is not None:
                behavior_intervals[current_behavior].append((start_time, timestamp))
            
            # 开始新的行为区间
            start_time = timestamp
            current_behavior = behavior
    
    return behavior_intervals


def create_behavior_colormap() -> Dict[str, str]:
    """
    创建行为颜色映射表
    
    Returns
    -------
    Dict[str, str]
        行为名称到颜色的映射
    """
    return {
        # 原有配色（保持不变）
        'Crack-seeds-shells': '#FF9500',    # 明亮橙色
        'Eat-feed': '#0066CC',              # 深蓝色
        'Eat-seed-kernels': '#00CC00',      # 亮绿色
        'Explore': '#FF0000',               # 鲜红色
        'Explore-search-seeds': '#9900FF',  # 亮紫色
        'Find-seeds': '#994C00',            # 深棕色
        'Get-feed': '#FF00CC',              # 亮粉色
        'Get-seeds': '#000000',             # 黑色
        'Grab-seeds': '#AACC00',            # 亮黄绿色
        'Groom': '#00CCFF',                 # 亮蓝绿色
        'Smell-feed': '#66B3FF',            # 亮蓝色
        'Smell-Get-seeds': '#33FF33',       # 鲜绿色
        'Store-seeds': '#FF6666',           # 亮红色
        'Water': '#CC99FF',                 # 亮紫色
        
        # 高架十字迷宫行为标签
        'Close-arm': '#8B0000',             # 深红色
        'Close-armed-Exp': '#CD5C5C',       # 印度红
        'Closed-arm-freezing': '#2F2F2F',  # 深灰色
        'Middle-zone': '#FFD700',           # 金色
        'Middle-zone-freezing': '#B8860B',  # 深金黄色
        'Open-arm': '#32CD32',              # 酸橙绿
        'Open-arm-exp': '#00FF7F',          # 春绿色
        'open-arm-freezing': '#006400',     # 深绿色
        'Open-arm-head dipping': '#7CFC00', # 草绿色
        
        # 家庭笼行为标签
        'Active': '#FF4500',                # 橙红色
        'Drink': '#1E90FF',                 # 道奇蓝
        'Move': '#32CD32',                  # 酸橙绿
        'Scratch': '#CD853F',               # 秘鲁色
        'Scratch + Groom': '#DDA0DD',       # 李子色
        'Sleep': '#2F4F4F',                 # 深灰绿色
        'Wake': '#FFD700',                  # 金色
        'zone-out': '#708090',              # 石板灰
        'CD1': '#FF1493'                    # 深粉色 - CD1特殊标记
    }


def generate_em_sort_heatmap(neural_data: pd.DataFrame, 
                           behavior_data: Optional[pd.Series],
                           config: EMSortHeatmapConfig) -> Tuple[Figure, Dict[str, Any]]:
    """
    生成EM排序热力图
    
    Parameters
    ----------
    neural_data : pd.DataFrame
        神经元数据
    behavior_data : Optional[pd.Series]
        行为数据（可选）
    config : EMSortHeatmapConfig
        配置对象
        
    Returns
    -------
    Tuple[Figure, Dict[str, Any]]
        生成的图形对象和分析信息
    """
    # 计算神经元排序顺序（基于全局数据，确保一致性）
    sorted_neurons = calculate_neuron_sorting(neural_data, config)
    
    # 标准化数据
    neural_data_standardized = (neural_data - neural_data.mean()) / neural_data.std()
    
    # 根据排序后的神经元顺序重新排列数据
    sorted_data = neural_data_standardized[sorted_neurons]
    
    # 提取行为区间
    behavior_intervals = extract_behavior_intervals(behavior_data) if behavior_data is not None else {}
    
    # 创建图形
    fig = plt.figure(figsize=(60, 25))
    
    # 使用GridSpec布局系统
    from matplotlib.gridspec import GridSpec
    if behavior_intervals:
        # 有行为数据时：上方行为时间线，下方热力图，右侧图例
        grid = GridSpec(2, 2, height_ratios=[0.5, 6], width_ratios=[6, 0.5], 
                       hspace=0.05, wspace=0.02, figure=fig)
        ax_heatmap = fig.add_subplot(grid[1, 0])
        ax_behavior = fig.add_subplot(grid[0, 0])
        ax_legend = fig.add_subplot(grid[1, 1])
    else:
        # 无行为数据时：只有热力图
        ax_heatmap = fig.add_subplot(111)
    
    # 绘制热力图
    heatmap = sns.heatmap(
        sorted_data.T, 
        cmap=config.colormap, 
        cbar=True, 
        vmin=config.vmin, 
        vmax=config.vmax, 
        ax=ax_heatmap
    )
    
    # 设置热力图的x轴范围
    ax_heatmap.set_xlim(-0.5, len(sorted_data.index) - 0.5)
    
    # 添加行为标记（如果有行为数据）
    if behavior_intervals:
        # 创建行为颜色映射
        behavior_colormap = create_behavior_colormap()
        
        # 设置行为子图的X轴范围与热力图完全匹配
        ax_behavior.set_xlim(-0.5, len(sorted_data.index) - 0.5)
        
        # 为每种行为绘制区间
        y_position = 0.5
        line_height = 0.8
        legend_patches = []
        
        for behavior, intervals in behavior_intervals.items():
            behavior_color = behavior_colormap.get(behavior, '#808080')  # 默认灰色
            
            for start_time, end_time in intervals:
                if start_time in sorted_data.index and end_time in sorted_data.index:
                    start_idx = sorted_data.index.get_loc(start_time)
                    end_idx = sorted_data.index.get_loc(end_time)
                    
                    if end_idx - start_idx > 0:
                        # 在行为标记子图中绘制区间
                        rect = plt.Rectangle(
                            (start_idx - 0.5, y_position - line_height/2), 
                            end_idx - start_idx, line_height, 
                            color=behavior_color, alpha=0.9, 
                            ec='black', linewidth=0.5
                        )
                        ax_behavior.add_patch(rect)
                        
                        # 在热力图中添加边界线
                        ax_heatmap.axvline(x=start_idx - 0.5, color='white', linestyle='--', linewidth=2, alpha=0.5)
                        ax_heatmap.axvline(x=end_idx - 0.5, color='white', linestyle='--', linewidth=2, alpha=0.5)
            
            # 创建图例补丁
            legend_patches.append(plt.Rectangle((0, 0), 1, 1, color=behavior_color, alpha=0.9, label=behavior))
        
        # 再次确认两个坐标轴的对齐情况
        # 唯一正确的做法是设置完全相同的范围
        ax_heatmap.set_xlim(-0.5, len(sorted_data.index) - 0.5)
        ax_behavior.set_xlim(-0.5, len(sorted_data.index) - 0.5)
        
        # 设置行为子图的Y轴范围，只显示一条线
        ax_behavior.set_ylim(0, 1)
        
        # 移除Y轴刻度和标签
        ax_behavior.set_yticks([])
        ax_behavior.set_yticklabels([])
        
        # 特别重要：移除X轴刻度，让它只在热图上显示
        ax_behavior.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        ax_behavior.set_title('Behavior Timeline', fontsize=40, pad=10)
        ax_behavior.set_xlabel('')  # Remove x-axis label, shared with the heatmap below

        # 更强制地隐藏X轴刻度和标签
        ax_behavior.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        # 确保行为图和热图水平对齐
        ax_behavior.set_anchor('SW')
        # 去除行为子图边框
        ax_behavior.spines['top'].set_visible(False)
        ax_behavior.spines['right'].set_visible(False)
        ax_behavior.spines['bottom'].set_visible(False)
        ax_behavior.spines['left'].set_visible(False)
        
        # 添加图例
        ax_legend.axis('off')
        if legend_patches:
            legend = ax_legend.legend(
                handles=legend_patches, 
                loc='center left', 
                fontsize=40, 
                title='Behavior Types', 
                title_fontsize=40, 
                ncol=1,
                frameon=True, 
                fancybox=True, 
                shadow=True, 
                bbox_to_anchor=(0, 0.5)
            )
    
    # 设置热力图标签
    ax_heatmap.set_xlabel('Time (s)', fontsize=40)
    ax_heatmap.set_ylabel('Neuron', fontsize=40)
    
    # 设置Y轴标签（神经元标签）
    ax_heatmap.set_yticklabels(ax_heatmap.get_yticklabels(), fontsize=23, fontweight='bold', rotation=0)
    ax_heatmap.set_xticklabels(ax_heatmap.get_xticklabels(), fontsize=30, fontweight='bold')
    
    # 设置X轴刻度，以10秒为间隔
    time_per_frame = 1.0 / config.sampling_rate
    min_stamp = sorted_data.index.min()
    max_stamp = sorted_data.index.max()
    min_seconds = max(0, min_stamp / config.sampling_rate)
    max_seconds = max_stamp / config.sampling_rate
    
    # 生成以10秒为间隔的时间刻度
    start_tick = int(min_seconds // 10) * 10
    time_ticks_seconds = np.arange(start_tick, max_seconds + 10, 10)
    
    xtick_positions = []
    xtick_labels = []
    
    for t in time_ticks_seconds:
        timestamp = t * config.sampling_rate
        if min_stamp <= timestamp <= max_stamp and timestamp in sorted_data.index:
            pos = sorted_data.index.get_loc(timestamp)
            xtick_positions.append(pos)
            xtick_labels.append(f'{int(t)}')
    
    if xtick_positions:
        ax_heatmap.set_xticks(xtick_positions)
        ax_heatmap.set_xticklabels(xtick_labels, fontsize=30, fontweight='bold', rotation=45)
    
    # 在保存前使用多重方法确保对齐
    if behavior_intervals:
        # 1. 强制更新布局
        fig.canvas.draw()
        
        # 2. 再次确认轴范围一致
        ax_heatmap.set_xlim(-0.5, len(sorted_data.index) - 0.5)
        ax_behavior.set_xlim(-0.5, len(sorted_data.index) - 0.5)
        
        # 3. 达到更精确的对齐
        # 获取热图的实际边界位置（像素单位）
        heatmap_bbox = ax_heatmap.get_position()
        behavior_bbox = ax_behavior.get_position()
        
        # 不能直接修改Bbox对象，需要创建新的
        # 使用Bbox的坐标创建新的位置，保持高度不变，但使用热图的宽度和水平位置
        from matplotlib.transforms import Bbox
        new_behavior_pos = Bbox([[heatmap_bbox.x0, behavior_bbox.y0], 
                                [heatmap_bbox.x0 + heatmap_bbox.width, behavior_bbox.y0 + behavior_bbox.height]])
        
        # 设置新的位置
        ax_behavior.set_position(new_behavior_pos)
    
    # 应用紧凑布局
    # 不使用tight_layout()，因为它与GridSpec布局不兼容
    # 而是使用之前设置的subplots_adjust()已经足够调整布局
    
    # 生成分析信息
    info = {
        'sort_method': config.sort_method,
        'total_neurons': int(len(sorted_neurons)),
        'time_range': {
            'start_stamp': float(min_stamp),
            'end_stamp': float(max_stamp),
            'start_seconds': float(min_seconds),
            'end_seconds': float(max_seconds),
            'duration_seconds': float(max_seconds - min_seconds)
        },
        'behavior_types': list(behavior_intervals.keys()) if behavior_intervals else [],
        'total_behavior_events': int(sum(len(intervals) for intervals in behavior_intervals.values()) if behavior_intervals else 0)
    }
    
    return fig, info


def analyze_em_sort_heatmap(data: pd.DataFrame, config: EMSortHeatmapConfig) -> Tuple[Figure, Dict[str, Any]]:
    """
    执行EM排序热力图分析
    
    Parameters
    ----------
    data : pd.DataFrame
        输入数据
    config : EMSortHeatmapConfig
        配置对象
        
    Returns
    -------
    Tuple[Figure, Dict[str, Any]]
        生成的图形对象和分析信息
    """
    # 加载和预处理数据
    neural_data, behavior_data = load_and_preprocess_data(data, config)
    
    # 生成热力图
    fig, info = generate_em_sort_heatmap(neural_data, behavior_data, config)
    
    return fig, info
