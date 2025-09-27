import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from scipy.signal import find_peaks
import base64
from io import BytesIO
from typing import Dict, List, Optional, Tuple, Any
import os
from pathlib import Path

# 设置matplotlib参数
plt.rcParams['lines.linewidth'] = 2.0
plt.rcParams['font.size'] = 12

class TraceConfig:
    """Trace图配置类"""
    def __init__(self):
        # 时间戳区间
        self.stamp_min: Optional[float] = None
        self.stamp_max: Optional[float] = None
        
        # 排序方式
        self.sort_method: str = 'peak'  # 'original', 'peak', 'calcium_wave', 'custom'
        self.custom_neuron_order: List[str] = []
        
        # Trace图显示参数
        self.trace_offset: float = 60.0
        self.scaling_factor: float = 80.0
        self.max_neurons: int = 60
        self.trace_alpha: float = 0.8
        self.line_width: float = 2.0
        
        # 采样率
        self.sampling_rate: float = 4.8
        
        # 钙爆发检测参数
        self.calcium_threshold: float = 2.0
        self.calcium_wave_threshold: float = 1.5
        self.min_prominence: float = 1.0
        self.min_rise_rate: float = 0.1
        self.max_fall_rate: float = 0.05

def load_trace_data(file_path: str) -> pd.DataFrame:
    """
    加载trace数据
    
    Parameters
    ----------
    file_path : str
        数据文件路径
        
    Returns
    -------
    pd.DataFrame
        加载的数据
    """
    try:
        data = pd.read_excel(file_path)
        return data
    except Exception as e:
        raise ValueError(f"无法加载数据文件: {str(e)}")

def sort_neurons_by_custom_order(data_columns: List[str], custom_order: List[str]) -> List[str]:
    """
    按自定义神经元顺序排序
    
    Parameters
    ----------
    data_columns : List[str]
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
    
    # 合并两部分：自定义顺序 + 剩余神经元（按大小排序）
    final_order = ordered_neurons + remaining_neurons
    
    return final_order

def detect_first_calcium_wave(neuron_data: pd.Series, config: TraceConfig) -> float:
    """
    检测神经元第一次真实钙波发生的时间点
    
    Parameters
    ----------
    neuron_data : pd.Series
        包含神经元活动的时间序列数据（标准化后）
    config : TraceConfig
        配置对象
        
    Returns
    -------
    float
        第一次真实钙波发生的时间点，如果没有检测到则返回数据最后一个时间点
    """
    try:
        # 记录数据范围用于调试
        data_min_time = neuron_data.index.min()
        data_max_time = neuron_data.index.max()
        print(f"DEBUG: 神经元数据时间范围: {data_min_time} - {data_max_time}")
        
        # 计算阈值（基于数据的标准差）
        threshold = config.calcium_wave_threshold
        print(f"DEBUG: 钙波检测阈值: {threshold}")
        
        # 使用find_peaks函数检测峰值
        peaks, properties = find_peaks(neuron_data, 
                                     height=threshold, 
                                     prominence=config.min_prominence,
                                     distance=5)  # 要求峰值之间至少间隔5个时间点
        
        print(f"DEBUG: 检测到 {len(peaks)} 个峰值")
        
        if len(peaks) == 0:
            # 如果没有检测到峰值，返回时间序列的最后一个点
            result_time = neuron_data.index[-1]
            print(f"DEBUG: 未检测到峰值，返回最后时间点: {result_time}")
            return result_time
        
        # 对每个峰值进行验证，确认是否为真实钙波（上升快，下降慢）
        for i, peak_idx in enumerate(peaks):
            print(f"DEBUG: 验证第 {i+1} 个峰值，索引: {peak_idx}, 时间: {neuron_data.index[peak_idx]}")
            
            # 确保峰值不在时间序列的开始或结束处
            if peak_idx <= 1 or peak_idx >= len(neuron_data) - 2:
                print(f"DEBUG: 峰值 {i+1} 在边界附近，跳过")
                continue
                
            # 计算峰值前的上升速率（取峰值前5个点或更少）
            pre_peak_idx = max(0, peak_idx - 5)
            rise_rate = (neuron_data.iloc[peak_idx] - neuron_data.iloc[pre_peak_idx]) / (peak_idx - pre_peak_idx)
            
            # 计算峰值后的下降速率（取峰值后10个点或更少）
            post_peak_idx = min(len(neuron_data) - 1, peak_idx + 10)
            if post_peak_idx <= peak_idx:
                print(f"DEBUG: 峰值 {i+1} 后向索引无效，跳过")
                continue
            
            fall_rate = (neuron_data.iloc[peak_idx] - neuron_data.iloc[post_peak_idx]) / (post_peak_idx - peak_idx)
            
            print(f"DEBUG: 峰值 {i+1} - 上升速率: {rise_rate:.4f}, 下降速率: {fall_rate:.4f}")
            print(f"DEBUG: 阈值检查 - 上升速率 > {config.min_rise_rate}: {rise_rate > config.min_rise_rate}")
            print(f"DEBUG: 阈值检查 - 下降速率 < {config.max_fall_rate}: {0 < fall_rate < config.max_fall_rate}")
            
            # 确认是否符合钙波特征：上升快，下降慢
            if rise_rate > config.min_rise_rate and 0 < fall_rate < config.max_fall_rate:
                # 找到第一个真实钙波，返回时间点
                result_time = neuron_data.index[peak_idx]
                print(f"DEBUG: 找到真实钙波，返回时间点: {result_time}")
                return result_time
            else:
                print(f"DEBUG: 峰值 {i+1} 不符合钙波特征，继续检查下一个")
        
        # 如果没有满足条件的钙波，返回时间序列的最后一个点
        result_time = neuron_data.index[-1]
        print(f"DEBUG: 未找到符合条件的钙波，返回最后时间点: {result_time}")
        return result_time
        
    except Exception as e:
        print(f"Error in detect_first_calcium_wave: {e}")
        # 如果出现错误，返回数据的中点作为默认值
        result_time = neuron_data.index[len(neuron_data) // 2]
        print(f"DEBUG: 异常处理，返回中点时间: {result_time}")
        return result_time

def calculate_neuron_sorting(trace_data: pd.DataFrame, config: TraceConfig) -> Tuple[pd.Index, Optional[Dict[str, float]]]:
    """
    计算神经元排序
    
    Parameters
    ----------
    trace_data : pd.DataFrame
        神经元活动数据
    config : TraceConfig
        配置对象
        
    Returns
    -------
    Tuple[pd.Index, Optional[Dict[str, float]]]
        排序后的神经元索引和峰值时间字典
    """
    if config.sort_method == 'original':
        # 原始顺序：保持数据原有的神经元顺序
        sorted_neurons = trace_data.columns
        peak_times_dict = None
        
    elif config.sort_method == 'peak':
        # 按峰值时间排序
        trace_data_standardized = (trace_data - trace_data.mean()) / trace_data.std()
        peak_times = trace_data_standardized.idxmax()
        sorted_neurons = peak_times.sort_values().index
        # 将numpy类型转换为Python原生类型，确保JSON序列化
        peak_times_dict = {k: float(v) for k, v in peak_times.to_dict().items()}
        
    elif config.sort_method == 'custom':
        # 自定义顺序排序
        sorted_neurons_list = sort_neurons_by_custom_order(trace_data.columns.tolist(), config.custom_neuron_order)
        sorted_neurons = pd.Index(sorted_neurons_list)
        peak_times_dict = None
        
    else:  # 'calcium_wave'
        # 按第一次真实钙波发生时间排序
        try:
            print(f"DEBUG: 开始钙波排序，原始数据形状: {trace_data.shape}")
            print(f"DEBUG: 原始数据时间范围: {trace_data.index.min()} - {trace_data.index.max()}")
            
            # 数据标准化
            trace_data_standardized = (trace_data - trace_data.mean()) / trace_data.std()
            print(f"DEBUG: 标准化后数据形状: {trace_data_standardized.shape}")
            
            # 检查标准化后是否有异常值
            has_inf = np.isinf(trace_data_standardized).any().any()
            has_nan = np.isnan(trace_data_standardized).any().any()
            print(f"DEBUG: 标准化后数据检查 - 包含Inf: {has_inf}, 包含NaN: {has_nan}")
            
            if has_inf or has_nan:
                print("WARNING: 标准化后数据包含异常值，使用峰值排序作为替代")
                peak_times = trace_data.idxmax()  # 使用原始数据而不是标准化数据
                sorted_neurons = peak_times.sort_values().index
                # 将numpy类型转换为Python原生类型，确保JSON序列化
                peak_times_dict = {k: float(v) for k, v in peak_times.to_dict().items()}
            else:
                first_wave_times = {}
                
                # 对每个神经元进行钙波检测
                for i, neuron in enumerate(trace_data_standardized.columns):
                    try:
                        print(f"DEBUG: 处理神经元 {i+1}/{len(trace_data_standardized.columns)}: {neuron}")
                        neuron_data = trace_data_standardized[neuron]
                        first_wave_times[neuron] = detect_first_calcium_wave(neuron_data, config)
                    except Exception as e:
                        print(f"Error processing neuron {neuron}: {e}")
                        # 如果单个神经元处理失败，使用数据中点作为默认值
                        first_wave_times[neuron] = trace_data.index[len(trace_data) // 2]
                
                # 转换为Series以便排序
                first_wave_times_series = pd.Series(first_wave_times)
                print(f"DEBUG: 钙波时间统计 - 最小值: {first_wave_times_series.min()}, 最大值: {first_wave_times_series.max()}")
                
                # 按第一次钙波时间排序
                sorted_neurons = first_wave_times_series.sort_values().index
                # 将numpy类型转换为Python原生类型，确保JSON序列化
                peak_times_dict = {k: float(v) for k, v in first_wave_times.items()}
                
                print(f"DEBUG: 钙波排序完成，排序后神经元数量: {len(sorted_neurons)}")
            
        except Exception as e:
            print(f"Error in calcium_wave sorting: {e}")
            # 如果钙波排序失败，回退到峰值排序
            print("DEBUG: 回退到峰值排序")
            peak_times = trace_data.idxmax()
            sorted_neurons = peak_times.sort_values().index
            # 将numpy类型转换为Python原生类型，确保JSON序列化
            peak_times_dict = {k: float(v) for k, v in peak_times.to_dict().items()}
    
    return sorted_neurons, peak_times_dict

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
        'CD1': '#FF1493',                   # 深粉色 - CD1特殊标记
        
        # 处理后的数据文件中的行为标签
        'Exp': '#FF6B6B',                   # 珊瑚红 - 探索行为
        'Gro': '#4ECDC4',                   # 青绿色 - 梳理行为
        'Clim': '#45B7D1',                  # 天蓝色 - 攀爬行为
        'Sta': '#96CEB4',                   # 薄荷绿 - 站立行为
        'Scra': '#FFEAA7',                  # 淡黄色 - 抓挠行为
        'Stiff': '#DDA0DD',                 # 李子色 - 僵硬行为
        'Trem': '#98D8C8',                  # 浅绿色 - 颤抖行为
        'Turn': '#F7DC6F'                   # 金黄色 - 转身行为
    }

def extract_behavior_intervals(behavior_data: pd.Series, sampling_rate: float) -> Dict[str, List[Tuple[float, float]]]:
    """
    提取行为区间信息
    
    Parameters
    ----------
    behavior_data : pd.Series
        行为标签时间序列
    sampling_rate : float
        采样率
        
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
                behavior_intervals[current_behavior].append((start_time / sampling_rate, extended_index[i-1] / sampling_rate))
            break
        
        # 跳过空值
        if pd.isna(behavior):
            # 如果之前有行为，则结束当前区间
            if start_time is not None and current_behavior is not None:
                behavior_intervals[current_behavior].append((start_time / sampling_rate, timestamp / sampling_rate))
                start_time = None
                current_behavior = None
            continue
        
        # 如果是新的行为类型或第一个行为
        if behavior != current_behavior:
            # 如果之前有行为，先结束当前区间
            if start_time is not None and current_behavior is not None:
                behavior_intervals[current_behavior].append((start_time / sampling_rate, timestamp / sampling_rate))
            
            # 开始新的行为区间
            start_time = timestamp
            current_behavior = behavior
    
    return behavior_intervals

def plot_to_base64(fig) -> str:
    """
    将matplotlib图形转换为base64编码字符串
    
    Parameters
    ----------
    fig : matplotlib.figure.Figure
        matplotlib图形对象
        
    Returns
    -------
    str
        base64编码的图形字符串
    """
    buffer = BytesIO()
    fig.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    buffer.close()
    plt.close(fig)  # 关闭图形以释放内存
    return f"data:image/png;base64,{image_base64}"

def generate_trace_plot(trace_data: pd.DataFrame, 
                       behavior_data: Optional[pd.Series],
                       config: TraceConfig) -> Tuple[str, Dict[str, Any]]:
    """
    生成Trace图
    
    Parameters
    ----------
    trace_data : pd.DataFrame
        神经元活动数据
    behavior_data : Optional[pd.Series]
        行为数据
    config : TraceConfig
        配置对象
        
    Returns
    -------
    Tuple[str, Dict[str, Any]]
        生成的图像base64字符串和相关信息
    """
    # 检查是否存在行为数据
    has_behavior = behavior_data is not None and not behavior_data.empty and behavior_data.dropna().unique().size > 0
    
    # 根据配置的时间戳区间筛选数据
    if config.stamp_min is not None or config.stamp_max is not None:
        min_stamp = config.stamp_min if config.stamp_min is not None else trace_data.index.min()
        max_stamp = config.stamp_max if config.stamp_max is not None else trace_data.index.max()
        trace_data = trace_data.loc[min_stamp:max_stamp]
        if has_behavior:
            behavior_data = behavior_data.loc[min_stamp:max_stamp]
    
    # 计算神经元排序
    sorted_neurons, peak_times_dict = calculate_neuron_sorting(trace_data, config)
    
    # 维度检查：确保排序后的神经元数量与原始数据匹配
    print(f"DEBUG: 维度检查 - 原始神经元数量: {len(trace_data.columns)}")
    print(f"DEBUG: 维度检查 - 排序后神经元数量: {len(sorted_neurons)}")
    print(f"DEBUG: 维度检查 - 排序方法: {config.sort_method}")
    
    if len(sorted_neurons) != len(trace_data.columns):
        print(f"ERROR: 神经元数量不匹配！原始: {len(trace_data.columns)}, 排序后: {len(sorted_neurons)}")
        # 如果数量不匹配，使用原始顺序
        sorted_neurons = trace_data.columns
        print("DEBUG: 使用原始神经元顺序作为替代")
    
    # 检查排序后的神经元是否都在原始数据中存在
    missing_neurons = set(sorted_neurons) - set(trace_data.columns)
    if missing_neurons:
        print(f"ERROR: 排序后的神经元中有 {len(missing_neurons)} 个在原始数据中不存在: {missing_neurons}")
        # 过滤掉不存在的神经元
        sorted_neurons = [neuron for neuron in sorted_neurons if neuron in trace_data.columns]
        print(f"DEBUG: 过滤后神经元数量: {len(sorted_neurons)}")
    
    # 根据排序重新排列数据
    if config.sort_method != 'original':
        try:
            sorted_trace_data = trace_data[sorted_neurons]
            print(f"DEBUG: 数据重排成功，形状: {sorted_trace_data.shape}")
        except Exception as e:
            print(f"ERROR: 数据重排失败: {e}")
            print("DEBUG: 使用原始数据作为替代")
            sorted_trace_data = trace_data
    else:
        sorted_trace_data = trace_data
    
    # 创建图形
    if has_behavior:
        # 如果有行为数据，使用2行2列的布局
        fig = plt.figure(figsize=(60, 25))
        grid = GridSpec(2, 2, height_ratios=[0.5, 6], width_ratios=[6, 0.5], hspace=0.05, wspace=0.02, figure=fig)
        ax_behavior = fig.add_subplot(grid[0, 0])
        ax_trace = fig.add_subplot(grid[1, 0])
        ax_legend = fig.add_subplot(grid[1, 1])
    else:
        # 没有行为数据，只创建一个图表
        fig = plt.figure(figsize=(60, 25))
        ax_trace = fig.add_subplot(111)
    
    # 绘制Trace图
    for i, column in enumerate(sorted_neurons):
        if i >= config.max_neurons:
            break
        
        # 计算当前神经元trace的垂直偏移量
        if config.sort_method in ['peak', 'calcium_wave']:
            # 对于峰值排序和钙波排序，使用反向位置（早期的在上方）
            position = config.max_neurons - i if i < config.max_neurons else 1
        else:
            # 对于原始顺序和自定义顺序，使用正向位置
            position = i + 1
        
        ax_trace.plot(
            sorted_trace_data.index / config.sampling_rate,  # x轴是时间(秒)
            sorted_trace_data[column] * config.scaling_factor + position * config.trace_offset,  # 应用缩放并偏移
            linewidth=config.line_width,
            alpha=config.trace_alpha,
            label=column
        )
        
        # 如果是峰值排序或钙波排序，标记峰值点
        if peak_times_dict is not None and column in peak_times_dict:
            peak_time = peak_times_dict[column] / config.sampling_rate
            ax_trace.scatter(
                peak_time, 
                sorted_trace_data.loc[peak_times_dict[column], column] * config.scaling_factor + position * config.trace_offset,
                color='red', s=30, zorder=3
            )
    
    # 设置Y轴标签
    total_positions = min(config.max_neurons, len(sorted_neurons))
    yticks = []
    ytick_labels = []
    
    for i, column in enumerate(sorted_neurons):
        if i >= config.max_neurons:
            break
        
        # 计算位置
        if config.sort_method in ['peak', 'calcium_wave']:
            position = config.max_neurons - i if i < config.max_neurons else 1
        else:
            position = i + 1
        
        yticks.append(position * config.trace_offset)
        ytick_labels.append(str(column))
    
    # 设置Y轴刻度和标签
    ax_trace.set_yticks(yticks)
    ax_trace.set_yticklabels(ytick_labels)
    
    # 设置X轴范围
    if config.stamp_min is not None and config.stamp_max is not None:
        min_seconds = config.stamp_min / config.sampling_rate
        max_seconds = config.stamp_max / config.sampling_rate
        min_seconds = max(0, min_seconds)
        ax_trace.set_xlim(min_seconds, max_seconds)
    else:
        min_seconds = max(0, sorted_trace_data.index.min() / config.sampling_rate)
        max_seconds = sorted_trace_data.index.max() / config.sampling_rate
        ax_trace.set_xlim(min_seconds, max_seconds)
    
    # 设置轴标签和标题
    ax_trace.set_xlabel('Time (seconds)', fontsize=50, fontweight='bold')
    
    # 根据排序方式设置不同的Y轴标签
    if config.sort_method == 'peak':
        ylabel = 'Neuron ID (Sorted by Peak Time)'
    elif config.sort_method == 'calcium_wave':
        ylabel = 'Neuron ID (Sorted by First Calcium Wave)'
    elif config.sort_method == 'custom':
        ylabel = 'Neuron ID (Custom Order)'
    else:
        ylabel = 'Neuron ID'
    ax_trace.set_ylabel(ylabel, fontsize=50, fontweight='bold')
    
    # 设置刻度标签字体大小
    ax_trace.tick_params(axis='x', labelsize=30, rotation=45)
    ax_trace.tick_params(axis='y', labelsize=23)
    
    # 设置X轴刻度间隔为10秒
    import matplotlib.ticker as ticker
    ax_trace.xaxis.set_major_locator(ticker.MultipleLocator(10))
    
    # 添加网格线
    ax_trace.grid(False)
    
    # 处理行为数据
    if has_behavior:
        behavior_colormap = create_behavior_colormap()
        behavior_intervals = extract_behavior_intervals(behavior_data, config.sampling_rate)
        
        # 创建图例补丁列表
        legend_patches = []
        
        # 将所有行为绘制在同一条水平线上
        y_position = 0.5
        line_height = 0.8
        
        # 设置行为子图的Y轴范围
        ax_behavior.set_ylim(0, 1)
        ax_behavior.set_yticks([])
        ax_behavior.set_yticklabels([])
        ax_behavior.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        ax_behavior.set_title('Behavior Timeline', fontsize=40, pad=10)
        ax_behavior.set_xlabel('')
        
        # 确保行为图和trace图水平对齐
        ax_behavior.set_xlim(ax_trace.get_xlim())
        ax_behavior.set_anchor('SW')
        
        # 去除行为子图边框
        ax_behavior.spines['top'].set_visible(False)
        ax_behavior.spines['right'].set_visible(False)
        ax_behavior.spines['bottom'].set_visible(False)
        ax_behavior.spines['left'].set_visible(False)
        
        # 为每种行为绘制区间
        for behavior, intervals in behavior_intervals.items():
            behavior_color = behavior_colormap.get(behavior, '#808080')
            
            for start_time, end_time in intervals:
                if end_time - start_time > 0:
                    # 在行为标记子图中绘制区间
                    rect = plt.Rectangle(
                        (start_time, y_position - line_height/2), 
                        end_time - start_time, line_height, 
                        color=behavior_color, alpha=0.9, 
                        ec='black', linewidth=0.5
                    )
                    ax_behavior.add_patch(rect)
                    
                    # 在trace图中添加区间边界垂直线
                    ax_trace.axvline(x=start_time, color='white', linestyle='--', linewidth=0.8, alpha=0.7)
                    ax_trace.axvline(x=end_time, color='white', linestyle='--', linewidth=0.8, alpha=0.7)
            
            # 添加到图例
            legend_patches.append(plt.Rectangle((0, 0), 1, 1, color=behavior_color, alpha=0.9, label=behavior))
        
        # 在单独的图例子图中添加图例
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
    
    # 强制更新图形以确保所有刻度标签都已生成
    fig.canvas.draw()
    
    # 设置刻度标签加粗
    for label in ax_trace.get_xticklabels():
        label.set_fontweight('bold')
    for label in ax_trace.get_yticklabels():
        label.set_fontweight('bold')
    
    # 转换为base64
    image_base64 = plot_to_base64(fig)
    
    # 生成信息
    info = {
        'sort_method': config.sort_method,
        'total_neurons': int(len(sorted_neurons)),
        'displayed_neurons': int(min(config.max_neurons, len(sorted_neurons))),
        'time_range': {
            'start_stamp': float(sorted_trace_data.index.min()),
            'end_stamp': float(sorted_trace_data.index.max()),
            'start_seconds': float(sorted_trace_data.index.min() / config.sampling_rate),
            'end_seconds': float(sorted_trace_data.index.max() / config.sampling_rate),
            'duration_seconds': float((sorted_trace_data.index.max() - sorted_trace_data.index.min()) / config.sampling_rate)
        },
        'behavior_types': list(behavior_intervals.keys()) if has_behavior else [],
        'total_behavior_events': int(sum(len(intervals) for intervals in behavior_intervals.values()) if has_behavior else 0),
        'peak_times': peak_times_dict if peak_times_dict else {}
    }
    
    return image_base64, info
