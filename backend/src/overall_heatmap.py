# 整体热力图生成模块
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvas
from scipy.signal import find_peaks
from scipy import stats
from typing import Optional, Tuple
import matplotlib

# 设置matplotlib后端为非交互式
matplotlib.use('Agg')

class OverallHeatmapConfig:
    """整体热力图配置类"""
    def __init__(self):
        # 时间戳区间默认值（None表示不限制）
        self.STAMP_MIN = None  # 最小时间戳
        self.STAMP_MAX = None  # 最大时间戳
        # 排序方式：'peak'（默认，按峰值时间排序）或'calcium_wave'（按第一次真实钙波发生时间排序）
        self.SORT_METHOD = 'peak'
        # 钙波检测参数
        self.CALCIUM_WAVE_THRESHOLD = 1.5  # 钙波阈值（标准差的倍数）
        self.MIN_PROMINENCE = 1.0  # 最小峰值突出度
        self.MIN_RISE_RATE = 0.1  # 最小上升速率
        self.MAX_FALL_RATE = 0.05  # 最大下降速率（下降应当比上升慢）
        # 图形参数
        self.FIGURE_SIZE = (25, 15)
        self.VMIN = -2
        self.VMAX = 2
        self.COLORMAP = 'viridis'

def detect_first_calcium_wave(neuron_data, config: OverallHeatmapConfig):
    """
    检测神经元第一次真实钙波发生的时间点
    
    参数:
    neuron_data -- 包含神经元活动的时间序列数据（标准化后）
    config -- 配置对象
    
    返回:
    first_wave_time -- 第一次真实钙波发生的时间点，如果没有检测到则返回数据最后一个时间点
    """
    # 计算阈值（基于数据的标准差）
    threshold = config.CALCIUM_WAVE_THRESHOLD
    
    # 使用find_peaks函数检测峰值
    peaks, properties = find_peaks(neuron_data, 
                                 height=threshold, 
                                 prominence=config.MIN_PROMINENCE,
                                 distance=5)  # 要求峰值之间至少间隔5个时间点
    
    if len(peaks) == 0:
        # 如果没有检测到峰值，返回时间序列的最后一个点
        return neuron_data.index[-1]
    
    # 对每个峰值进行验证，确认是否为真实钙波（上升快，下降慢）
    for peak_idx in peaks:
        # 确保峰值不在时间序列的开始或结束处
        if peak_idx <= 1 or peak_idx >= len(neuron_data) - 2:
            continue
            
        # 计算峰值前的上升速率（取峰值前5个点或更少）
        pre_peak_idx = max(0, peak_idx - 5)
        rise_rate = (neuron_data.iloc[peak_idx] - neuron_data.iloc[pre_peak_idx]) / (peak_idx - pre_peak_idx)
        
        # 计算峰值后的下降速率（取峰值后10个点或更少）
        post_peak_idx = min(len(neuron_data) - 1, peak_idx + 10)
        if post_peak_idx <= peak_idx:
            continue
        
        fall_rate = (neuron_data.iloc[peak_idx] - neuron_data.iloc[post_peak_idx]) / (post_peak_idx - peak_idx)
        
        # 确认是否符合钙波特征：上升快，下降慢
        if rise_rate > config.MIN_RISE_RATE and 0 < fall_rate < config.MAX_FALL_RATE:
            # 找到第一个真实钙波，返回时间点
            return neuron_data.index[peak_idx]
    
    # 如果没有满足条件的钙波，返回时间序列的最后一个点
    return neuron_data.index[-1]

def generate_overall_heatmap(data: pd.DataFrame, config: OverallHeatmapConfig) -> Tuple[Figure, dict]:
    """
    生成整体热力图
    
    参数:
    data -- 包含神经元数据的DataFrame，第一列为stamp（时间戳），其他列为神经元数据
    config -- 配置对象
    
    返回:
    fig -- matplotlib图形对象
    info -- 包含分析信息的字典
    """
    # 复制数据以避免修改原始数据
    day6_data = data.copy()
    
    # 将 'stamp' 列设置为索引
    if 'stamp' in day6_data.columns:
        day6_data = day6_data.set_index('stamp')
    
    # 根据配置的时间戳区间筛选数据
    if config.STAMP_MIN is not None or config.STAMP_MAX is not None:
        # 确定实际的最小值和最大值
        min_stamp = config.STAMP_MIN if config.STAMP_MIN is not None else day6_data.index.min()
        max_stamp = config.STAMP_MAX if config.STAMP_MAX is not None else day6_data.index.max()
        
        # 筛选数据，保留指定区间内的数据
        day6_data = day6_data.loc[min_stamp:max_stamp]
    
    # 检查是否存在 'behavior' 列
    has_behavior = 'behavior' in day6_data.columns
    
    # 分离 'behavior' 列（如果存在）
    if has_behavior:
        frame_lost = day6_data['behavior']
        day6_data = day6_data.drop(columns=['behavior'])
    
    # 数据标准化（Z-score 标准化）
    day6_data_standardized = (day6_data - day6_data.mean()) / day6_data.std()
    
    # 根据排序方式选择相应的排序算法
    if config.SORT_METHOD == 'peak':
        # 原始方法：按峰值时间排序
        # 对于每个神经元，找到其信号达到最大值的时间戳
        peak_times = day6_data_standardized.idxmax()
        
        # 将神经元按照峰值时间从早到晚排序
        sorted_neurons = peak_times.sort_values().index
        
        sort_method_str = "Sorted by peak time"
    else:  # 'calcium_wave'
        # 新方法：按第一次真实钙波发生时间排序
        first_wave_times = {}
        
        # 对每个神经元进行钙波检测
        for neuron in day6_data_standardized.columns:
            neuron_data = day6_data_standardized[neuron]
            first_wave_times[neuron] = detect_first_calcium_wave(neuron_data, config)
        
        # 转换为Series以便排序
        first_wave_times_series = pd.Series(first_wave_times)
        
        # 按第一次钙波时间排序
        sorted_neurons = first_wave_times_series.sort_values().index
        
        sort_method_str = "Sorted by first calcium wave time"
    
    # 根据排序后的神经元顺序重新排列 DataFrame 的列
    sorted_day6_data = day6_data_standardized[sorted_neurons]
    
    # 初始化行为标记变量
    behavior_indices = {}
    unique_behaviors = []
    
    # 只有当behavior列存在时才处理行为标签
    if has_behavior:
        # 获取所有不同的行为标签
        unique_behaviors = frame_lost.dropna().unique()
        
        # 对frame_lost进行处理，找出每种行为连续出现时的第一个时间点
        previous_behavior = None
        for timestamp, behavior in frame_lost.items():
            # 跳过空值
            if pd.isna(behavior):
                continue
            
            # 如果与前一个行为不同，则记录该时间点
            if behavior != previous_behavior:
                if behavior not in behavior_indices:
                    behavior_indices[behavior] = []
                behavior_indices[behavior].append(timestamp)
            
            previous_behavior = behavior
    
    # 创建图形和轴
    fig = plt.figure(figsize=config.FIGURE_SIZE)
    plt.subplots_adjust(left=0.05, right=0.98, top=0.9, bottom=0.15)
    
    # 绘制热图
    ax = sns.heatmap(sorted_day6_data.T, cmap=config.COLORMAP, cbar=True, 
                     vmin=config.VMIN, vmax=config.VMAX)
    
    # 只有当behavior列存在时才添加行为标记
    if has_behavior and len(unique_behaviors) > 0:
        # 颜色映射，为每种行为分配不同的颜色
        colors = plt.cm.tab20(np.linspace(0, 1, len(unique_behaviors)))
        color_map = {behavior: colors[i] for i, behavior in enumerate(unique_behaviors)}
        
        # 只标注CD1行为
        if 'CD1' in behavior_indices:
            for behavior_time in behavior_indices['CD1']:
                # 检查行为时间是否在排序后的数据索引中
                if behavior_time in sorted_day6_data.index:
                    # 获取对应的绘图位置
                    position = sorted_day6_data.index.get_loc(behavior_time)
                    # 绘制垂直线，白色虚线
                    ax.axvline(x=position, color='white', linestyle='--', linewidth=2)
                    # 添加文本标签，放在热图外部并使用黑色字体
                    plt.text(position + 0.5, -5, 'CD1', 
                            color='black', rotation=90, verticalalignment='top', 
                            fontsize=30, fontweight='bold')
    
    # 在第426个时间戳位置添加白色虚线（如果数据足够长）
    if len(sorted_day6_data.index) > 426:
        ax.axvline(x=426, color='white', linestyle='--', linewidth=4)
    
    # 生成标题
    title_text = f'Overall Neural Activity Heatmap ({sort_method_str})'
    if config.STAMP_MIN is not None or config.STAMP_MAX is not None:
        min_stamp = config.STAMP_MIN if config.STAMP_MIN is not None else day6_data.index.min()
        max_stamp = config.STAMP_MAX if config.STAMP_MAX is not None else day6_data.index.max()
        title_text += f' (Time: {min_stamp:.2f} to {max_stamp:.2f})'
    
    plt.title(title_text, fontsize=30)
    plt.xlabel('Time Points', fontsize=30)
    plt.ylabel('Neurons', fontsize=30)
    
    # 修改Y轴标签（神经元标签）的字体大小和粗细，设置为水平方向
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=20, rotation=0)
    
    # 设置X轴刻度，从0开始，以100为间隔
    numpoints = sorted_day6_data.shape[0]  # 获取数据点的总数
    xtick_positions = np.arange(0, numpoints, 100)  # 生成从0开始，间隔为100的刻度位置
    xtick_labels = xtick_positions  # 刻度标签就是位置值
    
    # 设置X轴刻度位置和标签
    ax.set_xticks(xtick_positions)
    ax.set_xticklabels(xtick_labels, fontsize=20)
    
    # 应用紧凑布局
    plt.tight_layout()
    
    # 准备返回信息
    info = {
        'sort_method': config.SORT_METHOD,
        'neuron_count': len(sorted_neurons),
        'time_points': len(sorted_day6_data.index),
        'has_behavior': has_behavior,
        'unique_behaviors': unique_behaviors.tolist() if has_behavior else [],
        'time_range': {
            'min': float(sorted_day6_data.index.min()),
            'max': float(sorted_day6_data.index.max())
        }
    }
    
    return fig, info