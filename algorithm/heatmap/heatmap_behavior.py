"""
特定行为前后时间窗口的神经元活动热力图分析工具

功能：
- 检测指定行为的发生时间点
- 提取从第一个行为开始前指定时间到第二个行为结束的神经元活动数据
- 生成标准化热力图显示钙离子浓度变化
- 支持同一行为或不同行为的组合分析
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
from typing import List, Tuple, Dict, Optional
from scipy.signal import find_peaks
from datetime import datetime

class BehaviorHeatmapConfig:
    """
    行为热力图配置类
    
    Attributes
    ----------
    INPUT_FILE : str
        输入数据文件路径
    OUTPUT_DIR : str
        输出目录路径
    START_BEHAVIOR : str
        起始行为类型
    END_BEHAVIOR : str
        结束行为类型
    PRE_BEHAVIOR_TIME : float
        行为开始前的时间（秒）
    SAMPLING_RATE : float
        采样率（Hz）
    MIN_BEHAVIOR_DURATION : float
        最小行为持续时间（秒）
    """
    
    def __init__(self):
        # 输入文件路径
        self.INPUT_FILE = '../../datasets/29790930糖水铁网糖水trace2.xlsx'
        # 输出目录
        self.OUTPUT_DIR = '../../graph/behavior_heatmaps'
        # 起始行为类型（分析从此行为开始前的时间）
        self.START_BEHAVIOR = 'Crack-seeds-shells'
        # 结束行为类型（分析到此行为结束时刻）
        self.END_BEHAVIOR = 'Eat-seed-kernels'
        # 行为开始前的时间（秒）
        self.PRE_BEHAVIOR_TIME = 10.0
        # 采样率（钙离子数据采样频率：4.8Hz）
        self.SAMPLING_RATE = 4.8
        # 最小行为持续时间（秒），用于过滤短暂的误标记
        self.MIN_BEHAVIOR_DURATION = 1.0
        # 热力图颜色范围（与heatmap_sort-EM.py保持一致）
        self.VMIN = -2
        self.VMAX = 2
        # 神经元排序方式：
        # 'global' - 使用全局排序（基于整个数据集）
        # 'local' - 每个热图单独排序
        # 'first' - 以第一个热图的排序为基准，后续热图使用相同顺序
        # 'custom' - 使用自定义神经元顺序
        self.SORTING_METHOD = 'first'
        
        # 自定义神经元排序顺序（仅在SORTING_METHOD='custom'时使用）
        self.CUSTOM_NEURON_ORDER = ['n53', 'n40', 'n29', 'n34', 'n4', 'n32', 'n25', 'n27', 'n22', 'n55', 'n21', 'n5', 'n19']
        # 配置优先级控制：True表示__init__中的设定优先级最高，False表示命令行参数优先
        self.INIT_CONFIG_PRIORITY = True

def parse_arguments() -> argparse.Namespace:
    """
    解析命令行参数
    
    Returns
    -------
    argparse.Namespace
        解析后的参数对象
    """
    parser = argparse.ArgumentParser(
        description='生成从指定行为开始前到另一行为结束的神经元活动热力图'
    )
    
    parser.add_argument(
        '--input', 
        type=str, 
        help='输入数据文件路径'
    )
    
    parser.add_argument(
        '--output-dir', 
        type=str, 
        help='输出目录路径'
    )
    
    parser.add_argument(
        '--start-behavior', 
        type=str, 
        default='Eat-seed-kernels',
        help='起始行为类型（默认：Eat-seed-kernels）'
    )
    
    parser.add_argument(
        '--end-behavior', 
        type=str, 
        default='Eat-seed-kernels',
        help='结束行为类型（默认：Eat-seed-kernels）'
    )
    
    parser.add_argument(
        '--pre-behavior-time', 
        type=float, 
        default=10.0,
        help='行为开始前的时间，秒（默认：10.0）'
    )
    
    parser.add_argument(
        '--min-duration', 
        type=float, 
        default=1.0,
        help='最小行为持续时间，秒（默认：1.0）'
    )
    
    parser.add_argument(
        '--sampling-rate', 
        type=float, 
        default=4.8,
        help='数据采样率，Hz（默认：4.8）'
    )
    
    parser.add_argument(
        '--sorting-method',
        type=str,
        choices=['global', 'local', 'first', 'custom'],
        help='神经元排序方式：global（全局排序），local（局部排序），first（首图排序），custom（自定义顺序）'
    )
    
    return parser.parse_args()

def convert_timestamps_to_seconds(data: pd.DataFrame, sampling_rate: float) -> pd.DataFrame:
    """
    将时间戳转换为秒为单位的时间序列
    
    Parameters
    ----------
    data : pd.DataFrame
        原始数据，索引为时间戳
    sampling_rate : float
        采样率（Hz）
        
    Returns
    -------
    pd.DataFrame
        转换后的数据，索引为秒
    """
    # 调试信息：查看数据索引的具体情况
    print(f"调试信息：数据索引类型: {type(data.index[0])}")
    print(f"调试信息：前5个索引值: {data.index[:5].tolist()}")
    print(f"调试信息：索引范围: {data.index.min()} - {data.index.max()}")
    
    # 检查时间戳是否为连续的数值序列
    import numbers
    is_numeric = isinstance(data.index[0], (int, float, numbers.Integral, numbers.Real)) or np.issubdtype(data.index.dtype, np.number)
    print(f"调试信息：isinstance检查结果: {is_numeric}")
    
    if is_numeric:
        # 计算时间间隔来判断是否需要转换
        time_intervals = np.diff(data.index[:10])  # 检查前10个时间间隔
        avg_interval = np.mean(time_intervals)
        
        # 如果平均时间间隔接近1/采样率，说明时间戳已经是秒
        expected_interval = 1.0 / sampling_rate
        
        print(f"调试信息：平均时间间隔: {avg_interval:.3f}, 期望秒间隔: {expected_interval:.3f}")
        print(f"调试信息：差值: {abs(avg_interval - expected_interval):.3f}, 阈值: {expected_interval * 0.1:.3f}")
        
        if abs(avg_interval - expected_interval) < expected_interval * 0.1:
            # 时间戳已经是秒，直接返回
            print(f"检测到时间戳已为秒格式（平均间隔: {avg_interval:.3f}s）")
            return data
        else:
            # 时间戳是数据点索引，需要转换为秒
            print(f"检测到时间戳为采样点索引（间隔: {avg_interval:.1f}点），转换为秒")
            
            # 正确的转换：将采样点索引转换为时间（秒）
            # 假设第一个采样点（索引1）对应时间0秒
            time_seconds = (data.index - data.index.min()) / sampling_rate
            
            data_converted = data.copy()
            data_converted.index = time_seconds
            
            print(f"时间转换完成: {data.index.min()}-{data.index.max()}(采样点) → {time_seconds.min():.2f}-{time_seconds.max():.2f}(秒)")
            return data_converted
    else:
        # 如果时间戳已经是时间格式，直接返回
        print(f"检测到时间戳为非数值格式，类型: {type(data.index[0])}")
        return data

def load_and_validate_data(file_path: str) -> pd.DataFrame:
    """
    加载和验证数据文件
    
    Parameters
    ----------
    file_path : str
        数据文件路径
        
    Returns
    -------
    pd.DataFrame
        加载的数据框
        
    Raises
    ------
    FileNotFoundError
        文件不存在时抛出
    ValueError
        数据格式不正确时抛出
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"数据文件不存在: {file_path}")
    
    # 加载数据
    data = pd.read_excel(file_path)
    
    # 验证必要列是否存在
    if 'stamp' not in data.columns:
        raise ValueError("数据文件缺少 'stamp' 列")
    
    if 'behavior' not in data.columns:
        raise ValueError("数据文件缺少 'behavior' 列")
    
    # 设置时间戳为索引
    data = data.set_index('stamp')
    
    return data

def detect_behavior_events(data: pd.DataFrame, 
                          behavior_type: str, 
                          min_duration: float,
                          sampling_rate: float) -> List[Tuple[float, float]]:
    """
    检测行为事件的开始和结束时间
    
    Parameters
    ----------
    data : pd.DataFrame
        包含行为标签的数据
    behavior_type : str
        目标行为类型
    min_duration : float
        最小行为持续时间（秒）
    sampling_rate : float
        采样率（Hz）
        
    Returns
    -------
    List[Tuple[float, float]]
        行为事件列表，每个元素为(开始时间, 结束时间)
    """
    behavior_column = data['behavior']
    events = []
    
    # 找到行为变化点
    behavior_changes = behavior_column != behavior_column.shift(1)
    change_indices = behavior_changes[behavior_changes].index
    
    current_behavior = None
    start_time = None
    
    for timestamp in change_indices:
        new_behavior = behavior_column.loc[timestamp]
        
        # 如果当前行为结束
        if current_behavior == behavior_type and new_behavior != behavior_type:
            if start_time is not None:
                duration = timestamp - start_time
                # 只保留持续时间足够长的行为事件
                if duration >= min_duration:
                    events.append((start_time, timestamp))
        
        # 如果目标行为开始
        if new_behavior == behavior_type and current_behavior != behavior_type:
            start_time = timestamp
        
        current_behavior = new_behavior
    
    # 处理最后一个行为事件（如果数据在行为期间结束）
    if current_behavior == behavior_type and start_time is not None:
        end_time = data.index[-1]
        duration = end_time - start_time
        if duration >= min_duration:
            events.append((start_time, end_time))
    
    return events

def find_behavior_pairs(data: pd.DataFrame,
                       start_behavior: str,
                       end_behavior: str,
                       min_duration: float,
                       sampling_rate: float) -> List[Tuple[float, float, float, float]]:
    """
    找到起始行为和结束行为的连续配对
    
    只有当起始行为和结束行为连续出现（或为同一行为）时才返回配对。
    
    Parameters
    ----------
    data : pd.DataFrame
        包含行为标签的数据
    start_behavior : str
        起始行为类型
    end_behavior : str
        结束行为类型
    min_duration : float
        最小行为持续时间（秒）
    sampling_rate : float
        采样率（Hz）
        
    Returns
    -------
    List[Tuple[float, float, float, float]]
        连续行为配对列表，每个元素为(起始行为开始时间, 起始行为结束时间, 结束行为开始时间, 结束行为结束时间)
    """
    # 检测起始行为事件
    start_events = detect_behavior_events(data, start_behavior, min_duration, sampling_rate)
    
    if not start_events:
        print(f"未找到 {start_behavior} 行为事件")
        return []
    
    behavior_pairs = []
    
    # 如果是同一行为，直接使用每个事件的开始和结束时间
    if start_behavior == end_behavior:
        print(f"分析同一行为: {start_behavior}")
        for start_time, end_time in start_events:
            behavior_pairs.append((start_time, start_time, end_time, end_time))
    else:
        # 如果是不同行为，需要检查行为的连续性
        print(f"分析连续行为: {start_behavior} → {end_behavior}")
        
        # 检测结束行为事件
        end_events = detect_behavior_events(data, end_behavior, min_duration, sampling_rate)
        
        if not end_events:
            print(f"未找到 {end_behavior} 行为事件")
            return []
        
        # 获取完整的行为序列，用于检查连续性
        behavior_column = data['behavior']
        
        for start_begin, start_end in start_events:
            # 检查起始行为结束后紧接着的下一个行为是否为结束行为
            continuous_pair = check_behavior_continuity(
                behavior_column, 
                start_begin, 
                start_end, 
                end_behavior, 
                min_duration
            )
            
            if continuous_pair:
                end_begin, end_end = continuous_pair
                behavior_pairs.append((start_begin, start_end, end_begin, end_end))
                print(f"找到连续行为配对: {start_behavior}({start_begin:.1f}s-{start_end:.1f}s) → {end_behavior}({end_begin:.1f}s-{end_end:.1f}s)")
            else:
                print(f"跳过非连续行为: {start_behavior}({start_begin:.1f}s-{start_end:.1f}s) 后没有紧接着 {end_behavior}")
    
    return behavior_pairs

def check_behavior_continuity(behavior_column: pd.Series,
                            start_begin: float,
                            start_end: float,
                            target_end_behavior: str,
                            min_duration: float) -> Optional[Tuple[float, float]]:
    """
    检查起始行为结束后是否紧接着目标结束行为
    
    Parameters
    ----------
    behavior_column : pd.Series
        行为标签时间序列
    start_begin : float
        起始行为开始时间
    start_end : float
        起始行为结束时间
    target_end_behavior : str
        目标结束行为类型
    min_duration : float
        最小行为持续时间（秒）
        
    Returns
    -------
    Optional[Tuple[float, float]]
        如果连续，返回(结束行为开始时间, 结束行为结束时间)；否则返回None
    """
    # 从起始行为结束时间开始查找下一个行为
    after_start_data = behavior_column[behavior_column.index > start_end]
    
    if after_start_data.empty:
        return None
    
    # 找到起始行为结束后的第一个非空行为标签
    next_behavior = None
    next_behavior_start = None
    
    for timestamp, behavior in after_start_data.items():
        if pd.notna(behavior):  # 跳过空值
            next_behavior = behavior
            next_behavior_start = timestamp
            break
    
    # 检查下一个行为是否是目标结束行为
    if next_behavior != target_end_behavior:
        return None
    
    # 找到这个结束行为的结束时间
    end_behavior_data = after_start_data[after_start_data == target_end_behavior]
    if end_behavior_data.empty:
        return None
    
    # 找到连续的结束行为区间
    behavior_start = next_behavior_start
    behavior_end = None
    
    # 从结束行为开始位置向后查找，直到行为改变
    current_time = behavior_start
    for timestamp, behavior in after_start_data[after_start_data.index >= behavior_start].items():
        if pd.notna(behavior) and behavior == target_end_behavior:
            current_time = timestamp
        elif pd.notna(behavior) and behavior != target_end_behavior:
            # 行为改变了，结束时间是上一个时间点
            behavior_end = current_time
            break
        # 如果是空值，继续当前行为
    
    # 如果没有找到行为改变点，说明数据在这个行为中结束
    if behavior_end is None:
        behavior_end = after_start_data.index[-1]
    
    # 检查持续时间是否满足最小要求
    duration = behavior_end - behavior_start
    if duration < min_duration:
        return None
    
    return (behavior_start, behavior_end)

def extract_behavior_sequence_data(data: pd.DataFrame,
                                 start_time: float,
                                 end_time: float,
                                 pre_behavior_time: float,
                                 sampling_rate: float = 4.8) -> Optional[pd.DataFrame]:
    """
    提取从行为开始前到行为结束的数据序列
    
    Parameters
    ----------
    data : pd.DataFrame
        完整的神经元数据（时间戳为索引）
    start_time : float
        起始行为开始时间戳
    end_time : float
        结束行为结束时间戳
    pre_behavior_time : float
        行为开始前的时间（秒）
    sampling_rate : float
        采样率（Hz），用于计算时间戳偏移量
        
    Returns
    -------
    Optional[pd.DataFrame]
        提取的数据序列，如果时间范围超出数据范围则返回None
    """
    # 将行为开始前的时间（秒）转换为时间戳偏移量
    pre_behavior_timestamps = pre_behavior_time * sampling_rate
    sequence_start = start_time - pre_behavior_timestamps
    sequence_end = end_time
    
    # 检查时间范围是否在数据范围内
    if sequence_start < data.index.min() or sequence_end > data.index.max():
        print(f"时间范围 {sequence_start:.1f}s - {sequence_end:.1f}s 超出数据范围 {data.index.min():.1f}s - {data.index.max():.1f}s")
        return None
    
    # 提取时间序列内的数据
    sequence_data = data.loc[sequence_start:sequence_end].copy()
    
    # 移除行为列（如果存在）
    if 'behavior' in sequence_data.columns:
        sequence_data = sequence_data.drop(columns=['behavior'])
    
    return sequence_data

def calculate_global_standardization_params(data: pd.DataFrame) -> Dict[str, Tuple[float, float]]:
    """
    计算整体数据的全局标准化参数（均值和标准差）
    
    Parameters
    ----------
    data : pd.DataFrame
        完整的神经元数据
        
    Returns
    -------
    Dict[str, Tuple[float, float]]
        每个神经元的全局标准化参数，格式为 {神经元名: (均值, 标准差)}
    """
    # 移除行为列（如果存在）
    neural_data = data.copy()
    if 'behavior' in neural_data.columns:
        neural_data = neural_data.drop(columns=['behavior'])
    
    # 只处理数值列
    numeric_columns = neural_data.select_dtypes(include=[np.number]).columns
    
    global_params = {}
    for col in numeric_columns:
        mean_val = neural_data[col].mean()
        std_val = neural_data[col].std()
        global_params[col] = (mean_val, std_val)
    
    return global_params

def standardize_neural_data(data: pd.DataFrame, 
                           global_params: Optional[Dict[str, Tuple[float, float]]] = None) -> pd.DataFrame:
    """
    对神经元数据进行Z-score标准化
    
    Parameters
    ----------
    data : pd.DataFrame
        原始神经元数据
    global_params : Optional[Dict[str, Tuple[float, float]]]
        全局标准化参数。如果提供，使用全局参数；否则基于当前数据计算
        
    Returns
    -------
    pd.DataFrame
        标准化后的数据
    """
    # 只对数值列进行标准化
    numeric_columns = data.select_dtypes(include=[np.number]).columns
    standardized_data = data.copy()
    
    for col in numeric_columns:
        if global_params is not None and col in global_params:
            # 使用全局标准化参数
            mean_val, std_val = global_params[col]
        else:
            # 基于当前数据计算标准化参数（原有逻辑）
            mean_val = data[col].mean()
            std_val = data[col].std()
        
        if std_val != 0:  # 避免除零错误
            standardized_data[col] = (data[col] - mean_val) / std_val
        else:
            standardized_data[col] = 0
    
    return standardized_data

def get_global_neuron_order(data: pd.DataFrame, 
                          global_standardization_params: Dict[str, Tuple[float, float]]) -> pd.Index:
    """
    根据整体数据计算全局神经元排序顺序（按峰值时间排序）
    
    Parameters
    ----------
    data : pd.DataFrame
        完整的神经元数据
    global_standardization_params : Dict[str, Tuple[float, float]]
        全局标准化参数
        
    Returns
    -------
    pd.Index
        按峰值时间排序的神经元顺序
    """
    # 移除行为列（如果存在）
    neural_data = data.copy()
    if 'behavior' in neural_data.columns:
        neural_data = neural_data.drop(columns=['behavior'])
    
    # 使用全局标准化参数对整体数据进行标准化
    neural_data_standardized = standardize_neural_data(neural_data, global_standardization_params)
    
    # 计算每个神经元的峰值时间
    peak_times = neural_data_standardized.idxmax()
    
    # 按峰值时间排序，返回神经元顺序
    sorted_neurons = peak_times.sort_values().index
    
    return sorted_neurons

def apply_global_neuron_order(data: pd.DataFrame, global_order: pd.Index) -> pd.DataFrame:
    """
    按照全局神经元顺序重新排列数据
    
    Parameters
    ----------
    data : pd.DataFrame
        神经元数据
    global_order : pd.Index
        全局神经元排序顺序
        
    Returns
    -------
    pd.DataFrame
        按全局顺序排列的数据
    """
    # 只保留在全局顺序中的神经元列
    available_neurons = [neuron for neuron in global_order if neuron in data.columns]
    
    return data[available_neurons]

def sort_neurons_by_local_peak_time(data: pd.DataFrame) -> pd.DataFrame:
    """
    按当前数据窗口内的神经元峰值时间排序（局部排序）
    
    Parameters
    ----------
    data : pd.DataFrame
        标准化的神经元数据
        
    Returns
    -------
    pd.DataFrame
        按峰值时间排序的数据
    """
    # 计算每个神经元在当前时间窗口内的峰值时间
    peak_times = data.idxmax()
    
    # 按峰值时间排序
    sorted_neurons = peak_times.sort_values().index
    
    return data[sorted_neurons]

def sort_neurons_by_custom_order(data: pd.DataFrame, custom_order: List[str]) -> pd.DataFrame:
    """
    按自定义神经元顺序排序
    
    指定神经元按给定顺序排在前面，剩余神经元按字符串排序排在后面
    
    Parameters
    ----------
    data : pd.DataFrame
        神经元数据
    custom_order : List[str]
        自定义的神经元顺序列表
        
    Returns
    -------
    pd.DataFrame
        按自定义顺序排列的数据
    """
    available_neurons = set(data.columns)
    
    # 首先按照自定义顺序排列存在的神经元
    ordered_neurons = []
    for neuron in custom_order:
        if neuron in available_neurons:
            ordered_neurons.append(neuron)
    
    # 找出剩余的神经元，按字符串大小顺序排列
    remaining_neurons = sorted(list(available_neurons - set(ordered_neurons)))
    
    # 合并两部分：自定义顺序 + 剩余神经元（按大小排序）
    final_order = ordered_neurons + remaining_neurons
    
    return data[final_order]

def create_behavior_sequence_heatmap(data: pd.DataFrame,
                                   start_behavior_time: float,
                                   end_behavior_time: float,
                                   start_behavior: str,
                                   end_behavior: str,
                                   pre_behavior_time: float,
                                   config: BehaviorHeatmapConfig,
                                   pair_index: int,
                                   global_neuron_order: Optional[pd.Index] = None,
                                   first_heatmap_order: Optional[pd.Index] = None) -> Tuple[plt.Figure, Optional[pd.Index]]:
    """
    创建行为序列的热力图
    
    Parameters
    ----------
    data : pd.DataFrame
        行为序列内的神经元数据
    start_behavior_time : float
        起始行为开始时间
    end_behavior_time : float
        结束行为结束时间
    start_behavior : str
        起始行为类型
    end_behavior : str
        结束行为类型
    pre_behavior_time : float
        行为开始前的时间
    config : BehaviorHeatmapConfig
        配置对象
    pair_index : int
        配对索引
    global_neuron_order : Optional[pd.Index]
        全局神经元排序顺序（仅在使用全局排序时需要）
    first_heatmap_order : Optional[pd.Index]
        第一个热图的排序顺序（仅在使用首图排序时需要）
        
    Returns
    -------
    Tuple[plt.Figure, Optional[pd.Index]]
        生成的图形对象和当前热图的神经元排序顺序
    """
    # 根据配置选择排序方式
    if config.SORTING_METHOD == 'global' and global_neuron_order is not None:
        # 使用全局排序
        data_ordered = apply_global_neuron_order(data, global_neuron_order)
        sorting_method = "global"
        current_order = global_neuron_order
    elif config.SORTING_METHOD == 'custom':
        # 使用自定义排序
        data_ordered = sort_neurons_by_custom_order(data, config.CUSTOM_NEURON_ORDER)
        sorting_method = "custom"
        current_order = data_ordered.columns
    elif config.SORTING_METHOD == 'first' and first_heatmap_order is not None:
        # 使用首图排序
        data_ordered = apply_global_neuron_order(data, first_heatmap_order)
        sorting_method = "first heatmap"
        current_order = first_heatmap_order
    elif config.SORTING_METHOD == 'first' and pair_index == 0:
        # 第一个热图，创建首图排序
        data_ordered = sort_neurons_by_local_peak_time(data)
        sorting_method = "first heatmap"
        current_order = data_ordered.columns
    else:
        # 使用局部排序
        data_ordered = sort_neurons_by_local_peak_time(data)
        sorting_method = "local"
        current_order = data_ordered.columns
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(16, 60))
    
    # 绘制热力图
    sns.heatmap(
        data_ordered.T,  # 转置：行为神经元，列为时间
        cmap='viridis',
        cbar=False,  # 显示颜色图例
        vmin=config.VMIN,
        vmax=config.VMAX,
        ax=ax
    )
    
    # 计算关键时间点的位置
    sequence_start_time = start_behavior_time - pre_behavior_time
    total_duration = end_behavior_time - sequence_start_time
    
    # 起始行为开始位置
    start_position = len(data_ordered.index) * pre_behavior_time / total_duration
    # 结束行为结束位置（序列结束）
    end_position = len(data_ordered.index) - 1
    
    # 在关键时间点画垂直线
    ax.axvline(x=start_position, color='black', linestyle='--', linewidth=5, alpha=0.9)
    ax.axvline(x=end_position, color='black', linestyle='-', linewidth=5, alpha=0.9)
    
    # 添加文本标注
    ax.text(start_position + 1, -3, f'{start_behavior} Start', 
           color='black', fontweight='bold', fontsize=12, 
           bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    ax.text(end_position - 10, -3, f'{end_behavior} End', 
           color='black', fontweight='bold', fontsize=12,
           bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    # 设置标题和标签
    if sorting_method == "global":
        sorting_description = "global peak time"
    elif sorting_method == "first heatmap":
        sorting_description = "first heatmap order"
    elif sorting_method == "custom":
        sorting_description = "custom order"
    else:
        sorting_description = "local peak time"
    
    if start_behavior == end_behavior:
        title = f'{start_behavior} Behavior Sequence #{pair_index + 1}\n'
        # title += f'Neural Activity: -{pre_behavior_time}s to End (Neurons sorted by {sorting_description})'
    else:
        title = f'{start_behavior} → {end_behavior} Behavior Sequence #{pair_index + 1}\n'
        # title += f'Neural Activity: {start_behavior} -{pre_behavior_time}s to {end_behavior} End (Neurons sorted by {sorting_description})'
    
    ax.set_title(title, fontsize=25, fontweight='bold')
    ax.set_xlabel('Time (seconds)', fontsize=30, fontweight='bold')
    ax.set_ylabel(f'Neurons (sorted by {sorting_description})', fontsize=30, fontweight='bold')
    
    # 设置X轴刻度标签（显示相对于起始行为开始的时间，以秒为单位，5秒为一个刻度）
    tick_positions, tick_labels = calculate_5second_ticks(
        data_ordered.index, 
        start_behavior_time, 
        config.SAMPLING_RATE
    )
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, fontsize=30, fontweight='bold')
    
    # 设置Y轴显示所有神经元编号
    neuron_positions = np.arange(len(data_ordered.columns))
    neuron_labels = data_ordered.columns.tolist()
    ax.set_yticks(neuron_positions)
    ax.set_yticklabels(neuron_labels, rotation=0, fontsize=30, fontweight='bold')
    
    plt.tight_layout()
    return fig, current_order

def create_average_sequence_heatmap(all_sequence_data: List[pd.DataFrame],
                                  start_behavior: str,
                                  end_behavior: str,
                                  pre_behavior_time: float,
                                  config: BehaviorHeatmapConfig,
                                  global_neuron_order: Optional[pd.Index] = None,
                                  first_heatmap_order: Optional[pd.Index] = None) -> plt.Figure:
    """
    创建所有行为序列的平均热力图
    
    Parameters
    ----------
    all_sequence_data : List[pd.DataFrame]
        所有行为序列数据的列表
    start_behavior : str
        起始行为类型
    end_behavior : str
        结束行为类型
    pre_behavior_time : float
        行为开始前的时间
    config : BehaviorHeatmapConfig
        配置对象
    global_neuron_order : Optional[pd.Index]
        全局神经元排序顺序（仅在使用全局排序时需要）
    first_heatmap_order : Optional[pd.Index]
        第一个热图的排序顺序（仅在使用首图排序时需要）
        
    Returns
    -------
    plt.Figure
        平均热力图
    """
    if not all_sequence_data:
        raise ValueError("没有有效的行为序列数据")
    
    # 根据配置选择排序方式获取公共神经元
    if config.SORTING_METHOD == 'global' and global_neuron_order is not None:
        # 使用全局排序：找到所有数据的公共神经元，并按照全局顺序排列
        all_available_neurons = set()
        for data in all_sequence_data:
            all_available_neurons.update(data.columns)
        
        # 按照全局顺序筛选出可用的神经元
        common_neurons = [neuron for neuron in global_neuron_order if neuron in all_available_neurons]
        
        # 确保每个序列数据都包含这些神经元
        for data in all_sequence_data:
            missing_neurons = set(common_neurons) - set(data.columns)
            if missing_neurons:
                common_neurons = [neuron for neuron in common_neurons if neuron in data.columns]
        
        sorting_method = "global"
    elif config.SORTING_METHOD == 'custom':
        # 使用自定义排序：基于自定义神经元顺序
        all_available_neurons = set()
        for data in all_sequence_data:
            all_available_neurons.update(data.columns)
        
        # 按照自定义顺序排列神经元：优先顺序 + 剩余按字符串排序
        ordered_neurons = []
        for neuron in config.CUSTOM_NEURON_ORDER:
            if neuron in all_available_neurons:
                ordered_neurons.append(neuron)
        
        # 添加剩余神经元（按字符串排序）
        remaining_neurons = sorted(list(all_available_neurons - set(ordered_neurons)))
        common_neurons = ordered_neurons + remaining_neurons
        
        # 确保每个序列数据都包含这些神经元
        for data in all_sequence_data:
            missing_neurons = set(common_neurons) - set(data.columns)
            if missing_neurons:
                common_neurons = [neuron for neuron in common_neurons if neuron in data.columns]
        
        sorting_method = "custom"
    elif config.SORTING_METHOD == 'first' and first_heatmap_order is not None:
        # 使用首图排序：基于第一个热图的排序顺序
        all_available_neurons = set()
        for data in all_sequence_data:
            all_available_neurons.update(data.columns)
        
        # 按照首图顺序筛选出可用的神经元
        common_neurons = [neuron for neuron in first_heatmap_order if neuron in all_available_neurons]
        
        # 确保每个序列数据都包含这些神经元
        for data in all_sequence_data:
            missing_neurons = set(common_neurons) - set(data.columns)
            if missing_neurons:
                common_neurons = [neuron for neuron in common_neurons if neuron in data.columns]
        
        sorting_method = "first heatmap"
    else:
        # 使用局部排序：基于平均数据计算排序
        # 先找到所有数据的公共神经元
        common_neurons = set(all_sequence_data[0].columns)
        for data in all_sequence_data[1:]:
            common_neurons &= set(data.columns)
        
        common_neurons = sorted(list(common_neurons))
        sorting_method = "local"
    
    # 确定目标长度（使用最短序列的长度以确保所有数据都有效）
    min_length = min(len(data) for data in all_sequence_data)
    aligned_data = []
    
    for data in all_sequence_data:
        # 只保留公共神经元
        data_subset = data[common_neurons]
        
        # 重采样到统一长度
        if len(data_subset) != min_length:
            # 创建新的索引
            new_index = np.linspace(0, len(data_subset)-1, min_length)
            original_index = np.arange(len(data_subset))
            
            # 对每个神经元进行插值
            resampled_data = np.zeros((min_length, len(common_neurons)))
            for j, neuron in enumerate(common_neurons):
                resampled_data[:, j] = np.interp(new_index, original_index, data_subset[neuron].values)
            
            aligned_data.append(resampled_data)
        else:
            aligned_data.append(data_subset.values)
    
    # 计算平均值
    average_data = np.mean(aligned_data, axis=0)
    
    # 创建平均数据的DataFrame
    # 使用相对于起始行为开始的时间戳形式，但在显示时会转换为秒
    time_relative_timestamps = np.linspace(-pre_behavior_time * config.SAMPLING_RATE, 0, min_length)  # 相对于起始行为开始的时间戳
    average_df = pd.DataFrame(average_data, 
                             index=time_relative_timestamps, 
                             columns=common_neurons)
    
    # 根据排序方式处理数据
    if sorting_method == "global":
        # 全局排序：数据已经按照全局顺序排列，无需再次排序
        average_df_sorted = average_df
    elif sorting_method == "first heatmap":
        # 首图排序：数据已经按照首图顺序排列，无需再次排序
        average_df_sorted = average_df
    elif sorting_method == "custom":
        # 自定义排序：数据已经按照自定义顺序排列，无需再次排序
        average_df_sorted = average_df
    else:
        # 局部排序：基于平均数据重新排序
        average_df_sorted = sort_neurons_by_local_peak_time(average_df)
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(16, 60))
    
    # 绘制热力图
    sns.heatmap(
        average_df_sorted.T,
        cmap='viridis',
        cbar=False,  # 显示颜色图例
        vmin=config.VMIN,
        vmax=config.VMAX,
        ax=ax
    )
    
    # 在起始行为开始时间点画垂直线
    start_position = len(average_df_sorted) * pre_behavior_time / (pre_behavior_time + 0)  # 在序列中的相对位置
    ax.axvline(x=start_position, color='black', linestyle='--', linewidth=5, alpha=0.9)
    
    # 添加文本标注
    ax.text(start_position + 1, -3, f'{start_behavior} Start', 
           color='black', fontweight='bold', fontsize=12,
           bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    # 设置标题和标签
    if sorting_method == "global":
        sorting_description = "global peak time"
    elif sorting_method == "first heatmap":
        sorting_description = "first heatmap order"
    elif sorting_method == "custom":
        sorting_description = "custom order"
    else:
        sorting_description = "local peak time"
    
    if start_behavior == end_behavior:
        title = f'Average {start_behavior} Behavior Sequence\n'
        # title += f'Neural Activity: -{pre_behavior_time}s to End (n={len(all_sequence_data)} sequences, Neurons sorted by {sorting_description})'
    else:
        title = f'Average {start_behavior} → {end_behavior} Behavior Sequence\n'
        # title += f'Neural Activity: {start_behavior} -{pre_behavior_time}s to {end_behavior} End (n={len(all_sequence_data)} sequences, Neurons sorted by {sorting_description})'
    
    ax.set_title(title, fontsize=15, fontweight='bold')
    ax.set_xlabel('Time (seconds)', fontsize=25, fontweight='bold')
    ax.set_ylabel(f'Neurons (sorted by {sorting_description})', fontsize=25, fontweight='bold')
    
    # 设置X轴刻度标签（以秒为单位，5秒为一个刻度）
    # 为平均热力图设置参考时间为0（起始行为开始时间）
    tick_positions, tick_labels = calculate_5second_ticks(
        average_df_sorted.index, 
        0,  # 参考时间为0，因为time_relative_timestamps已经是相对时间
        config.SAMPLING_RATE
    )
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, fontsize=25, fontweight='bold')
    
    # 设置Y轴显示所有神经元编号
    neuron_positions = np.arange(len(average_df_sorted.columns))
    neuron_labels = average_df_sorted.columns.tolist()
    ax.set_yticks(neuron_positions)
    ax.set_yticklabels(neuron_labels, rotation=0, fontsize=25, fontweight='bold')
    
    plt.tight_layout()
    return fig

def calculate_5second_ticks(data_index: pd.Index, 
                           reference_time: float, 
                           sampling_rate: float) -> Tuple[List[float], List[str]]:
    """
    计算基于5秒间隔的X轴刻度位置和标签
    
    Parameters
    ----------
    data_index : pd.Index
        数据的时间戳索引
    reference_time : float
        参考时间戳（如行为开始时间）
    sampling_rate : float
        采样率（Hz）
        
    Returns
    -------
    Tuple[List[float], List[str]]
        刻度位置和对应的标签
    """
    # 将时间戳转换为相对于参考时间的秒数
    time_points_seconds = (data_index - reference_time) / sampling_rate
    
    # 确定5秒刻度的范围
    min_time = time_points_seconds.min()
    max_time = time_points_seconds.max()
    
    # 生成5秒间隔的刻度点
    tick_times_seconds = np.arange(
        np.floor(min_time / 5) * 5,  # 起始点向下取整到5的倍数
        np.ceil(max_time / 5) * 5 + 5,  # 结束点向上取整到5的倍数
        5  # 5秒间隔
    )
    
    # 将秒数转换为热力图中的像素位置
    tick_positions = []
    tick_labels = []
    
    for tick_time in tick_times_seconds:
        if min_time <= tick_time <= max_time:
            # 计算该时间点在热力图中的位置
            relative_position = (tick_time - min_time) / (max_time - min_time)
            pixel_position = relative_position * (len(data_index) - 1)
            tick_positions.append(pixel_position)
            tick_labels.append(f'{tick_time:.0f}')
    
    return tick_positions, tick_labels

def main():
    """
    主函数：执行行为序列热力图分析流程
    """
    # 解析命令行参数
    args = parse_arguments()
    
    # 创建配置对象
    config = BehaviorHeatmapConfig()
    
    # 保存__init__中的行为设定
    init_start_behavior = config.START_BEHAVIOR
    init_end_behavior = config.END_BEHAVIOR
    init_priority = config.INIT_CONFIG_PRIORITY
    
    # 更新其他配置项
    if args.input:
        config.INPUT_FILE = args.input
    if args.output_dir:
        config.OUTPUT_DIR = args.output_dir
    if args.pre_behavior_time:
        config.PRE_BEHAVIOR_TIME = args.pre_behavior_time
    if args.min_duration:
        config.MIN_BEHAVIOR_DURATION = args.min_duration
    if args.sampling_rate:
        config.SAMPLING_RATE = args.sampling_rate
    
    # 处理排序方式参数
    if args.sorting_method:
        config.SORTING_METHOD = args.sorting_method
    
    # 行为设定优先级控制
    if init_priority:
        # __init__中的设定具有最高优先级
        config.START_BEHAVIOR = init_start_behavior
        config.END_BEHAVIOR = init_end_behavior
        print(f"使用起始行为: {config.START_BEHAVIOR}, 结束行为: {config.END_BEHAVIOR} (来源: __init__配置，优先级最高)")
    else:
        # 命令行参数优先
        if args.start_behavior:
            config.START_BEHAVIOR = args.start_behavior
        if args.end_behavior:
            config.END_BEHAVIOR = args.end_behavior
        print(f"使用起始行为: {config.START_BEHAVIOR}, 结束行为: {config.END_BEHAVIOR} (来源: 命令行参数)")
    
    # 创建输出目录
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    
    try:
        # 加载数据
        print(f"正在加载数据: {config.INPUT_FILE}")
        data = load_and_validate_data(config.INPUT_FILE)
        
        # 保持原始时间戳格式，不进行转换
        print(f"保持原始时间戳格式（采样率: {config.SAMPLING_RATE}Hz）")
        print(f"数据加载成功，包含 {len(data)} 个时间点，时间戳范围: {data.index.min():.0f} - {data.index.max():.0f}")
        
        # 计算全局标准化参数（确保所有片段使用相同的标准化参数）
        print("正在计算全局标准化参数...")
        global_standardization_params = calculate_global_standardization_params(data)
        print(f"全局标准化参数计算完成，共 {len(global_standardization_params)} 个神经元")
        
        # 根据配置决定是否计算全局神经元排序顺序
        global_neuron_order = None
        first_heatmap_order = None
        
        if config.SORTING_METHOD == 'global':
            print("正在计算全局神经元排序顺序...")
            global_neuron_order = get_global_neuron_order(data, global_standardization_params)
            print(f"全局排序完成，共 {len(global_neuron_order)} 个神经元按峰值时间排序")
        elif config.SORTING_METHOD == 'custom':
            print(f"使用自定义神经元排序模式")
            print(f"指定顺序: {config.CUSTOM_NEURON_ORDER}")
            print("剩余神经元将按字符串大小顺序排列在指定神经元下方")
        elif config.SORTING_METHOD == 'first':
            print("使用首图排序模式，将以第一个热图的排序为基准，后续热图使用相同神经元顺序")
        else:
            print("使用局部排序模式，每个热图将根据当前时间窗口内的峰值时间独立排序")
        
        # 查找连续行为配对
        if config.START_BEHAVIOR == config.END_BEHAVIOR:
            print(f"正在查找 {config.START_BEHAVIOR} 行为事件...")
        else:
            print(f"正在查找连续行为配对: {config.START_BEHAVIOR} → {config.END_BEHAVIOR}")
            print("注意：只有连续出现的行为配对才会被处理")
        
        behavior_pairs = find_behavior_pairs(
            data,
            config.START_BEHAVIOR,
            config.END_BEHAVIOR,
            config.MIN_BEHAVIOR_DURATION,
            config.SAMPLING_RATE
        )
        
        if not behavior_pairs:
            if config.START_BEHAVIOR == config.END_BEHAVIOR:
                print(f"未找到符合条件的 {config.START_BEHAVIOR} 行为事件")
            else:
                print(f"未找到符合条件的连续行为配对: {config.START_BEHAVIOR} → {config.END_BEHAVIOR}")
                print("提示：请检查这两个行为是否在数据中连续出现")
            return
        
        if config.START_BEHAVIOR == config.END_BEHAVIOR:
            print(f"找到 {len(behavior_pairs)} 个 {config.START_BEHAVIOR} 行为事件")
        else:
            print(f"找到 {len(behavior_pairs)} 个连续行为配对")
            print("所有配对都满足连续性要求")
        
        # 分析每个行为配对
        all_sequence_data = []
        
        for i, (start_begin, start_end, end_begin, end_end) in enumerate(behavior_pairs):
            if config.START_BEHAVIOR == config.END_BEHAVIOR:
                print(f"正在分析序列 {i+1}: {config.START_BEHAVIOR} (时间戳 {start_begin:.0f} - {end_end:.0f})")
                sequence_start_time = start_begin
                sequence_end_time = end_end
            else:
                print(f"正在分析序列 {i+1}: {config.START_BEHAVIOR} (时间戳 {start_begin:.0f} - {start_end:.0f}) → {config.END_BEHAVIOR} (时间戳 {end_begin:.0f} - {end_end:.0f})")
                sequence_start_time = start_begin
                sequence_end_time = end_end
            
            # 提取行为序列数据
            sequence_data = extract_behavior_sequence_data(
                data,
                sequence_start_time,
                sequence_end_time,
                config.PRE_BEHAVIOR_TIME,
                config.SAMPLING_RATE
            )
            
            if sequence_data is None:
                print(f"序列 {i+1} 的时间范围超出数据范围，跳过")
                continue
            
            # 使用全局标准化参数进行标准化（确保与整体热图一致的对比度）
            standardized_data = standardize_neural_data(sequence_data, global_standardization_params)
            
            # 根据配置选择排序方式（在创建热力图时才排序，这里保持原始数据）
            sorted_data = standardized_data
            
            # 保存用于平均计算
            all_sequence_data.append(sorted_data)
            
            # 创建单个序列的热力图
            fig, current_order = create_behavior_sequence_heatmap(
                sorted_data,
                sequence_start_time,
                sequence_end_time,
                config.START_BEHAVIOR,
                config.END_BEHAVIOR,
                config.PRE_BEHAVIOR_TIME,
                config,
                i,
                global_neuron_order,
                first_heatmap_order
            )
            
            # 如果是首图排序且这是第一个热图，保存排序顺序
            if config.SORTING_METHOD == 'first' and i == 0:
                first_heatmap_order = current_order
                print(f"第一个热图排序顺序已确定，共 {len(current_order)} 个神经元")
            
            # 保存图形
            if config.START_BEHAVIOR == config.END_BEHAVIOR:
                output_path = os.path.join(
                    config.OUTPUT_DIR,
                    f'{config.START_BEHAVIOR}_sequence_{i+1}_heatmap.png'
                )
            else:
                output_path = os.path.join(
                    config.OUTPUT_DIR,
                    f'{config.START_BEHAVIOR}_to_{config.END_BEHAVIOR}_sequence_{i+1}_heatmap.png'
                )
            
            fig.savefig(output_path, bbox_inches='tight', pad_inches=0.1, dpi=100)
            plt.close(fig)
            print(f"已保存: {output_path}")
        
        # 创建平均热力图
        if all_sequence_data:
            print("正在创建平均热力图...")
            avg_fig = create_average_sequence_heatmap(
                all_sequence_data,
                config.START_BEHAVIOR,
                config.END_BEHAVIOR,
                config.PRE_BEHAVIOR_TIME,
                config,
                global_neuron_order,
                first_heatmap_order
            )
            
            # 保存平均热力图
            if config.START_BEHAVIOR == config.END_BEHAVIOR:
                avg_output_path = os.path.join(
                    config.OUTPUT_DIR,
                    f'{config.START_BEHAVIOR}_average_sequence_heatmap.png'
                )
            else:
                avg_output_path = os.path.join(
                    config.OUTPUT_DIR,
                    f'{config.START_BEHAVIOR}_to_{config.END_BEHAVIOR}_average_sequence_heatmap.png'
                )
            
            avg_fig.savefig(avg_output_path, bbox_inches='tight', pad_inches=0.1, dpi=100)
            plt.close(avg_fig)
            print(f"已保存平均热力图: {avg_output_path}")
        
        print(f"分析完成！共处理 {len(all_sequence_data)} 个有效序列")
        
    except Exception as e:
        print(f"错误: {e}")
        raise

if __name__ == "__main__":
    main()

"""
使用示例和配置说明：

1. 分析同一行为（从开始前10秒到结束）：
   ```python
   class BehaviorHeatmapConfig:
       def __init__(self):
           self.START_BEHAVIOR = 'Eat-seed-kernels'  # 起始行为
           self.END_BEHAVIOR = 'Eat-seed-kernels'    # 结束行为（同一行为）
           self.PRE_BEHAVIOR_TIME = 10.0             # 行为开始前10秒
           self.SAMPLING_RATE = 4.8                  # 采样率（Hz）
           self.INIT_CONFIG_PRIORITY = True
   ```

2. 分析不同行为序列：
   ```python
   class BehaviorHeatmapConfig:
       def __init__(self):
           self.START_BEHAVIOR = 'Groom'           # 从梳理行为开始前10秒
           self.END_BEHAVIOR = 'Water'             # 到饮水行为结束
           self.PRE_BEHAVIOR_TIME = 10.0           # 行为开始前10秒
           self.SAMPLING_RATE = 4.8                # 采样率（Hz）
           self.INIT_CONFIG_PRIORITY = True
   ```

3. 命令行使用示例：
   ```bash
   # 分析同一行为（使用全局排序）
   python heatmap_behavior.py --start-behavior "Crack-seeds-shells" --end-behavior "Crack-seeds-shells" --pre-behavior-time 10 --sorting-method global

   # 分析不同行为序列（使用局部排序）
   python heatmap_behavior.py --start-behavior "Find-seeds" --end-behavior "Eat-seed-kernels" --pre-behavior-time 15 --sorting-method local
   
   # 分析行为序列（使用首图排序）
   python heatmap_behavior.py --start-behavior "Groom" --end-behavior "Water" --sorting-method first
   
   # 分析行为序列（使用自定义排序）
   python heatmap_behavior.py --start-behavior "Crack-seeds-shells" --end-behavior "Eat-seed-kernels" --sorting-method custom
   
   # 使用默认排序方式（根据配置类中的SORTING_METHOD设定）
   python heatmap_behavior.py --start-behavior "Groom" --end-behavior "Water"
   ```

4. 神经元排序方式控制：
   ```python
   # 使用全局排序（基于整个数据集，所有热图使用相同的神经元顺序，便于比较）
   self.SORTING_METHOD = 'global'
   
   # 使用局部排序（每个热图根据当前时间窗口独立排序，突出局部模式）
   self.SORTING_METHOD = 'local'
   
   # 使用首图排序（以第一个热图的排序为基准，后续热图使用相同顺序）
   self.SORTING_METHOD = 'first'
   
   # 使用自定义排序（按指定的神经元顺序排列，剩余神经元按字符串大小排序）
   self.SORTING_METHOD = 'custom'
   self.CUSTOM_NEURON_ORDER = ['n53', 'n40', 'n29', 'n34', 'n4', 'n32', 'n25', 'n27', 'n22', 'n55', 'n21', 'n5', 'n19']
   ```

5. 功能特点：
   - 支持同一行为的完整序列分析（开始前N秒到行为结束）
   - 支持不同行为的连续序列分析（行为A开始前N秒到行为B结束）
   - 严格的连续性检查：只有当起始行为结束后紧接着出现结束行为时才绘制热力图
   - 自动跳过非连续的行为配对，确保分析的生物学意义
   - 灵活的神经元排序：支持全局排序（一致性）或局部排序（突出局部特征）
   - 生成个体序列和平均序列的热力图
   - 在热力图上标注关键时间点（行为开始和结束）
   - 热力图粒度保持原始时间戳，横坐标显示为秒（5秒一刻度）

5. 可用的行为类型：
6. 排序方式对比：
   - **全局排序 (global)**：基于整个数据集计算排序，所有热图使用相同的神经元顺序，便于跨行为序列比较，神经元位置固定
   - **局部排序 (local)**：每个热图根据当前时间窗口内的峰值时间独立排序，突出各自的时序模式
   - **首图排序 (first)**：以第一个热图的排序为基准，后续所有热图使用相同的神经元顺序，既保留第一个序列的局部特征，又保持一致性便于比较。这种方式特别适合当第一个行为序列具有代表性，希望以此为参照来观察其他序列的神经元活动模式时使用
   - **自定义排序 (custom)**：按用户指定的神经元顺序排列（n53, n40, n29等），剩余神经元按字符串大小顺序排在指定神经元下方，适用于特定的科研需求或重点关注特定神经元的情况

7. 排序方式选择建议：
   - 需要严格比较不同序列时 → 使用全局排序
   - 关注每个序列自身的时序特征时 → 使用局部排序  
   - 以某个特定序列为参照进行比较时 → 使用首图排序
   - 需要重点关注特定神经元或遵循特定顺序时 → 使用自定义排序

8. 可用的行为类型：
   - 'Crack-seeds-shells', 'Eat-feed', 'Eat-seed-kernels', 'Explore'
   - 'Explore-search-seeds', 'Find-seeds', 'Get-feed', 'Get-seeds'
   - 'Grab-seeds', 'Groom', 'Smell-feed', 'Smell-Get-seeds'
   - 'Store-seeds', 'Water'
"""
