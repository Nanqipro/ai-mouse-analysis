import numpy as np
import pandas as pd
from scipy import signal
from scipy.signal import find_peaks, peak_widths, savgol_filter, butter, filtfilt
# 兼容不同版本的 NumPy
try:
    from numpy import trapezoid
except ImportError:
    from numpy import trapz as trapezoid
import matplotlib.pyplot as plt
import os
import argparse
import logging
import datetime
import sys
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from typing import Union, Literal

# 初始化日志记录器
def setup_logger(output_dir=None, prefix="element_extraction", capture_all_output=True):
    """
    设置日志记录器，将日志消息输出到控制台和文件
    
    参数:
        output_dir: 日志文件输出目录，默认为输出到当前脚本所在目录的logs文件夹
        prefix: 日志文件名称前缀
        capture_all_output: 是否捕获所有标准输出到日志文件
    
    返回:
        logger: 已配置好的日志记录器
    """
    # 创建日志记录器
    logger = logging.getLogger(prefix)
    logger.setLevel(logging.INFO)
    
    # 清除现有的处理器，避免重复添加
    if logger.handlers:
        for handler in logger.handlers:
            logger.removeHandler(handler)
    
    # 创建格式化器
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # 添加控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 如果指定了输出目录，则添加文件处理器
    if output_dir is None:
        # 默认在当前脚本目录下创建 logs 文件夹
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建日志文件名，包含时间戳
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_file = os.path.join(output_dir, f"{prefix}_{timestamp}.log")
    
    # 添加文件处理器
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # 创建输出重定向类，捕获所有的标准输出和错误输出
    class OutputRedirector:
        def __init__(self, original_stream, logger, level=logging.INFO):
            self.original_stream = original_stream
            self.logger = logger
            self.level = level
            self.buffer = ''
        
        def write(self, buf):
            self.original_stream.write(buf)
            self.buffer += buf
            if '\n' in buf:
                self.flush()
        
        def flush(self):
            if self.buffer.strip():
                for line in self.buffer.rstrip().split('\n'):
                    if line.strip():  # 只记录非空行
                        self.logger.log(self.level, f"OUTPUT: {line.rstrip()}")
            self.buffer = ''
    
    # 重定向标准输出和错误输出到日志文件
    if capture_all_output:
        sys.stdout = OutputRedirector(sys.stdout, logger, logging.INFO)
        sys.stderr = OutputRedirector(sys.stderr, logger, logging.ERROR)
    
    logger.info(f"日志文件创建于: {log_file}")
    return logger

def moving_average_smooth(data: Union[pd.Series, np.ndarray], window_size: int = 3) -> np.ndarray:
    """
    应用移动平均滤波平滑数据（与smooth_data.py保持一致）
    
    参数
    ------
    data : array-like
        输入的时间序列数据
    window_size : int, 可选
        滑动窗口大小，必须为奇数，默认为3
        
    返回
    ------
    np.ndarray
        平滑后的数据
    """
    if window_size % 2 == 0:
        logging.warning("窗口大小应为奇数，已自动加1")
        window_size += 1
    # 使用与 smooth_data.py 相同的实现，但返回 numpy 数组
    smoothed_series = pd.Series(data).rolling(window=window_size, center=True).mean().bfill().ffill()
    return smoothed_series.values

def butterworth_filter(
    data: np.ndarray,
    cutoff_freq: float = 20,
    fs: float = 4.8,
    order: int = 2,
    strength: float = 0.05
) -> np.ndarray:
    """
    应用Butterworth低通滤波器去除高频噪声（与smooth_data.py保持一致）
    
    参数
    ------
    data : np.ndarray
        输入信号数据
    cutoff_freq : float, 可选
        截止频率，值越小滤波效果越强，默认为20
    fs : float, 可选
        采样频率，默认为4.8Hz（钙离子浓度数据的实际采样频率）
    order : int, 可选
        滤波器阶数，阶数越高滤波效果越陡峭，默认为2
    strength : float, 可选
        滤波强度系数，范围0-1，值越大滤波效果越强，默认为0.05
        
    返回
    ------
    np.ndarray
        滤波后的数据
    """
    nyquist = fs * 0.5
    normal_cutoff = (cutoff_freq * strength) / nyquist
    if normal_cutoff >= 1.0:
        normal_cutoff = 0.99  # 防止截止频率超过奈奎斯特频率
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)

def normalize_neural_data(
    data: np.ndarray,
    method: Literal['standard', 'minmax', 'robust', 'log_standard', 'log_minmax'] = 'standard',
    feature_range: tuple = (0, 1)
) -> np.ndarray:
    """
    对神经元数据进行归一化处理
    
    参数
    ------
    data : np.ndarray
        输入的神经元数据
    method : str, 可选
        归一化方法，可选：
        'standard' - 标准化(Z-score)
        'minmax' - 最小最大值归一化
        'robust' - 稳健归一化（基于分位数）
        'log_standard' - 对数变换后的标准化
        'log_minmax' - 对数变换后的最小最大值归一化
        默认为'standard'
    feature_range : tuple, 可选
        用于minmax归一化的目标范围，默认为(0, 1)
        
    返回
    ------
    np.ndarray
        归一化后的数据
    """
    data_2d = data.reshape(-1, 1)  # 转换为2D数组以适配scikit-learn
    
    # 对数变换预处理
    if method.startswith('log_'):
        # 将数据平移到正数区间
        min_val = data_2d.min()
        shift = abs(min_val) + 1 if min_val <= 0 else 0
        data_2d = data_2d + shift
        # 应用对数变换
        data_2d = np.log1p(data_2d)
    
    # 选择归一化方法
    if method in ['standard', 'log_standard']:
        scaler = StandardScaler()
    elif method in ['minmax', 'log_minmax']:
        scaler = MinMaxScaler(feature_range=feature_range)
    elif method == 'robust':
        scaler = RobustScaler(quantile_range=(25, 75))
    else:
        raise ValueError(f"未知的归一化方法: {method}")
    
    # 应用归一化
    normalized_data = scaler.fit_transform(data_2d)
    return normalized_data.flatten()

def preprocess_neural_signal(
    data: np.ndarray,
    apply_moving_average: bool = True,
    moving_avg_window: int = 3,
    apply_butterworth: bool = True,
    butterworth_cutoff: float = 20,
    butterworth_strength: float = 0.05,
    apply_normalization: bool = False,
    normalization_method: str = 'standard',
    fs: float = 4.8
) -> np.ndarray:
    """
    对神经元信号进行预处理，包括平滑化和归一化（与smooth_data.py保持一致的参数）
    
    参数
    ------
    data : np.ndarray
        输入的神经元信号数据
    apply_moving_average : bool, 可选
        是否应用移动平均滤波，默认为True
    moving_avg_window : int, 可选
        移动平均窗口大小，默认为3
    apply_butterworth : bool, 可选
        是否应用Butterworth滤波，默认为True
    butterworth_cutoff : float, 可选
        Butterworth滤波器截止频率，默认为20
    butterworth_strength : float, 可选
        Butterworth滤波强度，默认为0.05
    apply_normalization : bool, 可选
        是否应用归一化，默认为False
    normalization_method : str, 可选
        归一化方法，默认为'standard'
    fs : float, 可选
        采样频率，默认为4.8Hz（钙离子浓度数据的实际采样频率）
        
    返回
    ------
    np.ndarray
        预处理后的数据
    """
    processed_data = data.copy()
    
    # 步骤1：移动平均滤波
    if apply_moving_average:
        processed_data = moving_average_smooth(processed_data, window_size=moving_avg_window)
    
    # 步骤2：Butterworth滤波
    if apply_butterworth:
        processed_data = butterworth_filter(
            processed_data, 
            cutoff_freq=butterworth_cutoff,
            fs=fs,
            strength=butterworth_strength
        )
    
    # 步骤3：数据归一化
    if apply_normalization:
        processed_data = normalize_neural_data(
            processed_data, 
            method=normalization_method
        )
    
    return processed_data

def detect_calcium_transients(data, fs=4.8, min_snr = 3.5, min_duration=12, smooth_window=31, 
                             peak_distance=24, baseline_percentile=8, max_duration=800,
                             detect_subpeaks=False, subpeak_prominence=0.15, 
                             subpeak_width=5, subpeak_distance=8, params=None, 
                             min_morphology_score=0.20, min_exp_decay_score=0.12,
                             filter_strength=1.0,
                             # 新增的预处理参数（与smooth_data.py保持一致）
                             apply_preprocessing=True,
                             apply_moving_average=True,
                             moving_avg_window=3,
                             apply_butterworth=True,
                             butterworth_cutoff=20,
                             butterworth_strength=0.05,
                             apply_normalization=False,
                             normalization_method='standard',
                             # Savitzky-Golay平滑控制
                             apply_savgol=True):
    """
    检测钙离子浓度数据中的钙爆发(calcium transients)，包括大波中的小波动
    增强对钙爆发形态的过滤，剔除不符合典型钙波特征的信号
    
    参数:
        data: 神经元荧光数据
        fs: 采样频率，默认为4.8Hz（钙离子浓度数据的实际采样频率）
        min_snr: 最小信噪比阈值，默认为3.5
        min_duration: 最小持续时间(采样点数)，默认为12
        smooth_window: Savitzky-Golay平滑窗口大小，默认为31
        peak_distance: 峰值之间的最小距离，默认为24
        baseline_percentile: 用于估计基线的百分位数，默认为8
        max_duration: 最大持续时间(采样点数)，默认为800
        detect_subpeaks: 是否检测子峰，默认为False
        subpeak_prominence: 子峰突出度阈值，默认为0.15
        subpeak_width: 子峰最小宽度，默认为5
        subpeak_distance: 子峰之间的最小距离，默认为8
        params: 手动指定的参数字典，如果提供则覆盖默认参数
        min_morphology_score: 最小形态评分阈值，默认为0.20
        min_exp_decay_score: 最小指数衰减评分阈值，默认为0.12
        filter_strength: 过滤强度系数(0.5-2.0)，控制所有阈值的整体强度，1.0为默认强度
                         <1.0: 降低所有阈值，检测更多潜在信号
                         >1.0: 提高所有阈值，更严格地过滤噪声
        apply_preprocessing: 是否应用预处理，默认为True
        apply_moving_average: 是否应用移动平均滤波，默认为True
        moving_avg_window: 移动平均窗口大小，默认为3
        apply_butterworth: 是否应用Butterworth滤波，默认为True
        butterworth_cutoff: Butterworth滤波器截止频率，默认为20（与smooth_data.py保持一致）
        butterworth_strength: Butterworth滤波强度，默认为0.05（与smooth_data.py保持一致）
        apply_normalization: 是否应用归一化，默认为False
        normalization_method: 归一化方法，默认为'standard'
        apply_savgol: 是否应用Savitzky-Golay平滑，默认为True
                         
    返回:
        transients: 检测到的钙爆发列表，每个元素包含开始、峰值、结束时间以及其他特征
        preprocessed_data: 预处理后的数据
    """
    
    # 步骤0：数据预处理（新增）
    if apply_preprocessing:
        preprocessed_data = preprocess_neural_signal(
            data=data,
            apply_moving_average=apply_moving_average,
            moving_avg_window=moving_avg_window,
            apply_butterworth=apply_butterworth,
            butterworth_cutoff=butterworth_cutoff,
            butterworth_strength=butterworth_strength,
            apply_normalization=apply_normalization,
            normalization_method=normalization_method,
            fs=fs
        )
        print(f"  预处理完成: 移动平均({apply_moving_average}), Butterworth({apply_butterworth}), 归一化({apply_normalization})")
    else:
        preprocessed_data = data.copy()
    
    # 根据filter_strength调整所有阈值参数
    if filter_strength != 1.0:
        # 调整与信号门槛相关的参数（高filter_strength时提高阈值）
        min_snr *= filter_strength
        min_morphology_score = min(0.9, min_morphology_score * filter_strength)
        min_exp_decay_score = min(0.9, min_exp_decay_score * filter_strength)
        
        # 调整与信号特征相关的参数（高filter_strength时更严格）
        if filter_strength > 1.0:
            min_duration = int(min_duration * (1 + (filter_strength - 1) * 0.5))
            smooth_window = int(smooth_window * (1 + (filter_strength - 1) * 0.3))
            peak_distance = int(peak_distance * (1 + (filter_strength - 1) * 0.2))
            subpeak_prominence *= filter_strength
            subpeak_width = int(subpeak_width * filter_strength)
            subpeak_distance = int(subpeak_distance * filter_strength)
        else:  # filter_strength < 1.0
            min_duration = max(10, int(min_duration * filter_strength))
            smooth_window = max(21, int(smooth_window * (1 - (1 - filter_strength) * 0.3)))
            peak_distance = max(10, int(peak_distance * filter_strength))  # 降低下限至10，更好地检测相邻钙波
            subpeak_prominence = max(0.1, subpeak_prominence * filter_strength)
            subpeak_width = max(5, int(subpeak_width * filter_strength))
            subpeak_distance = max(8, int(subpeak_distance * filter_strength))
    
    # 获取数据特征和信噪比信息，用于调整参数
    data_mean = np.mean(data)
    data_std = np.std(data)
    robust_range = np.percentile(data, 98) - np.percentile(data, 5)
    
    # 使用更高级的噪声评估方法
    median_val = np.median(data)
    mad = np.median(np.abs(data - median_val))
    signal_noise_ratio = robust_range / (mad * 1.4826) if mad > 0 else 0
    
    # 评估数据的基线稳定性
    sorted_data = np.sort(data)
    lower_half = sorted_data[:len(sorted_data)//3]  # 使用下三分之一作为基线估计
    baseline_variability = np.std(lower_half) / np.mean(lower_half) if np.mean(lower_half) > 0 else 0
    
    # 如果提供了自定义参数，覆盖默认参数
    # 确保params是字典类型
    if params is None:
        params = {}
    
    # 使用自定义参数覆盖默认值，确保所有必要参数都存在
    params['min_snr'] = params.get('min_snr', min_snr)
    params['min_duration'] = params.get('min_duration', min_duration)
    params['smooth_window'] = params.get('smooth_window', smooth_window)
    params['peak_distance'] = params.get('peak_distance', peak_distance)
    params['baseline_percentile'] = params.get('baseline_percentile', baseline_percentile)
    params['max_duration'] = params.get('max_duration', max_duration)
    params['subpeak_prominence'] = params.get('subpeak_prominence', subpeak_prominence)
    params['subpeak_width'] = params.get('subpeak_width', subpeak_width)
    params['subpeak_distance'] = params.get('subpeak_distance', subpeak_distance)
    params['min_morphology_score'] = params.get('min_morphology_score', min_morphology_score)
    params['min_exp_decay_score'] = params.get('min_exp_decay_score', min_exp_decay_score)
    
    # 从参数字典中提取值，以确保我们使用更新后的值
    min_snr = params['min_snr']
    min_duration = params['min_duration']
    smooth_window = params['smooth_window']
    peak_distance = params['peak_distance']
    baseline_percentile = params['baseline_percentile']
    max_duration = params['max_duration']
    subpeak_prominence = params['subpeak_prominence']
    subpeak_width = params['subpeak_width']
    subpeak_distance = params['subpeak_distance']
    min_morphology_score = params['min_morphology_score']
    min_exp_decay_score = params['min_exp_decay_score']
    
    # 1. 应用Savitzky-Golay平滑滤波器（在预处理之后）
    if apply_savgol and smooth_window > 1:
        # 确保smooth_window是奇数
        if smooth_window % 2 == 0:
            smooth_window += 1
        
        # 确保smooth_window不超过数据长度
        if smooth_window >= len(preprocessed_data):
            # 如果数据长度太短，调整smooth_window为数据长度的一半（确保为奇数）
            if len(preprocessed_data) <= 3:
                # 数据点太少，无法进行有效分析
                return [], preprocessed_data.copy()
            else:
                smooth_window = min(len(preprocessed_data) - 1, 
                                  max(3, int(len(preprocessed_data) // 2)))
                # 确保为奇数
                if smooth_window % 2 == 0:
                    smooth_window -= 1
                print(f"  Warning: Data length ({len(preprocessed_data)}) after preprocessing is less than smooth window (original {smooth_window}), adjusted to {smooth_window}")
        
        smoothed_data = signal.savgol_filter(preprocessed_data, smooth_window, 3)
        print(f"  Savitzky-Golay smoothing applied (window={smooth_window})")
    else:
        smoothed_data = preprocessed_data.copy()
        if not apply_savgol:
            print(f"  Savitzky-Golay smoothing disabled")
    
    # 2. 估计基线和噪声水平
    baseline = np.percentile(smoothed_data, baseline_percentile)
    noise_level = np.std(smoothed_data[smoothed_data < np.percentile(smoothed_data, 50)])
    
    # 3. 检测主要峰值 - 使用更强的峰值筛选条件
    threshold = baseline + min_snr * noise_level
    
    # 增加prominence参数来要求峰值必须明显突出于背景
    # 调整prominence_threshold，随filter_strength变化
    prominence_factor = 1.2 * filter_strength  # 进一步降低prominence_factor以捕获更多微弱峰值
    prominence_threshold = noise_level * prominence_factor
    
    # 增加width参数来过滤太窄的峰值（可能是尖刺噪声）
    min_width = min_duration // 6  # 进一步降低最小宽度要求，改为1/6
    
    # 首次检测所有可能的峰值，使用较低的distance要求
    initial_peaks, peak_props = find_peaks(smoothed_data, height=threshold, 
                                          prominence=prominence_threshold, width=min_width)
    
    # 对找到的峰值使用更智能的选择策略，而不是简单地应用固定距离
    # 如果没有检测到峰值，返回空列表
    if len(initial_peaks) == 0:
        return [], smoothed_data
    
    # 通过检查峰值之间的谷值深度来决定是否保留相邻峰值
    peaks = []
    for i, peak_idx in enumerate(initial_peaks):
        # 第一个峰值总是保留
        if i == 0:
            peaks.append(peak_idx)
            continue
            
        # 检查与之前添加的最后一个峰的距离
        last_peak = peaks[-1]
        if peak_idx - last_peak < peak_distance:
            # 如果距离太近，检查两个峰之间的谷值深度
            valley_idx = last_peak + np.argmin(smoothed_data[last_peak:peak_idx+1])
            valley_value = smoothed_data[valley_idx]
            
            # 计算谷值相对于两个峰的深度
            left_peak_height = smoothed_data[last_peak] - baseline
            right_peak_height = smoothed_data[peak_idx] - baseline
            min_peak_height = min(left_peak_height, right_peak_height)
            
            # 计算谷值深度占峰值高度的比例
            valley_depth = smoothed_data[valley_idx] - baseline
            valley_depth_ratio = valley_depth / min_peak_height if min_peak_height > 0 else 1.0
            
            # 如果谷值足够深（低于峰值高度的70%），则认为是两个独立的钙波，保留当前峰值
            # 否则，只保留较高的一个峰值
            if valley_depth_ratio < 0.8:
                peaks.append(peak_idx)
            elif smoothed_data[peak_idx] > smoothed_data[last_peak]:
                # 如果当前峰更高，替换上一个峰
                peaks[-1] = peak_idx
        else:
            # 距离足够远，保留当前峰
            peaks.append(peak_idx)
    
    # 转换为numpy数组
    peaks = np.array(peaks)
    
    # 如果没有剩余的峰值，返回空列表
    if len(peaks) == 0:
        return [], smoothed_data
    
    # 4. 分析每个钙爆发
    transients = []
    
    # 第一遍检测：主要钙爆发
    for i, peak_idx in enumerate(peaks):
        # 寻找左侧边界（从峰值向左搜索）
        start_idx = peak_idx
        # 向左搜索到信号低于基线或达到最大距离或达到前一个峰值的右边界
        left_limit = 0 if i == 0 else peaks[i-1]
        while start_idx > left_limit and smoothed_data[start_idx] > baseline:
            start_idx -= 1
            # 如果搜索范围过大，在局部最小值处停止
            if peak_idx - start_idx > max_duration:
                # 找到从peak_idx向左max_duration点范围内的局部最小值
                local_min_idx = start_idx + np.argmin(smoothed_data[start_idx:start_idx+max_duration])
                start_idx = local_min_idx
                break
        
        # 寻找右侧边界（从峰值向右搜索）
        end_idx = peak_idx
        # 向右搜索到信号低于基线或达到最大距离或达到下一个峰值的左边界
        right_limit = len(smoothed_data) - 1 if i == len(peaks) - 1 else peaks[i+1]
        while end_idx < right_limit and smoothed_data[end_idx] > baseline:
            end_idx += 1
            # 如果搜索范围过大，在局部最小值处停止
            if end_idx - peak_idx > max_duration:
                # 找到从peak_idx向右max_duration点范围内的局部最小值
                search_end = min(end_idx + max_duration, len(smoothed_data))
                if peak_idx < search_end - 1:
                    local_min_idx = peak_idx + np.argmin(smoothed_data[peak_idx:search_end])
                    end_idx = local_min_idx
                break
        
        # 如果峰值之间的信号始终高于基线，则使用峰值之间的最低点作为分界
        if i < len(peaks) - 1 and end_idx >= peaks[i+1]:
            # 寻找两个峰值之间的最低点作为分界
            valley_idx = peak_idx + np.argmin(smoothed_data[peak_idx:peaks[i+1]])
            end_idx = valley_idx
        
        if i > 0 and start_idx <= peaks[i-1]:
            # 寻找两个峰值之间的最低点作为分界
            valley_idx = peaks[i-1] + np.argmin(smoothed_data[peaks[i-1]:peak_idx])
            start_idx = valley_idx
            
        # 计算持续时间
        duration = (end_idx - start_idx) / fs
        
        # 如果持续时间太短，跳过此峰值
        if (end_idx - start_idx) < min_duration:
            continue
            
        # 验证峰值形状 - 在信号尖峰处添加额外验证（放宽要求以提高敏感度）
        # 计算峰值前后区域的斜率变化，检查是否符合典型钙爆发特征
        peak_region_start = max(start_idx, peak_idx - 5)
        peak_region_end = min(end_idx, peak_idx + 5)
        
        # 检查峰值前后的斜率变化，真实钙爆发通常在峰值处有明显的斜率变化
        # 注释掉严格的斜率验证，以提高检测敏感度
        # if peak_region_start < peak_idx and peak_idx < peak_region_end:
        #     pre_slope = np.mean(np.diff(smoothed_data[peak_region_start:peak_idx+1]))
        #     post_slope = np.mean(np.diff(smoothed_data[peak_idx:peak_region_end+1]))
        #     
        #     # 前斜率应为正（上升），后斜率应为负（下降）
        #     if pre_slope <= 0 or post_slope >= 0:
        #         continue  # 峰值前后斜率不符合预期，可能是噪声
                
        # 计算特征
        peak_value = smoothed_data[peak_idx]
        amplitude = peak_value - baseline
        
        # 计算半高宽 (FWHM)
        half_max = baseline + amplitude / 2
        widths, width_heights, left_ips, right_ips = peak_widths(smoothed_data, [peak_idx], rel_height=0.5)
        fwhm = widths[0] / fs
        
        # 上升和衰减时间
        rise_time = (peak_idx - start_idx) / fs
        decay_time = (end_idx - peak_idx) / fs
        
        # 计算峰面积 (AUC)
        segment = smoothed_data[start_idx:end_idx+1] - baseline
        auc = trapezoid(segment, dx=1.0/fs)
        
        # 新增：计算典型钙波形态特征评分
        # 1. 上升期陡峭、下降期缓慢的特征 - 钙波通常上升快，下降慢
        rise_decay_ratio = rise_time / decay_time if decay_time > 0 else float('inf')
        
        # 2. 计算波形对称性 - 钙波通常是非对称的（快速上升，缓慢下降）
        # 理想钙波的对称性应该较低
        left_half = smoothed_data[start_idx:peak_idx+1] - baseline
        right_half = smoothed_data[peak_idx:end_idx+1] - baseline
        
        # 对齐左右两侧长度
        min_half_len = min(len(left_half), len(right_half))
        if min_half_len > 3:  # 确保有足够的点来计算
            left_half_resampled = np.interp(
                np.linspace(0, 1, min_half_len),
                np.linspace(0, 1, len(left_half)),
                left_half
            )
            right_half_resampled = np.interp(
                np.linspace(0, 1, min_half_len),
                np.linspace(0, 1, len(right_half)),
                right_half[::-1]  # 反转右半部分
            )
            
            # 计算两侧差异作为非对称度量
            asymmetry = np.sum(np.abs(left_half_resampled - right_half_resampled)) / np.sum(left_half_resampled)
        else:
            # 如果半峰太短，则使用默认值
            asymmetry = 0
            
        # 3. 计算上升沿的平滑性和单调性
        if peak_idx > start_idx + 2:
            # 使用一阶差分评估平滑性
            rise_segment = smoothed_data[start_idx:peak_idx+1]
            rise_diff = np.diff(rise_segment)
            
            # 负值比例表示非单调上升的程度
            non_monotonic_ratio = np.sum(rise_diff < 0) / len(rise_diff) if len(rise_diff) > 0 else 1
            
            # 计算一阶差分的波动性 - 平滑的上升沿应有较小的波动
            rise_smoothness = np.std(rise_diff) / np.mean(rise_diff) if np.mean(rise_diff) > 0 else float('inf')
        else:
            non_monotonic_ratio = 1
            rise_smoothness = float('inf')
            
        # 4. 下降沿的指数衰减特性
        if end_idx > peak_idx + 3:
            # 提取衰减部分并归一化
            decay_segment = smoothed_data[peak_idx:end_idx+1] - baseline
            decay_segment = decay_segment / decay_segment[0]  # 归一化
            
            # 对数变换前过滤无效值
            valid_mask = decay_segment > 0
            if np.any(valid_mask):
                # 只对有效值(正值)取对数，防止出现警告
                log_decay = np.zeros_like(decay_segment)
                log_decay[valid_mask] = np.log(decay_segment[valid_mask])
                
                # 使用线性拟合，计算拟合度
                x = np.arange(len(log_decay))
                valid_idx = np.where(valid_mask)[0]
                if len(valid_idx) > 1:  # 确保有足够的有效点拟合
                    try:
                        # 只使用有效点进行拟合
                        slope, intercept = np.polyfit(valid_idx, log_decay[valid_idx], 1)
                        # 计算R²作为衰减指数特性指标
                        y_pred = slope * valid_idx + intercept
                        ss_tot = np.sum((log_decay[valid_idx] - np.mean(log_decay[valid_idx]))**2)
                        ss_res = np.sum((log_decay[valid_idx] - y_pred)**2)
                        exp_decay_score = 1 - ss_res/ss_tot if ss_tot > 0 else 0
                    except:
                        exp_decay_score = 0
                else:
                    exp_decay_score = 0
            else:
                exp_decay_score = 0
        else:
            exp_decay_score = 0
            
        # 5. 钙爆发持续时间/宽度比例 - 钙爆发通常有一定的形态比例
        duration_width_ratio = duration / fwhm if fwhm > 0 else 0
        
        # 组合计算一个形态评分：0-1之间，越高表示越符合典型钙波特征
        # 理想情况下：rise_decay_ratio应该小，asymmetry应该大，
        # non_monotonic_ratio应该小，rise_smoothness应该小，
        # exp_decay_score应该大，duration_width_ratio在适当范围
        
        # 定义理想的参数范围（更宽松的标准）
        ideal_rise_decay_ratio = 0.35  # 放宽理想的上升/衰减时间比例，从0.28改为0.35
        min_asymmetry = 0.15  # 降低最小非对称度要求，从0.22改为0.15
        max_non_monotonic = 0.35  # 放宽上升沿非单调性最大允许值，从0.25改为0.35
        ideal_duration_width_ratio = 3.0  # 调整理想的持续时间/宽度比，从2.6改为3.0
        min_exp_decay_score = 0.15  # 降低最小指数衰减评分，从0.25改为0.15
        
        # 计算各指标的评分（使用更宽松的评分函数）
        rise_decay_score = np.exp(-1.5 * abs(rise_decay_ratio - ideal_rise_decay_ratio)) if rise_decay_ratio < 1.5 else 0  # 放宽上限和衰减系数
        asymmetry_score = min(asymmetry / min_asymmetry, 1) if min_asymmetry > 0 else 0
        monotonic_score = 1 - min(non_monotonic_ratio / max_non_monotonic, 1)
        duration_ratio_score = np.exp(-0.3 * abs(duration_width_ratio - ideal_duration_width_ratio))  # 降低衰减系数
        
        # 综合形态评分 (0-1)，使用折中的权重配置
        morphology_score = (
            0.28 * rise_decay_score +     # 上升/衰减比例的权重
            0.2 * asymmetry_score +      # 非对称性的权重
            0.17 * monotonic_score +     # 单调性的权重
            0.22 * exp_decay_score +     # 指数衰减特性的权重
            0.13 * duration_ratio_score  # 持续时间比例的权重
        )
        
        # 设置适中的形态评分阈值，过滤不符合典型钙波形态的峰值
        min_morphology_score = 0.25  # 进一步降低最低形态评分要求，从0.45降到0.25
        
        # 考虑总体形态评分和关键特征
        if morphology_score < min_morphology_score:
            continue  # 跳过此峰值，因为总体形态不符合典型钙波特征
        
        # 添加关键特征的检查，但使用较为宽松的条件
        if exp_decay_score < min_exp_decay_score:  # 指数衰减特性是钙波的关键特征
            continue  # 跳过指数衰减不明显的峰值
            
        if rise_decay_ratio > 1.2:  # 放宽上升时间要求，从0.8改为1.2
            continue  # 跳过上升过慢的峰值
            
        if non_monotonic_ratio > 0.7:  # 放宽非单调性要求，从0.5改为0.7
            continue  # 跳过上升沿过于不规则的峰值
        
        # 检测波形的子峰值（如果启用）
        subpeaks = []
        if detect_subpeaks and (end_idx - start_idx) > 3 * subpeak_width:
            # 计算当前波形区间
            wave_segment = smoothed_data[start_idx:end_idx+1]
            
            # 计算相对突出度阈值（基于主峰的振幅）
            abs_prominence = subpeak_prominence * amplitude
            
            # 在此波形内找到所有局部峰值
            sub_peaks, sub_properties = find_peaks(
                wave_segment,
                prominence=abs_prominence,
                width=subpeak_width,
                distance=subpeak_distance
            )
            
            # 转换为原始数据索引
            sub_peaks = sub_peaks + start_idx
            
            # 排除与主峰相同的峰
            sub_peaks = [sp for sp in sub_peaks if abs(sp - peak_idx) > subpeak_distance]
            
            # 记录子峰特征
            for sp_idx in sub_peaks:
                # 计算子峰特征
                sp_value = smoothed_data[sp_idx]
                sp_amplitude = sp_value - baseline
                
                # 子峰的半高宽特性
                try:
                    sp_widths, _, sp_left_ips, sp_right_ips = peak_widths(
                        smoothed_data, [sp_idx], rel_height=0.5
                    )
                    sp_fwhm = sp_widths[0] / fs
                    
                    # 找出子峰的边界（局部最小值点）
                    # 向左寻找局部最小值
                    sp_start = sp_idx
                    left_search_limit = max(start_idx, sp_idx - max_duration//2)
                    while sp_start > left_search_limit:
                        if sp_start == left_search_limit + 1 or smoothed_data[sp_start] <= smoothed_data[sp_start-1]:
                            break
                        sp_start -= 1
                    
                    # 向右寻找局部最小值
                    sp_end = sp_idx
                    right_search_limit = min(end_idx, sp_idx + max_duration//2)
                    while sp_end < right_search_limit:
                        if sp_end == right_search_limit - 1 or smoothed_data[sp_end] <= smoothed_data[sp_end+1]:
                            break
                        sp_end += 1
                    
                    # 子峰持续时间
                    sp_duration = (sp_end - sp_start) / fs
                    
                    # 上升和衰减时间
                    sp_rise_time = (sp_idx - sp_start) / fs
                    sp_decay_time = (sp_end - sp_idx) / fs
                    
                    # 计算子峰面积
                    sp_segment = smoothed_data[sp_start:sp_end+1] - baseline
                    sp_auc = trapezoid(sp_segment, dx=1.0/fs)
                    
                    # 添加子峰信息
                    subpeaks.append({
                        'index': sp_idx,
                        'value': sp_value,
                        'amplitude': sp_amplitude,
                        'start_idx': sp_start,
                        'end_idx': sp_end,
                        'duration': sp_duration,
                        'fwhm': sp_fwhm,
                        'rise_time': sp_rise_time,
                        'decay_time': sp_decay_time,
                        'auc': sp_auc
                    })
                except Exception as e:
                    # 子峰分析失败，跳过该子峰
                    pass
        
        # 收集主波形对象特征
        wave_type = "complex" if len(subpeaks) > 0 else "simple"
        subpeaks_count = len(subpeaks)
        
        # 存储此次钙爆发的特征
        transient = {
            'start_idx': start_idx,
            'peak_idx': peak_idx,
            'end_idx': end_idx,
            'amplitude': amplitude,
            'peak_value': peak_value,
            'baseline': baseline,
            'duration': duration,
            'fwhm': fwhm,
            'rise_time': rise_time,
            'decay_time': decay_time,
            'auc': auc,
            'snr': amplitude / noise_level,
            'wave_type': wave_type,
            'subpeaks_count': subpeaks_count,
            'subpeaks': subpeaks,
            'morphology_score': morphology_score,  # 添加形态评分
            'rise_decay_ratio': rise_decay_ratio,
            'asymmetry': asymmetry,
            'exp_decay_score': exp_decay_score
        }
        
        transients.append(transient)
    
    # 检测是否有复杂波形或组合波形（不同特征的波）
    wave_types = {t['wave_type'] for t in transients}
    complex_waves = [t for t in transients if t['wave_type'] == 'complex']
    
    # 输出波形分类统计
    if len(transients) > 0:
        print(f"总共检测到 {len(transients)} 个钙爆发，其中：")
        print(f"  - 简单波形: {len(transients) - len(complex_waves)} 个")
        print(f"  - 复合波形: {len(complex_waves)} 个 (含有子峰)")
        
        # 计算子峰总数
        total_subpeaks = sum(t['subpeaks_count'] for t in transients)
        if total_subpeaks > 0:
            print(f"  - 子峰总数: {total_subpeaks} 个")
    
    return transients, smoothed_data

def extract_calcium_features(neuron_data, fs=4.8, visualize=False, detect_subpeaks=False, params=None, filter_strength=1.0,
                           # 新增的预处理参数（与smooth_data.py保持一致）
                           apply_preprocessing=True,
                           apply_moving_average=True,
                           moving_avg_window=3,
                           apply_butterworth=True,
                           butterworth_cutoff=20,
                           butterworth_strength=0.05,
                           apply_normalization=False,
                           normalization_method='standard',
                           apply_savgol=True):
    """
    从钙离子浓度数据中提取关键特征
    
    参数
    ----------
    neuron_data : numpy.ndarray 或 pandas.Series
        神经元钙离子浓度时间序列数据
    fs : float, 可选
        采样频率，默认为4.8Hz（钙离子浓度数据的实际采样频率）
    visualize : bool, 可选
        是否可视化结果，默认为False
    detect_subpeaks : bool, 可选
        是否检测大波中的小波峰，默认为False
    params : dict, 可选
        自定义参数字典，可覆盖默认参数
    filter_strength : float, 可选
        过滤强度调节参数，值越大过滤越强，默认为1.0
        可以调整此参数来平衡检测灵敏度和假阳性率
    apply_preprocessing : bool, 可选
        是否应用预处理，默认为True
    apply_moving_average : bool, 可选
        是否应用移动平均滤波，默认为True
    moving_avg_window : int, 可选
        移动平均窗口大小，默认为3
    apply_butterworth : bool, 可选
        是否应用Butterworth滤波，默认为True
    butterworth_cutoff : float, 可选
        Butterworth滤波器截止频率，默认为0.1
    butterworth_strength : float, 可选
        Butterworth滤波强度，默认为0.5
    apply_normalization : bool, 可选
        是否应用归一化，默认为False
    normalization_method : str, 可选
        归一化方法，默认为'standard'
        
    返回
    -------
    features : dict
        计算得到的特征统计数据
    transients : list of dict
        每个钙爆发的特征参数字典列表
    """
    if isinstance(neuron_data, pd.Series):
        data = neuron_data.values
    else:
        data = neuron_data
    
    # 检测钙爆发
    transients, smoothed_data = detect_calcium_transients(
        data, fs=fs, detect_subpeaks=detect_subpeaks, params=params, filter_strength=filter_strength,
        apply_preprocessing=apply_preprocessing,
        apply_moving_average=apply_moving_average,
        moving_avg_window=moving_avg_window,
        apply_butterworth=apply_butterworth,
        butterworth_cutoff=butterworth_cutoff,
        butterworth_strength=butterworth_strength,
        apply_normalization=apply_normalization,
        normalization_method=normalization_method,
        apply_savgol=apply_savgol
    )
    
    # 如果没有检测到钙爆发，返回空特征
    if len(transients) == 0:
        return {
            'num_transients': 0,
            'mean_amplitude': np.nan,
            'mean_duration': np.nan,
            'mean_fwhm': np.nan,
            'mean_rise_time': np.nan,
            'mean_decay_time': np.nan,
            'mean_auc': np.nan,
            'frequency': 0,
            'complex_waves_ratio': 0,
            'subpeaks_per_wave': 0
        }, []
    
    # 计算特征统计值
    amplitudes = [t['amplitude'] for t in transients]
    durations = [t['duration'] for t in transients]
    fwhms = [t['fwhm'] for t in transients]
    rise_times = [t['rise_time'] for t in transients]
    decay_times = [t['decay_time'] for t in transients]
    aucs = [t['auc'] for t in transients]
    
    # 统计复杂波形比例
    complex_waves = [t for t in transients if t['wave_type'] == 'complex']
    complex_waves_ratio = len(complex_waves) / len(transients) if len(transients) > 0 else 0
    
    # 计算每个波的平均子峰数
    subpeaks_per_wave = sum(t['subpeaks_count'] for t in transients) / len(transients) if len(transients) > 0 else 0
    
    # 记录总体特征
    total_time = len(data) / fs  # 总时间（秒）
    features = {
        'num_transients': len(transients),
        'mean_amplitude': np.mean(amplitudes),
        'mean_duration': np.mean(durations),
        'mean_fwhm': np.mean(fwhms),
        'mean_rise_time': np.mean(rise_times),
        'mean_decay_time': np.mean(decay_times),
        'mean_auc': np.mean(aucs),
        'frequency': len(transients) / total_time,  # 每秒事件数
        'complex_waves_ratio': complex_waves_ratio,
        'subpeaks_per_wave': subpeaks_per_wave
    }
    
    # 可视化（如果需要）
    if visualize:
        visualize_calcium_transients(data, smoothed_data, transients, fs)
    
    return features, transients

def visualize_calcium_transients_individual(raw_data, smoothed_data, transients_dict, fs=4.8,
                                           neuron_ids=None, save_dir=None, show_subpeaks=True,
                                           line_width=2.0, figsize=None,
                                           auto_figsize=False,
                                           layout='separate'):
    """
    为每个神经元单独生成 trace 图，标注检测到的钙波
    
    参数
    ----------
    raw_data : pandas.DataFrame or dict
        原始钙离子浓度数据
    smoothed_data : pandas.DataFrame or dict
        平滑后的数据
    transients_dict : dict
        检测到的钙爆发特征字典，键为神经元ID，值为transients列表
    fs : float, 可选
        采样频率，默认为4.8Hz
    neuron_ids : list, 可选
        要显示的神经元ID列表，默认为None（显示所有）
    save_dir : str, 可选
        保存目录，默认为None
    show_subpeaks : bool, 可选
        是否显示子峰，默认为True
    line_width : float, 可选
        trace线条宽度，默认为2.0
    figsize : tuple, 可选
        单个图像尺寸，默认为根据时间长度自适应（最少20x6）
    layout : str, 可选
        布局方式：'separate'（每个神经元独立文件）或'subplots'（多个子图在一个文件中）
        
    返回
    -------
    saved_files : list
        保存的文件路径列表
    """
    import matplotlib.cm as cm
    from matplotlib.colors import Normalize
    from matplotlib.patches import Patch
    
    # 转换数据格式
    if isinstance(raw_data, pd.DataFrame):
        raw_data_dict = {col: raw_data[col].values for col in raw_data.columns}
        smoothed_data_dict = {col: smoothed_data[col].values for col in smoothed_data.columns}
    elif isinstance(raw_data, dict):
        raw_data_dict = raw_data
        smoothed_data_dict = smoothed_data
    else:
        raw_data_dict = {'neuron_0': raw_data}
        smoothed_data_dict = {'neuron_0': smoothed_data}
    
    # 确定要显示的神经元
    if neuron_ids is None:
        neuron_ids = list(raw_data_dict.keys())
    
    # 生成时间轴
    time_points = len(list(raw_data_dict.values())[0])
    time_axis = np.arange(time_points) / fs
    duration_seconds = time_axis[-1] if len(time_axis) > 0 else 0
    
    # 图像尺寸控制
    if auto_figsize:
        auto_width = max(24.0, duration_seconds / 6.0)
        auto_height = 6.0
        base_figsize = (auto_width, auto_height)
    else:
        base_figsize = figsize if figsize is not None else (20.0, 6.0)
    
    saved_files = []
    
    if layout == 'separate':
        # 方案A：每个神经元保存为独立文件
        for neuron_id in neuron_ids:
            if neuron_id not in raw_data_dict:
                print(f"警告: 神经元 {neuron_id} 不在数据中，跳过")
                continue
            
            # 创建单独的图
            fig, ax = plt.subplots(figsize=base_figsize)
            
            raw_trace = raw_data_dict[neuron_id]
            smoothed_trace = smoothed_data_dict[neuron_id]
            
            # 绘制原始数据（半透明）
            ax.plot(time_axis, raw_trace,
                   linewidth=line_width * 0.7,
                   alpha=0.4,
                   color='gray',
                   label='Raw Data')
            
            # 绘制平滑后的数据
            ax.plot(time_axis, smoothed_trace,
                   linewidth=line_width,
                   alpha=0.9,
                   color='black',
                   label='Smoothed Data')
            
            # 标注钙波
            if neuron_id in transients_dict:
                transients = transients_dict[neuron_id]
                
                for j, transient in enumerate(transients):
                    start_time = transient['start_idx'] / fs
                    end_time = transient['end_idx'] / fs
                    peak_time = transient['peak_idx'] / fs
                    peak_value = transient['peak_value']
                    
                    # Determine color based on wave type
                    wave_type = transient.get('wave_type', 'simple')
                    if wave_type == 'complex':
                        color = '#9B59B6'  # Purple - Complex wave (with subpeaks)
                    else:
                        color = '#F39C12'  # Yellow - Simple wave
                    
                    # 绘制钙波区间（半透明背景）
                    ax.axvspan(start_time, end_time,
                              alpha=0.25, color=color,
                              label='Calcium Wave' if j == 0 else "")
                    
                    # 标记区间边界（垂直虚线）
                    ax.axvline(start_time, color=color, linestyle='--',
                              linewidth=1.5, alpha=0.7)
                    ax.axvline(end_time, color=color, linestyle='--',
                              linewidth=1.5, alpha=0.7)
                    
                    # 标记峰值
                    ax.scatter(peak_time, peak_value,
                              color=color, s=120, marker='o', zorder=5,
                              edgecolors='black', linewidth=1.5,
                              label='Peak' if j == 0 else "")
                    
                    # 添加钙波编号和评分
                    if 'morphology_score' in transient:
                        score_text = f"#{j+1}\nScore:{transient['morphology_score']:.2f}"
                    else:
                        score_text = f"#{j+1}"
                    
                    y_offset = (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.05
                    ax.text(peak_time, peak_value + y_offset,
                           score_text,
                           horizontalalignment='center',
                           verticalalignment='bottom',
                           fontsize=11,
                           fontweight='bold',
                           bbox=dict(boxstyle='round,pad=0.3',
                                   facecolor=color, alpha=0.3,
                                   edgecolor='black', linewidth=0.5))
                    
                    # 标记子峰
                    if show_subpeaks and 'subpeaks' in transient and transient['subpeaks']:
                        for sp in transient['subpeaks']:
                            sp_time = sp['index'] / fs
                            sp_value = sp['value']
                            ax.scatter(sp_time, sp_value,
                                      color='magenta', s=80, marker='*', zorder=5,
                                      edgecolors='black', linewidth=0.5,
                                      label='Subpeak' if j == 0 else "")
            
            # 设置标题和标签
            transients_list = transients_dict.get(neuron_id, [])
            n_waves = len(transients_list)
            n_simple = sum(1 for t in transients_list if t.get('wave_type') == 'simple')
            n_complex = sum(1 for t in transients_list if t.get('wave_type') == 'complex')
            ax.set_title(f'Neuron {neuron_id} - Calcium Transient Detection ({n_waves} waves: {n_simple} simple, {n_complex} complex)',
                        fontsize=16, fontweight='bold', pad=15)
            ax.set_xlabel('Time (seconds)', fontsize=14, fontweight='bold')
            ax.set_ylabel('Calcium Signal Intensity', fontsize=14, fontweight='bold')
            
            # 设置刻度
            ax.tick_params(axis='both', labelsize=12)
            for label in ax.get_xticklabels() + ax.get_yticklabels():
                label.set_fontweight('bold')
            
            # 添加网格
            ax.grid(True, alpha=0.3, linestyle='--')
            
            # 添加图例（去重）
            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            
            # 添加波形类型说明
            from matplotlib.patches import Patch
            wave_legend = [
                Patch(facecolor='#F39C12', alpha=0.25, label='Simple Wave'),
                Patch(facecolor='#9B59B6', alpha=0.25, label='Complex Wave (with subpeaks)')
            ]
            all_handles = list(by_label.values()) + wave_legend
            all_labels = list(by_label.keys()) + ['Simple Wave', 'Complex Wave (with subpeaks)']
            
            ax.legend(all_handles, all_labels,
                     loc='upper right', fontsize=11, framealpha=0.9)
            
            plt.tight_layout()
            
            # 保存图像
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, f'{neuron_id}_trace.png')
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                saved_files.append(save_path)
                print(f"  已保存: {save_path}")
                plt.close()
            else:
                plt.show()
    
    else:  # layout == 'subplots'
        # 方案B：所有神经元作为子图在一个文件中
        n_neurons = len(neuron_ids)
        n_cols = 1
        n_rows = n_neurons
        
        subplot_width = base_figsize[0]
        height_per_plot = base_figsize[1]
        fig_height = height_per_plot * n_neurons
        fig, axes = plt.subplots(n_rows, n_cols,
                                figsize=(subplot_width, fig_height),
                                squeeze=False)
        
        for idx, neuron_id in enumerate(neuron_ids):
            ax = axes[idx, 0]
            
            if neuron_id not in raw_data_dict:
                ax.text(0.5, 0.5, f'神经元 {neuron_id} 数据缺失',
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'Neuron {neuron_id}')
                continue
            
            raw_trace = raw_data_dict[neuron_id]
            smoothed_trace = smoothed_data_dict[neuron_id]
            
            # 绘制数据
            ax.plot(time_axis, raw_trace, linewidth=line_width * 0.7,
                   alpha=0.4, color='gray', label='Raw')
            ax.plot(time_axis, smoothed_trace, linewidth=line_width,
                   alpha=0.9, color='black', label='Smoothed')
            
            # 标注钙波
            if neuron_id in transients_dict:
                transients = transients_dict[neuron_id]
                
                for j, transient in enumerate(transients):
                    start_time = transient['start_idx'] / fs
                    end_time = transient['end_idx'] / fs
                    peak_time = transient['peak_idx'] / fs
                    peak_value = transient['peak_value']
                    
                    # Determine color based on wave type
                    wave_type = transient.get('wave_type', 'simple')
                    if wave_type == 'complex':
                        color = '#9B59B6'  # Purple - Complex wave (with subpeaks)
                    else:
                        color = '#F39C12'  # Yellow - Simple wave
                    
                    ax.axvspan(start_time, end_time, alpha=0.25, color=color)
                    ax.axvline(start_time, color=color, linestyle='--', linewidth=1, alpha=0.7)
                    ax.axvline(end_time, color=color, linestyle='--', linewidth=1, alpha=0.7)
                    ax.scatter(peak_time, peak_value, color=color, s=80,
                              marker='o', zorder=5, edgecolors='black', linewidth=1)
                    
                    # 标注编号
                    y_offset = (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.03
                    ax.text(peak_time, peak_value + y_offset, f'#{j+1}',
                           ha='center', va='bottom', fontsize=9, fontweight='bold')
                    
                    # 子峰
                    if show_subpeaks and 'subpeaks' in transient and transient['subpeaks']:
                        for sp in transient['subpeaks']:
                            ax.scatter(sp['index'] / fs, sp['value'],
                                      color='magenta', s=50, marker='*', zorder=5)
            
            # 设置标题和标签
            transients_list = transients_dict.get(neuron_id, [])
            n_waves = len(transients_list)
            n_simple = sum(1 for t in transients_list if t.get('wave_type') == 'simple')
            n_complex = sum(1 for t in transients_list if t.get('wave_type') == 'complex')
            ax.set_title(f'Neuron {neuron_id} ({n_waves} waves: {n_simple}S/{n_complex}C)',
                        fontsize=13, fontweight='bold')
            ax.set_ylabel('Signal', fontsize=11, fontweight='bold')
            ax.tick_params(axis='both', labelsize=10)
            ax.grid(True, alpha=0.3, linestyle='--')
            
            # 只在最后一个子图显示x轴标签
            if idx == n_neurons - 1:
                ax.set_xlabel('Time (seconds)', fontsize=11, fontweight='bold')
            else:
                ax.set_xticklabels([])
        
        plt.tight_layout()
        
        # 保存
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, 'all_neurons_subplots.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            saved_files.append(save_path)
            print(f"已保存多子图: {save_path}")
            plt.close()
        else:
            plt.show()
    
    return saved_files


def visualize_calcium_transients_trace(raw_data, smoothed_data, transients_dict, fs=4.8, 
                                       neuron_ids=None, save_path=None, show_subpeaks=True,
                                       trace_offset=60, scaling_factor=80, line_width=2.0,
                                       figsize=None,
                                       auto_figsize=False):
    """
    绘制类似 show_trace.py 风格的 trace 图，标注检测到的钙波区间
    
    参数
    ----------
    raw_data : pandas.DataFrame or dict
        原始钙离子浓度数据，可以是DataFrame（列为神经元）或字典（键为神经元ID）
    smoothed_data : pandas.DataFrame or dict
        平滑后的数据，格式同raw_data
    transients_dict : dict
        检测到的钙爆发特征字典，键为神经元ID，值为transients列表
    fs : float, 可选
        采样频率，默认为4.8Hz
    neuron_ids : list, 可选
        要显示的神经元ID列表，默认为None（显示所有）
    save_path : str, 可选
        图像保存路径，默认为None（不保存）
    show_subpeaks : bool, 可选
        是否显示子峰，默认为True
    trace_offset : float, 可选
        不同神经元trace之间的垂直偏移量，默认为60
    scaling_factor : float, 可选
        信号振幅缩放因子，默认为80
    line_width : float, 可选
        trace线条宽度，默认为2.0
    figsize : tuple, 可选
        图像尺寸，默认为(60, 25)
    """
    # 转换数据格式
    if isinstance(raw_data, pd.DataFrame):
        raw_data_dict = {col: raw_data[col].values for col in raw_data.columns}
        smoothed_data_dict = {col: smoothed_data[col].values for col in smoothed_data.columns}
    elif isinstance(raw_data, dict):
        raw_data_dict = raw_data
        smoothed_data_dict = smoothed_data
    else:
        # 单个神经元数据
        raw_data_dict = {'neuron_0': raw_data}
        smoothed_data_dict = {'neuron_0': smoothed_data}
    
    # 确定要显示的神经元
    if neuron_ids is None:
        neuron_ids = list(raw_data_dict.keys())
    
    # 生成时间轴
    time_points = len(list(raw_data_dict.values())[0])
    time_axis = np.arange(time_points) / fs
    duration_seconds = time_axis[-1] if len(time_axis) > 0 else 0
    
    # 图像尺寸控制
    if auto_figsize:
        auto_width = max(60.0, duration_seconds / 5.0)
        auto_height = 25.0
        figsize = (auto_width, auto_height)
    else:
        figsize = figsize if figsize is not None else (60.0, 25.0)
    
    # 创建图形
    fig, ax = plt.subplots(figsize=figsize)
    
    # 颜色映射
    import matplotlib.cm as cm
    from matplotlib.colors import Normalize
    
    # 绘制每个神经元的trace
    for i, neuron_id in enumerate(neuron_ids):
        if neuron_id not in raw_data_dict:
            print(f"警告: 神经元 {neuron_id} 不在数据中，跳过")
            continue
        
        position = len(neuron_ids) - i  # 反向排列，第一个在上方
        raw_trace = raw_data_dict[neuron_id]
        smoothed_trace = smoothed_data_dict[neuron_id]
        
        # 绘制原始数据（半透明）
        ax.plot(time_axis, 
                raw_trace * scaling_factor + position * trace_offset,
                linewidth=line_width * 0.5,
                alpha=0.3,
                color='gray',
                label=f'{neuron_id} (raw)' if i == 0 else "")
        
        # 绘制平滑后的数据
        ax.plot(time_axis,
                smoothed_trace * scaling_factor + position * trace_offset,
                linewidth=line_width,
                alpha=0.8,
                color='black',
                label=f'{neuron_id} (smoothed)' if i == 0 else "")
        
        # 标注检测到的钙波区间
        if neuron_id in transients_dict:
            transients = transients_dict[neuron_id]
            
            for j, transient in enumerate(transients):
                start_time = transient['start_idx'] / fs
                end_time = transient['end_idx'] / fs
                peak_time = transient['peak_idx'] / fs
                peak_value = transient['peak_value']
                
                # 根据波形类型确定颜色
                wave_type = transient.get('wave_type', 'simple')
                if wave_type == 'complex':
                    color = '#9B59B6'  # 紫色 - 复合峰（有子峰）
                else:
                    color = '#F39C12'  # 黄色 - 单峰
                
                # 绘制钙波区间（用半透明背景色）
                ax.axvspan(start_time, end_time, 
                          ymin=(position - 0.5) * trace_offset / (len(neuron_ids) * trace_offset),
                          ymax=(position + 0.5) * trace_offset / (len(neuron_ids) * trace_offset),
                          alpha=0.2, color=color)
                
                # 标记区间边界（垂直虚线）
                ax.axvline(start_time, 
                          ymin=(position - 0.5) / len(neuron_ids),
                          ymax=(position + 0.5) / len(neuron_ids),
                          color=color, linestyle='--', linewidth=1.5, alpha=0.7)
                ax.axvline(end_time,
                          ymin=(position - 0.5) / len(neuron_ids),
                          ymax=(position + 0.5) / len(neuron_ids),
                          color=color, linestyle='--', linewidth=1.5, alpha=0.7)
                
                # 标记峰值位置
                ax.scatter(peak_time,
                          peak_value * scaling_factor + position * trace_offset,
                          color=color, s=100, marker='o', zorder=5,
                          edgecolors='black', linewidth=1.5)
                
                # 添加钙波编号和形态评分
                if 'morphology_score' in transient:
                    score_text = f"{j+1}\n{transient['morphology_score']:.2f}"
                else:
                    score_text = f"{j+1}"
                
                ax.text(peak_time, 
                       peak_value * scaling_factor + position * trace_offset + trace_offset * 0.15,
                       score_text,
                       horizontalalignment='center',
                       verticalalignment='bottom',
                       fontsize=10,
                       fontweight='bold',
                       color=color)
                
                # 标记子峰（如果有且启用显示）
                if show_subpeaks and 'subpeaks' in transient and transient['subpeaks']:
                    for subpeak in transient['subpeaks']:
                        sp_time = subpeak['index'] / fs
                        sp_value = subpeak['value']
                        ax.scatter(sp_time,
                                  sp_value * scaling_factor + position * trace_offset,
                                  color='magenta', s=60, marker='*', zorder=5,
                                  edgecolors='black', linewidth=0.5)
    
    # 设置Y轴标签
    yticks = []
    ytick_labels = []
    for i, neuron_id in enumerate(neuron_ids):
        position = len(neuron_ids) - i
        yticks.append(position * trace_offset)
        ytick_labels.append(str(neuron_id))
    
    ax.set_yticks(yticks)
    ax.set_yticklabels(ytick_labels)
    
    # 设置轴标签
    ax.set_xlabel('Time (seconds)', fontsize=40, fontweight='bold')
    ax.set_ylabel('Neuron ID', fontsize=40, fontweight='bold')
    ax.set_title('Calcium Transient Detection Results', fontsize=45, fontweight='bold', pad=20)
    
    # 设置刻度标签字体
    ax.tick_params(axis='x', labelsize=25, rotation=45)
    ax.tick_params(axis='y', labelsize=20)
    
    # 设置刻度标签加粗
    for label in ax.get_xticklabels():
        label.set_fontweight('bold')
    for label in ax.get_yticklabels():
        label.set_fontweight('bold')
    
    # 添加网格
    ax.grid(False)
    
    # 设置X轴范围
    ax.set_xlim(0, time_axis[-1])
    
    # 添加图例说明
    from matplotlib.patches import Patch
    legend_elements = [
        plt.Line2D([0], [0], color='gray', linewidth=line_width*0.5, alpha=0.3, label='Raw Data'),
        plt.Line2D([0], [0], color='black', linewidth=line_width, alpha=0.8, label='Smoothed Data'),
        Patch(facecolor='#F39C12', alpha=0.2, label='Simple Wave'),
        Patch(facecolor='#9B59B6', alpha=0.2, label='Complex Wave (with subpeaks)'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#F39C12', 
                   markersize=10, markeredgecolor='black', markeredgewidth=1.5, label='Peak'),
    ]
    if show_subpeaks:
        legend_elements.append(
            plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='magenta',
                      markersize=12, markeredgecolor='black', markeredgewidth=0.5, label='Subpeak')
        )
    
    ax.legend(handles=legend_elements, loc='upper right', fontsize=20, framealpha=0.9)
    
    plt.tight_layout()
    
    # 保存图像
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Trace图已保存到: {save_path}")
    
    return fig, ax


def visualize_calcium_transients(raw_data, smoothed_data, transients, fs=4.8):
    """
    可视化钙离子浓度数据和检测到的钙爆发，包括形态评分信息
    
    参数
    ----------
    raw_data : numpy.ndarray
        原始钙离子浓度数据
    smoothed_data : numpy.ndarray
        平滑后的数据
    transients : list of dict
        检测到的钙爆发特征列表
    fs : float, 可选
        采样频率，默认为4.8Hz（钙离子浓度数据的实际采样频率）
    """
    time = np.arange(len(raw_data)) / fs
    plt.figure(figsize=(14, 10))
    
    # 创建三个子图：原始数据、放大的波峰和形态评分分布
    ax1 = plt.subplot(3, 1, 1)
    ax1.plot(time, raw_data, 'k-', alpha=0.4, label='Raw data')
    ax1.plot(time, smoothed_data, 'b-', label='Smoothed data')
    
    # 用颜色梯度表示形态评分
    import matplotlib.cm as cm
    from matplotlib.colors import Normalize
    
    # 获取所有形态评分
    scores = [t.get('morphology_score', 0.5) for t in transients]
    norm = Normalize(vmin=0.4, vmax=1.0)  # 评分范围从0.4到1.0
    cmap = cm.viridis
    
    # 标记钙爆发
    for i, t in enumerate(transients):
        # 使用形态评分来确定颜色，评分越高颜色越亮
        morphology_score = t.get('morphology_score', 0.5)
        color = cmap(norm(morphology_score))
        marker_size = 8 + morphology_score * 5  # 根据评分调整标记大小
        
        # 标记主峰值
        ax1.plot(t['peak_idx']/fs, t['peak_value'], 'o', color=color, markersize=marker_size)
        
        # 标记开始和结束
        ax1.axvline(x=t['start_idx']/fs, color=color, linestyle='--', alpha=0.5)
        ax1.axvline(x=t['end_idx']/fs, color=color, linestyle='--', alpha=0.5)
        
        # 添加编号和评分
        ax1.text(t['peak_idx']/fs, t['peak_value']*1.05, 
                 f"{i+1}\n{morphology_score:.2f}", 
                 horizontalalignment='center', verticalalignment='bottom',
                 fontsize=8)
        
        # 标记子峰（如果有）
        if 'subpeaks' in t and t['subpeaks']:
            for sp in t['subpeaks']:
                ax1.plot(sp['index']/fs, sp['value'], '*', color='magenta', markersize=8)
    
    # 添加颜色条以显示形态评分范围
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax1, orientation='vertical', pad=0.01)
    cbar.set_label('形态评分')
    
    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel('Calcium Signal Intensity')
    ax1.set_title(f'Detected {len(transients)} calcium transient events')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # 选择一个高形态评分的波形进行放大显示
    if transients:
        ax2 = plt.subplot(3, 1, 2)
        
        # 尝试选择形态评分最高的波形
        if 'morphology_score' in transients[0]:
            t = max(transients, key=lambda x: x.get('morphology_score', 0))
        else:
            # 如果没有形态评分，则选择最大振幅的波形
            t = max(transients, key=lambda x: x['amplitude'])
        
        # 计算放大区域
        margin = 20  # 左右额外显示的点数
        zoom_start = max(0, t['start_idx'] - margin)
        zoom_end = min(len(raw_data), t['end_idx'] + margin)
        
        # 绘制放大区域
        zoom_time = time[zoom_start:zoom_end]
        ax2.plot(zoom_time, raw_data[zoom_start:zoom_end], 'k-', alpha=0.4, label='Raw data')
        ax2.plot(zoom_time, smoothed_data[zoom_start:zoom_end], 'b-', label='Smoothed data')
        
        # 计算高度百分比标记点
        peak_val = t['peak_value']
        baseline = t['baseline']
        amplitude = peak_val - baseline
        heights = [0.25, 0.5, 0.75]  # 25%, 50%, 75%的高度
        
        # 标记主峰
        ms = t.get('morphology_score', 0.5)
        color = cmap(norm(ms))
        ax2.plot(t['peak_idx']/fs, peak_val, 'o', color=color, markersize=10, 
                label=f'Peak (Score: {ms:.2f})')
        
        # 标记不同高度百分比
        for h in heights:
            h_val = baseline + h * amplitude
            ax2.axhline(y=h_val, color='gray', linestyle=':', alpha=0.5)
            ax2.text(zoom_time[0], h_val, f'{int(h*100)}%', 
                    verticalalignment='center', fontsize=8)
        
        # 标记基线
        ax2.axhline(y=baseline, color='r', linestyle='-', alpha=0.5, label='Baseline')
        
        # 添加关键特征标注
        rise_time = t['rise_time']
        decay_time = t['decay_time']
        fwhm = t['fwhm']
        ratio = t.get('rise_decay_ratio', 0)
        asymm = t.get('asymmetry', 0)
        
        # 在图上标记这些特征
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        feature_text = (f"Rise: {rise_time:.2f}s\nDecay: {decay_time:.2f}s\n"
                        f"FWHM: {fwhm:.2f}s\nRise/Decay: {ratio:.2f}\n"
                        f"Asymmetry: {asymm:.2f}")
        ax2.text(0.05, 0.95, feature_text, transform=ax2.transAxes, 
                fontsize=9, verticalalignment='top', bbox=props)
        
        # 标记子峰
        if 'subpeaks' in t and t['subpeaks']:
            for sp in t['subpeaks']:
                ax2.plot(sp['index']/fs, sp['value'], '*', color='magenta', markersize=8)
                # 标记子峰边界
                ax2.axvline(x=sp['start_idx']/fs, color='c', linestyle=':', alpha=0.7)
                ax2.axvline(x=sp['end_idx']/fs, color='m', linestyle=':', alpha=0.7)
        
        ax2.set_xlabel('Time (seconds)')
        ax2.set_ylabel('Calcium Signal Intensity')
        ax2.set_title(f'High-quality calcium wave (Score: {ms:.2f})')
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)
        
        # 第三个子图：形态评分分布直方图
        ax3 = plt.subplot(3, 1, 3)
        
        # 提取所有波形的形态特征
        if 'morphology_score' in transients[0]:
            morphology_scores = [t.get('morphology_score', 0) for t in transients]
            rise_decay_ratios = [t.get('rise_decay_ratio', 0) for t in transients]
            asymmetries = [t.get('asymmetry', 0) for t in transients]
            
            # 绘制形态评分直方图
            ax3.hist(morphology_scores, bins=15, alpha=0.7, color='skyblue', 
                    label='Morphology Scores')
            ax3.set_xlabel('Morphology Score')
            ax3.set_ylabel('Frequency')
            ax3.set_title('Distribution of Calcium Wave Morphology Scores')
            
            # 添加中位数标记
            median_score = np.median(morphology_scores)
            ax3.axvline(x=median_score, color='r', linestyle='--', label=f'Median: {median_score:.2f}')
            
            # 添加统计信息文本框
            stats_text = (f"Total waves: {len(transients)}\n"
                        f"Median score: {median_score:.2f}\n"
                        f"Mean rise/decay ratio: {np.mean(rise_decay_ratios):.2f}\n"
                        f"Mean asymmetry: {np.mean(asymmetries):.2f}")
            ax3.text(0.95, 0.95, stats_text, transform=ax3.transAxes, 
                    fontsize=9, horizontalalignment='right', 
                    verticalalignment='top', bbox=props)
            
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        else:
            # 如果没有形态评分，则显示振幅分布
            amplitudes = [t['amplitude'] for t in transients]
            ax3.hist(amplitudes, bins=15, alpha=0.7, color='skyblue', 
                    label='Amplitude Distribution')
            ax3.set_xlabel('Amplitude')
            ax3.set_ylabel('Frequency')
            ax3.set_title('Distribution of Calcium Wave Amplitudes')
            ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def process_multiple_neurons(data_df, neuron_columns, fs=4.8):
    """
    处理多个神经元的钙离子数据并提取特征
    
    参数
    ----------
    data_df : pandas.DataFrame
        包含多个神经元数据的DataFrame
    neuron_columns : list of str
        要处理的神经元列名列表
    fs : float, 可选
        采样频率，默认为4.8Hz（钙离子浓度数据的实际采样频率）
        
    返回
    -------
    results : pandas.DataFrame
        每个神经元的特征统计结果
    """
    results = []
    
    for neuron in neuron_columns:
        print(f"处理神经元 {neuron}...")
        features, _ = extract_calcium_features(data_df[neuron], fs=fs)
        features['neuron'] = neuron
        results.append(features)
    
    return pd.DataFrame(results)

def analyze_behavior_specific_features(data_df, neuron_columns, behavior_col='behavior', fs=4.8):
    """
    分析不同行为条件下的钙离子特征
    
    参数
    ----------
    data_df : pandas.DataFrame
        包含神经元数据和行为标签的DataFrame
    neuron_columns : list of str
        要处理的神经元列名列表
    behavior_col : str, 可选
        行为标签列名，默认为'behavior'
    fs : float, 可选
        采样频率，默认为4.8Hz（钙离子浓度数据的实际采样频率）
        
    返回
    -------
    results : pandas.DataFrame
        不同行为条件下每个神经元的特征统计结果
    """
    # 获取所有行为类型
    behaviors = data_df[behavior_col].unique()
    results = []
    
    for neuron in neuron_columns:
        for behavior in behaviors:
            # 获取特定行为下的数据
            behavior_data = data_df[data_df[behavior_col] == behavior][neuron]
            
            if len(behavior_data) > 0:
                print(f"分析神经元 {neuron} 在行为 '{behavior}' 下的特征...")
                features, _ = extract_calcium_features(behavior_data, fs=fs)
                features['neuron'] = neuron
                features['behavior'] = behavior
                results.append(features)
    
    return pd.DataFrame(results)

def estimate_neuron_params(neuron_data, filter_strength=1.0):
    """
    估计神经元数据的最佳参数配置
    
    参数:
        neuron_data: 神经元荧光数据
        filter_strength: 过滤强度系数(0.5-2.0)，控制所有阈值的整体强度
                        <1.0: 降低阈值，更宽松的标准
                        >1.0: 提高阈值，更严格的标准
    
    返回:
        params: 参数字典
    """
    # 直接返回用户在detect_calcium_transients函数中设定的细粒度参数
    # 这些参数已经针对更细粒度的钙波检测进行了优化
    params = {
        'min_snr': 3.5,                    # 降低信噪比要求，检测更微弱的钙波
        'min_duration': 12,                # 允许更短的钙波事件
        'smooth_window': 31,               # 减少平滑，保留更多细节
        'peak_distance': 24,               # 峰值之间的最小距离（与UI界面一致）
        'baseline_percentile': 8,         # 使用更低的基线估计
        'max_duration': 800,               # 保持较大的最大持续时间
        'subpeak_prominence': 0.15,         # 降低子峰检测阈值
        'subpeak_width': 5,                # 允许更窄的子峰
        'subpeak_distance': 8,            # 允许更密集的子峰
        'min_morphology_score': 0.20,     # 放宽形态要求
        'min_exp_decay_score': 0.12       # 放宽衰减特性要求
    }
    
    # 根据filter_strength进行微调（保持原有的filter_strength功能）
    if filter_strength != 1.0:
        # 调整主要的检测阈值
        params['min_snr'] *= filter_strength
        params['min_morphology_score'] = min(0.9, params['min_morphology_score'] * filter_strength)
        params['min_exp_decay_score'] = min(0.9, params['min_exp_decay_score'] * filter_strength)
        
        # 调整时间相关参数
        if filter_strength > 1.0:
            # 更严格的设置
            params['min_duration'] = int(params['min_duration'] * (1 + (filter_strength - 1) * 0.5))
            params['smooth_window'] = int(params['smooth_window'] * (1 + (filter_strength - 1) * 0.3))
            params['peak_distance'] = int(params['peak_distance'] * (1 + (filter_strength - 1) * 0.2))
            params['subpeak_prominence'] *= filter_strength
            params['subpeak_width'] = int(params['subpeak_width'] * filter_strength)
            params['subpeak_distance'] = int(params['subpeak_distance'] * filter_strength)
        else:  # filter_strength < 1.0
            # 更宽松的设置
            params['min_duration'] = max(10, int(params['min_duration'] * filter_strength))
            params['smooth_window'] = max(21, int(params['smooth_window'] * (1 - (1 - filter_strength) * 0.3)))
            params['peak_distance'] = max(5, int(params['peak_distance'] * filter_strength))  # 降低下限至5，允许更密集检测
            params['subpeak_prominence'] = max(0.1, params['subpeak_prominence'] * filter_strength)
            params['subpeak_width'] = max(3, int(params['subpeak_width'] * filter_strength))
            params['subpeak_distance'] = max(5, int(params['subpeak_distance'] * filter_strength))
    
    # 打印当前使用的细粒度参数，便于调试
    print(f"  - 使用细粒度参数: min_snr={params['min_snr']:.2f}, "  
          f"baseline={params['baseline_percentile']}, min_duration={params['min_duration']}, "
          f"smooth={params['smooth_window']}, peak_distance={params['peak_distance']}")
    print(f"  - 形态评分阈值: morphology={params['min_morphology_score']:.2f}, "
          f"exp_decay={params['min_exp_decay_score']:.2f}")
    print(f"  - 子峰参数: prominence={params['subpeak_prominence']:.2f}, "
          f"width={params['subpeak_width']}, distance={params['subpeak_distance']}")
    
    return params

def analyze_all_neurons_transients(data_df, neuron_columns, fs=4.8, save_path=None, adaptive_params=True, start_id=1, 
                            file_info=None, filter_strength=1.0,
                            # 新增的预处理参数（与smooth_data.py保持一致）
                            apply_preprocessing=True,
                            apply_moving_average=True,
                            moving_avg_window=3,
                            apply_butterworth=True,
                            butterworth_cutoff=20,
                            butterworth_strength=0.05,
                            apply_normalization=False,
                            normalization_method='standard',
                            apply_savgol=True,
                            # 新增的可视化参数
                            visualize_trace=False,
                            trace_save_path=None,
                            max_neurons_visualize=20,
                            individual_plots=False,
                            plot_layout='separate',
                            trace_figsize=None,
                            individual_figsize=None,
                            auto_trace_figsize=False,
                            auto_individual_figsize=False):
    """
    分析所有神经元的钙爆发并为每个爆发分配唯一ID
    
    参数
    ----------
    data_df : pandas.DataFrame
        包含多个神经元数据的DataFrame
    neuron_columns : list of str
        要处理的神经元列名列表
    fs : float, 可选
        采样频率，默认为4.8Hz（钙离子浓度数据的实际采样频率）
    save_path : str, 可选
        Excel文件保存路径，默认为None（不保存）
    adaptive_params : bool, 可选
        是否为每个神经元使用自适应参数，默认为True
    start_id : int, 可选
        钙爆发ID的起始编号，默认为1
    file_info : dict, 可选
        原始文件的信息，包含绝对路径、相对路径和文件名
    filter_strength : float, 可选
        过滤强度系数(0.5-2.0)，控制所有阈值的整体强度，1.0为默认强度
    apply_preprocessing : bool, 可选
        是否应用预处理，默认为True
    apply_moving_average : bool, 可选
        是否应用移动平均滤波，默认为True
    moving_avg_window : int, 可选
        移动平均窗口大小，默认为3
    apply_butterworth : bool, 可选
        是否应用Butterworth滤波，默认为True
    butterworth_cutoff : float, 可选
        Butterworth滤波器截止频率，默认为20（与smooth_data.py保持一致）
    butterworth_strength : float, 可选
        Butterworth滤波强度，默认为0.05（与smooth_data.py保持一致）
    apply_normalization : bool, 可选
        是否应用归一化，默认为False
    normalization_method : str, 可选
        归一化方法，默认为'standard'
    visualize_trace : bool, 可选
        是否生成trace可视化图，默认为False
    trace_save_path : str, 可选
        trace图保存路径，默认为None（使用默认路径）
    max_neurons_visualize : int, 可选
        最多可视化的神经元数量，默认为20
    individual_plots : bool, 可选
        是否为每个神经元生成独立的trace图，默认为False
    plot_layout : str, 可选
        布局方式：'separate'（每个神经元独立文件）或'subplots'（多子图），默认为'separate'
    trace_figsize : tuple, 可选
        整体Trace图的固定尺寸(宽, 高)，默认为None（使用默认比例）
    individual_figsize : tuple, 可选
        单神经元Trace图的固定尺寸(宽, 高)，默认为None（使用默认比例）
    auto_trace_figsize : bool, 可选
        是否对整体Trace图使用自适应尺寸，默认为False
    auto_individual_figsize : bool, 可选
        是否对单神经元Trace图使用自适应尺寸，默认为False
        
    返回
    -------
    all_transients_df : pandas.DataFrame
        包含所有神经元所有钙爆发特征的DataFrame
    next_id : int
        下一个可用的钙爆发ID
    """
    all_transients = []
    transient_id = start_id  # 使用传入的起始ID
    
    # 用于存储原始数据和平滑数据（如果需要可视化）
    raw_data_dict = {}
    smoothed_data_dict = {}
    transients_dict = {}
    
    for neuron in neuron_columns:
        print(f"处理神经元 {neuron} 的钙爆发...")
        neuron_data = data_df[neuron].values
        
        # 如果启用自适应参数，为每个神经元生成自定义参数
        custom_params = None
        if adaptive_params:
            print(f"  估计神经元 {neuron} 的最优参数...")
            custom_params = estimate_neuron_params(neuron_data, filter_strength)
        
        # 检测钙爆发（启用子峰检测）
        transients, smoothed_data = detect_calcium_transients(
            neuron_data, fs=fs, params=custom_params, filter_strength=filter_strength,
            detect_subpeaks=True,  # 启用子峰检测
            apply_preprocessing=apply_preprocessing,
            apply_moving_average=apply_moving_average,
            moving_avg_window=moving_avg_window,
            apply_butterworth=apply_butterworth,
            butterworth_cutoff=butterworth_cutoff,
            butterworth_strength=butterworth_strength,
            apply_normalization=apply_normalization,
            normalization_method=normalization_method,
            apply_savgol=apply_savgol
        )
        
        # 如果需要可视化，存储数据
        if visualize_trace:
            raw_data_dict[neuron] = neuron_data
            smoothed_data_dict[neuron] = smoothed_data
            transients_dict[neuron] = transients
        
        # 为该神经元的每个钙爆发分配ID并添加到列表
        for t in transients:
            t['neuron'] = neuron
            t['transient_id'] = transient_id
            
            # 添加原始文件信息（如果提供）
            if file_info:
                t['source_file'] = file_info['filename']
                t['source_path'] = file_info['rel_path']
                t['source_abs_path'] = file_info['abs_path']
                
            all_transients.append(t)
            transient_id += 1
    
    # 如果没有检测到钙爆发，返回空DataFrame
    if len(all_transients) == 0:
        print("未检测到任何钙爆发")
        return pd.DataFrame(), transient_id
    
    # 创建DataFrame
    all_transients_df = pd.DataFrame(all_transients)
    
    # 如果指定了保存路径，则保存到Excel
    if save_path:
        # 确保保存目录存在
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        all_transients_df.to_excel(save_path, index=False)
        print(f"成功将所有钙爆发数据保存到: {save_path}")
    
    # 生成trace可视化图
    if visualize_trace and len(transients_dict) > 0:
        print(f"\n正在生成钙波检测trace可视化图...")
        
        # 限制可视化的神经元数量
        if max_neurons_visualize is None:
            neurons_to_visualize = neuron_columns
            print(f"  Will generate trace plots for all {len(neuron_columns)} neurons")
        else:
            neurons_to_visualize = neuron_columns[:max_neurons_visualize]
            print(f"  Will generate trace plots for the first {min(max_neurons_visualize, len(neuron_columns))} neurons")
        
        if individual_plots:
            # Method A: Generate individual trace plots for each neuron
            print(f"  Mode: Individual file for each neuron")
            
            # 确定保存目录
            if save_path is not None:
                base_dir = os.path.dirname(save_path)
                save_dir = os.path.join(base_dir, "individual_traces")
            else:
                save_dir = "individual_traces"
            
            try:
                saved_files = visualize_calcium_transients_individual(
                    raw_data=raw_data_dict,
                    smoothed_data=smoothed_data_dict,
                    transients_dict=transients_dict,
                    fs=fs,
                    neuron_ids=neurons_to_visualize,
                    save_dir=save_dir,
                    show_subpeaks=True,
                    layout=plot_layout,
                    figsize=individual_figsize,
                    auto_figsize=auto_individual_figsize
                )
                # Count simple and complex waves
                total_simple = 0
                total_complex = 0
                for neuron_id in neurons_to_visualize:
                    if neuron_id in transients_dict:
                        for t in transients_dict[neuron_id]:
                            if t.get('wave_type') == 'complex':
                                total_complex += 1
                            else:
                                total_simple += 1
                
                print(f"✓ Generated trace plots for {len(saved_files)} neurons")
                abs_save_dir = os.path.abspath(save_dir)
                print(f"  Save location: {abs_save_dir}/")
                print(f"  Sample file: {os.path.basename(saved_files[0]) if saved_files else 'n1_trace.png'}")
                print(f"  Wave statistics: {total_simple} simple waves, {total_complex} complex waves (with subpeaks)")
            except Exception as e:
                print(f"警告: 生成独立trace图时出错: {str(e)}")
                import traceback
                traceback.print_exc()
        else:
            # Original method: Stacked display
            print(f"  Mode: Stacked display")
            
            # 确定保存路径
            if trace_save_path is None and save_path is not None:
                # 如果没有指定trace保存路径，根据Excel保存路径生成
                base_path = os.path.splitext(save_path)[0]
                trace_save_path = f"{base_path}_trace_visualization.png"
            elif trace_save_path is None:
                # 如果都没有指定，使用默认路径
                trace_save_path = "calcium_transients_trace.png"
            
            try:
                visualize_calcium_transients_trace(
                    raw_data=raw_data_dict,
                    smoothed_data=smoothed_data_dict,
                    transients_dict=transients_dict,
                    fs=fs,
                    neuron_ids=neurons_to_visualize,
                    save_path=trace_save_path,
                    show_subpeaks=True,
                    figsize=trace_figsize,
                    auto_figsize=auto_trace_figsize
                )
                print(f"✓ Trace visualization generated: {trace_save_path}")
            except Exception as e:
                print(f"警告: 生成trace可视化图时出错: {str(e)}")
                import traceback
                traceback.print_exc()
    
    return all_transients_df, transient_id

def analyze_behavior_calcium_frequency(data_df, neuron_columns, behavior_col='behavior', fs=4.8, 
                                      save_path=None, filter_strength=1.0, adaptive_params=True,
                                      # 新增的预处理参数（与smooth_data.py保持一致）
                                      apply_preprocessing=True,
                                      apply_moving_average=True,
                                      moving_avg_window=3,
                                      apply_butterworth=True,
                                      butterworth_cutoff=20,
                                      butterworth_strength=0.05,
                                      apply_normalization=False,
                                      normalization_method='standard'):
    """
    分析不同行为标签下神经元的钙波频次，并生成CSV表格
    
    参数
    ----------
    data_df : pandas.DataFrame
        包含神经元数据和行为标签的DataFrame
    neuron_columns : list of str
        要处理的神经元列名列表
    behavior_col : str, 可选
        行为标签列名，默认为'behavior'
    fs : float, 可选
        采样频率，默认为4.8Hz（钙离子浓度数据的实际采样频率）
    save_path : str, 可选
        CSV文件保存路径，默认为None（不保存）
    filter_strength : float, 可选
        过滤强度系数(0.5-2.0)，控制检测灵敏度
    adaptive_params : bool, 可选
        是否为每个神经元使用自适应参数，默认为True
        
    返回
    -------
    freq_df : pandas.DataFrame
        包含每个神经元在不同行为标签下钙波频次的DataFrame
    """
    # 确保数据中包含行为标签列
    if behavior_col not in data_df.columns:
        print(f"错误: 数据中不包含行为标签列 '{behavior_col}'")
        return pd.DataFrame()
    
    # 获取所有行为类型
    behaviors = data_df[behavior_col].unique()
    print(f"数据中包含 {len(behaviors)} 种行为标签: {behaviors}")
    
    # 准备结果容器
    frequency_data = []
    
    # 遍历所有神经元
    for neuron in neuron_columns:
        print(f"处理神经元 {neuron} 的钙波频次...")
        neuron_data = data_df[neuron].values
        
        # 如果启用自适应参数，为每个神经元生成自定义参数
        custom_params = None
        if adaptive_params:
            print(f"  估计神经元 {neuron} 的最优参数...")
            custom_params = estimate_neuron_params(neuron_data, filter_strength)
        
        # 计算总体频次（整个时间序列）
        total_transients, _ = detect_calcium_transients(
            neuron_data, fs=fs, params=custom_params, filter_strength=filter_strength,
            detect_subpeaks=True,  # 启用子峰检测
            apply_preprocessing=apply_preprocessing,
            apply_moving_average=apply_moving_average,
            moving_avg_window=moving_avg_window,
            apply_butterworth=apply_butterworth,
            butterworth_cutoff=butterworth_cutoff,
            butterworth_strength=butterworth_strength,
            apply_normalization=apply_normalization,
            normalization_method=normalization_method,
            apply_savgol=True  # 默认启用Savgol平滑
        )
        total_time = len(neuron_data) / fs  # 总时间（秒）
        total_freq = len(total_transients) / total_time if total_time > 0 else 0
        
        # 记录神经元的总体频次
        neuron_result = {
            'neuron': neuron,
            'behavior': 'ALL',
            'calcium_events': len(total_transients),
            'duration_seconds': total_time,
            'frequency_hz': total_freq
        }
        frequency_data.append(neuron_result)
        
        # 遍历每种行为标签
        for behavior in behaviors:
            # 获取该行为标签的索引
            behavior_indices = data_df[data_df[behavior_col] == behavior].index
            
            if len(behavior_indices) > 0:
                # 提取该行为下神经元数据
                behavior_data = data_df.loc[behavior_indices, neuron].values
                behavior_time = len(behavior_indices) / fs  # 行为持续时间（秒）
                
                # 检查行为数据长度是否足够进行分析
                if len(behavior_data) < 10:  # 至少需要10个点才能进行有效分析
                    print(f"  警告: 神经元 {neuron} 在行为 '{behavior}' 下的数据点({len(behavior_data)})太少，无法进行有效分析")
                    # 记录没有检测到钙波的结果
                    neuron_behavior_data = {
                        'neuron': neuron,
                        'behavior': behavior,
                        'calcium_events': 0,
                        'duration_seconds': behavior_time,
                        'frequency_hz': 0
                    }
                    frequency_data.append(neuron_behavior_data)
                    continue
                
                # 检测该行为下的钙爆发
                try:
                    behavior_transients, _ = detect_calcium_transients(
                        behavior_data, fs=fs, params=custom_params, filter_strength=filter_strength,
                        detect_subpeaks=True,  # 启用子峰检测
                        apply_preprocessing=apply_preprocessing,
                        apply_moving_average=apply_moving_average,
                        moving_avg_window=moving_avg_window,
                        apply_butterworth=apply_butterworth,
                        butterworth_cutoff=butterworth_cutoff,
                        butterworth_strength=butterworth_strength,
                        apply_normalization=apply_normalization,
                        normalization_method=normalization_method,
                        apply_savgol=True  # 默认启用Savgol平滑
                    )
                    behavior_freq = len(behavior_transients) / behavior_time if behavior_time > 0 else 0
                except Exception as e:
                    print(f"  错误: 无法分析神经元 {neuron} 在行为 '{behavior}' 下的钙波: {str(e)}")
                    behavior_transients = []
                    behavior_freq = 0
                
                # 记录该行为下的频次
                neuron_behavior_data = {
                    'neuron': neuron,
                    'behavior': behavior,
                    'calcium_events': len(behavior_transients),
                    'duration_seconds': behavior_time,
                    'frequency_hz': behavior_freq
                }
                frequency_data.append(neuron_behavior_data)
    
    # 创建DataFrame
    freq_df = pd.DataFrame(frequency_data)
    
    # 如果指定了保存路径，则保存到CSV
    if save_path:
        # 确保保存目录存在
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        freq_df.to_csv(save_path, index=False)
        print(f"成功将神经元钙波频次数据保存到: {save_path}")
    
    return freq_df

def analyze_behavior_total_calcium_frequency(data_df, neuron_columns, behavior_col='behavior', fs=4.8, 
                                           save_path=None, filter_strength=1.0, adaptive_params=True,
                                           # 新增的预处理参数（与smooth_data.py保持一致）
                                           apply_preprocessing=True,
                                           apply_moving_average=True,
                                           moving_avg_window=3,
                                           apply_butterworth=True,
                                           butterworth_cutoff=20,
                                           butterworth_strength=0.05,
                                           apply_normalization=False,
                                           normalization_method='standard'):
    """
    按行为标签统计所有神经元的钙波总次数，忽略神经元个体差异
    
    参数
    ----------
    data_df : pandas.DataFrame
        包含神经元数据和行为标签的DataFrame
    neuron_columns : list of str
        要处理的神经元列名列表
    behavior_col : str, 可选
        行为标签列名，默认为'behavior'
    fs : float, 可选
        采样频率，默认为4.8Hz（钙离子浓度数据的实际采样频率）
    save_path : str, 可选
        CSV文件保存路径，默认为None（不保存）
    filter_strength : float, 可选
        过滤强度系数(0.5-2.0)，控制检测灵敏度
    adaptive_params : bool, 可选
        是否为每个神经元使用自适应参数，默认为True
        
    返回
    -------
    behavior_freq_df : pandas.DataFrame
        包含每种行为下钙波总次数和频率的DataFrame
    """
    # 确保数据中包含行为标签列
    if behavior_col not in data_df.columns:
        print(f"错误: 数据中不包含行为标签列 '{behavior_col}'")
        return pd.DataFrame()
    
    # 获取所有行为类型
    behaviors = data_df[behavior_col].unique()
    print(f"数据中包含 {len(behaviors)} 种行为标签: {behaviors}")
    
    # 准备结果容器
    behavior_frequency_data = []
    
    # 计算总体频次（所有神经元，整个时间序列）
    print("计算总体钙波频次...")
    total_calcium_events = 0
    total_time = len(data_df) / fs  # 总时间（秒）
    
    for neuron in neuron_columns:
        neuron_data = data_df[neuron].values
        
        # 如果启用自适应参数，为每个神经元生成自定义参数
        custom_params = None
        if adaptive_params:
            custom_params = estimate_neuron_params(neuron_data, filter_strength)
        
        # 检测该神经元的钙爆发
        neuron_transients, _ = detect_calcium_transients(
            neuron_data, fs=fs, params=custom_params, filter_strength=filter_strength,
            detect_subpeaks=True,  # 启用子峰检测
            apply_preprocessing=apply_preprocessing,
            apply_moving_average=apply_moving_average,
            moving_avg_window=moving_avg_window,
            apply_butterworth=apply_butterworth,
            butterworth_cutoff=butterworth_cutoff,
            butterworth_strength=butterworth_strength,
            apply_normalization=apply_normalization,
            normalization_method=normalization_method,
            apply_savgol=True  # 默认启用Savgol平滑
        )
        total_calcium_events += len(neuron_transients)
    
    total_freq = total_calcium_events / total_time if total_time > 0 else 0
    
    # 记录总体频次
    total_result = {
        'behavior': 'ALL',
        'total_calcium_events': total_calcium_events,
        'duration_seconds': total_time,
        'frequency_hz': total_freq,
        'neuron_count': len(neuron_columns),
        'avg_frequency_per_neuron': total_freq / len(neuron_columns) if len(neuron_columns) > 0 else 0
    }
    behavior_frequency_data.append(total_result)
    
    print(f"总体统计: {total_calcium_events} 个钙波事件，频率 {total_freq:.4f} Hz")
    
    # 遍历每种行为标签
    for behavior in behaviors:
        print(f"处理行为 '{behavior}' 的钙波频次...")
        
        # 获取该行为标签的索引
        behavior_indices = data_df[data_df[behavior_col] == behavior].index
        
        if len(behavior_indices) == 0:
            print(f"  警告: 行为 '{behavior}' 没有对应的数据点")
            continue
            
        behavior_time = len(behavior_indices) / fs  # 行为持续时间（秒）
        behavior_calcium_events = 0
        
        # 检查行为数据长度是否足够进行分析
        if len(behavior_indices) < 10:  # 至少需要10个点才能进行有效分析
            print(f"  警告: 行为 '{behavior}' 的数据点({len(behavior_indices)})太少，无法进行有效分析")
            # 记录没有检测到钙波的结果
            behavior_result = {
                'behavior': behavior,
                'total_calcium_events': 0,
                'duration_seconds': behavior_time,
                'frequency_hz': 0,
                'neuron_count': len(neuron_columns),
                'avg_frequency_per_neuron': 0
            }
            behavior_frequency_data.append(behavior_result)
            continue
        
        # 统计该行为下所有神经元的钙波事件
        for neuron in neuron_columns:
            # 提取该行为下神经元数据
            behavior_data = data_df.loc[behavior_indices, neuron].values
            
            # 如果启用自适应参数，为每个神经元生成自定义参数
            custom_params = None
            if adaptive_params:
                custom_params = estimate_neuron_params(behavior_data, filter_strength)
            
            # 检测该行为下的钙爆发
            try:
                behavior_transients, _ = detect_calcium_transients(
                    behavior_data, fs=fs, params=custom_params, filter_strength=filter_strength,
                    detect_subpeaks=True,  # 启用子峰检测
                    apply_preprocessing=apply_preprocessing,
                    apply_moving_average=apply_moving_average,
                    moving_avg_window=moving_avg_window,
                    apply_butterworth=apply_butterworth,
                    butterworth_cutoff=butterworth_cutoff,
                    butterworth_strength=butterworth_strength,
                    apply_normalization=apply_normalization,
                    normalization_method=normalization_method,
                    apply_savgol=True  # 默认启用Savgol平滑
                )
                behavior_calcium_events += len(behavior_transients)
            except Exception as e:
                print(f"  错误: 无法分析神经元 {neuron} 在行为 '{behavior}' 下的钙波: {str(e)}")
                continue
        
        # 计算该行为的频率
        behavior_freq = behavior_calcium_events / behavior_time if behavior_time > 0 else 0
        avg_freq_per_neuron = behavior_freq / len(neuron_columns) if len(neuron_columns) > 0 else 0
        
        # 记录该行为下的频次
        behavior_result = {
            'behavior': behavior,
            'total_calcium_events': behavior_calcium_events,
            'duration_seconds': behavior_time,
            'frequency_hz': behavior_freq,
            'neuron_count': len(neuron_columns),
            'avg_frequency_per_neuron': avg_freq_per_neuron
        }
        behavior_frequency_data.append(behavior_result)
        
        print(f"  行为 '{behavior}': {behavior_calcium_events} 个钙波事件，频率 {behavior_freq:.4f} Hz，平均每神经元 {avg_freq_per_neuron:.4f} Hz")
    
    # 创建DataFrame
    behavior_freq_df = pd.DataFrame(behavior_frequency_data)
    
    # 如果指定了保存路径，则保存到CSV
    if save_path:
        # 确保保存目录存在
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        behavior_freq_df.to_csv(save_path, index=False)
        print(f"成功将行为钙波频次数据保存到: {save_path}")
    
    return behavior_freq_df

if __name__ == "__main__":
    """
    从Excel文件加载神经元数据并进行特征提取的示例
    """
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='神经元钙离子特征提取工具')
    parser.add_argument('--data', type=str, default='../datasets/Day6_with_behavior_labels_filled.xlsx',
                        help='数据文件路径，支持.xlsx格式')
    parser.add_argument('--output', type=str, default=None,
                        help='输出目录，不指定则根据数据集名称自动生成')
    parser.add_argument('--filter_strength', type=float, default=1.0,
                        help='过滤强度系数(0.5-2.0)。<1.0更宽松，>1.0更严格。默认为1.0')
    parser.add_argument('--behavior_col', type=str, default='behavior',
                        help='行为标签列名，不指定则不进行行为相关分析')
    
    # 新增预处理参数
    parser.add_argument('--disable_preprocessing', action='store_true',
                        help='禁用预处理（默认启用）')
    parser.add_argument('--disable_moving_average', action='store_true',
                        help='禁用移动平均滤波（默认启用）')
    parser.add_argument('--moving_avg_window', type=int, default=3,
                        help='移动平均窗口大小（默认为3）')
    parser.add_argument('--disable_butterworth', action='store_true',
                        help='禁用Butterworth滤波（默认启用）')
    parser.add_argument('--butterworth_cutoff', type=float, default=20,
                        help='Butterworth滤波器截止频率（默认为20，与smooth_data.py一致）')
    parser.add_argument('--butterworth_strength', type=float, default=0.05,
                        help='Butterworth滤波强度（默认为0.05，与smooth_data.py一致）')
    parser.add_argument('--enable_normalization', action='store_true',
                        help='启用归一化（默认禁用）')
    parser.add_argument('--normalization_method', type=str, default='standard',
                        choices=['standard', 'minmax', 'robust', 'log_standard', 'log_minmax'],
                        help='归一化方法（默认为standard）')
    parser.add_argument('--disable_savgol', action='store_true',
                        help='禁用Savitzky-Golay平滑（默认启用）')
    parser.add_argument('--smooth_window', type=int, default=31,
                        help='Savitzky-Golay平滑窗口大小（默认为31，需为奇数）')
    
    # 新增可视化参数
    parser.add_argument('--visualize_trace', action='store_true',
                        help='生成钙波检测trace可视化图（默认禁用）')
    parser.add_argument('--max_neurons_visualize', type=int, default=None,
                        help='最多可视化的神经元数量（默认为所有神经元）')
    parser.add_argument('--individual_plots', action='store_true', default=True,
                        help='为每个神经元生成独立的trace图（默认启用）')
    parser.add_argument('--disable_individual_plots', action='store_true',
                        help='禁用独立trace图生成')
    parser.add_argument('--plot_layout', type=str, default='separate',
                        choices=['separate', 'subplots'],
                        help='布局方式：separate（独立文件）或subplots（多子图），默认为separate')
    parser.add_argument('--trace_fig_width', type=float, default=None,
                        help='整体Trace图的宽度（英寸），默认使用固定比例')
    parser.add_argument('--trace_fig_height', type=float, default=None,
                        help='整体Trace图的高度（英寸），默认使用固定比例')
    parser.add_argument('--individual_fig_width', type=float, default=None,
                        help='单神经元Trace图的宽度（英寸），默认使用固定比例')
    parser.add_argument('--individual_fig_height', type=float, default=None,
                        help='单神经元Trace图的高度（英寸），默认使用固定比例')
    parser.add_argument('--auto_trace_figsize', action='store_true',
                        help='整体Trace图使用自适应尺寸')
    parser.add_argument('--auto_individual_figsize', action='store_true',
                        help='单神经元Trace图使用自适应尺寸')
    
    args = parser.parse_args()
    
    # 处理禁用参数
    if args.disable_individual_plots:
        args.individual_plots = False
    
    # 如果启用了 individual_plots，自动启用 visualize_trace
    if args.individual_plots:
        args.visualize_trace = True
    
    # 处理图像尺寸参数
    trace_figsize = None
    if args.trace_fig_width is not None or args.trace_fig_height is not None:
        trace_width = args.trace_fig_width if args.trace_fig_width is not None else 60.0
        trace_height = args.trace_fig_height if args.trace_fig_height is not None else 25.0
        trace_figsize = (trace_width, trace_height)
    
    individual_figsize = None
    if args.individual_fig_width is not None or args.individual_fig_height is not None:
        indiv_width = args.individual_fig_width if args.individual_fig_width is not None else 20.0
        indiv_height = args.individual_fig_height if args.individual_fig_height is not None else 6.0
        individual_figsize = (indiv_width, indiv_height)
    
    # 检查文件是否存在
    if os.path.exists(args.data):
        try:
            # 从指定路径加载Excel数据
            print(f"正在从 {args.data} 加载数据...")
            df = pd.read_excel(args.data)
            # 清理列名（去除可能的空格）
            df.columns = [col.strip() for col in df.columns]
            print(f"成功加载数据，共 {len(df)} 行")
            
            # 提取神经元列
            neuron_columns = [col for col in df.columns if col.startswith('n') and col[1:].isdigit()]
            print(f"Detected {len(neuron_columns)} neuron data columns")
            
            # 显示预处理配置
            print(f"\n预处理配置:")
            print(f"  - 预处理: {'启用' if not args.disable_preprocessing else '禁用'}")
            if not args.disable_preprocessing:
                print(f"  - 移动平均滤波: {'启用' if not args.disable_moving_average else '禁用'} (窗口大小: {args.moving_avg_window})")
                print(f"  - Butterworth滤波: {'启用' if not args.disable_butterworth else '禁用'} (截止频率: {args.butterworth_cutoff}, 强度: {args.butterworth_strength})")
                print(f"  - 归一化: {'启用' if args.enable_normalization else '禁用'} (方法: {args.normalization_method})")
            print(f"  - Savitzky-Golay平滑: {'启用' if not args.disable_savgol else '禁用'} (窗口: {args.smooth_window})")
            print(f"  - 过滤强度: {args.filter_strength}")
            
            print(f"\n可视化配置:")
            print(f"  - 独立trace图: {'启用' if args.individual_plots else '禁用'} (默认启用)")
            if args.individual_plots:
                if args.max_neurons_visualize is None:
                    print(f"  - 可视化神经元数: 所有神经元 ({len(neuron_columns)}个)")
                else:
                    print(f"  - 最多可视化神经元数: {args.max_neurons_visualize}")
                if args.auto_individual_figsize:
                    individual_size_text = "自适应"
                elif individual_figsize:
                    individual_size_text = f"{individual_figsize[0]}x{individual_figsize[1]} in"
                else:
                    individual_size_text = "默认 (20x6 in)"
                print(f"  - 单神经元图尺寸: {individual_size_text}")
            if args.auto_trace_figsize:
                trace_size_text = "自适应"
            elif trace_figsize:
                trace_size_text = f"{trace_figsize[0]}x{trace_figsize[1]} in"
            else:
                trace_size_text = "默认 (60x25 in)"
            print(f"  - 整体Trace图尺寸: {trace_size_text}")
            
            # 根据数据文件名生成输出目录
            if args.output is None:
                # 提取数据文件名（不含扩展名）
                data_basename = os.path.basename(args.data)
                dataset_name = os.path.splitext(data_basename)[0]
                output_dir = f"../results/{dataset_name}"
            else:
                output_dir = args.output
            
            print(f"输出目录设置为: {output_dir}")
            
            # 确保保存目录存在
            os.makedirs(output_dir, exist_ok=True)
            
            # 准备文件信息字典
            abs_path = os.path.abspath(args.data)
            try:
                # 计算相对路径
                rel_path = os.path.relpath(abs_path, os.path.dirname(__file__))
            except ValueError:
                # 如果无法计算相对路径，使用绝对路径
                rel_path = abs_path
            
            file_info = {
                'filename': data_basename,
                'abs_path': abs_path,
                'rel_path': rel_path
            }
            
            # 分析并保存所有钙爆发数据，启用自适应参数
            all_transients_path = f"{output_dir}/all_neurons_transients.xlsx"
            all_transients, next_id = analyze_all_neurons_transients(
                df, neuron_columns, save_path=all_transients_path, adaptive_params=True,
                file_info=file_info, filter_strength=args.filter_strength,
                # 传递预处理参数
                apply_preprocessing=not args.disable_preprocessing,
                apply_moving_average=not args.disable_moving_average,
                moving_avg_window=args.moving_avg_window,
                apply_butterworth=not args.disable_butterworth,
                butterworth_cutoff=args.butterworth_cutoff,
                butterworth_strength=args.butterworth_strength,
                apply_normalization=args.enable_normalization,
                normalization_method=args.normalization_method,
                apply_savgol=not args.disable_savgol,
                # 传递可视化参数
                visualize_trace=args.visualize_trace,
                trace_save_path=None,  # 自动生成路径
                max_neurons_visualize=args.max_neurons_visualize,
                individual_plots=args.individual_plots,
                plot_layout=args.plot_layout,
                trace_figsize=trace_figsize,
                individual_figsize=individual_figsize,
                auto_trace_figsize=args.auto_trace_figsize,
                auto_individual_figsize=args.auto_individual_figsize
            )
            print(f"\n共检测到 {len(all_transients)} 个钙爆发")
            
            # 输出文件位置总结
            print(f"\n" + "="*70)
            print("生成的文件：")
            print(f"  1. 钙波数据: {os.path.abspath(all_transients_path)}")
            if args.individual_plots:
                individual_traces_dir = os.path.join(os.path.dirname(all_transients_path), "individual_traces")
                print(f"  2. 独立trace图: {os.path.abspath(individual_traces_dir)}/")
            print("="*70)
            
            # 如果指定了行为标签列，则进行行为相关分析
            if args.behavior_col and args.behavior_col in df.columns:
                print(f"\n执行行为相关分析，使用行为标签列: {args.behavior_col}")
                
                # 1. 原始的按神经元分别计算的频率分析
                behavior_freq_path = f"{output_dir}/behavior_calcium_frequency.csv"
                freq_df = analyze_behavior_calcium_frequency(
                    df, neuron_columns, behavior_col=args.behavior_col, fs=4.8,
                    save_path=behavior_freq_path, filter_strength=args.filter_strength,
                    # 传递预处理参数
                    apply_preprocessing=not args.disable_preprocessing,
                    apply_moving_average=not args.disable_moving_average,
                    moving_avg_window=args.moving_avg_window,
                    apply_butterworth=not args.disable_butterworth,
                    butterworth_cutoff=args.butterworth_cutoff,
                    butterworth_strength=args.butterworth_strength,
                    apply_normalization=args.enable_normalization,
                    normalization_method=args.normalization_method
                )
                print(f"成功生成按神经元的行为钙波频次分析结果")
                
                # 2. 新的按行为统计总钙波次数的分析
                behavior_total_freq_path = f"{output_dir}/behavior_total_calcium_frequency.csv"
                total_freq_df = analyze_behavior_total_calcium_frequency(
                    df, neuron_columns, behavior_col=args.behavior_col, fs=4.8,
                    save_path=behavior_total_freq_path, filter_strength=args.filter_strength,
                    # 传递预处理参数
                    apply_preprocessing=not args.disable_preprocessing,
                    apply_moving_average=not args.disable_moving_average,
                    moving_avg_window=args.moving_avg_window,
                    apply_butterworth=not args.disable_butterworth,
                    butterworth_cutoff=args.butterworth_cutoff,
                    butterworth_strength=args.butterworth_strength,
                    apply_normalization=args.enable_normalization,
                    normalization_method=args.normalization_method
                )
                print(f"成功生成按行为统计的总钙波频次分析结果")
            else:
                if args.behavior_col:
                    print(f"\n警告: 指定的行为标签列 '{args.behavior_col}' 不存在于数据中，跳过行为相关分析")
                else:
                    print("\n未指定行为标签列，跳过行为相关分析")
            
        except Exception as e:
            print(f"加载或处理数据时出错: {str(e)}")
            import traceback
            traceback.print_exc()
    else:
        print(f"错误: 找不到数据文件 '{args.data}'，请检查文件路径")