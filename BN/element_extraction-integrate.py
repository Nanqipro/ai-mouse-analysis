import numpy as np
import pandas as pd
from scipy import signal
from scipy.signal import find_peaks, peak_widths, savgol_filter
from numpy import trapezoid
import matplotlib.pyplot as plt
import os
import argparse
import glob
import logging
import datetime
import sys

# 初始化日志记录器
def setup_logger(output_dir=None, prefix="element_extraction-integrate", capture_all_output=True):
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

def detect_calcium_transients(data, fs=0.7, min_snr=3.5, min_duration=12, smooth_window=31, 
                             peak_distance=5, baseline_percentile=8, max_duration=800,
                             detect_subpeaks=False, subpeak_prominence=0.15, 
                             subpeak_width=5, subpeak_distance=8, params=None, 
                             min_morphology_score=0.20, min_exp_decay_score=0.12,
                             filter_strength=1.0):
    """
    检测钙离子浓度数据中的钙爆发(calcium transients)，包括大波中的小波动
    增强对钙爆发形态的过滤，剔除不符合典型钙波特征的信号
    
    参数:
        data: 神经元荧光数据
        fs: 采样频率，默认为0.7Hz
        min_snr: 最小信噪比阈值，默认为3.5
        min_duration: 最小持续时间(采样点数)，默认为12
        smooth_window: 平滑窗口大小，默认为31
        peak_distance: 峰值之间的最小距离，默认为5（已降低以更好地检测相邻钙波）
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
                         
    返回:
        transients: 检测到的钙爆发列表，每个元素包含开始、峰值、结束时间以及其他特征
    """
    
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
    
    # 1. 应用平滑滤波器
    if smooth_window > 1:
        # 确保smooth_window是奇数
        if smooth_window % 2 == 0:
            smooth_window += 1
        smoothed_data = signal.savgol_filter(data, smooth_window, 3)
    else:
        smoothed_data = data.copy()
    
    # 2. 估计基线和噪声水平
    baseline = np.percentile(smoothed_data, baseline_percentile)
    noise_level = np.std(smoothed_data[smoothed_data < np.percentile(smoothed_data, 50)])
    
    # 3. 检测主要峰值 - 使用更强的峰值筛选条件
    threshold = baseline + min_snr * noise_level
    
    # 增加prominence参数来要求峰值必须明显突出于背景
    # 调整prominence_threshold，随filter_strength变化
    prominence_factor = 1.2 * filter_strength  # 与element_extraction.py保持一致的突出度因子
    prominence_threshold = noise_level * prominence_factor
    
    # 增加width参数来过滤太窄的峰值（可能是尖刺噪声）
    min_width = min_duration // 6  # 与element_extraction.py保持一致的最小宽度要求
    
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
            
            # 如果谷值足够深（低于峰值高度的80%），则认为是两个独立的钙波，保留当前峰值
            # 否则，只保留较高的一个峰值
            if valley_depth_ratio < 0.8:  # 与element_extraction.py保持一致的谷值深度要求
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
        ideal_rise_decay_ratio = 0.35  # 放宽理想的上升/衰减时间比例，与element_extraction.py一致
        min_asymmetry = 0.15  # 降低最小非对称度要求，与element_extraction.py一致
        max_non_monotonic = 0.35  # 放宽上升沿非单调性最大允许值，与element_extraction.py一致
        ideal_duration_width_ratio = 3.0  # 调整理想的持续时间/宽度比，与element_extraction.py一致
        min_exp_decay_score = 0.15  # 降低最小指数衰减评分，与element_extraction.py一致
        
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
        min_morphology_score = 0.25  # 进一步降低最低形态评分要求，与element_extraction.py一致
        
        # 考虑总体形态评分和关键特征
        if morphology_score < min_morphology_score:
            continue  # 跳过此峰值，因为总体形态不符合典型钙波特征
        
        # 添加关键特征的检查，但使用较为宽松的条件
        if exp_decay_score < min_exp_decay_score:  # 指数衰减特性是钙波的关键特征
            continue  # 跳过指数衰减不明显的峰值
            
        if rise_decay_ratio > 1.2:  # 放宽上升时间要求，与element_extraction.py一致
            continue  # 跳过上升过慢的峰值
            
        if non_monotonic_ratio > 0.7:  # 放宽非单调性要求，与element_extraction.py一致
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

def extract_calcium_features(neuron_data, fs=0.7, visualize=False, detect_subpeaks=False, params=None, filter_strength=1.0):
    """
    从钙离子浓度数据中提取关键特征
    
    参数
    ----------
    neuron_data : numpy.ndarray 或 pandas.Series
        神经元钙离子浓度时间序列数据
    fs : float, 可选
        采样频率，默认为0.7Hz
    visualize : bool, 可选
        是否可视化结果，默认为False
    detect_subpeaks : bool, 可选
        是否检测大波中的小波峰，默认为False
    params : dict, 可选
        自定义参数字典
    filter_strength : float, 可选
        过滤强度系数(0.5-2.0)，控制所有阈值的整体强度，1.0为默认强度
        
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
    transients, smoothed_data = detect_calcium_transients(data, fs=fs, detect_subpeaks=detect_subpeaks, params=params, filter_strength=filter_strength)
    
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

def visualize_calcium_transients(raw_data, smoothed_data, transients, fs=1.0, max_duration=350):
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
        采样频率，默认为1.0Hz
    max_duration : int, 可选
        钙爆发最大持续时间（采样点数），默认为350
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
        
        # 子峰的半高宽特性
        try:
            sp_widths, _, sp_left_ips, sp_right_ips = peak_widths(
                smoothed_data, [t['peak_idx']], rel_height=0.5
            )
            sp_fwhm = sp_widths[0] / fs
            
            # 找出子峰的边界（局部最小值点）
            # 向左寻找局部最小值
            sp_start = t['peak_idx']
            left_search_limit = max(t['start_idx'], t['peak_idx'] - max_duration//2)
            while sp_start > left_search_limit:
                if sp_start == left_search_limit + 1 or smoothed_data[sp_start] <= smoothed_data[sp_start-1]:
                    break
                sp_start -= 1
            
            # 向右寻找局部最小值
            sp_end = t['peak_idx']
            right_search_limit = min(t['end_idx'], t['peak_idx'] + max_duration//2)
            while sp_end < right_search_limit:
                if sp_end == right_search_limit - 1 or smoothed_data[sp_end] <= smoothed_data[sp_end+1]:
                    break
                sp_end += 1
            
            # 子峰持续时间
            sp_duration = (sp_end - sp_start) / fs
            
            # 上升和衰减时间
            sp_rise_time = (t['peak_idx'] - sp_start) / fs
            sp_decay_time = (sp_end - t['peak_idx']) / fs
            
            # 计算子峰面积
            sp_segment = smoothed_data[sp_start:sp_end+1] - baseline
            sp_auc = trapezoid(sp_segment, dx=1.0/fs)
            
            # 添加子峰信息
            subpeaks = [{
                'index': t['peak_idx'],
                'value': t['peak_value'],
                'amplitude': amplitude,
                'start_idx': sp_start,
                'end_idx': sp_end,
                'duration': sp_duration,
                'fwhm': sp_fwhm,
                'rise_time': sp_rise_time,
                'decay_time': sp_decay_time,
                'auc': sp_auc
            }]
        except Exception as e:
            # 子峰分析失败，跳过该子峰
            pass
        
        # 标记主峰
        ms = t.get('morphology_score', 0.5)
        color = cmap(norm(ms))
        ax2.plot(t['peak_idx']/fs, peak_val, 'o', color=color, markersize=10, 
                label=f'Peak (Score: {ms:.2f})')
        
        # 标记不同高度百分比
        heights = [0.25, 0.5, 0.75]  # 25%, 50%, 75%的高度
        
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

def process_multiple_neurons(data_df, neuron_columns, fs=1.0):
    """
    处理多个神经元的钙离子数据并提取特征
    
    参数
    ----------
    data_df : pandas.DataFrame
        包含多个神经元数据的DataFrame
    neuron_columns : list of str
        要处理的神经元列名列表
    fs : float, 可选
        采样频率，默认为1.0Hz
        
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

def analyze_behavior_specific_features(data_df, neuron_columns, behavior_col='behavior', fs=1.0):
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
        采样频率，默认为1.0Hz
        
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
    # 直接返回与element_extraction.py一致的细粒度参数
    # 这些参数已经针对更细粒度的钙波检测进行了优化
    params = {
        'min_snr': 3.5,                    # 降低信噪比要求，检测更微弱的钙波
        'min_duration': 12,                # 允许更短的钙波事件
        'smooth_window': 31,               # 减少平滑，保留更多细节
        'peak_distance': 5,                # 允许检测更密集的钙波
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

def analyze_all_neurons_transients(data_df, neuron_columns, fs=0.7, save_path=None, adaptive_params=True, start_id=1, 
                            file_info=None, filter_strength=1.0):
    """
    分析所有神经元的钙爆发并为每个爆发分配唯一ID
    
    参数
    ----------
    data_df : pandas.DataFrame
        包含多个神经元数据的DataFrame
    neuron_columns : list of str
        要处理的神经元列名列表
    fs : float, 可选
        采样频率，默认为0.7Hz
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
        
    返回
    -------
    all_transients_df : pandas.DataFrame
        包含所有神经元所有钙爆发特征的DataFrame
    next_id : int
        下一个可用的钙爆发ID
    """
    all_transients = []
    transient_id = start_id  # 使用传入的起始ID
    
    for neuron in neuron_columns:
        print(f"处理神经元 {neuron} 的钙爆发...")
        neuron_data = data_df[neuron].values
        
        # 如果启用自适应参数，为每个神经元生成自定义参数
        custom_params = None
        if adaptive_params:
            print(f"  估计神经元 {neuron} 的最优参数...")
            custom_params = estimate_neuron_params(neuron_data, filter_strength)
        
        # 检测钙爆发
        transients, smoothed_data = detect_calcium_transients(neuron_data, fs=fs, params=custom_params, filter_strength=filter_strength)
        
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
    
    return all_transients_df, transient_id

if __name__ == "__main__":
    """
    从多个Excel文件加载神经元数据并进行特征提取
    """
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='神经元钙离子特征提取工具')
    parser.add_argument('--data', type=str, nargs='+', default=['../datasets/processed_EMtrace.xlsx'],
                        help='数据文件路径列表，支持.xlsx格式，可以指定多个文件')
    parser.add_argument('--output_dir', type=str, default='../results/all_datasets_transients_noCD1',
                        help='输出基础目录，结果将保存在此目录下对应的子文件夹中')
    parser.add_argument('--combine', action='store_true',
                        help='是否将所有文件的结果合并到一个总表中')
    parser.add_argument('--filter_strength', type=float, default=1.0,
                        help='过滤强度系数(0.5-2.0)。<1.0更宽松，>1.0更严格。默认为1.0')
    parser.add_argument('--root_dir', type=str, default=None,
                        help='项目根目录，用于生成相对路径，默认为当前工作目录的上一级')
    parser.add_argument('--log_dir', type=str, default=None,
                        help='日志文件保存目录，默认为输出目录下的logs文件夹')
    args = parser.parse_args()
    
    # 设置日志输出目录
    log_dir = args.log_dir
    if log_dir is None:
        log_dir = os.path.join(args.output_dir, 'logs')
    
    # 初始化日志记录器
    logger = setup_logger(log_dir, "element_extraction-integrate")
    
    # 设置项目根目录，用于计算相对路径
    if args.root_dir:
        root_dir = args.root_dir
    else:
        # 默认使用当前工作目录的上一级
        root_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))
    
    logger.info(f"使用项目根目录: {root_dir}")
    
    # 展开输入文件路径列表（支持通配符）
    input_files = []
    for path in args.data:
        if os.path.isdir(path):
            # 如果是目录，获取目录中的所有Excel文件
            excel_files = glob.glob(os.path.join(path, "*.xlsx"))
            input_files.extend(excel_files)
        elif '*' in path:
            # 如果包含通配符，展开匹配的文件
            matched_files = glob.glob(path)
            input_files.extend(matched_files)
        elif os.path.exists(path):
            # 单个文件
            input_files.append(path)
    
    if not input_files:
        print("错误: 未找到任何匹配的输入文件")
        exit(1)
    
    print(f"共找到 {len(input_files)} 个输入文件:")
    for file in input_files:
        print(f"  - {file}")
    
    # 用于可能的合并结果
    all_datasets_transients = []
    
    # 全局钙爆发ID计数器，确保唯一性
    next_transient_id = 1
    
    # 处理每个输入文件
    for input_file in input_files:
        if not os.path.exists(input_file):
            print(f"警告: 文件 '{input_file}' 不存在，跳过处理")
            continue
            
        try:
            # 从指定路径加载Excel数据
            print(f"\n正在处理文件: {input_file}")
            df = pd.read_excel(input_file)
            # 清理列名（去除可能的空格）
            df.columns = [col.strip() for col in df.columns]
            print(f"成功加载数据，共 {len(df)} 行")
            
            # 提取神经元列
            neuron_columns = [col for col in df.columns if col.startswith('n') and col[1:].isdigit()]
            print(f"检测到 {len(neuron_columns)} 个神经元数据列")
            
            # 提取数据文件名（不含扩展名）
            data_basename = os.path.basename(input_file)
            dataset_name = os.path.splitext(data_basename)[0]
            output_dir = os.path.join(args.output_dir, dataset_name)
            save_path = os.path.join(output_dir, f"{dataset_name}_transients.xlsx")
            
            print(f"输出目录设置为: {output_dir}")
            
            # 确保保存目录存在
            os.makedirs(output_dir, exist_ok=True)
            
            # 准备文件信息字典
            abs_path = os.path.abspath(input_file)
            try:
                # 计算相对路径
                rel_path = os.path.relpath(abs_path, root_dir)
            except ValueError:
                # 如果无法计算相对路径，使用绝对路径
                rel_path = abs_path
            
            file_info = {
                'filename': data_basename,
                'abs_path': abs_path,
                'rel_path': rel_path
            }
            
            print(f"原始文件路径信息:")
            print(f"  - 文件名: {file_info['filename']}")
            print(f"  - 相对路径: {file_info['rel_path']}")
            print(f"  - 绝对路径: {file_info['abs_path']}")
            
            # 分析并保存所有钙爆发数据，启用自适应参数，传递当前的ID计数器和文件信息
            transients_df, next_transient_id = analyze_all_neurons_transients(
                df, neuron_columns, save_path=save_path, adaptive_params=True, 
                start_id=next_transient_id, file_info=file_info,
                filter_strength=args.filter_strength
            )
            
            if len(transients_df) > 0:
                print(f"共检测到 {len(transients_df)} 个钙爆发，已保存到 {save_path}")
                
                # 添加数据集标识，用于合并结果
                if args.combine and not transients_df.empty:
                    transients_df['dataset'] = dataset_name
                    all_datasets_transients.append(transients_df)
            else:
                print("未检测到任何钙爆发")
            
        except Exception as e:
            print(f"处理文件 '{input_file}' 时出错: {str(e)}")
    
    # 如果需要合并所有结果
    if args.combine and all_datasets_transients:
        try:
            combined_df = pd.concat(all_datasets_transients, ignore_index=True)
            combined_output_path = os.path.join(args.output_dir, "all_datasets_transients.xlsx")
            
            # 确认源文件路径列已经包含
            if 'source_file' not in combined_df.columns:
                print("警告: 合并结果中未包含 'source_file' 列")
            if 'source_path' not in combined_df.columns:
                print("警告: 合并结果中未包含 'source_path' 列")
            if 'source_abs_path' not in combined_df.columns:
                print("警告: 合并结果中未包含 'source_abs_path' 列")
                
            # 可选：添加索引列，方便查找
            combined_df['index'] = range(1, len(combined_df) + 1)
            
            # 保存合并结果
            combined_df.to_excel(combined_output_path, index=False)
            print(f"\n已将所有数据集的钙爆发结果合并，共 {len(combined_df)} 条记录")
            print(f"合并结果已保存到: {combined_output_path}")
            
            # 输出列信息以便确认路径信息已正确包含
            print(f"合并表包含以下列:")
            for col in combined_df.columns:
                print(f"  - {col}")
        except Exception as e:
            print(f"合并结果时出错: {str(e)}")