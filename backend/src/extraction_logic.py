import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, peak_widths
from scipy.optimize import curve_fit
from scipy.integrate import trapezoid
import os
import json
import base64
from io import BytesIO
from typing import List, Dict, Any, Tuple, Optional
import sys

# 导入新的钙波检测算法
# 将element_extraction.py所在目录添加到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
element_extraction_path = os.path.join(project_root, 'element_extraction.py')
if os.path.exists(element_extraction_path):
    sys.path.insert(0, project_root)
    from element_extraction import (
        detect_calcium_transients as new_detect_calcium_transients,
        extract_calcium_features as new_extract_calcium_features,
        preprocess_neural_signal
    )
    USE_NEW_ALGORITHM = True
else:
    USE_NEW_ALGORITHM = False
    print("警告: 未找到element_extraction.py，使用旧算法")

def detect_calcium_transients(data, fs=4.8, params=None):
    """
    检测钙成像数据中的钙瞬变。
    使用新的element_extraction.py算法。
    所有参数通过'params'字典传递。
    """
    if USE_NEW_ALGORITHM:
        # 使用新算法
        default_params = {
            'min_snr': 3.5, 
            'min_duration': 12, 
            'peak_distance': 24,
            'baseline_percentile': 8, 
            'max_duration': 800, 
            'filter_strength': 1.0,
            'smooth_window': 31,
            'min_morphology_score': 0.20,
            'min_exp_decay_score': 0.12,
            'apply_preprocessing': True,
            'apply_moving_average': True,
            'moving_avg_window': 3,
            'apply_butterworth': True,
            'butterworth_cutoff': 20,
            'butterworth_strength': 0.05,
            'apply_normalization': False,
            'normalization_method': 'standard',
            'apply_savgol': True
        }
        if params is None:
            params = {}
        
        run_params = default_params.copy()
        run_params.update(params)
        
        # 调用新算法
        transients, smoothed_data = new_detect_calcium_transients(
            data,
            fs=fs,
            min_snr=run_params.get('min_snr', 3.5),
            min_duration=run_params.get('min_duration', 12),
            smooth_window=run_params.get('smooth_window', 31),
            peak_distance=run_params.get('peak_distance', 24),
            baseline_percentile=run_params.get('baseline_percentile', 8),
            max_duration=run_params.get('max_duration', 800),
            detect_subpeaks=False,
            params=run_params,
            min_morphology_score=run_params.get('min_morphology_score', 0.20),
            min_exp_decay_score=run_params.get('min_exp_decay_score', 0.12),
            filter_strength=run_params.get('filter_strength', 1.0),
            apply_preprocessing=run_params.get('apply_preprocessing', True),
            apply_moving_average=run_params.get('apply_moving_average', True),
            moving_avg_window=run_params.get('moving_avg_window', 3),
            apply_butterworth=run_params.get('apply_butterworth', True),
            butterworth_cutoff=run_params.get('butterworth_cutoff', 20),
            butterworth_strength=run_params.get('butterworth_strength', 0.05),
            apply_normalization=run_params.get('apply_normalization', False),
            normalization_method=run_params.get('normalization_method', 'standard'),
            apply_savgol=run_params.get('apply_savgol', True)
        )
        
        # 转换格式以兼容旧接口
        converted_transients = []
        for t in transients:
            converted_transients.append({
                'start': int(t['start_idx']),
                'peak': int(t['peak_idx']),
                'end': int(t['end_idx']),
                'amplitude': float(t['amplitude']),
                'duration': float(t['duration']),
                'fwhm': float(t['fwhm']) if not np.isnan(t['fwhm']) else np.nan,
                'rise_time': float(t['rise_time']),
                'decay_time': float(t['decay_time']),
                'auc': float(t['auc']),
                'snr': float(t['snr']),
                # 添加新算法特有的字段
                'start_idx': int(t['start_idx']),
                'peak_idx': int(t['peak_idx']),
                'end_idx': int(t['end_idx']),
                'start_time': float(t['start_idx'] / fs),
                'peak_time': float(t['peak_idx'] / fs),
                'end_time': float(t['end_idx'] / fs),
                'peak_value': float(t['peak_value']),
                'baseline': float(t['baseline']),
                'morphology_score': float(t.get('morphology_score', 0)),
                'wave_type': t.get('wave_type', 'simple')
            })
        
        return converted_transients, smoothed_data
    else:
        # 使用旧算法（向后兼容）
        default_params = {
            'min_snr': 3.5, 'min_duration': 12, 'peak_distance': 24,
            'baseline_percentile': 8, 'max_duration': 800, 'filter_strength': 1.0,
            'start_deriv_threshold_sd': 4.0, 'end_exp_fit_factor_m': 3.5
        }
        if params is None:
            params = {}
        
        run_params = default_params.copy()
        run_params.update(params)

        # 使用局部变量以便于访问
        p = run_params
        min_snr, min_duration, peak_distance = p['min_snr'], p['min_duration'], p['peak_distance']
        baseline_percentile, max_duration, filter_strength = p['baseline_percentile'], p['max_duration'], p['filter_strength']
        start_deriv_threshold_sd, end_exp_fit_factor_m = p['start_deriv_threshold_sd'], p['end_exp_fit_factor_m']

        # 直接使用原始数据，不进行平滑处理
        raw_data = data.copy()
        
        # 计算基线和噪声水平（使用原始数据）
        baseline = np.percentile(raw_data, baseline_percentile)
        noise_level = np.std(raw_data[raw_data < np.percentile(raw_data, 50)])
        if noise_level == 0:
            noise_level = 1e-9

        # --- 起始点检测设置（使用原始数据） ---
        dF_dt = np.gradient(raw_data)
        baseline_deriv_mask = raw_data < np.percentile(raw_data, 50)
        baseline_derivative = dF_dt[baseline_deriv_mask]
        deriv_mean = np.mean(baseline_derivative)
        deriv_std = np.std(baseline_derivative)
        deriv_threshold = deriv_mean + start_deriv_threshold_sd * deriv_std

        # --- 峰值检测（使用原始数据） ---
        threshold = baseline + min_snr * noise_level
        prominence_threshold = noise_level * 1.2 * filter_strength
        min_width_frames = min_duration // 6
        peaks, peak_props = find_peaks(raw_data, height=threshold, prominence=prominence_threshold, width=min_width_frames, distance=peak_distance)
        
        if len(peaks) == 0:
            return [], raw_data

        # --- 结束点检测设置 ---
        def exp_decay(t, A, tau, C):
            return A * np.exp(-t / tau) + C
            
        transients = []
        for i, peak_idx in enumerate(peaks):
            # --- 起始点检测：从峰值向左查找 ---
            start_idx = peak_idx
            # 对于第一个峰值，允许搜索到数据开始；对于后续峰值，限制在前一个峰值之后
            left_limit = 0 if i == 0 else peaks[i-1]
            
            # 从峰值向左搜索，寻找信号开始上升的点
            while start_idx > left_limit:
                # 检查导数是否小于阈值（表示信号开始上升）
                if dF_dt[start_idx] < deriv_threshold:
                    break
                start_idx -= 1
            
            # 进一步向左搜索，找到信号真正开始上升的起点
            # 寻找信号值接近基线或开始上升的点
            # 对于第一个峰值，如果信号在数据开始就很高，需要更智能的检测
            if i == 0:
                # 对于第一个峰值，寻找信号真正开始上升的起点
                # 如果当前起始点仍然很高，继续向左搜索直到找到更合适的起始点
                original_start = start_idx
                while start_idx > left_limit and raw_data[start_idx] > baseline * 1.1:
                    # 检查是否找到了局部最小值或信号开始上升的点
                    if start_idx > left_limit + 1:
                        # 检查前一个点是否更低，如果是则继续
                        if raw_data[start_idx - 1] < raw_data[start_idx]:
                            start_idx -= 1
                        else:
                            # 如果前一个点更高，检查是否找到了合适的起始点
                            # 寻找从当前位置向左的局部最小值
                            search_start = max(left_limit, start_idx - 10)  # 向前搜索最多10个点
                            if search_start < start_idx:
                                local_min_idx = search_start + np.argmin(raw_data[search_start:start_idx+1])
                                if raw_data[local_min_idx] < raw_data[start_idx] * 0.8:  # 如果局部最小值明显更低
                                    start_idx = local_min_idx
                            break
                    else:
                        break
                
                # 如果搜索后起始点仍然在数据开始且信号很高，尝试找到更好的起始点
                if start_idx == left_limit and raw_data[start_idx] > baseline * 1.5:
                    # 在峰值前寻找最低点作为起始点
                    search_range = min(20, peak_idx - left_limit)  # 搜索范围不超过20个点
                    if search_range > 0:
                        min_idx = left_limit + np.argmin(raw_data[left_limit:left_limit + search_range])
                        if raw_data[min_idx] < raw_data[start_idx] * 0.9:  # 如果找到明显更低的点
                            start_idx = min_idx
            else:
                # 对于后续峰值，使用原来的逻辑
                while start_idx > left_limit and raw_data[start_idx] > baseline * 1.1:
                    start_idx -= 1
            
            # --- 新的结束点检测 ---
            prelim_end_idx = peak_idx
            right_limit = len(raw_data) - 1 if i == len(peaks) - 1 else peaks[i+1]
            while prelim_end_idx < right_limit and raw_data[prelim_end_idx] > baseline:
                prelim_end_idx += 1
            
            end_idx = prelim_end_idx
            decay_data = raw_data[peak_idx:prelim_end_idx]
            if len(decay_data) > 3:
                t_decay = np.arange(len(decay_data))
                try:
                    initial_A = raw_data[peak_idx] - baseline
                    initial_tau = max(1.0, len(decay_data) / 2)
                    initial_C = baseline
                    popt, _ = curve_fit(
                        exp_decay, t_decay, decay_data, 
                        p0=(initial_A, initial_tau, initial_C),
                        maxfev=5000,
                        bounds=([0, 1e-9, -np.inf], [np.inf, np.inf, np.inf])
                    )
                    A_fit, tau_fit, C_fit = popt
                    if 0 < tau_fit < len(data):
                        calculated_end_idx = peak_idx + int(end_exp_fit_factor_m * tau_fit)
                        end_idx = min(calculated_end_idx, right_limit, len(raw_data) - 1)
                except (RuntimeError, ValueError):
                    pass
            
            # 边界最终检查
            if i < len(peaks) - 1 and end_idx >= peaks[i+1]:
                # 如果结束点超过了下一个峰值，使用两个峰值之间的最低点
                end_idx = peak_idx + np.argmin(raw_data[peak_idx:peaks[i+1]])
            
            if i > 0 and start_idx <= peaks[i-1]:
                # 如果起始点在前一个峰值之前或等于前一个峰值，使用两个峰值之间的最低点
                valley_idx = peaks[i-1] + np.argmin(raw_data[peaks[i-1]:peak_idx])
                start_idx = valley_idx

            duration_frames = end_idx - start_idx
            if not (min_duration <= duration_frames <= max_duration):
                continue
                
            # 特征计算（使用原始数据）
            amplitude = raw_data[peak_idx] - baseline
            widths_info = peak_widths(raw_data, [peak_idx], rel_height=0.5)

            transients.append({
                'start': int(start_idx), 
                'peak': int(peak_idx), 
                'end': int(end_idx), 
                'amplitude': float(amplitude),
                'duration': float(duration_frames / fs),
                'fwhm': float(widths_info[0][0] / fs) if len(widths_info[0]) > 0 else np.nan,
                'rise_time': float((peak_idx - start_idx) / fs),
                'decay_time': float((end_idx - peak_idx) / fs),
                'auc': float(trapezoid(raw_data[start_idx:end_idx] - baseline, dx=1/fs)),
                'snr': float(amplitude / noise_level)
            })
            
        return transients, raw_data

def extract_calcium_features(neuron_data, fs=4.8, visualize=False, params=None):
    """
    提取钙信号特征
    
    Args:
        neuron_data: 神经元数据
        fs: 采样频率，默认4.8Hz
        visualize: 是否生成可视化图表
        params: 参数字典
    
    Returns:
        特征表格、图表对象、原始数据
    """
    if USE_NEW_ALGORITHM and params is not None:
        # 使用新算法
        features, transients = new_extract_calcium_features(
            neuron_data,
            fs=fs,
            visualize=False,
            detect_subpeaks=False,
            params=params,
            filter_strength=params.get('filter_strength', 1.0),
            apply_preprocessing=params.get('apply_preprocessing', True),
            apply_moving_average=params.get('apply_moving_average', True),
            moving_avg_window=params.get('moving_avg_window', 3),
            apply_butterworth=params.get('apply_butterworth', True),
            butterworth_cutoff=params.get('butterworth_cutoff', 20),
            butterworth_strength=params.get('butterworth_strength', 0.05),
            apply_normalization=params.get('apply_normalization', False),
            normalization_method=params.get('normalization_method', 'standard'),
            apply_savgol=params.get('apply_savgol', True)
        )
        
        if not transients:
            return pd.DataFrame(), None, neuron_data if isinstance(neuron_data, np.ndarray) else neuron_data.values
        
        # 转换格式
        feature_table = pd.DataFrame(transients)
        
        # 获取原始数据
        if isinstance(neuron_data, pd.Series):
            raw_data = neuron_data.values
        else:
            raw_data = neuron_data
        
        if visualize:
            fig = visualize_calcium_transients(raw_data, transients, fs=fs)
            return feature_table, fig, raw_data
        return feature_table, None, raw_data
    else:
        # 使用旧算法
        if params is None:
            params = {}
        transients, raw_data = detect_calcium_transients(neuron_data, fs=fs, params=params)
        if not transients:
            return pd.DataFrame(), None, raw_data
        feature_table = pd.DataFrame(transients)
        if visualize:
            fig = visualize_calcium_transients(raw_data, transients, fs=fs)
            return feature_table, fig, raw_data
        return feature_table, None, raw_data

def visualize_calcium_transients(raw_data, transients, fs=4.8):
    """
    可视化钙瞬变检测结果（直接使用原始数据，不进行平滑处理）
    兼容新旧两种数据格式
    
    Args:
        raw_data: 原始数据
        transients: 检测到的瞬变列表
        fs: 采样频率
    
    Returns:
        图表对象
    """
    # 设置中文字体支持
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    plt.rcParams['font.size'] = 10  # 设置字体大小
    
    fig, ax = plt.subplots(figsize=(20, 6))
    time_axis = np.arange(len(raw_data)) / fs
    ax.plot(time_axis, raw_data, color='blue', linewidth=1.5, label='原始信号')
    for i, transient in enumerate(transients):
        # 兼容新旧两种格式
        if 'start_idx' in transient:
            # 新格式：使用 start_idx, peak_idx, end_idx
            start_idx = transient['start_idx']
            peak_idx = transient['peak_idx']
            end_idx = transient['end_idx']
            start_time = transient.get('start_time', start_idx / fs)
            peak_time = transient.get('peak_time', peak_idx / fs)
            end_time = transient.get('end_time', end_idx / fs)
            peak_value = transient.get('peak_value', raw_data[peak_idx])
        else:
            # 旧格式：使用 start, peak, end
            start_idx = transient['start']
            peak_idx = transient['peak']
            end_idx = transient['end']
            start_time = start_idx / fs
            peak_time = peak_idx / fs
            end_time = end_idx / fs
            peak_value = raw_data[peak_idx]
        
        # 确保索引在有效范围内
        start_idx = max(0, min(len(raw_data) - 1, start_idx))
        peak_idx = max(0, min(len(raw_data) - 1, peak_idx))
        end_idx = max(0, min(len(raw_data) - 1, end_idx))
        
        # 绘制黄色事件特征区域（从起始点到结束点）
        ax.axvspan(start_time, end_time, color='yellow', alpha=0.3, label='事件特征区域' if i == 0 else "")
        
        # 标记峰值（红色圆点）
        ax.plot(peak_time, peak_value, 'ro', markersize=6, label='峰值' if i == 0 else "")
        
        # 标记起始点（绿色竖线）
        ax.axvline(x=start_time, color='green', linestyle='--', alpha=0.8, linewidth=2, label='起始点' if i == 0 else "")
        
        # 标记结束点（蓝色竖线）
        ax.axvline(x=end_time, color='blue', linestyle='--', alpha=0.8, linewidth=2, label='结束点' if i == 0 else "")
        
        # 在起始点添加文本标注
        start_value = raw_data[start_idx] if start_idx < len(raw_data) else raw_data[-1]
        ax.text(start_time, start_value, f'起始{i+1}', 
                fontsize=8, color='green', ha='center', va='bottom')
    ax.set_title("检测到的钙瞬变（原始数据）")
    ax.set_xlabel(f"时间 (秒, fs={fs}Hz)")
    ax.set_ylabel("荧光强度 (a.u.)")
    ax.legend()
    return fig

def analyze_all_neurons_transients(data_df, neuron_columns, fs=4.8, start_id=1, file_info=None, params=None):
    """
    分析所有神经元的钙瞬变
    
    Args:
        data_df: 数据DataFrame
        neuron_columns: 神经元列名列表
        fs: 采样频率
        start_id: 起始事件ID
        file_info: 文件信息字典
        params: 参数字典
    
    Returns:
        合并的特征表格和下一个起始ID
    """
    all_transients = []
    for neuron in neuron_columns:
        feature_table, _, _ = extract_calcium_features(data_df[neuron].values, fs=fs, params=params)
        if not feature_table.empty:
            feature_table['neuron_id'] = neuron
            feature_table['event_id'] = range(start_id, start_id + len(feature_table))
            start_id += len(feature_table)
            all_transients.append(feature_table)
    if not all_transients:
        return pd.DataFrame(), start_id
    final_table = pd.concat(all_transients, ignore_index=True)
    if file_info:
        for key, value in file_info.items():
            final_table[key] = value
    return final_table, start_id

def get_interactive_data(file_path: str, neuron_id: str) -> Dict[str, Any]:
    """
    获取用于交互式图表的原始数据
    
    Args:
        file_path: 文件路径
        neuron_id: 神经元ID
    
    Returns:
        包含时间和数据的字典
    """
    try:
        # 读取数据 - 适配element_extraction.py格式（直接读取Excel文件）
        data = pd.read_excel(file_path)
        # 清理列名（去除可能的空格）
        data.columns = [col.strip() for col in data.columns]
        
        if neuron_id not in data.columns:
            raise ValueError(f"神经元 {neuron_id} 不存在")
        
        # 获取神经元数据
        neuron_data = data[neuron_id].values
        
        # 生成时间轴（假设采样频率为4.8Hz）
        time_points = np.arange(len(neuron_data)) / 4.8
        
        return {
            'time': time_points.tolist(),
            'data': neuron_data.tolist(),
            'neuron_id': neuron_id,
            'total_points': len(neuron_data)
        }
        
    except Exception as e:
        raise Exception(f"获取交互式数据失败: {str(e)}")

def detect_from_peak(file_path: str, neuron_id: str, peak_time: float, fs: float = 4.8, params: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    从峰值点自动检测事件特征（起始点、结束点等）
    使用新的element_extraction.py算法
    
    Args:
        file_path: 文件路径
        neuron_id: 神经元ID
        peak_time: 峰值时间（秒）
        fs: 采样频率
        params: 参数字典
    
    Returns:
        检测结果
    """
    try:
        # 读取数据
        data = pd.read_excel(file_path)
        data.columns = [col.strip() for col in data.columns]
        
        if neuron_id not in data.columns:
            raise ValueError(f"神经元 {neuron_id} 不存在")
        
        # 获取神经元数据
        neuron_data = data[neuron_id].values
        
        # 转换峰值时间到索引
        peak_idx = int(peak_time * fs)
        peak_idx = max(0, min(len(neuron_data) - 1, peak_idx))
        
        if params is None:
            params = {}
        
        # 准备参数字典，确保包含所有必要的参数
        detection_params = {
            'min_duration': params.get('min_duration', 12),
            'max_duration': params.get('max_duration', 800),
            'min_snr': params.get('min_snr', 3.5),
            'smooth_window': params.get('smooth_window', 31),
            'peak_distance': params.get('peak_distance', 24),
            'baseline_percentile': params.get('baseline_percentile', 8),
            'filter_strength': params.get('filter_strength', 1.0),
            'min_morphology_score': params.get('min_morphology_score', 0.20),
            'min_exp_decay_score': params.get('min_exp_decay_score', 0.12),
            'apply_preprocessing': params.get('apply_preprocessing', True),
            'apply_moving_average': params.get('apply_moving_average', True),
            'moving_avg_window': params.get('moving_avg_window', 3),
            'apply_butterworth': params.get('apply_butterworth', True),
            'butterworth_cutoff': params.get('butterworth_cutoff', 20),
            'butterworth_strength': params.get('butterworth_strength', 0.05),
            'apply_normalization': params.get('apply_normalization', False),
            'normalization_method': params.get('normalization_method', 'standard'),
            'apply_savgol': params.get('apply_savgol', True)
        }
        
        # 定义最大允许的时间距离（秒），超过这个距离的事件将被忽略
        max_time_distance = params.get('max_time_distance', 5.0)  # 默认5秒，更严格
        max_frame_distance = int(max_time_distance * fs)
        
        # 在点击位置附近的小范围内检测事件（而不是整个数据）
        # 搜索窗口：点击位置前后各 max_time_distance 秒
        search_window_frames = max_frame_distance
        start_search = max(0, peak_idx - search_window_frames)
        end_search = min(len(neuron_data), peak_idx + search_window_frames)
        
        search_data = neuron_data[start_search:end_search]
        
        # 在搜索范围内检测事件
        transients, _ = detect_calcium_transients(search_data, fs=fs, params=detection_params)
        
        # 找到最接近点击峰值的事件
        best_transient = None
        min_distance = float('inf')
        
        for transient in transients:
            # 调整索引到原始数据
            adjusted_peak_idx = transient['peak_idx'] + start_search
            distance = abs(adjusted_peak_idx - peak_idx)
            
            # 只考虑在合理时间范围内的事件（不超过max_time_distance）
            if distance <= max_frame_distance:
                if distance < min_distance:
                    min_distance = distance
                    best_transient = transient.copy()
                    # 调整所有索引到原始数据
                    best_transient['start_idx'] = transient['start_idx'] + start_search
                    best_transient['peak_idx'] = transient['peak_idx'] + start_search
                    best_transient['end_idx'] = transient['end_idx'] + start_search
        
        # 如果仍然没有找到，尝试扩大搜索范围
        if best_transient is None:
            # 扩大搜索范围到前后各10秒
            expanded_time_distance = 10.0
            expanded_frame_distance = int(expanded_time_distance * fs)
            start_search = max(0, peak_idx - expanded_frame_distance)
            end_search = min(len(neuron_data), peak_idx + expanded_frame_distance)
            
            search_data = neuron_data[start_search:end_search]
            
            # 在扩大的范围内检测
            transients, _ = detect_calcium_transients(search_data, fs=fs, params=detection_params)
            
            for transient in transients:
                # 调整索引到原始数据
                adjusted_peak_idx = transient['peak_idx'] + start_search
                distance = abs(adjusted_peak_idx - peak_idx)
                
                # 只考虑在合理时间范围内的事件（不超过expanded_time_distance）
                if distance <= expanded_frame_distance:
                    if distance < min_distance:
                        min_distance = distance
                        best_transient = transient.copy()
                        # 调整所有索引到原始数据
                        best_transient['start_idx'] = transient['start_idx'] + start_search
                        best_transient['peak_idx'] = transient['peak_idx'] + start_search
                        best_transient['end_idx'] = transient['end_idx'] + start_search
            
            # 如果仍然没有找到，手动计算一个简单的事件
            if best_transient is None:
                baseline = np.percentile(neuron_data, detection_params.get('baseline_percentile', 8))
                noise_level = np.std(neuron_data[neuron_data < np.percentile(neuron_data, 50)])
                if noise_level == 0:
                    noise_level = 1e-9
                threshold = baseline + detection_params.get('min_snr', 3.5) * noise_level
                
                # 向左搜索起始点
                start_idx = peak_idx
                while start_idx > 0 and neuron_data[start_idx] > threshold:
                    start_idx -= 1
                
                # 向右搜索结束点
                end_idx = peak_idx
                while end_idx < len(neuron_data) - 1 and neuron_data[end_idx] > threshold:
                    end_idx += 1
                
                # 计算特征
                amplitude = neuron_data[peak_idx] - baseline
                duration = (end_idx - start_idx) / fs
                rise_time = (peak_idx - start_idx) / fs
                decay_time = (end_idx - peak_idx) / fs
                
                try:
                    widths_info = peak_widths(neuron_data, [peak_idx], rel_height=0.5)
                    fwhm = float(widths_info[0][0] / fs) if len(widths_info[0]) > 0 else np.nan
                except:
                    fwhm = np.nan
                
                segment = neuron_data[start_idx:end_idx+1] - baseline
                auc = float(trapezoid(segment, dx=1.0/fs))
                snr = float(amplitude / noise_level)
                
                best_transient = {
                    'start_idx': int(start_idx),
                    'peak_idx': int(peak_idx),
                    'end_idx': int(end_idx),
                    'amplitude': float(amplitude),
                    'duration': float(duration),
                    'fwhm': fwhm,
                    'rise_time': float(rise_time),
                    'decay_time': float(decay_time),
                    'auc': float(auc),
                    'snr': float(snr),
                    'start_time': float(start_idx / fs),
                    'peak_time': float(peak_idx / fs),
                    'end_time': float(end_idx / fs),
                    'peak_value': float(neuron_data[peak_idx]),
                    'baseline': float(baseline),
                    'morphology_score': 0.0,
                    'wave_type': 'simple'
                }
        
        # 验证找到的事件确实在点击位置附近（双重检查）
        if best_transient is not None:
            result_peak_idx = best_transient['peak_idx']
            result_peak_time = result_peak_idx / fs
            clicked_peak_time = peak_idx / fs
            time_distance = abs(result_peak_time - clicked_peak_time)
            
            # 如果时间距离超过10秒，说明检测有误，使用手动计算
            if time_distance > 10.0:
                print(f"警告：检测到的事件距离点击位置过远 ({time_distance:.2f}秒，点击位置: {clicked_peak_time:.2f}秒，检测到: {result_peak_time:.2f}秒)，将使用手动计算")
                # 使用点击位置手动计算事件
                baseline = np.percentile(neuron_data, detection_params.get('baseline_percentile', 8))
                noise_level = np.std(neuron_data[neuron_data < np.percentile(neuron_data, 50)])
                if noise_level == 0:
                    noise_level = 1e-9
                threshold = baseline + detection_params.get('min_snr', 3.5) * noise_level
                
                # 向左搜索起始点
                start_idx = peak_idx
                while start_idx > 0 and neuron_data[start_idx] > threshold:
                    start_idx -= 1
                
                # 向右搜索结束点
                end_idx = peak_idx
                while end_idx < len(neuron_data) - 1 and neuron_data[end_idx] > threshold:
                    end_idx += 1
                
                # 计算特征
                amplitude = neuron_data[peak_idx] - baseline
                duration = (end_idx - start_idx) / fs
                rise_time = (peak_idx - start_idx) / fs
                decay_time = (end_idx - peak_idx) / fs
                
                try:
                    widths_info = peak_widths(neuron_data, [peak_idx], rel_height=0.5)
                    fwhm = float(widths_info[0][0] / fs) if len(widths_info[0]) > 0 else np.nan
                except:
                    fwhm = np.nan
                
                segment = neuron_data[start_idx:end_idx+1] - baseline
                auc = float(trapezoid(segment, dx=1.0/fs))
                snr = float(amplitude / noise_level)
                
                best_transient = {
                    'start_idx': int(start_idx),
                    'peak_idx': int(peak_idx),
                    'end_idx': int(end_idx),
                    'amplitude': float(amplitude),
                    'duration': float(duration),
                    'fwhm': fwhm,
                    'rise_time': float(rise_time),
                    'decay_time': float(decay_time),
                    'auc': float(auc),
                    'snr': float(snr),
                    'start_time': float(start_idx / fs),
                    'peak_time': float(peak_idx / fs),
                    'end_time': float(end_idx / fs),
                    'peak_value': float(neuron_data[peak_idx]),
                    'baseline': float(baseline),
                    'morphology_score': 0.0,
                    'wave_type': 'simple'
                }
        
        # 确保所有值都是Python原生类型
        result = {
            'start_idx': int(best_transient['start_idx']),
            'peak_idx': int(best_transient['peak_idx']),
            'end_idx': int(best_transient['end_idx']),
            'amplitude': float(best_transient['amplitude']),
            'duration': float(best_transient['duration']),
            'fwhm': best_transient.get('fwhm', np.nan) if 'fwhm' in best_transient else np.nan,
            'rise_time': float(best_transient['rise_time']),
            'decay_time': float(best_transient['decay_time']),
            'auc': float(best_transient['auc']),
            'snr': float(best_transient['snr']),
            'start_time': float(best_transient.get('start_time', best_transient['start_idx'] / fs)),
            'peak_time': float(best_transient.get('peak_time', best_transient['peak_idx'] / fs)),
            'end_time': float(best_transient.get('end_time', best_transient['end_idx'] / fs)),
            'peak_value': float(best_transient.get('peak_value', neuron_data[best_transient['peak_idx']])),
            'baseline': float(best_transient.get('baseline', np.percentile(neuron_data, 8))),
            'morphology_score': float(best_transient.get('morphology_score', 0.0)),
            'wave_type': best_transient.get('wave_type', 'simple'),
            'is_manual': True  # 标记为手动添加
        }
        
        # 处理NaN值
        if np.isnan(result['fwhm']):
            result['fwhm'] = None
        
        return {
            'success': True,
            'transient': result
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            'success': False,
            'error': str(e)
        }

def extract_manual_range(file_path: str, neuron_id: str, start_time: float, end_time: float, params: Dict[str, Any]) -> Dict[str, Any]:
    """
    基于用户选择的时间范围进行钙事件提取
    
    Args:
        file_path: 文件路径
        neuron_id: 神经元ID
        start_time: 开始时间（秒）
        end_time: 结束时间（秒）
        params: 参数字典
    
    Returns:
        提取结果
    """
    try:
        # 读取数据
        data = pd.read_excel(file_path)
        
        if neuron_id not in data.columns:
            raise ValueError(f"神经元 {neuron_id} 不存在")
        
        # 获取神经元数据
        neuron_data = data[neuron_id].values
        fs = params.get('fs', 4.8)
        
        # 转换时间到索引
        start_idx = int(start_time * fs)
        end_idx = int(end_time * fs)
        
        # 确保索引在有效范围内
        start_idx = max(0, start_idx)
        end_idx = min(len(neuron_data), end_idx)
        
        if start_idx >= end_idx:
            raise ValueError("无效的时间范围")
        
        # 提取选定范围的数据
        selected_data = neuron_data[start_idx:end_idx]
        
        # 检测钙瞬变（直接使用原始数据，不进行平滑处理）
        transients, raw_data = detect_calcium_transients(selected_data, fs, params)
        
        # 调整瞬变的时间索引（相对于原始数据）
        adjusted_transients = []
        for transient in transients:
            adjusted_transient = {}
            # 确保所有值都是Python原生类型，避免JSON序列化错误
            adjusted_transient['start'] = int(transient['start'] + start_idx)
            adjusted_transient['peak'] = int(transient['peak'] + start_idx)
            adjusted_transient['end'] = int(transient['end'] + start_idx)
            adjusted_transient['amplitude'] = float(transient['amplitude'])
            adjusted_transient['duration'] = float(transient['duration'])
            try:
                fwhm_val = float(transient['fwhm'])
                adjusted_transient['fwhm'] = fwhm_val if not np.isnan(fwhm_val) else None
            except (ValueError, TypeError):
                adjusted_transient['fwhm'] = None
            adjusted_transient['rise_time'] = float(transient['rise_time'])
            adjusted_transient['decay_time'] = float(transient['decay_time'])
            adjusted_transient['auc'] = float(transient['auc'])
            adjusted_transient['snr'] = float(transient['snr'])
            adjusted_transient['start_time'] = float(adjusted_transient['start'] / fs)
            adjusted_transient['peak_time'] = float(adjusted_transient['peak'] / fs)
            adjusted_transient['end_time'] = float(adjusted_transient['end'] / fs)
            adjusted_transients.append(adjusted_transient)
        
        # 生成可视化（使用原始数据）
        fig = visualize_calcium_transients(
            raw_data, transients, fs
        )
        
        # 转换图表为base64
        buffer = BytesIO()
        fig.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        plot_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close(fig)
        
        return {
            'success': True,
            'neuron_id': neuron_id,
            'time_range': {'start': start_time, 'end': end_time},
            'transients_count': len(adjusted_transients),
            'transients': adjusted_transients,
            'features': adjusted_transients,  # 添加features字段以兼容前端显示
            'plot': plot_base64
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

def run_batch_extraction(file_paths, output_dir, fs=4.8, **kwargs):
    """
    批量提取钙信号特征
    
    Args:
        file_paths: 文件路径列表
        output_dir: 输出目录
        fs: 采样频率
        **kwargs: 其他参数
    
    Returns:
        输出文件路径
    """
    all_results = []
    current_event_id = 1
    for file_path in file_paths:
        try:
            # 适配element_extraction.py格式（直接读取Excel文件）
            data_df = pd.read_excel(file_path)
            # 清理列名（去除可能的空格）
            data_df.columns = [col.strip() for col in data_df.columns]
            
            # 提取神经元列（以'n'开头且后面跟数字的列）
            neuron_columns = [col for col in data_df.columns if col.startswith('n') and col[1:].isdigit()]
            
            file_info = {'source_file': os.path.basename(file_path)}
            result_df, next_start_id = analyze_all_neurons_transients(
                data_df=data_df, neuron_columns=neuron_columns, fs=fs,
                start_id=current_event_id, file_info=file_info, params=kwargs
            )
            if not result_df.empty:
                all_results.append(result_df)
                current_event_id = next_start_id
        except Exception as e:
            print(f"处理文件 {os.path.basename(file_path)} 时出错: {e}")
            continue
    if not all_results:
        return None
    final_df = pd.concat(all_results, ignore_index=True)
    output_filename = f"batch_run_{pd.Timestamp.now().strftime('%Y%m%d-%H%M%S')}_features.xlsx"
    output_path = os.path.join(output_dir, output_filename)
    final_df.to_excel(output_path, index=False)
    return output_path