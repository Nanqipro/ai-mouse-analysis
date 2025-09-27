import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, savgol_filter, peak_widths
from scipy.optimize import curve_fit
from scipy.integrate import trapezoid
from scipy import signal
import os
import json
import base64
from io import BytesIO
from typing import List, Dict, Any, Tuple, Optional

def detect_calcium_transients(data, fs=4.8, params=None):
    """
    检测钙成像数据中的钙瞬变。
    所有参数通过'params'字典传递。
    此版本使用导数检测起始点，使用指数拟合检测结束点。
    """
    default_params = {
        'min_snr': 3.5, 'min_duration': 12, 'smooth_window': 31, 'peak_distance': 24,
        'baseline_percentile': 8, 'max_duration': 800, 'detect_subpeaks': False,
        'subpeak_prominence': 0.15, 'subpeak_width': 5, 'subpeak_distance': 8,
        'min_morphology_score': 0.20, 'min_exp_decay_score': 0.12, 'filter_strength': 1.0,
        'start_deriv_threshold_sd': 4.0, 'end_exp_fit_factor_m': 3.5
    }
    if params is None:
        params = {}
    
    run_params = default_params.copy()
    run_params.update(params)

    # 使用局部变量以便于访问
    p = run_params
    min_snr, min_duration, smooth_window, peak_distance = p['min_snr'], p['min_duration'], p['smooth_window'], p['peak_distance']
    baseline_percentile, max_duration, filter_strength = p['baseline_percentile'], p['max_duration'], p['filter_strength']
    start_deriv_threshold_sd, end_exp_fit_factor_m = p['start_deriv_threshold_sd'], p['end_exp_fit_factor_m']

    if smooth_window > 1 and smooth_window % 2 == 0:
        smooth_window += 1
    smoothed_data = signal.savgol_filter(data, smooth_window, 3) if smooth_window > 1 else data.copy()
    
    baseline = np.percentile(smoothed_data, baseline_percentile)
    noise_level = np.std(smoothed_data[smoothed_data < np.percentile(smoothed_data, 50)])
    if noise_level == 0:
        noise_level = 1e-9

    # --- 起始点检测设置 ---
    dF_dt = np.gradient(smoothed_data)
    baseline_deriv_mask = smoothed_data < np.percentile(smoothed_data, 50)
    baseline_derivative = dF_dt[baseline_deriv_mask]
    deriv_mean = np.mean(baseline_derivative)
    deriv_std = np.std(baseline_derivative)
    deriv_threshold = deriv_mean + start_deriv_threshold_sd * deriv_std

    # --- 峰值检测 ---
    threshold = baseline + min_snr * noise_level
    prominence_threshold = noise_level * 1.2 * filter_strength
    min_width_frames = min_duration // 6
    peaks, peak_props = find_peaks(smoothed_data, height=threshold, prominence=prominence_threshold, width=min_width_frames, distance=peak_distance)
    
    if len(peaks) == 0:
        return [], smoothed_data

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
            while start_idx > left_limit and smoothed_data[start_idx] > baseline * 1.1:
                # 检查是否找到了局部最小值或信号开始上升的点
                if start_idx > left_limit + 1:
                    # 检查前一个点是否更低，如果是则继续
                    if smoothed_data[start_idx - 1] < smoothed_data[start_idx]:
                        start_idx -= 1
                    else:
                        # 如果前一个点更高，检查是否找到了合适的起始点
                        # 寻找从当前位置向左的局部最小值
                        search_start = max(left_limit, start_idx - 10)  # 向前搜索最多10个点
                        if search_start < start_idx:
                            local_min_idx = search_start + np.argmin(smoothed_data[search_start:start_idx+1])
                            if smoothed_data[local_min_idx] < smoothed_data[start_idx] * 0.8:  # 如果局部最小值明显更低
                                start_idx = local_min_idx
                        break
                else:
                    break
            
            # 如果搜索后起始点仍然在数据开始且信号很高，尝试找到更好的起始点
            if start_idx == left_limit and smoothed_data[start_idx] > baseline * 1.5:
                # 在峰值前寻找最低点作为起始点
                search_range = min(20, peak_idx - left_limit)  # 搜索范围不超过20个点
                if search_range > 0:
                    min_idx = left_limit + np.argmin(smoothed_data[left_limit:left_limit + search_range])
                    if smoothed_data[min_idx] < smoothed_data[start_idx] * 0.9:  # 如果找到明显更低的点
                        start_idx = min_idx
        else:
            # 对于后续峰值，使用原来的逻辑
            while start_idx > left_limit and smoothed_data[start_idx] > baseline * 1.1:
                start_idx -= 1
        
        # --- 新的结束点检测 ---
        prelim_end_idx = peak_idx
        right_limit = len(smoothed_data) - 1 if i == len(peaks) - 1 else peaks[i+1]
        while prelim_end_idx < right_limit and smoothed_data[prelim_end_idx] > baseline:
            prelim_end_idx += 1
        
        end_idx = prelim_end_idx
        decay_data = smoothed_data[peak_idx:prelim_end_idx]
        if len(decay_data) > 3:
            t_decay = np.arange(len(decay_data))
            try:
                initial_A = smoothed_data[peak_idx] - baseline
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
                    end_idx = min(calculated_end_idx, right_limit, len(smoothed_data) - 1)
            except (RuntimeError, ValueError):
                pass
        
        # 边界最终检查
        if i < len(peaks) - 1 and end_idx >= peaks[i+1]:
            # 如果结束点超过了下一个峰值，使用两个峰值之间的最低点
            end_idx = peak_idx + np.argmin(smoothed_data[peak_idx:peaks[i+1]])
        
        if i > 0 and start_idx <= peaks[i-1]:
            # 如果起始点在前一个峰值之前或等于前一个峰值，使用两个峰值之间的最低点
            valley_idx = peaks[i-1] + np.argmin(smoothed_data[peaks[i-1]:peak_idx])
            start_idx = valley_idx

        duration_frames = end_idx - start_idx
        if not (min_duration <= duration_frames <= max_duration):
            continue
            
        # 特征计算
        amplitude = smoothed_data[peak_idx] - baseline
        widths_info = peak_widths(smoothed_data, [peak_idx], rel_height=0.5)

        transients.append({
            'start': int(start_idx), 
            'peak': int(peak_idx), 
            'end': int(end_idx), 
            'amplitude': float(amplitude),
            'duration': float(duration_frames / fs),
            'fwhm': float(widths_info[0][0] / fs) if len(widths_info[0]) > 0 else np.nan,
            'rise_time': float((peak_idx - start_idx) / fs),
            'decay_time': float((end_idx - peak_idx) / fs),
            'auc': float(trapezoid(smoothed_data[start_idx:end_idx] - baseline, dx=1/fs)),
            'snr': float(amplitude / noise_level)
        })
        
    return transients, smoothed_data

def extract_calcium_features(neuron_data, fs=4.8, visualize=False, params=None):
    """
    提取钙信号特征
    
    Args:
        neuron_data: 神经元数据
        fs: 采样频率，默认4.8Hz
        visualize: 是否生成可视化图表
        params: 参数字典
    
    Returns:
        特征表格、图表对象、平滑数据
    """
    if params is None:
        params = {}
    transients, smoothed_data = detect_calcium_transients(neuron_data, fs=fs, params=params)
    if not transients:
        return pd.DataFrame(), None, smoothed_data
    feature_table = pd.DataFrame(transients)
    if visualize:
        fig = visualize_calcium_transients(neuron_data, smoothed_data, transients, fs=fs)
        return feature_table, fig, smoothed_data
    return feature_table, None, smoothed_data

def visualize_calcium_transients(raw_data, smoothed_data, transients, fs=4.8):
    """
    可视化钙瞬变检测结果
    
    Args:
        raw_data: 原始数据
        smoothed_data: 平滑数据
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
    ax.plot(time_axis, raw_data, color='grey', alpha=0.6, label='原始信号')
    ax.plot(time_axis, smoothed_data, color='blue', label='平滑信号')
    for i, transient in enumerate(transients):
        start_time, peak_time, end_time = transient['start']/fs, transient['peak']/fs, transient['end']/fs
        
        # 绘制黄色事件特征区域（从起始点到结束点）
        ax.axvspan(start_time, end_time, color='yellow', alpha=0.3, label='事件特征区域' if i == 0 else "")
        
        # 标记峰值（红色圆点）
        ax.plot(peak_time, smoothed_data[transient['peak']], 'ro', markersize=6, label='峰值' if i == 0 else "")
        
        # 标记起始点（绿色竖线）
        ax.axvline(x=start_time, color='green', linestyle='--', alpha=0.8, linewidth=2, label='起始点' if i == 0 else "")
        
        # 标记结束点（蓝色竖线）
        ax.axvline(x=end_time, color='blue', linestyle='--', alpha=0.8, linewidth=2, label='结束点' if i == 0 else "")
        
        # 在起始点添加文本标注
        ax.text(start_time, smoothed_data[transient['start']], f'起始{i+1}', 
                fontsize=8, color='green', ha='center', va='bottom')
    ax.set_title("检测到的钙瞬变")
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
        
        # 检测钙瞬变
        transients, smoothed_data = detect_calcium_transients(selected_data, fs, params)
        
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
        
        # 生成可视化
        fig = visualize_calcium_transients(
            selected_data, smoothed_data, transients, fs
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